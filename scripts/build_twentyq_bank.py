#!/usr/bin/env python3
"""Build a FRONTIER secret bank: secrets the solver wins sometimes, not always.

Why this exists. A GRPO group in this trainer is the K episodes played on ONE
secret, so its advantages are identically zero whenever the group is all-win or
all-loss. The v7 headroom probe measured what the bank actually delivers:

    usable_group_rate = 0.125       (3 of 24 secrets)
    per-secret wins   = cow 8/8, penguin 7/8, television 4/8, bread 1/8,
                        and 0/8 for the other twenty

Per-secret win probability is bimodal — nearly 0 or nearly 1 — so the variance
GRPO needs lives BETWEEN secrets, where it cannot reach it, and not WITHIN a
secret, where it can. Seven eighths of every iteration's compute produced no
gradient at all. That single number explains the flat solver in v5 (49/57
all-loss groups), v6 and v7 without appealing to reward design.

Difficulty calibration was supposed to prevent this and could not: the creator
is asked to hit a target guess rate for an opponent whose actual competence it
cannot observe, and v6 measured obscurity as reward-optimal for it anyway.

So this script measures instead of asking. It generates candidates from the
creator model, vets them, PLAYS them, and keeps only the band where the solver
is genuinely uncertain. The output is a warm-start scaffold, not the end state:
the creator is meant to take this job over (priority two), and the measured
rates here are the supervision target that makes that trainable.

Stages:
  generate  creator rollouts per category with rolling exclusions, high temp
  dedup     normalized + edit-distance-1 (schema.repeat_matches)
  vet       validity judge, FAIL-CLOSED (v7 shipped 21/206 unparseable verdicts
            through fail_open, including the non-word "Okoupee")
  calibrate K real episodes per candidate, base solver vs base answerer
  select    keep lo <= measured win rate <= hi

Calibration costs hours, so the report file doubles as a checkpoint: it is
rewritten once vetting finishes and again after every measured candidate, and
a rerun with --resume (the default) reloads it and measures only what is
missing. The checkpoint carries a signature of everything that would change
the numbers (model, judge, categories, generation settings, max_turns,
question_retries, episodes_per_candidate, seed); a mismatch is reported and
ignored rather than silently merged into a second population. The band is
deliberately NOT part of the signature — selection is cheap and recomputed
from the measured rates, so it can be retuned without remeasuring.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.backends import get_backend  # noqa: E402
from twin.config import Config, ModelConfig  # noqa: E402
from twin.games.twentyq.episode import run_episodes_batched  # noqa: E402
from twin.games.twentyq.judge import judge_secret_validity  # noqa: E402
from twin.games.twentyq.prompts import (  # noqa: E402
    ANSWERER_SYSTEM,
    CREATOR_SYSTEM,
    GUESSER_SYSTEM,
    JUDGE_SYSTEM,
    answerer_user,
    creator_secret_user,
    guesser_user,
)
from twin.games.twentyq.schema import (  # noqa: E402
    Secret,
    SecretParseError,
    normalize_guess,
    parse_secret,
    repeat_matches,
)
from twin.games.twentyq.trainer import load_validation_secret_set  # noqa: E402


def dedup(candidates: list[Secret], reserved: list[str]) -> list[Secret]:
    """Drop exact/near duplicates and anything colliding with the held-out set.

    repeat_matches is the creator repeat gate's matcher: normalized equality
    plus edit-distance-1 for 5+ character strings, on both raw and ASCII-folded
    forms. It is deliberately broader than guess matching — it is what caught
    the v4.5 near-miss evasions ("Okpi" for "Okapi")."""
    kept: list[Secret] = []
    for cand in candidates:
        name = normalize_guess(cand.secret)
        if not name:
            continue
        if any(repeat_matches(cand.secret, r) for r in reserved):
            continue
        if any(repeat_matches(cand.secret, k.secret) for k in kept):
            continue
        kept.append(cand)
    return kept


def signature_of(args, cfg) -> dict:
    """Everything that would change the measured numbers.

    A checkpoint is reusable only when all of it matches: resuming across a
    different player model or turn budget would blend two populations into one
    bank while the file still claims a single set of conditions."""
    qcfg = cfg.twentyq
    return {
        "model": str(cfg.model.path),
        "judge_model": str(args.judge_model),
        "categories": list(qcfg.categories),
        "rollouts_per_category": args.rollouts_per_category,
        "targets": [float(t) for t in args.targets],
        "episodes_per_candidate": args.episodes_per_candidate,
        "max_turns": qcfg.max_turns,
        "question_retries": qcfg.question_retries,
        "holdout": str(args.holdout),
        "seed": args.seed,
    }


def generate_candidates(base, cfg, args) -> tuple[list[Secret], int]:
    """Stage 1: creator rollouts per category, with rolling exclusions."""
    qcfg = cfg.twentyq
    candidates: list[Secret] = []
    n_parse_fail = 0
    for category in qcfg.categories:
        seen_names: list[str] = []
        for i in range(args.rollouts_per_category):
            # Sweep the dictated target, but only over its EASY end. The band
            # is selected by measurement afterwards, so the generator's job is
            # coverage, not accuracy — yet the gap between what is dictated and
            # what comes out is enormous, so most of the scale is dead weight.
            # Measured on the v7 run (196 draws, 174 distinct secrets), which
            # dictated a 90% guess rate throughout:
            #
            #     mean per-secret win rate   0.034
            #     secrets at exactly 0/K     148 of 174
            #     secrets in [0.125, 0.875]  9.2%
            #
            # A dictated 0.9 landed near 0.03. Targets at 0.7 and 0.5 therefore
            # buy nothing but hours of calibration spent confirming 0/8 — they
            # were in this default until the number above was measured. (That
            # run also carried the flat-mode prompt bug: an unconditional
            # "not so obvious ... not so obscure" clause that contradicted its
            # own target. It is fixed, so treat 9.2% as a floor on yield.)
            target_rate = args.targets[i % len(args.targets)]
            user = creator_secret_user(
                category, rank=i % max(1, qcfg.n_secrets),
                n_secrets=qcfg.n_secrets,
                difficulty=round(1.0 - target_rate, 2), target_rate=target_rate,
                recent=seen_names[-qcfg.recent_secret_window:],
                difficulty_mode="flat",
            )
            prompt = base.render(user, system=CREATOR_SYSTEM,
                                 enable_thinking=qcfg.creator_thinking)
            gen = base.generate(
                prompt, max_tokens=qcfg.secret_max_tokens,
                temp=cfg.gen.creator_temp, top_p=cfg.gen.top_p,
                suppress_thinking=not qcfg.creator_thinking,
            )
            try:
                secret = parse_secret(gen.text, default_category=category)
            except SecretParseError:
                n_parse_fail += 1
                continue
            secret.category = category
            candidates.append(secret)
            seen_names.append(secret.secret)
        print(f"  {category}: {len(candidates)} raw so far "
              f"({n_parse_fail} parse fails)", flush=True)
    return candidates, n_parse_fail


def vet_candidates(unique: list[Secret], backend, cfg,
                   args) -> tuple[list[Secret], list[dict]]:
    """Stage 2: validity-judge every candidate, fail-closed.

    The judge is loaded and dropped inside this function on purpose. It is a
    separate multi-GB model, and holding it in the caller's scope would keep it
    resident through the hours of calibration that follow, on a box whose GPU
    and CPU share one memory pool."""
    print(f"loading judge: {args.judge_model}", flush=True)
    judge_model = backend.load_base(
        ModelConfig(path=args.judge_model, enable_thinking=False), cfg.compute)

    def judge_once(question: str, *, terse: bool) -> str:
        # The stock validity prompt says "Think briefly", which under a token
        # cap can spend the whole budget before reaching the verdict line. That
        # is harmless under fail_open and fatal under fail_closed, so the second
        # attempt forbids preamble outright.
        if terse:
            question += ("\n\nReply with ONLY the verdict line and nothing "
                         "else. Do not explain.")
        prompt = judge_model.render(question, system=JUDGE_SYSTEM,
                                    enable_thinking=False)
        return judge_model.generate(
            prompt, max_tokens=384, temp=0.0, top_p=1.0,
            suppress_thinking=True,
        ).text

    n_terse_rescues = 0

    def judge_oracle(question: str) -> str:
        nonlocal n_terse_rescues
        reply = judge_once(question, terse=False)
        if "VERDICT" in (reply or "").upper():
            return reply
        n_terse_rescues += 1
        return judge_once(question, terse=True)

    vetted: list[Secret] = []
    rejected: list[dict] = []
    for secret in unique:
        # FAIL-CLOSED: v7 passed 21/206 unparseable verdicts as VALID under
        # fail_open, and the failures clustered on exactly the corrupted
        # non-word secrets that most needed rejecting.
        result = judge_secret_validity(secret, judge_oracle, mode="fail_closed")
        if result.ok:
            vetted.append(secret)
        else:
            rejected.append({"secret": secret.secret,
                             "category": secret.category,
                             "reason": result.detail})
    vet_rate = len(vetted) / max(1, len(unique))
    print(f"vetted {len(vetted)}/{len(unique)} valid "
          f"({len(rejected)} rejected, {n_terse_rescues} needed a terse retry)",
          flush=True)
    # Calibration is the expensive stage (hours). If vetting collapsed, that is
    # a judge/prompt failure, not a statement about the candidates — stop here
    # rather than spend the night measuring whatever survived.
    if vet_rate < 0.30:
        for r in rejected[:10]:
            print(f"  REJECTED {r['secret']!r}: {r['reason']}", flush=True)
        raise SystemExit(
            f"vet rate {vet_rate:.1%} is implausibly low — the judge is "
            f"probably not emitting parseable verdicts. Inspect the reasons "
            f"above before spending hours on calibration.")

    del judge_model
    gc.collect()
    return vetted, rejected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs/twentyq-v7-solver-only.yaml")
    ap.add_argument("--rollouts-per-category", type=int, default=170,
                    help="creator draws per category before dedup")
    ap.add_argument("--targets", nargs="+", type=float,
                    default=[0.95, 0.9, 0.8],
                    help="dictated guess-rate targets cycled during generation")
    ap.add_argument("--episodes-per-candidate", type=int, default=8,
                    help="K games used to measure each candidate's win rate")
    ap.add_argument("--band", nargs=2, type=float, default=[0.125, 0.875],
                    metavar=("LO", "HI"),
                    help="keep candidates with LO <= win rate <= HI")
    ap.add_argument("--model", default=None,
                    help="override the config's player model (the bake-off "
                         "winner). MUST match the model the bank will be "
                         "trained with — the measured rates describe one "
                         "policy and mean nothing for another.")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--holdout", type=Path,
                    default=ROOT / "data/twentyq-validation-v2.json",
                    help="secrets that must NOT enter the bank")
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="cap candidates entering calibration (0 = all)")
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="reuse a matching --report checkpoint and measure "
                         "only the candidates still missing from it")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "data/twentyq-bank-v1.json")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "tq-runs/bank-v1-calibration.json")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.model:
        cfg.model.path = args.model
    qcfg = cfg.twentyq
    lo, hi = args.band
    max_turns = qcfg.max_turns

    signature = signature_of(args, cfg)
    prior: dict | None = None
    if args.resume and args.report.exists():
        try:
            loaded = json.loads(args.report.read_text())
        except json.JSONDecodeError:
            loaded = {}
        if loaded.get("signature") == signature:
            prior = loaded
        elif loaded:
            print(f"ignoring checkpoint {args.report}: it was written under "
                  f"different conditions (pass --report NEW_PATH to keep both, "
                  f"or --no-resume to overwrite)", flush=True)

    holdout_meta, holdout = load_validation_secret_set(args.holdout)
    reserved = [s.secret for s in holdout]
    print(f"holding out {len(reserved)} validation secrets", flush=True)

    backend = get_backend(cfg.compute.backend)
    print(f"loading player model: {cfg.model.path}", flush=True)
    base = backend.load_base(cfg.model, cfg.compute)
    backend.seed(args.seed)

    # ---- stages 1+2: generate, dedup, vet ---------------------------------
    if prior is not None:
        vetted = [Secret.from_dict(d) for d in prior["vetted"]]
        rejected = list(prior.get("rejected", []))
        n_generated = int(prior.get("n_generated", 0))
        n_unique = int(prior.get("n_unique", 0))
        measured = list(prior.get("candidates", []))
        print(f"resuming from {args.report}: {len(vetted)} vetted candidates, "
              f"{len(measured)} already measured", flush=True)
    else:
        candidates, n_parse_fail = generate_candidates(base, cfg, args)
        unique = dedup(candidates, reserved)
        print(f"generated {len(candidates)} raw ({n_parse_fail} parse fails) "
              f"-> {len(unique)} unique (holdout-safe)", flush=True)
        vetted, rejected = vet_candidates(unique, backend, cfg, args)
        n_generated, n_unique = len(candidates), len(unique)
        measured = []
    if args.max_candidates:
        vetted = vetted[: args.max_candidates]

    # ---- stage 3: calibrate ----------------------------------------------
    def guesser_batch(category: str):
        def fn(requests):
            prompts = [base.render(guesser_user(category, qa, i, max_turns),
                                   system=GUESSER_SYSTEM,
                                   enable_thinking=qcfg.guesser_thinking)
                       for qa, i in requests]
            return base.generate_batch(
                prompts, max_tokens=qcfg.question_max_tokens,
                temp=cfg.gen.solver_temp, top_p=cfg.gen.top_p,
                completion_batch_size=qcfg.generation_batch_size,
                suppress_thinking=not qcfg.guesser_thinking,
            )
        return fn

    def answerer_batch(secret: Secret):
        def fn(requests):
            prompts = [base.render(
                answerer_user(secret.secret, secret.category, q),
                system=ANSWERER_SYSTEM,
                enable_thinking=qcfg.answerer_thinking) for q, _ in requests]
            return [g.text for g in base.generate_batch(
                prompts, max_tokens=qcfg.answer_max_tokens,
                temp=cfg.gen.oracle_temp, top_p=cfg.gen.top_p,
                completion_batch_size=qcfg.generation_batch_size,
                suppress_thinking=not qcfg.answerer_thinking,
            )]
        return fn

    def write_report(elapsed: float) -> None:
        args.report.write_text(json.dumps({
            "signature": signature,
            "config": str(args.config), "model": cfg.model.path,
            "band": [lo, hi],
            "episodes_per_candidate": args.episodes_per_candidate,
            "max_turns": max_turns, "question_retries": qcfg.question_retries,
            "seed": args.seed, "n_generated": n_generated,
            "n_unique": n_unique, "n_vetted": len(vetted),
            "rejected": rejected,
            # The vetted roster is what makes this file a resumable checkpoint
            # rather than a log: without it a rerun would have to regenerate and
            # re-judge every candidate to know what is still unmeasured.
            "vetted": [s.to_dict() for s in vetted],
            "elapsed_s": round(elapsed, 1),
            "candidates": measured,
        }, indent=2) + "\n")

    elapsed_base = float(prior.get("elapsed_s", 0.0)) if prior else 0.0
    started = time.time()
    # Checkpoint generate+vet BEFORE the hours start, so a kill during the very
    # first candidate still costs nothing but that candidate.
    write_report(elapsed_base)
    done = {m["secret"] for m in measured}
    for idx, secret in enumerate(vetted):
        if secret.secret in done:
            continue
        episodes = run_episodes_batched(
            guesser_batch(secret.category), answerer_batch(secret), secret,
            n_episodes=args.episodes_per_candidate,
            max_turns=max_turns,
            # Calibrate under the SAME retry policy training will use, or the
            # measured rate describes a game nobody plays.
            question_retries=qcfg.question_retries,
        )
        wins = sum(e.guessed for e in episodes)
        rate = wins / len(episodes)
        measured.append({
            "secret": secret.secret,
            "category": secret.category,
            "notes": secret.notes,
            "measured_guess_rate": round(rate, 4),
            "wins": wins,
            "n_episodes": len(episodes),
            "format_ended": sum(e.ended == "format" for e in episodes),
            "mean_turns": round(
                sum(e.turns_used for e in episodes) / len(episodes), 3),
            "in_band": bool(lo <= rate <= hi),
        })
        print(f"[{idx + 1}/{len(vetted)}] {secret.secret:<28} "
              f"{wins}/{len(episodes)} "
              f"{'IN' if lo <= rate <= hi else '  '}", flush=True)
        write_report(elapsed_base + time.time() - started)

    # ---- stage 4: select --------------------------------------------------
    # Recomputed from the measured rate rather than read off the stored flag:
    # a resumed checkpoint may have been written under a different --band, and
    # the band is deliberately excluded from the signature so it can be retuned
    # without remeasuring anything.
    in_band = [m for m in measured if lo <= m["measured_guess_rate"] <= hi]
    by_cat: dict[str, int] = defaultdict(int)
    secrets_out = []
    for m in in_band:
        by_cat[m["category"]] += 1
        secrets_out.append({
            "secret_id": f"bank-v1-{m['category'].split()[0]}-"
                         f"{by_cat[m['category']] - 1:03d}",
            "secret": m["secret"],
            "category": m["category"],
            # Difficulty is now MEASURED, not dictated: 1 - the observed win
            # rate. This is the field the creator has to learn to predict.
            "difficulty": round(1.0 - m["measured_guess_rate"], 3),
            "measured_guess_rate": m["measured_guess_rate"],
            "notes": m["notes"],
        })

    args.output.write_text(json.dumps({
        "name": args.output.stem,
        "version": 1,
        "description": (
            f"Frontier bank: creator-generated candidates, judge-vetted "
            f"fail-closed, then CALIBRATED by playing {args.episodes_per_candidate} "
            f"real episodes each with the base model; kept only where the "
            f"measured win rate is in [{lo}, {hi}] so every GRPO group has "
            f"both a win and a loss to compare. Disjoint from "
            f"{holdout_meta.get('name', args.holdout.name)}. "
            f"Rates measured at max_turns={max_turns}, "
            f"question_retries={qcfg.question_retries}; they drift as the "
            f"solver improves and must be re-measured."),
        "source_model": str(cfg.model.path),
        "band": [lo, hi],
        "secrets": secrets_out,
    }, indent=2) + "\n")

    print(f"\nbank: {len(in_band)}/{len(measured)} candidates in band "
          f"[{lo}, {hi}] -> {args.output}")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:<20} {n}")
    print(f"calibration report -> {args.report}")


if __name__ == "__main__":
    main()
