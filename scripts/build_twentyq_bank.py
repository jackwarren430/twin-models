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

Stages, each cached to the output file so a killed run resumes:
  generate  creator rollouts per category with rolling exclusions, high temp
  dedup     normalized + edit-distance-1 (schema.repeat_matches)
  vet       validity judge, FAIL-CLOSED (v7 shipped 21/206 unparseable verdicts
            through fail_open, including the non-word "Okoupee")
  calibrate K real episodes per candidate, base solver vs base answerer
  select    keep lo <= measured win rate <= hi
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs/twentyq-v7-solver-only.yaml")
    ap.add_argument("--rollouts-per-category", type=int, default=80,
                    help="creator draws per category before dedup")
    ap.add_argument("--episodes-per-candidate", type=int, default=8,
                    help="K games used to measure each candidate's win rate")
    ap.add_argument("--band", nargs=2, type=float, default=[0.125, 0.875],
                    metavar=("LO", "HI"),
                    help="keep candidates with LO <= win rate <= HI")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--holdout", type=Path,
                    default=ROOT / "data/twentyq-validation-v2.json",
                    help="secrets that must NOT enter the bank")
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="cap candidates entering calibration (0 = all)")
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "data/twentyq-bank-v1.json")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "tq-runs/bank-v1-calibration.json")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    qcfg = cfg.twentyq
    lo, hi = args.band
    max_turns = qcfg.max_turns

    holdout_meta, holdout = load_validation_secret_set(args.holdout)
    reserved = [s.secret for s in holdout]
    print(f"holding out {len(reserved)} validation secrets", flush=True)

    backend = get_backend(cfg.compute.backend)
    print(f"loading player model: {cfg.model.path}", flush=True)
    base = backend.load_base(cfg.model, cfg.compute)
    backend.seed(args.seed)

    # ---- stage 1: generate ------------------------------------------------
    candidates: list[Secret] = []
    n_parse_fail = 0
    for category in qcfg.categories:
        seen_names: list[str] = []
        for i in range(args.rollouts_per_category):
            # The rank/target dictation is irrelevant here (nothing is being
            # calibrated against it yet) but keeping the real creator prompt
            # means the bank is drawn from the same distribution the creator
            # will later have to reproduce.
            user = creator_secret_user(
                category, rank=i % max(1, qcfg.n_secrets),
                n_secrets=qcfg.n_secrets, difficulty=0.5, target_rate=0.5,
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

    unique = dedup(candidates, reserved)
    print(f"generated {len(candidates)} raw -> {len(unique)} unique "
          f"(holdout-safe)", flush=True)

    # ---- stage 2: vet -----------------------------------------------------
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

    measured: list[dict] = []
    started = time.time()
    for idx, secret in enumerate(vetted):
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
        args.report.write_text(json.dumps({
            "config": str(args.config), "model": cfg.model.path,
            "band": [lo, hi], "episodes_per_candidate": args.episodes_per_candidate,
            "max_turns": max_turns, "question_retries": qcfg.question_retries,
            "seed": args.seed, "n_generated": len(candidates),
            "n_unique": len(unique), "n_vetted": len(vetted),
            "rejected": rejected, "elapsed_s": round(time.time() - started, 1),
            "candidates": measured,
        }, indent=2) + "\n")

    # ---- stage 4: select --------------------------------------------------
    in_band = [m for m in measured if m["in_band"]]
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
