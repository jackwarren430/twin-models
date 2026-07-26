#!/usr/bin/env python3
"""Where does the 21-questions win rate actually go? A no-training headroom probe.

Motivation (v7 validation-transcript audit). gemma-4-E2B does not fail at 21
questions by running out of ideas — it fails by LOCKING. Two observed episodes,
both at the base model with no adapter:

    giraffe  turns 16-19: "Is the animal a hoofed ungulate?" x4  -> lost
    penguin  turns 10-19: "Is the animal a parrot?"          x10 -> WON on turn 20

The penguin episode is the tell: the model knew the answer the whole time and
named it the instant the prompt said "this is your LAST turn". Across the
lineage mean_turns_on_success is ~18 of a 21 budget, i.e. essentially every win
is the forced final guess and the turns before it are spent, not used.

That makes the measured 13.5% a floor set by two fixable things rather than a
ceiling set by capability, and this probe separates them:

  dedup   turns that repeat an already-asked question are resampled
          (episode.run_episodes_batched question_retries). Isolates how much
          win rate is lost purely to the lock.
  oracle  a stronger frozen model answers instead of the base model. The v7
          giraffe transcript has the base oracle saying YES to "hoofed mammal"
          and NO to "hoofed ungulate" about the same secret, so some fraction
          of the loss is the environment lying rather than the guesser failing.

Arms are crossed, so the two effects can be read jointly and separately. No
gradients anywhere: every arm is the frozen base model, and `baseline` is a
straight reproduction of the historical validation number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.backends import get_backend  # noqa: E402
from twin.config import Config, ModelConfig  # noqa: E402
from twin.games.twentyq.episode import (  # noqa: E402
    normalize_question,
    run_episodes_batched,
)
from twin.games.twentyq.prompts import (  # noqa: E402
    ANSWERER_SYSTEM,
    GUESSER_SYSTEM,
    answerer_user,
    guesser_user,
)
from twin.games.twentyq.trainer import load_validation_secret_set  # noqa: E402

# label -> (question_retries, oracle key). "self" = the base model answers its
# own secret, exactly as in training; a model id names a separate frozen oracle.
DEFAULT_ARMS = [
    "baseline",
    "dedup",
    "oracle",
    "dedup+oracle",
]

ARM_SPECS = {
    "baseline":     {"question_retries": 0, "oracle": "self"},
    "dedup":        {"question_retries": 4, "oracle": "self"},
    "oracle":       {"question_retries": 0, "oracle": "judge"},
    "dedup+oracle": {"question_retries": 4, "oracle": "judge"},
}


def summarize(rows: list[dict]) -> dict:
    wins = [r for r in rows if r["guessed"]]
    n = max(1, len(rows))
    # A win on the final turn is a forced guess the prompt demanded, not a
    # decision the policy made; tracking it separately keeps "learned to guess
    # when confident" from hiding inside the headline rate.
    forced = [r for r in wins if r["won_on_last_turn"]]
    return {
        "n_episodes": len(rows),
        "guessed": len(wins),
        "guess_rate": round(len(wins) / n, 6),
        "wins_on_last_turn": len(forced),
        "early_wins": len(wins) - len(forced),
        "early_win_rate": round((len(wins) - len(forced)) / n, 6),
        "format_ended": sum(r["ended"] == "format" for r in rows),
        "format_rate": round(sum(r["ended"] == "format" for r in rows) / n, 6),
        "mean_turns": round(sum(r["turns"] for r in rows) / n, 6),
        "mean_turns_on_success": (round(sum(r["turns"] for r in wins) / len(wins), 6)
                                  if wins else None),
        # The lock, measured directly: what fraction of asked questions were
        # content-word duplicates of an earlier one in the same episode.
        "mean_distinct_question_ratio": round(
            sum(r["distinct_ratio"] for r in rows) / n, 6),
        "mean_repeat_turns": round(sum(r["repeat_turns"] for r in rows) / n, 6),
        "mean_retries": round(sum(r["retries"] for r in rows) / n, 6),
        "mean_answer_format_fails": round(
            sum(r["answer_format_fails"] for r in rows) / n, 6),
    }


def episode_row(secret, ep, max_turns: int) -> dict:
    questions = [t.content for t in ep.turns if t.kind == "question"]
    keys = {normalize_question(q, category=secret.category) for q in questions}
    return {
        "secret_id": secret.secret_id,
        "secret": secret.secret,
        "category": secret.category,
        "guessed": ep.guessed,
        "ended": ep.ended,
        "turns": ep.turns_used,
        "won_on_last_turn": bool(ep.guessed and ep.turns_used >= max_turns),
        "n_questions": len(questions),
        "n_distinct_questions": len(keys),
        "distinct_ratio": round(len(keys) / max(1, len(questions)), 6),
        "repeat_turns": ep.n_repeat_turns,
        "retries": ep.n_question_retries,
        "answer_format_fails": ep.n_answer_format_fails,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/twentyq-v7-solver-only.yaml")
    parser.add_argument("--secret-set", type=Path,
                        default=ROOT / "data/twentyq-validation-v2.json")
    parser.add_argument("--episodes-per-secret", type=int, default=8)
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    parser.add_argument("--judge-model", default="Qwen/Qwen3-8B",
                        help="frozen oracle for the 'oracle' arms")
    parser.add_argument("--model", default=None,
                        help="override cfg.model.path (base-model bake-off)")
    parser.add_argument("--max-secrets", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tq-runs/probe-headroom-v1.json")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.model:
        cfg.model.path = args.model
    qcfg = cfg.twentyq
    meta, secrets = load_validation_secret_set(args.secret_set)
    if args.max_secrets:
        secrets = secrets[: args.max_secrets]
    max_turns = qcfg.validation_max_turns

    arms = []
    for label in args.arms:
        if label not in ARM_SPECS:
            raise SystemExit(f"unknown arm {label!r}; known: {sorted(ARM_SPECS)}")
        arms.append({"label": label, **ARM_SPECS[label]})

    backend = get_backend(cfg.compute.backend)
    print(f"loading guesser base: {cfg.model.path}", flush=True)
    base = backend.load_base(cfg.model, cfg.compute)

    judge = None
    if any(a["oracle"] == "judge" for a in arms):
        print(f"loading frozen oracle: {args.judge_model}", flush=True)
        judge = backend.load_base(
            ModelConfig(path=args.judge_model, enable_thinking=False), cfg.compute)

    def make_guesser_batch(category: str):
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

    def make_answerer_batch(secret, which: str):
        model = base if which == "self" else judge
        def fn(requests):
            prompts = [model.render(
                answerer_user(secret.secret, secret.category, q),
                system=ANSWERER_SYSTEM, enable_thinking=qcfg.answerer_thinking)
                for q, _ in requests]
            gens = model.generate_batch(
                prompts, max_tokens=qcfg.answer_max_tokens,
                temp=cfg.gen.oracle_temp, top_p=cfg.gen.top_p,
                completion_batch_size=qcfg.generation_batch_size,
                suppress_thinking=not qcfg.answerer_thinking,
            )
            return [g.text for g in gens]
        return fn

    result = {
        "config": str(args.config),
        "model": str(cfg.model.path),
        "judge_model": args.judge_model,
        "secret_set": meta,
        "episodes_per_secret": args.episodes_per_secret,
        "max_turns": max_turns,
        "seed": args.seed,
        "sampler": {"temp": cfg.gen.solver_temp, "top_p": cfg.gen.top_p},
        "arms": [],
    }

    for arm_index, arm in enumerate(arms):
        # Same seed offset per arm index, so arms differ by intervention only.
        backend.seed(args.seed + arm_index)
        rows: list[dict] = []
        started = time.time()
        for secret in secrets:
            episodes = run_episodes_batched(
                make_guesser_batch(secret.category),
                make_answerer_batch(secret, arm["oracle"]),
                secret,
                n_episodes=args.episodes_per_secret,
                max_turns=max_turns,
                question_retries=arm["question_retries"],
            )
            rows.extend(episode_row(secret, ep, max_turns) for ep in episodes)
            won = sum(e.guessed for e in episodes)
            print(f"[{arm['label']}] {secret.secret}: {won}/{len(episodes)} won",
                  flush=True)
        summary = summarize(rows)
        summary["by_category"] = {
            c: summarize([r for r in rows if r["category"] == c])
            for c in sorted({r["category"] for r in rows})
        }
        summary.update({
            "label": arm["label"],
            "question_retries": arm["question_retries"],
            "oracle": arm["oracle"],
            "wall_seconds": round(time.time() - started, 1),
            "episodes": rows,
        })
        result["arms"].append(summary)
        print(f"== {arm['label']}: guess_rate={summary['guess_rate']:.3f} "
              f"(early {summary['early_win_rate']:.3f}) "
              f"fmt={summary['format_rate']:.3f} "
              f"distinct_q={summary['mean_distinct_question_ratio']:.3f} "
              f"({summary['wall_seconds']:.0f}s)", flush=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
