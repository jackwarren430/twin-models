#!/usr/bin/env python3
"""Sweep base-model guesser sampling (top_k / min_p) on the validation set.

Motivation (exp-fullv6 closeout finding): every sampled rollout in the lineage
silently inherited top_k=64 from gemma-4-E2B's generation_config.json — never
set in a config, never recorded in run metadata. Format fails ran 3-40% in
sampled training rollouts yet 0% under greedy validation at every step, i.e.
they are purely sampling-tail events, so the truncation policy is a live
stability lever (and a diversity/exploration trade-off for GRPO).

This probe plays SAMPLED episodes with the BASE model (no adapters) on the
stationary validation set under a grid of guesser sampler settings. The
answerer is held at the training oracle settings (temp 0.2, config-default
truncation) so only the guesser's sampler varies. Per setting it reports win
rate, format-ended rate, and turn counts, overall and by category.
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
from twin.config import Config  # noqa: E402
from twin.games.twentyq.episode import run_episodes_batched  # noqa: E402
from twin.games.twentyq.prompts import (  # noqa: E402
    ANSWERER_SYSTEM,
    GUESSER_SYSTEM,
    answerer_user,
    guesser_user,
)
from twin.games.twentyq.trainer import load_validation_secret_set  # noqa: E402

# temp/top_p default to the training solver settings; top_k=None/min_p=None
# mean "inherit the model generation_config" (== every historical run);
# top_k=0 disables the top-k warper. min_p arms disable top_k and top_p so the
# min_p truncation is read cleanly, not stacked under two other warpers.
DEFAULT_SETTINGS = [
    "top_k=64",                       # explicit baseline == historical implicit
    "top_k=40",
    "top_k=20",
    "top_k=0",                        # top_p 0.95 only
    "top_k=0,top_p=1.0,min_p=0.05",
    "top_k=0,top_p=1.0,min_p=0.10",
]


def parse_setting(spec: str, *, default_temp: float, default_top_p: float) -> dict:
    setting = {"label": spec, "temp": default_temp, "top_p": default_top_p,
               "top_k": None, "min_p": None}
    for pair in spec.split(","):
        key, value = pair.split("=", 1)
        key = key.strip()
        if key not in ("temp", "top_p", "top_k", "min_p"):
            raise ValueError(f"unknown sampler key {key!r} in {spec!r}")
        setting[key] = int(value) if key == "top_k" else float(value)
    return setting


def summarize(rows: list[dict]) -> dict:
    wins = [r for r in rows if r["guessed"]]
    return {
        "n_episodes": len(rows),
        "guessed": len(wins),
        "guess_rate": round(len(wins) / max(1, len(rows)), 6),
        "format_ended": sum(r["ended"] == "format" for r in rows),
        "format_rate": round(sum(r["ended"] == "format" for r in rows)
                             / max(1, len(rows)), 6),
        "mean_turns": round(sum(r["turns"] for r in rows) / max(1, len(rows)), 6),
        "mean_turns_on_success": (round(sum(r["turns"] for r in wins)
                                        / len(wins), 6) if wins else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/twentyq-full-v6.yaml")
    parser.add_argument("--secret-set", type=Path,
                        default=ROOT / "data/twentyq-validation-v2.json")
    parser.add_argument("--episodes-per-secret", type=int, default=8)
    parser.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS,
                        metavar="KEY=VAL[,KEY=VAL...]")
    parser.add_argument("--max-secrets", type=int, default=0,
                        help="limit secrets for smoke runs (0 = all)")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tq-runs/probe-base-sampling-v1.json")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    qcfg = cfg.twentyq
    settings = [parse_setting(spec, default_temp=cfg.gen.solver_temp,
                              default_top_p=cfg.gen.top_p)
                for spec in args.settings]
    meta, secrets = load_validation_secret_set(args.secret_set)
    if args.max_secrets:
        secrets = secrets[: args.max_secrets]
    max_turns = qcfg.validation_max_turns

    backend = get_backend(cfg.compute.backend)
    print(f"loading {cfg.model.path}", flush=True)
    base = backend.load_base(cfg.model, cfg.compute)

    def make_guesser_batch(category: str, sampler: dict):
        def fn(requests):
            users = [guesser_user(category, qa_pairs, turn_index, max_turns)
                     for qa_pairs, turn_index in requests]
            prompts = [base.render(u, system=GUESSER_SYSTEM,
                                   enable_thinking=qcfg.guesser_thinking)
                       for u in users]
            return base.generate_batch(
                prompts, max_tokens=qcfg.question_max_tokens,
                temp=sampler["temp"], top_p=sampler["top_p"],
                top_k=sampler["top_k"], min_p=sampler["min_p"],
                completion_batch_size=qcfg.generation_batch_size,
            )
        return fn

    def make_answerer_batch(secret):
        def fn(requests):
            prompts = [base.render(
                answerer_user(secret.secret, secret.category, question),
                system=ANSWERER_SYSTEM,
                enable_thinking=qcfg.answerer_thinking) for question, _ in requests]
            gens = base.generate_batch(
                prompts, max_tokens=qcfg.answer_max_tokens,
                temp=cfg.gen.oracle_temp, top_p=cfg.gen.top_p,
                completion_batch_size=qcfg.generation_batch_size,
            )
            return [g.text for g in gens]
        return fn

    result = {
        "config": str(args.config),
        "model": str(cfg.model.path),
        "adapter": None,
        "secret_set": meta,
        "episodes_per_secret": args.episodes_per_secret,
        "max_turns": max_turns,
        "answerer": {"temp": cfg.gen.oracle_temp, "top_p": cfg.gen.top_p},
        "seed": args.seed,
        "settings": [],
    }

    for setting_index, sampler in enumerate(settings):
        backend.seed(args.seed + setting_index)
        rows: list[dict] = []
        started = time.time()
        for secret in secrets:
            episodes = run_episodes_batched(
                make_guesser_batch(secret.category, sampler),
                make_answerer_batch(secret),
                secret,
                n_episodes=args.episodes_per_secret,
                max_turns=max_turns,
            )
            for ep in episodes:
                rows.append({
                    "secret_id": secret.secret_id,
                    "secret": secret.secret,
                    "category": secret.category,
                    "guessed": ep.guessed,
                    "ended": ep.ended,
                    "turns": ep.turns_used,
                })
            print(f"[{sampler['label']}] {secret.secret}: "
                  f"{sum(e.guessed for e in episodes)}/{len(episodes)} won, "
                  f"{sum(e.ended == 'format' for e in episodes)} fmt", flush=True)
        summary = summarize(rows)
        summary["by_category"] = {
            category: summarize([r for r in rows if r["category"] == category])
            for category in sorted({r["category"] for r in rows})
        }
        summary.update({
            "sampler": {k: sampler[k] for k in ("temp", "top_p", "top_k", "min_p")},
            "label": sampler["label"],
            "wall_seconds": round(time.time() - started, 1),
            "episodes": rows,
        })
        result["settings"].append(summary)
        print(f"== {sampler['label']}: guess_rate={summary['guess_rate']:.3f} "
              f"fmt_rate={summary['format_rate']:.3f} "
              f"mean_turns={summary['mean_turns']:.1f} "
              f"({summary['wall_seconds']:.0f}s)", flush=True)
        # Incremental write: a killed probe keeps every completed setting.
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
