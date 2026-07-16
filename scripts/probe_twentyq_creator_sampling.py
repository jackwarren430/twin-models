#!/usr/bin/env python3
"""Probe TwentyQ creator diversity for one prompt across sampling settings."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.backends import get_backend  # noqa: E402
from twin.config import Config  # noqa: E402
from twin.games.twentyq.prompts import (  # noqa: E402
    CREATOR_SYSTEM,
    creator_secret_user,
)
from twin.games.twentyq.schema import (  # noqa: E402
    SecretParseError,
    normalize_guess,
    parse_secret,
)


SETTINGS = [(0.9, 0.95), (1.1, 0.98), (1.2, 0.98), (1.3, 1.0)]


def summarize(texts: list[str]) -> dict:
    rows = []
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    parse_failures = 0
    for text in texts:
        try:
            secret = parse_secret(text, default_category="food")
        except SecretParseError as exc:
            parse_failures += 1
            rows.append({"parsed": False, "error": str(exc), "completion": text})
            continue
        normalized = normalize_guess(secret.secret)
        counts[normalized] += 1
        display.setdefault(normalized, secret.secret)
        rows.append({
            "parsed": True,
            "secret": secret.secret,
            "normalized": normalized,
            "category": secret.category,
        })
    parsed = len(texts) - parse_failures
    apple = counts["apple"]
    return {
        "n_samples": len(texts),
        "n_parsed": parsed,
        "parse_rate": parsed / len(texts),
        "n_unique": len(counts),
        "unique_rate_among_parsed": len(counts) / parsed if parsed else 0.0,
        "apple_count": apple,
        "apple_rate_among_parsed": apple / parsed if parsed else 0.0,
        "top_secrets": [
            {"secret": display[key], "count": count, "rate": count / parsed}
            for key, count in counts.most_common(15)
        ],
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/twentyq-full-v3.yaml")
    parser.add_argument("--checkpoint", type=Path, default=(
        ROOT / "checkpoints/twentyq-fullv3-ctrl/adapter_A_step30.safetensors"
    ))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--settings", nargs="+",
        default=[f"{temperature}:{top_p}" for temperature, top_p in SETTINGS],
        metavar="TEMP:TOP_P",
    )
    parser.add_argument("--output", type=Path, default=(
        ROOT / "tq-runs/q-fullv3-ctrl-creator-sampling.json"
    ))
    args = parser.parse_args()
    settings = []
    for value in args.settings:
        temperature, top_p = value.split(":", 1)
        settings.append((float(temperature), float(top_p)))

    cfg = Config.from_yaml(args.config)
    backend = get_backend(cfg.compute.backend)
    backend.seed(cfg.train.seed)
    print(f"loading {cfg.model.path}", flush=True)
    base = backend.load_base(cfg.model, cfg.compute)
    adapters = backend.build_adapters(base, cfg.lora)
    adapters.activate("A")

    user = creator_secret_user(
        "food", rank=0, n_secrets=cfg.twentyq.n_secrets,
        difficulty=0.0, target_rate=cfg.rewards.target_hi,
    )
    prompt = base.render(
        user, system=CREATOR_SYSTEM,
        enable_thinking=cfg.twentyq.creator_thinking,
    )
    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "adapter": "A (control creator)",
        "prompt_user": user,
        "samples_per_setting": args.samples,
        "seed_per_setting": args.seed,
        "settings": [],
    }

    for phase in ("initial", "step30"):
        if phase == "step30":
            print(f"loading final checkpoint {args.checkpoint}", flush=True)
            adapters.load("A", str(args.checkpoint))
            adapters.activate("A")
        for temperature, top_p in settings:
            print(
                f"{phase}: temperature={temperature:.1f} top_p={top_p:.2f} "
                f"n={args.samples}",
                flush=True,
            )
            generated = base.generate_batch(
                [prompt] * args.samples,
                max_tokens=cfg.twentyq.secret_max_tokens,
                temp=temperature,
                top_p=top_p,
                seed=args.seed,
                completion_batch_size=args.batch_size,
            )
            summary = summarize([item.text for item in generated])
            summary.update({
                "phase": phase,
                "temperature": temperature,
                "top_p": top_p,
            })
            result["settings"].append(summary)
            print(
                f"  parsed={summary['n_parsed']}/{args.samples} "
                f"unique={summary['n_unique']} "
                f"apple={summary['apple_count']}/{summary['n_parsed']}",
                flush=True,
            )

    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
