#!/usr/bin/env python3
"""Verify that the thinking flag is honoured on the REAL model (DESIGN §8).

gemma-4-E2B opens a ``<|channel>thought`` span whether or not the chat
template was rendered with ``enable_thinking=True`` — the template flag only
declines to INVITE reasoning, it cannot forbid it. The fix is a decode-level
ban on the reasoning-open markers, so this probe checks the property that
actually matters end-to-end, per role, on the Spark:

    thinking OFF -> zero reasoning markers, zero think-share
    thinking ON  -> reasoning markers present

It also reports completion length, because suppressed thinking is the
difference between a 400-token deliberation that never reaches the contract
line and a 12-token answer (the q-shakeout-01 failure mode).

    .venv/bin/python -u scripts/probe_thinking_suppression.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.backends import get_backend  # noqa: E402
from twin.config import Config  # noqa: E402
from twin.games.twentyq.prompts import (  # noqa: E402
    ANSWERER_SYSTEM,
    GUESSER_SYSTEM,
    answerer_user,
    guesser_user,
)
from twin.think import THINK_OPEN_MARKERS, strip_think, think_share  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs/twentyq-v7-solver-only.yaml")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "tq-runs/probe-thinking-suppression.json")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    backend = get_backend(cfg.compute.backend)
    print(f"backend={backend.name} loading {cfg.model.path}", flush=True)
    base = backend.load_base(cfg.model, cfg.compute)

    roles = {
        "answerer": (ANSWERER_SYSTEM,
                     answerer_user("cow", "animal", "Is it a mammal?"),
                     cfg.gen.oracle_temp),
        "guesser": (GUESSER_SYSTEM,
                    guesser_user("animal", [("Is it a mammal?", "YES")], 1, 21),
                    cfg.gen.solver_temp),
    }

    result = {"model": str(cfg.model.path), "backend": backend.name,
              "markers": list(THINK_OPEN_MARKERS), "samples": args.samples,
              "rows": []}
    # (label, enable_thinking, suppress_thinking). "historical" is the v1..v6
    # decoding policy — the template declines to invite reasoning and nothing
    # enforces it — kept as an arm so the change this flag makes is measured,
    # not assumed.
    arms = [
        ("off", False, True),
        ("on", True, False),
        ("historical", False, False),
    ]
    for role, (system, user, temp) in roles.items():
        for label, think, suppress in arms:
            backend.seed(7)
            prompt = base.render(user, system=system, enable_thinking=think)
            gens = base.generate_batch(
                [prompt] * args.samples, max_tokens=args.max_tokens,
                temp=temp, top_p=cfg.gen.top_p,
                completion_batch_size=args.samples,
                # "off"/"on" are exactly what BaseTrainer._generate_batch does
                # with the resolved flag: template invites, decoder enforces.
                suppress_thinking=suppress,
            )
            texts = [g.text for g in gens]
            row = {
                "role": role,
                "arm": label,
                "enable_thinking": think,
                "suppress_thinking": suppress,
                "template_has_think_token": "<|think|>" in prompt,
                "n_with_marker": sum(
                    any(m.lower() in t.lower() for m in THINK_OPEN_MARKERS)
                    for t in texts),
                "mean_think_share": round(
                    sum(think_share(t) for t in texts) / max(1, len(texts)), 4),
                "mean_completion_tokens": round(
                    sum(len(g.completion_tokens) for g in gens)
                    / max(1, len(gens)), 1),
                "sample_visible": strip_think(texts[0]).strip()[:160],
                "sample_raw_head": texts[0][:160],
            }
            result["rows"].append(row)
            print(f"[{role}:{label}] enable_thinking={think} "
                  f"suppress={suppress} | markers {row['n_with_marker']}/"
                  f"{args.samples} | think_share={row['mean_think_share']:.3f} "
                  f"| tokens={row['mean_completion_tokens']:.0f}", flush=True)
            print(f"    visible: {row['sample_visible']!r}", flush=True)

    by_arm = {a: [r for r in result["rows"] if r["arm"] == a]
              for a in ("off", "on", "historical")}
    result["verdict"] = {
        "off_is_clean": all(r["n_with_marker"] == 0 for r in by_arm["off"]),
        "on_thinks": all(r["n_with_marker"] > 0 for r in by_arm["on"]),
        "historical_thought_unbidden": any(
            r["n_with_marker"] > 0 for r in by_arm["historical"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nverdict: {result['verdict']}\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
