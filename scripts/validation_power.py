#!/usr/bin/env python3
"""What effect could this validation set actually have detected?

A null result only means something if the instrument could have seen the effect
in question. v8 is the cautionary case: it reported a flat validation series
against a training gain of +0.148, and those two facts were treated as being in
tension for most of the run. They are not — the instrument's minimum detectable
effect was LARGER than the gain, so a perfect-transfer world and a zero-transfer
world would have produced the same flat series.

The comparison is PAIRED: adapters A and B are scored on the same secrets at
every checkpoint, so the right noise estimate is the spread of the per-secret
difference B-A, not the spread of either rate. That matters enormously here.
Per-secret rates run 0/K to K/K, and in an unpaired calculation all of that
between-secret spread counts as noise; pairing removes it. For v8 the unpaired
formula demands ~188 secrets to resolve +0.05 where the paired one needs ~136,
and the gap widens as the set gets more bimodal.

Two views are reported because they answer different questions and can only
mislead separately:

  ALL   every secret in the set. This is the metric as actually reported. Dead
        secrets contribute an exact zero difference at every checkpoint, which
        shrinks the standard deviation but also dilutes any real effect by the
        live fraction. Both effects are real and they partly cancel.
  LIVE  secrets that are not 0/K for both adapters at every checkpoint. This is
        the instrument's true resolution, and the number to size a new set on,
        because a band-selected set is live BY CONSTRUCTION.

    python scripts/validation_power.py tq-runs/q-v8-bank-solver.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# 80% power, two-sided alpha=0.05.
Z_SUM = 1.96 + 0.8416


def load_validation(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == "validation":
            rows.append(r)
    return sorted(rows, key=lambda r: r["step"])


def per_secret_rates(record: dict) -> dict[str, dict[str, float]]:
    """{adapter: {secret: win rate}} for one checkpoint.

    The validation record stores one row per EPISODE, not per secret, so a
    naive dict-by-name silently keeps only the last of K episodes and reports a
    rate of 0 or 1 for everything."""
    out: dict[str, dict[str, float]] = {}
    for name, metrics in record["adapters"].items():
        agg: dict[str, list[bool]] = defaultdict(list)
        for row in metrics["secrets"]:
            agg[row["secret"]].append(bool(row["guessed"]))
        out[name] = {s: sum(v) / len(v) for s, v in agg.items()}
    return out


def live_secrets(records: list[dict]) -> set[str]:
    """Secrets that are not 0 for BOTH adapters at EVERY checkpoint.

    A secret nobody ever wins carries no information about improvement, in
    exactly the way an all-loss GRPO group carries no gradient."""
    ever: dict[str, bool] = {}
    for rec in records:
        rates = per_secret_rates(rec)
        for name, by_secret in rates.items():
            for s, r in by_secret.items():
                ever[s] = ever.get(s, False) or r > 0.0
    return {s for s, seen in ever.items() if seen}


def paired_sd(records: list[dict], a: str, b: str,
              restrict: set[str] | None = None) -> float:
    """Mean over checkpoints of the per-secret sd of (B - A)."""
    sds = []
    for rec in records:
        rates = per_secret_rates(rec)
        if a not in rates or b not in rates:
            continue
        names = [s for s in rates[b] if restrict is None or s in restrict]
        if len(names) < 2:
            continue
        sds.append(statistics.stdev([rates[b][s] - rates[a][s] for s in names]))
    return statistics.mean(sds) if sds else float("nan")


def mde(sd: float, n: int) -> float:
    """Smallest true difference detectable at 80% power, two-sided 0.05."""
    if n <= 0 or sd != sd:
        return float("nan")
    return Z_SUM * sd / (n ** 0.5)


def n_required(sd: float, effect: float) -> float:
    return (Z_SUM * sd / effect) ** 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--control", default="A", help="frozen adapter")
    ap.add_argument("--solver", default="B")
    ap.add_argument("--effect", type=float, default=None,
                    help="the effect actually in play, e.g. the within-secret "
                         "training gain; reported as detectable or not")
    args = ap.parse_args()

    records = load_validation(args.jsonl)
    if not records:
        print(f"no validation records in {args.jsonl}", file=sys.stderr)
        return 1

    all_secrets = set(per_secret_rates(records[0])[args.solver])
    live = live_secrets(records)
    dead = sorted(all_secrets - live)

    sd_all = paired_sd(records, args.control, args.solver)
    sd_live = paired_sd(records, args.control, args.solver, restrict=live)

    print(f"=== {args.jsonl.name} ===")
    print(f"{len(records)} checkpoints, {len(all_secrets)} secrets, "
          f"{len(live)} live, {len(dead)} dead")
    if dead:
        print(f"  dead: {', '.join(dead)}")
    print(f"\npaired sd of (per-secret {args.solver} - {args.control})")
    print(f"  ALL  {sd_all:.4f}   over {len(all_secrets)} secrets")
    print(f"  LIVE {sd_live:.4f}   over {len(live)} secrets")

    print("\nMINIMUM DETECTABLE EFFECT of this instrument (80% power, p<0.05)")
    m_all = mde(sd_all, len(all_secrets))
    m_live = mde(sd_live, len(live))
    print(f"  ALL  {m_all:+.4f}  <- on the pooled rate as reported")
    print(f"  LIVE {m_live:+.4f}  <- on the live subset")

    if args.effect is not None:
        live_frac = len(live) / max(1, len(all_secrets))
        diluted = args.effect * live_frac
        print(f"\nAgainst a true effect of {args.effect:+.4f} on live secrets "
              f"(and 0 on dead, since they are unwinnable):")
        print(f"  pooled effect would be {args.effect:+.4f} x {live_frac:.2f} "
              f"= {diluted:+.4f}, vs MDE {m_all:+.4f}"
              f"  -> {'DETECTABLE' if diluted > m_all else 'INVISIBLE'}")
        print(f"  live-subset effect      {args.effect:+.4f}"
              f"        vs MDE {m_live:+.4f}"
              f"  -> {'DETECTABLE' if args.effect > m_live else 'INVISIBLE'}")
        if diluted <= m_all and args.effect <= m_live:
            print("\n  A flat series from this instrument is therefore "
                  "UNINFORMATIVE about\n  an effect of that size: perfect "
                  "transfer and zero transfer both\n  produce it. Do not read "
                  "the null as evidence of no transfer.")

    print(f"\nSIZING A NEW SET (all-live by construction, paired, "
          f"sd={sd_live:.4f})")
    print(f"  {'effect':>8} {'secrets needed':>16}")
    for d in (0.15, 0.10, 0.07, 0.05, 0.03):
        print(f"  {d:>+8.2f} {n_required(sd_live, d):>16.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
