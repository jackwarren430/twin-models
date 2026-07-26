#!/usr/bin/env python3
"""Read a headroom probe and answer the three questions that decide the run.

1. WHERE DOES THE EPISODE GO? Win / budget-exhausted / format-killed, per arm.
   A format kill is not a loss the policy earned; it is an episode deleted.

2. IS THERE A GRPO GRADIENT AT ALL? A GRPO group is the K episodes played on
   one secret, and its advantages are all zero when the group is all-win or
   all-loss. Per-secret win rates in this task are strongly bimodal (v7: cow
   8/8, penguin 7/8, nearly everything else 0/8), so most groups carry no
   signal whatever the reward says. `usable_group_rate` is the fraction of
   secrets with at least one win AND at least one loss — the real budget of
   trainable episodes, and the number to maximize when choosing a bank.

3. HOW MUCH BUDGET IS THE TURN LIMIT BUYING? Win-rate-at-T is recoverable from
   the winning turn index without replaying anything, so the cost of shrinking
   max_turns (which is the dominant per-iteration cost) can be read directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def group_stats(rows: list[dict]) -> dict:
    by_secret: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_secret[r["secret_id"]].append(r)
    usable = all_win = all_loss = 0
    trainable_episodes = 0
    for eps in by_secret.values():
        wins = sum(e["guessed"] for e in eps)
        if wins == 0:
            all_loss += 1
        elif wins == len(eps):
            all_win += 1
        else:
            usable += 1
            trainable_episodes += len(eps)
    n = max(1, len(by_secret))
    return {
        "n_secrets": len(by_secret),
        "usable_groups": usable,
        "usable_group_rate": round(usable / n, 4),
        "all_loss_groups": all_loss,
        "all_win_groups": all_win,
        "trainable_episodes": trainable_episodes,
        "trainable_episode_rate": round(trainable_episodes / max(1, len(rows)), 4),
    }


def win_rate_at(rows: list[dict], budget: int) -> float:
    """Win rate had the turn budget been `budget`. A win recorded at turn T
    survives any budget >= T; nothing else changes, because a shorter budget
    only removes turns after the win."""
    return sum(r["guessed"] and r["turns"] <= budget for r in rows) / max(1, len(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probes", nargs="+", type=Path)
    ap.add_argument("--budgets", nargs="+", type=int,
                    default=[6, 9, 12, 15, 18, 21])
    args = ap.parse_args()

    for path in args.probes:
        data = json.loads(path.read_text())
        print(f"\n=== {path.name} "
              f"({data.get('model', '?').split('/')[-1][:40]}) ===")
        arms = data.get("arms") or data.get("settings") or []
        hdr = (f"{'arm':<14} {'win':>16} {'early':>7} {'fmt':>7} "
               f"{'budget':>7} {'distinctQ':>9} {'usable_grp':>10} {'trainEp':>8}")
        print(hdr)
        print("-" * len(hdr))
        for arm in arms:
            rows = arm["episodes"]
            n = len(rows)
            w = sum(r["guessed"] for r in rows)
            lo, hi = wilson(w, n)
            g = group_stats(rows)
            budget_end = sum(r["ended"] == "budget" for r in rows) / max(1, n)
            print(f"{arm['label']:<14} "
                  f"{w/n:>6.3f}[{lo:.2f},{hi:.2f}] "
                  f"{arm.get('early_win_rate', float('nan')):>7.3f} "
                  f"{arm.get('format_rate', 0):>7.3f} "
                  f"{budget_end:>7.3f} "
                  f"{arm.get('mean_distinct_question_ratio', float('nan')):>9.3f} "
                  f"{g['usable_group_rate']:>10.3f} "
                  f"{g['trainable_episode_rate']:>8.3f}")

        print(f"\n  win rate vs turn budget "
              f"(what shrinking max_turns would cost):")
        print(f"  {'arm':<14}" + "".join(f"{b:>8}" for b in args.budgets))
        for arm in arms:
            rows = arm["episodes"]
            cells = "".join(f"{win_rate_at(rows, b):>8.3f}" for b in args.budgets)
            print(f"  {arm['label']:<14}{cells}")

        print(f"\n  per-secret win counts (the bimodality):")
        for arm in arms:
            by_secret: dict[str, list[dict]] = defaultdict(list)
            for r in arm["episodes"]:
                by_secret[r["secret"]].append(r)
            won = {s: sum(e["guessed"] for e in eps)
                   for s, eps in by_secret.items()}
            k = len(next(iter(by_secret.values())))
            live = {s: c for s, c in won.items() if c}
            print(f"  {arm['label']:<14} K={k}  "
                  f"nonzero: " + ", ".join(
                      f"{s}={c}" for s, c in sorted(
                          live.items(), key=lambda kv: -kv[1])) or "(none)")


if __name__ == "__main__":
    sys.exit(main())
