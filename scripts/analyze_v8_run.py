#!/usr/bin/env python3
"""Read a twentyq run's JSONL and report the things that actually decide it.

Deliberately does NOT plot per-iteration `guess%`. A twentyq iteration draws a
fresh set of secrets, so that series largely measures which secrets came up --
three "trends" read off it during v7 had to be retracted. Judge on:

  validation   adapter B vs the FROZEN adapter A, with Wilson intervals. A is
               the control: it cannot learn, so its spread across checkpoints
               is this instrument's empirical noise floor. B rising while A
               stays flat is the result; both drifting is an artifact.
  usable_group_rate
               the fraction of secrets whose K episodes contain both a win and
               a loss. The binding constraint (DESIGN §9), and NOT monotonic in
               win rate -- as the solver improves, secrets leave the band from
               the top and a static bank decays.
  bank drift   per-secret win rates now vs the rates the bank was calibrated
               at. Quantifies that decay directly.

    python scripts/analyze_v8_run.py tq-runs/q-v8-bank-solver.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--bank", type=Path,
                    default=Path("data/twentyq-bank-v1.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.run.read_text().splitlines() if l.strip()]
    meta = next((r for r in rows if "config" in r and "run" in r), {})
    iters = [r for r in rows if "usable_group_rate" in r]
    vals = [r for r in rows if r.get("mode") == "twentyq_validation"]

    print(f"=== {args.run.name} ===")
    if meta:
        print(f"source={meta.get('secret_source')} bank={meta.get('bank_path')} "
              f"credit={meta.get('credit')} retries={meta.get('question_retries')} "
              f"val_eps={meta.get('validation_episodes')}")

    # ---- validation: B vs the frozen control -----------------------------
    if vals:
        print(f"\nVALIDATION  (n per adapter = episodes x secrets; A is FROZEN "
              f"= noise floor)")
        print(f"{'step':>5}  {'A (control)':>22}  {'B (solver)':>22}  {'B-A':>7}")
        a_rates = []
        for v in vals:
            cells = {}
            for name in ("A", "B"):
                m = v["adapters"].get(name)
                if not m:
                    continue
                rate = m["guess_rate"]
                lo, hi = m.get("guess_rate_ci95", (float("nan"),) * 2)
                n = len(m.get("secrets", []))
                cells[name] = (rate, lo, hi, n)
            if "A" in cells:
                a_rates.append(cells["A"][0])
            a = cells.get("A"); b = cells.get("B")
            fa = f"{a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]" if a else "-"
            fb = f"{b[0]:.3f} [{b[1]:.3f},{b[2]:.3f}]" if b else "-"
            diff = f"{b[0]-a[0]:+.3f}" if a and b else "-"
            print(f"{v['step']:>5}  {fa:>22}  {fb:>22}  {diff:>7}")
        if len(a_rates) > 1:
            spread = max(a_rates) - min(a_rates)
            print(f"\n  control spread across {len(a_rates)} checkpoints: "
                  f"{spread:.3f}  (min {min(a_rates):.3f} max {max(a_rates):.3f})")
            print(f"  -> B must beat its step-0 value by MORE than this to mean "
                  f"anything.")

    # ---- gradient supply --------------------------------------------------
    if iters:
        print(f"\nGRADIENT SUPPLY  (usable_group_rate: groups with both a win "
              f"and a loss)")
        vals_u = [r["usable_group_rate"] for r in iters]
        print(f"  iterations {len(iters)}  mean {sum(vals_u)/len(vals_u):.3f}  "
              f"min {min(vals_u):.3f}  max {max(vals_u):.3f}")
        print(f"  v1..v7 secret source measured 0.125 — "
              f"{(sum(vals_u)/len(vals_u))/0.125:.1f}x")
        w = max(1, len(iters) // 6)
        print(f"\n  {'window':>12} {'usable':>8} {'train win':>10} {'fmt/ep':>8}")
        for i in range(0, len(iters), w):
            ch = iters[i:i + w]
            u = sum(r["usable_group_rate"] for r in ch) / len(ch)
            g = sum(r["guess_rate_mean"] for r in ch) / len(ch)
            f = sum(r["episodes"]["format_ended"] for r in ch) / sum(
                r["episodes"]["total"] for r in ch)
            print(f"  {f'{i}-{i+len(ch)-1}':>12} {u:>8.3f} {g:>10.3f} {f:>8.3f}")

    # ---- bank drift -------------------------------------------------------
    if iters and args.bank.exists():
        cal = {s["secret"]: s["measured_guess_rate"]
               for s in json.loads(args.bank.read_text())["secrets"]}
        seen: dict[str, list[float]] = defaultdict(list)
        for r in iters:
            for s in r.get("secrets", []):
                if s.get("guess_rate") is not None:
                    seen[s["secret"]].append(s["guess_rate"])
        rows_d = [(name, cal[name], sum(v) / len(v), len(v))
                  for name, v in seen.items() if name in cal]
        if rows_d:
            drift = sum(now - was for _, was, now, _ in rows_d) / len(rows_d)
            out = sum(1 for _, _, now, _ in rows_d if not 0.0 < now < 1.0)
            print(f"\nBANK DRIFT  ({len(rows_d)} of {len(cal)} secrets played)")
            print(f"  mean win rate now - calibrated: {drift:+.3f}")
            print(f"  secrets now all-win or all-loss: {out}/{len(rows_d)} "
                  f"({out/len(rows_d):.1%})  <- these have stopped teaching")
            movers = sorted(rows_d, key=lambda r: r[2] - r[1], reverse=True)
            print("  biggest risers:  " + ", ".join(
                f"{n} {w:.2f}->{x:.2f}" for n, w, x, _ in movers[:5]))
            print("  biggest fallers: " + ", ".join(
                f"{n} {w:.2f}->{x:.2f}" for n, w, x, _ in movers[-5:]))


if __name__ == "__main__":
    main()
