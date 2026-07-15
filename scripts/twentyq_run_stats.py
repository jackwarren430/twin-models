#!/usr/bin/env python
"""Aggregate twentyq training-run stats from one or two run JSONL logs.

    python scripts/twentyq_run_stats.py tq-runs/q-full-rot.jsonl \
        [tq-runs/q-full-ctrl.jsonl] [--md OUT.md]

Reads the per-iteration records written by scripts/train_twentyq.py and reports
the experiment metrics: solver success rates, creator difficulty-gradient
adherence (guess-rate-by-rank vs the dictated target ramp), and update sizes
(solver/creator grad-norm, adapter drift). With two logs it prints a side-by-side
first/last + mean comparison (rotation arm vs the swap_interval=0 control).
"""

import argparse
import json
from pathlib import Path
from statistics import mean


def load(path: str):
    """Return (meta, [iteration records]) from a run JSONL."""
    meta, recs = {}, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("type") == "meta":
            meta = r
        else:
            recs.append(r)
    recs.sort(key=lambda r: r.get("iter", 0))
    return meta, recs


def _g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def per_iter_table(recs) -> str:
    hdr = (f"{'it':>3} {'guess%':>6} {'g/tot':>6} {'void':>4} {'unaud':>5} "
           f"{'rgrad':>6} {'s_gnorm':>7} {'c_gnorm':>7} {'driftA':>7} "
           f"{'Rs':>6} {'Rc':>6} {'parse':>5} {'valid':>5} {'swaps':>5} {'mem':>5}")
    lines = [hdr, "-" * len(hdr)]
    for r in recs:
        ep = r.get("episodes", {}) or {}
        lines.append(
            f"{r.get('iter',0):>3} "
            f"{r.get('guess_rate_mean',0):>6.3f} "
            f"{str(ep.get('guessed','?'))+'/'+str(ep.get('total','?')):>6} "
            f"{ep.get('void',0):>4} {ep.get('unauditable',0):>5} "
            f"{r.get('r_gradient',0):>6.3f} "
            f"{_g(r,'solver_update','grad_norm',default=0):>7.3f} "
            f"{_g(r,'creator_update','grad_norm',default=0):>7.3f} "
            f"{_g(r,'adapter_drift','A',default=0):>7.4f} "
            f"{r.get('solver_reward_mean',0):>6.3f} "
            f"{r.get('creator_reward_mean',0):>6.3f} "
            f"{r.get('parse_ok_rate',0):>5.2f} "
            f"{r.get('validity_rate',0):>5.2f} "
            f"{r.get('n_swaps',0):>5} "
            f"{r.get('peak_mem_gb','?'):>5}")
    return "\n".join(lines)


def difficulty_gradient(recs) -> str:
    """Guess-rate per difficulty rank vs the dictated target ramp, per iter.
    Adherence = solve rate should DECLINE across ranks toward target_by_rank."""
    lines = ["  it | guess_rates_by_rank            (target_by_rank)"]
    for r in recs:
        gr = r.get("guess_rates_by_rank")
        tg = r.get("target_by_rank")
        if gr is None:
            continue
        grs = " ".join(f"{x:.2f}" for x in gr)
        tgs = " ".join(f"{x:.2f}" for x in tg) if tg else "-"
        lines.append(f"  {r.get('iter',0):>2} | {grs:30} ({tgs})")
    return "\n".join(lines)


def summarize(name, meta, recs) -> str:
    if not recs:
        return f"### {name}\n(no iteration records)\n"
    first, last = recs[0], recs[-1]
    half = max(1, len(recs) // 2)
    early = recs[:half]
    late = recs[half:]

    def m(rs, *keys):
        vals = [_g(r, *keys) for r in rs if _g(r, *keys) is not None]
        return mean(vals) if vals else float("nan")

    out = [f"### {name}  ({meta.get('run','?')}, swap_interval={meta.get('swap_interval','?')}, "
           f"credit={meta.get('credit','?')}, iters={len(recs)})", ""]
    out.append(f"- solver guess% : first={first.get('guess_rate_mean',0):.3f} "
               f"last={last.get('guess_rate_mean',0):.3f} "
               f"| early-half={m(early,'guess_rate_mean'):.3f} late-half={m(late,'guess_rate_mean'):.3f}")
    out.append(f"- r_gradient    : first={first.get('r_gradient',0):.3f} "
               f"last={last.get('r_gradient',0):.3f} "
               f"| early-half={m(early,'r_gradient'):.3f} late-half={m(late,'r_gradient'):.3f}")
    out.append(f"- solver R      : early-half={m(early,'solver_reward_mean'):.3f} "
               f"late-half={m(late,'solver_reward_mean'):.3f}")
    out.append(f"- creator R     : early-half={m(early,'creator_reward_mean'):.3f} "
               f"late-half={m(late,'creator_reward_mean'):.3f}")
    out.append(f"- solver gnorm  : mean={m(recs,'solver_update','grad_norm'):.3f} "
               f"| creator gnorm mean={m(recs,'creator_update','grad_norm'):.3f}")
    out.append(f"- adapter drift : A last={_g(last,'adapter_drift','A'):.4f} "
               f"B last={_g(last,'adapter_drift','B'):.4f}")
    out.append(f"- parse/valid   : mean parse={m(recs,'parse_ok_rate'):.3f} "
               f"valid={m(recs,'validity_rate'):.3f}")
    out.append(f"- peak_mem_gb   : max={max((r.get('peak_mem_gb') or 0) for r in recs):.1f}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="run JSONL(s): rotation [control]")
    ap.add_argument("--md", default=None, help="also write the report to this markdown file")
    args = ap.parse_args()

    blocks = []
    for i, path in enumerate(args.logs):
        meta, recs = load(path)
        name = f"ARM {chr(65+i)}"
        blocks.append(f"## {name}: {path}\n")
        blocks.append(summarize(name, meta, recs))
        blocks.append("\n#### per-iteration\n```\n" + per_iter_table(recs) + "\n```")
        blocks.append("\n#### difficulty-gradient (guess rate by rank)\n```\n"
                      + difficulty_gradient(recs) + "\n```\n")

    report = "\n".join(blocks)
    print(report)
    if args.md:
        Path(args.md).write_text(report)
        print(f"\n[written to {args.md}]")


if __name__ == "__main__":
    main()
