#!/usr/bin/env python
"""Run the held-out capability benchmark against the base and/or trained adapters.

The absolute-capability complement to ``analyze_run.py`` (which only watches the
self-play loop's internal curves). Scores a fixed, hand-verified item set spanning
math / coding / knowledge / reasoning and prints a base-vs-A-vs-B comparison
table, then writes a JSON report under ``runs/``.

    # frozen base only (sanity / reference bar)
    conda run -n twin-models python scripts/benchmark.py --config configs/base.yaml

    # base vs a trained checkpoint
    conda run -n twin-models python scripts/benchmark.py --config configs/base.yaml \
        --adapters base,A --checkpoint-a checkpoints/adapter_A_step1000.safetensors

    # quick smoke: 2 items per category, greedy
    conda run -n twin-models python scripts/benchmark.py --config configs/tiny.yaml --limit 2
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.bench import (  # noqa: E402
    build_report,
    format_table,
    load_dataset,
    make_model_solver,
    run_items,
    write_report,
)
from twin.bench.dataset import category_sort_key  # noqa: E402
from twin.config import Config  # noqa: E402
from twin.models import Adapters, TwinBase  # noqa: E402


def _limit_per_category(items, limit):
    """Keep at most ``limit`` items per category (stable order) — for fast smokes."""
    if not limit or limit <= 0:
        return items
    counts: dict[str, int] = {}
    kept = []
    for it in items:
        if counts.get(it.category, 0) < limit:
            kept.append(it)
            counts[it.category] = counts.get(it.category, 0) + 1
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--data", default=None,
                    help="benchmark data dir or file (default: data/bench/)")
    ap.add_argument("--adapters", default="base",
                    help="comma list of adapters to evaluate: base,A,B (default: base)")
    ap.add_argument("--checkpoint-a", default=None,
                    help="safetensors to load into adapter A before eval")
    ap.add_argument("--checkpoint-b", default=None,
                    help="safetensors to load into adapter B before eval")
    ap.add_argument("--categories", default=None,
                    help="comma list to restrict (e.g. math,coding)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max items per category (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="generation budget (default: config solver_max_tokens)")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="sampling temperature (default 0.0 = greedy/reproducible)")
    ap.add_argument("--thinking", dest="thinking", action="store_true", default=None,
                    help="force Qwen3 thinking mode on (default: from config)")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false",
                    help="force thinking mode off")
    ap.add_argument("--out", default=None, help="report JSON path (default: runs/bench-<ts>.json)")
    ap.add_argument("--no-items", action="store_true",
                    help="omit per-item detail from the JSON report")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    categories = args.categories.split(",") if args.categories else None
    items = load_dataset(args.data, categories=categories)
    items = _limit_per_category(items, args.limit)
    cats = sorted({it.category for it in items}, key=category_sort_key)
    print(f"Loaded {len(items)} benchmark items across {len(cats)} categories: "
          f"{', '.join(cats)}")

    want = [a.strip() for a in args.adapters.split(",") if a.strip()]
    for a in want:
        if a not in ("base", "A", "B"):
            ap.error(f"--adapters must be from base,A,B; got {a!r}")

    print(f"Loading base: {cfg.model.path}")
    base = TwinBase(cfg.model.path)
    adapters = Adapters.from_config(base.model, cfg.lora)
    if args.checkpoint_a:
        adapters.load("A", args.checkpoint_a)
        print(f"  loaded adapter A <- {args.checkpoint_a}")
    if args.checkpoint_b:
        adapters.load("B", args.checkpoint_b)
        print(f"  loaded adapter B <- {args.checkpoint_b}")

    enable_thinking = cfg.model.enable_thinking if args.thinking is None else args.thinking
    max_tokens = args.max_tokens or cfg.gen.solver_max_tokens
    print(f"  thinking={enable_thinking} max_tokens={max_tokens} temp={args.temp}\n")

    per_adapter = {}
    for name in want:
        t0 = time.time()
        solver = make_model_solver(
            base, adapters, name,
            max_tokens=max_tokens, temp=args.temp, top_p=cfg.gen.top_p,
            enable_thinking=enable_thinking, seed=cfg.train.seed,
        )
        done = {"n": 0}

        def progress(res, _name=name, _done=done):
            _done["n"] += 1
            flag = "✓" if res.correct else "✗"
            print(f"  [{_name}] {_done['n']:>3}/{len(items)} {flag} "
                  f"{res.id} ({res.category}/{res.vtype})")

        print(f"== adapter {name} ==")
        results = run_items(items, solver, on_result=progress,
                            keep_response=not args.no_items)
        per_adapter[name] = results
        acc = sum(r.correct for r in results) / max(1, len(results))
        print(f"   {name}: {acc:.1%} ({sum(r.correct for r in results)}/{len(results)}) "
              f"in {time.time() - t0:.0f}s\n")

    report = build_report(
        per_adapter,
        meta={
            "config": args.config, "model": cfg.model.path,
            "enable_thinking": enable_thinking, "max_tokens": max_tokens,
            "temp": args.temp, "checkpoint_a": args.checkpoint_a,
            "checkpoint_b": args.checkpoint_b,
        },
        include_items=not args.no_items,
    )

    print(format_table(report))

    out = Path(args.out) if args.out else (
        Path(cfg.paths.runs) / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    write_report(report, out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
