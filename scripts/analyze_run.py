#!/usr/bin/env python
"""Analyze a self-play run's JSONL log into Sprint-4 study curves (DESIGN.md §11).

    conda run -n twin-models python scripts/analyze_run.py runs/<run>.jsonl
    conda run -n twin-models python scripts/analyze_run.py --latest --plots

Reads the iteration records logged by the trainer and writes, next to the log
(or under ``--out``):

  * ``<run>_series.csv``    — one row per iteration, every tracked metric
  * ``<run>_summary.json``  — solve-rate curve + linearity + final/mean stats
  * ``*.png`` (with ``--plots``, only if matplotlib is installed)

and prints a terminal report (ASCII sparklines + the solve-rate-vs-difficulty
curve and its linearity). No model load; fast.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.analysis import (  # noqa: E402
    load_iterations,
    render_plots,
    report_text,
    summarize_run,
    time_series,
    write_csv,
    write_summary_json,
)
from twin.config import Config  # noqa: E402


def _latest_run() -> Path | None:
    runs_dir = Path(Config().paths.runs)
    if not runs_dir.is_dir():
        return None
    logs = sorted(runs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="path to a run JSONL")
    ap.add_argument("--latest", action="store_true",
                    help="use the most recent log under the configured runs/ dir")
    ap.add_argument("--out", default=None,
                    help="output directory (default: alongside the log)")
    ap.add_argument("--plots", action="store_true",
                    help="also write PNG plots (requires matplotlib)")
    args = ap.parse_args()

    log = Path(args.log) if args.log else _latest_run()
    if args.latest and not args.log:
        log = _latest_run()
    if log is None or not log.exists():
        ap.error("no log file: pass a path or use --latest with a non-empty runs/ dir")

    records = load_iterations(log)
    if not records:
        print(f"No iteration records in {log}", file=sys.stderr)
        return 1

    outdir = Path(args.out) if args.out else log.parent
    stem = log.stem
    csv_path = write_csv(time_series(records), outdir / f"{stem}_series.csv")
    summary = summarize_run(records)
    json_path = write_summary_json(summary, outdir / f"{stem}_summary.json")

    print(report_text(records))
    print(f"\nwrote {csv_path}")
    print(f"wrote {json_path}")

    if args.plots:
        pngs = render_plots(records, outdir)
        if pngs:
            for p in pngs:
                print(f"wrote {p}")
        else:
            print("(matplotlib not installed — skipped PNG plots; CSV/JSON written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
