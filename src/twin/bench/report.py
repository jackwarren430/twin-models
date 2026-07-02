"""Aggregate per-item results into a report (per category, per type, overall).

Produces three things from a set of :class:`~twin.bench.runner.ItemResult`s:

* :func:`aggregate` — accuracy buckets for one adapter (overall / by category /
  by verification type),
* :func:`build_report` — a JSON-serializable report across several adapters
  (base vs A vs B), which :func:`write_report` saves,
* :func:`format_table` — a terminal comparison table with deltas vs the base, so
  "did self-play beat the frozen base?" is a one-glance answer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from twin.bench.dataset import category_sort_key
from twin.bench.runner import ItemResult


def _bucket(correct: int, n: int) -> dict:
    return {"n": n, "correct": correct, "accuracy": round(correct / n, 4) if n else 0.0}


def _group(results: list[ItemResult], key) -> dict[str, dict]:
    out: dict[str, list[ItemResult]] = {}
    for r in results:
        out.setdefault(key(r), []).append(r)
    return {
        k: _bucket(sum(r.correct for r in rs), len(rs))
        for k, rs in out.items()
    }


def aggregate(results: list[ItemResult]) -> dict:
    """Overall / by-category / by-type accuracy buckets for one adapter."""
    return {
        "overall": _bucket(sum(r.correct for r in results), len(results)),
        "by_category": _group(results, lambda r: r.category),
        "by_type": _group(results, lambda r: r.vtype),
    }


def build_report(
    per_adapter: dict[str, list[ItemResult]],
    *,
    meta: dict | None = None,
    include_items: bool = True,
) -> dict:
    """Assemble the full multi-adapter report dict.

    ``per_adapter`` maps an adapter label ('base' | 'A' | 'B') to its item
    results (all adapters must have been run on the same item set)."""
    categories = sorted(
        {r.category for rs in per_adapter.values() for r in rs}, key=category_sort_key
    )
    n_items = max((len(rs) for rs in per_adapter.values()), default=0)
    adapters: dict[str, dict] = {}
    for name, results in per_adapter.items():
        entry = aggregate(results)
        if include_items:
            entry["items"] = [r.to_dict() for r in results]
        adapters[name] = entry
    return {
        "benchmark": "twin-bench",
        "version": 1,
        "time": time.time(),
        "n_items": n_items,
        "categories": categories,
        "adapters": adapters,
        "meta": meta or {},
    }


def write_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    return path


# --------------------------------------------------------------------------- #
# terminal rendering
# --------------------------------------------------------------------------- #
def _adapter_order(names: list[str]) -> list[str]:
    """base first, then A, B, then anything else alphabetically."""
    priority = {"base": 0, "A": 1, "B": 2}
    return sorted(names, key=lambda n: (priority.get(n, 3), n))


def _cell(entry: dict, base_entry: dict | None) -> str:
    if entry is None or entry.get("n", 0) == 0:
        return "—"
    acc = entry["accuracy"]
    txt = f"{acc:.0%}"
    if base_entry is not None and base_entry is not entry and base_entry.get("n", 0):
        delta = acc - base_entry["accuracy"]
        txt += f" ({delta:+.0%})"
    return txt


def format_table(report: dict) -> str:
    """A fixed-width table: rows = categories + OVERALL, columns = adapters.

    Non-base columns annotate each accuracy with its delta vs the base adapter,
    so improvement (or regression) from self-play is immediately visible."""
    adapters = _adapter_order(list(report.get("adapters", {})))
    if not adapters:
        return "(no adapters in report)"
    cats = report.get("categories", [])
    base = report["adapters"].get("base")
    header = ["category", *adapters]

    def row_for(label: str, getter) -> list[str]:
        cells = [label]
        for name in adapters:
            entry = getter(report["adapters"][name])
            base_e = None if name == "base" or base is None else getter(base)
            cells.append(_cell(entry, base_e))
        return cells

    body = [row_for(c, lambda a, c=c: a["by_category"].get(c, {"n": 0})) for c in cats]
    overall = row_for("OVERALL", lambda a: a["overall"])

    all_rows = [header, *body, overall]
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(header))]

    def fmt(row: list[str]) -> str:
        return "  ".join(row[j].ljust(widths[j]) for j in range(len(row))).rstrip()

    rule = "  ".join("-" * w for w in widths)
    lines = [fmt(header), rule, *(fmt(r) for r in body), rule, fmt(overall)]

    base_name = adapters[0]
    counts = ", ".join(
        f"{c}={report['adapters'][base_name]['by_category'].get(c, {}).get('n', 0)}"
        for c in cats
    )
    title = f"twin-bench — {report.get('n_items', 0)} items ({counts})"
    return title + "\n" + "\n".join(lines)
