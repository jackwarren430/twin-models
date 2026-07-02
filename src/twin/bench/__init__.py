"""Held-out capability benchmark (math / coding / knowledge / reasoning).

The absolute-capability counterpart to the in-training curves: a fixed,
hand-verified item set scored against the frozen base and/or the trained A/B
adapters, so a run can answer "did self-play make the model better in absolute
terms?" — not just "is the solver keeping up with the creator?".

Pipeline (all pure except the model-backed solver):

    load_dataset(...) -> [BenchItem]            # data/bench/*.json
    make_model_solver(base, adapters, name)     # solve_fn(item) -> response text
    run_items(items, solve_fn)  -> [ItemResult] # grade each via grade()
    build_report({name: results}) -> dict       # aggregate per category/type
    format_table(report)        -> str          # terminal comparison table

Driven by ``scripts/benchmark.py``.
"""

from twin.bench.dataset import (
    BenchDataError,
    BenchItem,
    load_dataset,
    load_file,
)
from twin.bench.grade import grade
from twin.bench.report import aggregate, build_report, format_table, write_report
from twin.bench.runner import (
    ItemResult,
    build_prompt,
    make_model_solver,
    run_items,
)

__all__ = [
    "BenchItem",
    "BenchDataError",
    "load_dataset",
    "load_file",
    "grade",
    "ItemResult",
    "build_prompt",
    "run_items",
    "make_model_solver",
    "aggregate",
    "build_report",
    "format_table",
    "write_report",
]
