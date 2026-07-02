"""Held-out benchmark items and their loader.

The benchmark is the absolute-capability counterpart to the in-training curves
(``twin.analysis.curves``). Where those measure the solver against the creator's
own moving distribution, the benchmark measures the model against a *fixed*,
hand-verified set so we can say whether self-play actually made it better at
math / coding / knowledge / reasoning in absolute terms.

Data lives as JSON under ``data/bench/`` — one file per category, each of the
shape::

    {"category": "math", "items": [ {item}, ... ]}

An *item* is one graded question. The ``verification.type`` (not the category)
decides how it is prompted and graded — categories are just reporting buckets,
so a "reasoning" file can freely mix numeric and short-answer items:

    math_numeric  reference ``answer`` checked by SymPy (twin.verifiers.verify_math)
    code          model writes a function; hidden ``tests`` run in the sandbox
    mcq           ``choices`` + letter ``answer``; graded on the chosen letter
    exact         short ``answer`` (+ optional ``aliases``) by normalized match

This module is pure data + parsing (no model, no mlx), so it loads and validates
without touching the weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Repo-default location of the benchmark data, resolved relative to the source
# tree so the CLI works regardless of the caller's CWD.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "bench"

VALID_TYPES = {"math_numeric", "code", "mcq", "exact"}
# Canonical category order for stable reporting (unknown categories sort after).
CATEGORY_ORDER = ["math", "coding", "knowledge", "reasoning"]


class BenchDataError(ValueError):
    """Raised when a benchmark data file is malformed."""


@dataclass
class BenchItem:
    """One graded benchmark question."""

    id: str
    category: str
    prompt: str
    verification: dict[str, Any]
    answer: str = ""
    choices: list[str] = field(default_factory=list)
    subcategory: str = ""
    entry_point: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def vtype(self) -> str:
        return str(self.verification.get("type", "")).strip().lower()

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, category: str) -> "BenchItem":
        if "prompt" not in d or not str(d.get("prompt", "")).strip():
            raise BenchDataError(f"item missing non-empty 'prompt': {d!r}")
        verification = dict(d.get("verification", {}) or {})
        vtype = str(verification.get("type", "")).strip().lower()
        if vtype not in VALID_TYPES:
            raise BenchDataError(
                f"item {d.get('id', '?')!r}: verification.type must be one of "
                f"{sorted(VALID_TYPES)}, got {vtype!r}"
            )
        choices = list(d.get("choices", []) or [])
        if vtype == "mcq" and len(choices) < 2:
            raise BenchDataError(f"mcq item {d.get('id', '?')!r} needs >= 2 'choices'")
        if not str(d.get("answer", "")).strip() and vtype != "code":
            raise BenchDataError(f"item {d.get('id', '?')!r} missing 'answer'")
        return cls(
            id=str(d.get("id") or ""),
            category=str(d.get("category", category)),
            prompt=str(d["prompt"]).strip(),
            verification=verification,
            answer=str(d.get("answer", "")).strip(),
            choices=[str(c) for c in choices],
            subcategory=str(d.get("subcategory", "")),
            entry_point=str(d.get("entry_point", "")),
            metadata=dict(d.get("metadata", {}) or {}),
        )


def load_file(path: str | Path) -> list[BenchItem]:
    """Parse one ``{category, items}`` JSON file into ``BenchItem``s.

    Items missing an ``id`` are auto-numbered ``<category>-<n>`` so every item is
    addressable. Raises :class:`BenchDataError` on a malformed file."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise BenchDataError(f"{path}: invalid JSON: {e}") from e
    if not isinstance(data, dict) or "items" not in data:
        raise BenchDataError(f"{path}: expected an object with an 'items' list")
    category = str(data.get("category", path.stem))
    items: list[BenchItem] = []
    for i, raw in enumerate(data["items"]):
        if not isinstance(raw, dict):
            raise BenchDataError(f"{path}: item {i} is not an object")
        item = BenchItem.from_dict(raw, category=category)
        if not item.id:
            item.id = f"{category}-{i:03d}"
        items.append(item)
    return items


def load_dataset(
    source: str | Path | None = None,
    *,
    categories: list[str] | None = None,
) -> list[BenchItem]:
    """Load all benchmark items from a directory of JSON files (or a single file).

    ``source`` defaults to :data:`DEFAULT_DATA_DIR`. ``categories`` optionally
    restricts to those category names. Duplicate ids across files raise."""
    src = Path(source) if source is not None else DEFAULT_DATA_DIR
    if src.is_file():
        files = [src]
    elif src.is_dir():
        files = sorted(src.glob("*.json"))
    else:
        raise BenchDataError(f"benchmark data source not found: {src}")
    if not files:
        raise BenchDataError(f"no .json benchmark files under {src}")

    items: list[BenchItem] = []
    seen: set[str] = set()
    wanted = set(categories) if categories else None
    for f in files:
        for item in load_file(f):
            if wanted is not None and item.category not in wanted:
                continue
            if item.id in seen:
                raise BenchDataError(f"duplicate item id across files: {item.id!r}")
            seen.add(item.id)
            items.append(item)
    if wanted:
        missing = wanted - {it.category for it in items}
        if missing:
            raise BenchDataError(f"requested categories with no items: {sorted(missing)}")
    return items


def category_sort_key(category: str) -> tuple[int, str]:
    """Stable ordering: known categories in :data:`CATEGORY_ORDER`, then alpha."""
    try:
        return (CATEGORY_ORDER.index(category), "")
    except ValueError:
        return (len(CATEGORY_ORDER), category)
