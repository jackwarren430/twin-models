"""Benchmark dataset loading + validation. Fast: no model."""

import json

import pytest

from twin.bench.dataset import (
    DEFAULT_DATA_DIR,
    VALID_TYPES,
    BenchDataError,
    BenchItem,
    category_sort_key,
    load_dataset,
    load_file,
)


def test_shipped_dataset_loads_and_is_well_formed():
    items = load_dataset()  # default data/bench
    assert len(items) >= 40
    cats = {it.category for it in items}
    assert {"math", "coding", "knowledge", "reasoning"} <= cats
    ids = [it.id for it in items]
    assert len(ids) == len(set(ids)), "ids must be unique across files"
    for it in items:
        assert it.prompt.strip()
        assert it.vtype in VALID_TYPES
        if it.vtype == "mcq":
            assert len(it.choices) >= 2
            assert it.answer.strip().upper() in {chr(ord("A") + i) for i in range(len(it.choices))}
        elif it.vtype == "code":
            assert it.verification.get("tests", "").strip()
        else:
            assert it.answer.strip()


def test_each_shipped_file_parses():
    for f in sorted(DEFAULT_DATA_DIR.glob("*.json")):
        assert load_file(f), f"{f} produced no items"


def test_category_filter_and_missing_category():
    only_math = load_dataset(categories=["math"])
    assert only_math and all(it.category == "math" for it in only_math)
    with pytest.raises(BenchDataError):
        load_dataset(categories=["does-not-exist"])


def test_auto_ids_assigned(tmp_path):
    p = tmp_path / "mini.json"
    p.write_text(json.dumps({
        "category": "math",
        "items": [
            {"prompt": "1+1?", "answer": "2", "verification": {"type": "math_numeric"}},
            {"prompt": "2+2?", "answer": "4", "verification": {"type": "math_numeric"}},
        ],
    }))
    items = load_file(p)
    assert [it.id for it in items] == ["math-000", "math-001"]


def test_bad_type_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({
        "category": "x",
        "items": [{"prompt": "q", "answer": "a", "verification": {"type": "vibes"}}],
    }))
    with pytest.raises(BenchDataError):
        load_file(p)


def test_mcq_requires_choices(tmp_path):
    p = tmp_path / "bad_mcq.json"
    p.write_text(json.dumps({
        "category": "k",
        "items": [{"prompt": "q", "answer": "A", "verification": {"type": "mcq"}}],
    }))
    with pytest.raises(BenchDataError):
        load_file(p)


def test_duplicate_ids_across_files_rejected(tmp_path):
    common = {"prompt": "q", "answer": "1", "verification": {"type": "math_numeric"}, "id": "dup"}
    (tmp_path / "a.json").write_text(json.dumps({"category": "math", "items": [common]}))
    (tmp_path / "b.json").write_text(json.dumps({"category": "math", "items": [common]}))
    with pytest.raises(BenchDataError):
        load_dataset(tmp_path)


def test_category_sort_key_orders_known_then_alpha():
    cats = ["zeta", "coding", "math", "reasoning", "knowledge"]
    assert sorted(cats, key=category_sort_key) == [
        "math", "coding", "knowledge", "reasoning", "zeta",
    ]


def test_benchitem_vtype_is_normalized():
    it = BenchItem(id="x", category="math", prompt="q",
                   verification={"type": "MATH_NUMERIC"}, answer="1")
    assert it.vtype == "math_numeric"
