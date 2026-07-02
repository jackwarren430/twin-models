"""Benchmark runner + report aggregation. Fast: fake solver, no model."""

from twin.bench.dataset import BenchItem
from twin.bench.report import aggregate, build_report, format_table
from twin.bench.runner import build_prompt, run_items
from twin.prompts import SOLVER_SYSTEM


def _items():
    return [
        BenchItem(id="m1", category="math", prompt="2+2?", answer="4",
                  verification={"type": "math_numeric"}),
        BenchItem(id="k1", category="knowledge", prompt="capital?", answer="B",
                  choices=["A-town", "B-city", "C-ville"], verification={"type": "mcq"}),
        BenchItem(id="r1", category="reasoning", prompt="yes?", answer="yes",
                  verification={"type": "exact"}),
    ]


# ----- prompt routing ------------------------------------------------------
def test_build_prompt_routes_by_type():
    items = _items()
    sys_math, user_math = build_prompt(items[0])
    assert sys_math == SOLVER_SYSTEM and "2+2?" in user_math

    sys_mcq, user_mcq = build_prompt(items[1])
    assert "letter" in sys_mcq.lower()
    assert "A) A-town" in user_mcq and "B) B-city" in user_mcq  # choices rendered

    code_item = BenchItem(id="c1", category="coding", prompt="write add",
                          verification={"type": "code", "tests": "assert add(1,1)==2"})
    sys_code, user_code = build_prompt(code_item)
    assert "code block" in sys_code.lower() and "python" in user_code.lower()


# ----- run_items -----------------------------------------------------------
def _canned_solver(answers):
    def solve(item):
        return answers[item.id]
    return solve


def test_run_items_grades_each():
    items = _items()
    solver = _canned_solver({
        "m1": "ANSWER: 4",          # correct
        "k1": "ANSWER: A",          # wrong (ref B)
        "r1": "ANSWER: yes",        # correct
    })
    results = run_items(items, solver)
    by_id = {r.id: r for r in results}
    assert by_id["m1"].correct
    assert not by_id["k1"].correct
    assert by_id["r1"].correct
    assert by_id["m1"].response == "ANSWER: 4"  # response retained by default


def test_run_items_captures_solver_error():
    items = _items()[:1]

    def boom(item):
        raise RuntimeError("model exploded")

    results = run_items(items, boom)
    assert len(results) == 1
    assert not results[0].correct
    assert "model exploded" in results[0].error


def test_run_items_on_result_callback_and_drop_response():
    items = _items()
    seen = []
    run_items(items, _canned_solver({"m1": "ANSWER: 4", "k1": "ANSWER: B", "r1": "ANSWER: no"}),
              on_result=seen.append, keep_response=False)
    assert len(seen) == 3
    assert all(r.response == "" for r in seen)


# ----- aggregation + report ------------------------------------------------
def test_aggregate_buckets():
    items = _items()
    results = run_items(items, _canned_solver(
        {"m1": "ANSWER: 4", "k1": "ANSWER: B", "r1": "ANSWER: yes"}))
    agg = aggregate(results)
    assert agg["overall"] == {"n": 3, "correct": 3, "accuracy": 1.0}
    assert agg["by_category"]["math"]["accuracy"] == 1.0
    assert set(agg["by_type"]) == {"math_numeric", "mcq", "exact"}


def test_build_report_and_table_show_delta_vs_base():
    items = _items()
    base = run_items(items, _canned_solver(
        {"m1": "ANSWER: 5", "k1": "ANSWER: A", "r1": "ANSWER: no"}))  # 0/3
    trained = run_items(items, _canned_solver(
        {"m1": "ANSWER: 4", "k1": "ANSWER: B", "r1": "ANSWER: yes"}))  # 3/3

    report = build_report({"base": base, "A": trained}, meta={"config": "x"})
    assert report["n_items"] == 3
    assert report["adapters"]["base"]["overall"]["accuracy"] == 0.0
    assert report["adapters"]["A"]["overall"]["accuracy"] == 1.0
    assert report["categories"][0] == "math"  # canonical ordering

    table = format_table(report)
    assert "OVERALL" in table
    assert "math" in table and "knowledge" in table and "reasoning" in table
    assert "base" in table and "A" in table
    assert "+100%" in table  # A improved from 0% to 100% over base


def test_format_table_base_only():
    items = _items()
    base = run_items(items, _canned_solver(
        {"m1": "ANSWER: 4", "k1": "ANSWER: B", "r1": "ANSWER: yes"}))
    table = format_table(build_report({"base": base}))
    assert "OVERALL" in table and "100%" in table
