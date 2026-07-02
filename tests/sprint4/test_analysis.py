"""Sprint 4 — run-analysis curves. Fast: no model.

Exercises the pure functions in ``twin.analysis.curves`` on synthetic iteration
records (the shape ``SelfPlayTrainer`` logs), so the curve maths is verified
without loading the base or running a real loop.
"""

import json

from twin.analysis import (
    linearity,
    load_iterations,
    moving_average,
    report_text,
    solve_rate_curve,
    sparkline,
    summarize_run,
    time_series,
    write_csv,
    write_summary_json,
)
from twin.log import JsonlLogger


def _rec(it, *, suites=None, **over):
    rec = {
        "type": "iteration", "iter": it,
        "creator_reward_mean": 0.1 * it, "solver_reward_mean": 1.0 - 0.05 * it,
        "solve_rate_mean": 0.5, "parse_ok_rate": 1.0, "r_gradient_mean": 0.8,
        "creator_update": {"kl": 0.001 * it, "grad_norm": 0.2},
        "solver_update": {"kl": 0.002, "grad_norm": 0.3},
        "creator_oracle_calls": 0, "solver_oracle_calls": it,
        "creator_tool_calls": 2,
        "adapter_norm": {"A": 1.0 + it, "B": 1.0 + 0.5 * it},
        "adapter_drift": {"A": 0.1 * it, "B": 0.05 * it},
        "suites": suites if suites is not None else [
            {"parsed": True, "solve_rates_by_rank": [1.0, 0.75, 0.5, 0.25, 0.0]},
        ],
    }
    rec.update(over)
    return rec


def _records(n=6):
    return [_rec(i) for i in range(n)]


# ----- time_series ---------------------------------------------------------
def test_time_series_aligns_and_pulls_nested_keys():
    ts = time_series(_records(4))
    assert ts["iter"] == [0, 1, 2, 3]
    assert ts["creator_kl"] == [0.0, 0.001, 0.002, 0.003]
    assert abs(ts["adapter_drift_A"][-1] - 0.3) < 1e-9
    assert ts["solver_oracle_calls"] == [0, 1, 2, 3]
    # every series stays the same length as the run
    assert all(len(v) == 4 for v in ts.values())


def test_time_series_missing_keys_become_none():
    recs = [{"type": "iteration", "iter": 0}]  # bare record, older log shape
    ts = time_series(recs)
    assert ts["creator_kl"] == [None]
    assert ts["adapter_drift_A"] == [None]


# ----- solve_rate_curve ----------------------------------------------------
def test_solve_rate_curve_averages_by_rank_over_variable_lengths():
    recs = [_rec(0, suites=[
        {"parsed": True, "solve_rates_by_rank": [1.0, 0.5, 0.0]},
        {"parsed": True, "solve_rates_by_rank": [0.6, 0.4]},  # shorter suite
    ])]
    curve = solve_rate_curve(recs)
    # rank 0: (1.0+0.6)/2, rank 1: (0.5+0.4)/2, rank 2: only first suite -> 0.0
    assert curve == [0.8, 0.45, 0.0]


def test_solve_rate_curve_last_window():
    recs = [
        _rec(0, suites=[{"parsed": True, "solve_rates_by_rank": [0.0, 0.0]}]),
        _rec(1, suites=[{"parsed": True, "solve_rates_by_rank": [1.0, 1.0]}]),
    ]
    assert solve_rate_curve(recs, last=1) == [1.0, 1.0]


# ----- linearity -----------------------------------------------------------
def test_linearity_of_perfect_descending_ramp():
    lin = linearity([1.0, 0.75, 0.5, 0.25, 0.0])
    assert lin["pearson_r"] < -0.999
    assert abs(lin["slope"] - (-0.25)) < 1e-9
    assert lin["r2"] > 0.999
    assert lin["ramp_mse"] < 1e-12  # this IS the target ramp


def test_linearity_of_flat_curve_is_zero_slope():
    lin = linearity([0.5, 0.5, 0.5, 0.5])
    assert abs(lin["slope"]) < 1e-12
    assert lin["pearson_r"] == 0.0  # no variance in y
    assert lin["ramp_mse"] > 0.0    # flat != the ramp


def test_linearity_edge_cases():
    assert linearity([])["n"] == 0
    one = linearity([0.7])
    assert one["n"] == 1 and one["slope"] == 0.0


# ----- helpers -------------------------------------------------------------
def test_moving_average_trailing_window():
    assert moving_average([1, 2, 3, 4, 5], 2) == [1.0, 1.5, 2.5, 3.5, 4.5]


def test_moving_average_skips_none():
    out = moving_average([None, 2, None, 4], 10)
    assert out[0] is None
    assert out[1] == 2.0
    assert out[-1] == 3.0  # mean of 2 and 4


def test_sparkline_handles_flat_and_none():
    assert sparkline([]) == ""
    assert set(sparkline([5, 5, 5])) == {"▁"}
    sp = sparkline([0, None, 1])
    assert sp[1] == " " and len(sp) == 3


# ----- summarize / serialize ----------------------------------------------
def test_summarize_run_is_json_serializable_and_has_curve():
    summary = summarize_run(_records(5))
    json.dumps(summary)  # must not raise
    assert summary["n_iterations"] == 5
    assert summary["linearity"]["pearson_r"] < -0.999
    assert abs(summary["final"]["adapter_drift_A"] - 0.4) < 1e-9
    assert summary["mean"]["solver_oracle_calls"] == 2.0  # mean of 0..4


def test_report_text_runs_and_mentions_curve():
    txt = report_text(_records(4))
    assert "solve-rate vs difficulty" in txt
    assert "linearity" in txt


def test_report_text_empty():
    assert report_text([]) == "(no iterations)"


# ----- round-trip through a JSONL file ------------------------------------
def test_load_iterations_skips_meta_and_sorts(tmp_path):
    log = tmp_path / "run.jsonl"
    with JsonlLogger(log, meta={"run": "t"}) as lg:
        lg.log(_rec(2))
        lg.log(_rec(0))
        lg.log(_rec(1))
    recs = load_iterations(log)
    assert [r["iter"] for r in recs] == [0, 1, 2]  # sorted, meta dropped


def test_write_csv_and_summary(tmp_path):
    recs = _records(3)
    csv_path = write_csv(time_series(recs), tmp_path / "series.csv")
    json_path = write_summary_json(summarize_run(recs), tmp_path / "summary.json")
    rows = csv_path.read_text().strip().splitlines()
    assert rows[0].split(",")[0] == "iter"
    assert len(rows) == 1 + 3  # header + 3 iterations
    assert json.loads(json_path.read_text())["n_iterations"] == 3
