# Sprint 4 tests — scale & study

Fast (model-free) tests for the Sprint-4 base loop:

- `test_analysis.py` — `twin.analysis.curves` on synthetic iteration records:
  time series, solve-rate-vs-difficulty curve, linearity (slope / Pearson r /
  R² / ramp-MSE), sparklines, summary JSON, CSV/JSON round-trip.
- `test_adapter_drift.py` — the pure tree maths behind `Adapters.global_norm` /
  `drift_from` (`tree_global_norm`, `tree_l2_distance`) on synthetic trees.
- `test_config_scale.py` — `base.yaml` turns thinking ON with bigger budgets;
  `tiny.yaml` keeps it off (DESIGN §12 Q4).
- `test_thinking_parsing.py` — the creator/solver parsers survive a Qwen3
  `<think>…</think>` prefix.

Model-gated (loads ~6 GB; the "separate e2e suite"):

- `test_e2e_model.py` — a 2-iteration real-base loop: the new metrics are
  logged, the analysis module consumes the produced log, the frozen base is
  untouched, and thinking mode runs with bounded KL. Marked `model` + skipped
  unless `TWIN_RUN_MODEL_TESTS=1`.

Run:
- fast: `python scripts/run_tests.py --sprint 4`
- e2e:  `python scripts/run_tests.py --only-model --sprint 4`

Convention: files here are auto-tagged `sprint4` by `tests/conftest.py`; model
tests add `pytest.mark.model` + the `TWIN_RUN_MODEL_TESTS` skipif.
