# Sprint 4 tests — scale & study (planned)

Drop `test_*.py` files here when Sprint 4 lands (bigger configs, longer runs,
curves, ablations — see DESIGN.md §11).

Convention (matches Sprint 1 & 2):
- Any file in this folder is auto-tagged with the `sprint4` marker by
  `tests/conftest.py` — no `pytestmark` boilerplate needed.
- Start the module docstring with `"""Sprint 4 — <area>. <Fast: no model.>"""`.
- Tests that load the real base must add `pytest.mark.model` + the
  `TWIN_RUN_MODEL_TESTS` skipif (copy the header from
  `tests/sprint1/test_adapters.py`).

Run them with `python scripts/run_tests.py --sprint 4`.
