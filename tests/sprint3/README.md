# Sprint 3 tests — RL loop (planned)

Drop `test_*.py` files here when Sprint 3 lands (GRPO step, `roles/manager.py`,
prompt templates, `SelfPlayTrainer`, JSONL logging — see DESIGN.md §11).

Convention (matches Sprint 1 & 2):
- Any file in this folder is auto-tagged with the `sprint3` marker by
  `tests/conftest.py` — no `pytestmark` boilerplate needed.
- Start the module docstring with `"""Sprint 3 — <area>. <Fast: no model.>"""`.
- Tests that load the real base must add `pytest.mark.model` + the
  `TWIN_RUN_MODEL_TESTS` skipif (copy the header from
  `tests/sprint1/test_adapters.py`).

Run them with `python scripts/run_tests.py --sprint 3`.
