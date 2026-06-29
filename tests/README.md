# Tests

Organized **one directory per sprint**, kept separate so each sprint's coverage
stands on its own:

```
tests/
  conftest.py          auto-tags each test with its sprintN marker (by folder)
  sprint1/             foundation: config, schema, adapters (model test)
  sprint2/             scoring: calc, sandbox, protocol, verifiers, rewards
  sprint3/             RL loop            (placeholder — see its README)
  sprint4/             scale & study      (placeholder — see its README)
```

## Conventions

- **Location = marker.** A file under `tests/sprintN/` is automatically tagged
  with the `sprintN` pytest marker (handled in `conftest.py`); no per-file
  boilerplate. Select a sprint with `-m sprintN`.
- **Module docstring** starts with `"""Sprint N — <area>. <Fast: no model.>"""`.
- **Fast by default.** Almost every test runs without the model. Tests that load
  the real ~6 GB base are marked `model` and skipped unless
  `TWIN_RUN_MODEL_TESTS=1` (see `sprint1/test_adapters.py`).
- **Unique file names** across the whole tree (pytest imports them flat).

## Running

Use the central runner — `scripts/run_tests.py` (full `--help` lists everything):

```bash
conda run -n twin-models python scripts/run_tests.py                 # all fast tests
conda run -n twin-models python scripts/run_tests.py --sprint 2 -v   # one sprint, verbose
conda run -n twin-models python scripts/run_tests.py --sprint 1 2    # several sprints
conda run -n twin-models python scripts/run_tests.py -k consistency  # filter by name
conda run -n twin-models python scripts/run_tests.py --model         # include model tests
conda run -n twin-models python scripts/run_tests.py --only-model    # only model tests
conda run -n twin-models python scripts/run_tests.py --list          # list, don't run
conda run -n twin-models python scripts/run_tests.py --failed        # rerun last failures
```

Plain `pytest` still works (`conda run -n twin-models python -m pytest`); the
runner is just a convenience layer over it that knows about the sprint markers.
