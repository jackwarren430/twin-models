# Sprint 3 tests — RL loop

Fast, model-free coverage of the self-play GRPO loop (DESIGN.md §5/§8):

- `test_grpo.py` — group advantages, k3 KL, grad global-norm + clip, and a real
  `grpo_update` step on a **tiny toy LM** (8-token vocab) that proves the
  `value_and_grad` → `optimizer.update` path moves log-probs in the advantage
  direction. Milliseconds, no base model.
- `test_roles.py` — `RoleManager` assignment, swap cadence, warmup, optional blend.
- `test_prompts.py` — creator/solver prompt builders + theme picking.
- `test_extract.py` — solver final-answer extraction + oracle-call counting.
- `test_logging.py` — `JsonlLogger` round-trip.
- `test_rewards_scored_mask.py` — void-problem masking in `creator_reward`.

The **end-to-end** check (loads the real 6 GB base, runs the loop for two
iterations, asserts rewards finite + KL bounded) lives in
`scripts/smoke_test_sprint_3.py`, not here, so the suite stays fast.

Convention (matches Sprint 1 & 2):
- Any file in this folder is auto-tagged with the `sprint3` marker by
  `tests/conftest.py` — no `pytestmark` boilerplate needed.
- Start the module docstring with `"""Sprint 3 — <area>. <Fast: no model.>"""`.

Run them with `python scripts/run_tests.py --sprint 3`.
