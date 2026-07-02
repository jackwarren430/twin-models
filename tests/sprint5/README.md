# Sprint 5 tests — mini-01 audit fixes

Fast (model-free) tests for the fixes that came out of the mini-01 run audit
(DESIGN.md Sprint 5, EXPERIMENTS.md mini-01 entry):

- `test_gradient_scaling.py` — `r_gradient` is scaled by the scored fraction so
  a mostly-void suite can no longer out-earn an honest fully-consistent ramp
  (the subset-target exploit mini-01 exposed). Complements the updated
  `tests/sprint3/test_rewards_scored_mask.py`.
- `test_check_predicate_cert.py` — the robust math verification certificate:
  `"a = b"` / `"a == b"` / `"Eq(a, b)"` / bare-expression / multi-relation
  forms, multi-symbol tuple answers, tolerance on `Eq`, rejection of trivial
  `"x = <answer>"` certs, and the locked parse namespace (model text can't
  reach builtins through `parse_expr`, in `check_predicate`, `verify_math`
  and `calc`).
- `test_prompts_verification.py` — the creator contract asks math problems for
  the `verification.check`/`symbol` certificate (and doesn't ask coding, whose
  pipeline lands with the coding-domain fix).
- `test_seeding.py` — `SelfPlayTrainer` seeds MLX's global RNG from
  `train.seed`, making generation sampling reproducible run-to-run.
