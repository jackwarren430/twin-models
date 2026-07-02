# Sprint 6 tests — advantage-degeneracy fixes

Fast (model-free) tests for the fixes queued from the mini-01 advantage-
degeneracy diagnosis (DESIGN.md §11 Sprint 6, EXPERIMENTS.md mini-01 entry):
~60% of mini-01's solver rollout compute produced zero gradient because G_c=2
tied constantly, ÷std standardization mapped any non-tie to ±1, and the 1→0
target ramp made endpoint solver groups zero-variance by design.

- `test_advantages.py` — `group_advantages` mean-baseline default (Dr.GRPO
  `A = R − mean(R)`, magnitude preserved), legacy `mode="std"` ablation arm,
  and the `all_zero_advantages` tie detector (float-residue tolerant).
- `test_target_band.py` — `ProblemSuite.target_curve(n, hi, lo)` interior band
  and `RewardEngine` honoring `rewards.target_hi`/`target_lo` (defaults keep
  the original 1→0 ramp — the band is an opt-in ablation, configs/mini2.yaml).
- `test_zero_adv_skip.py` — `SelfPlayTrainer._grpo` returns a
  `skipped_zero_adv` no-op for a fully tied batch *before* touching the model
  (no reference pass, no backward), and still runs the update when any
  advantage is nonzero.
- `test_config_sprint6.py` — new config knobs (`train.adv_mode`,
  `rewards.target_hi/lo`) default correctly and `configs/mini2.yaml` (the
  mini-02 run config) parses with G_c=4 + interior band + mean advantages.

G_c 2→4 itself is a config-only change (configs/mini2.yaml); nothing to test
beyond the parse.
