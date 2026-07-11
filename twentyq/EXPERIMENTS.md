# 21-questions run history

Run log for the twentyq mode (spec: `twentyq/DESIGN.md`). Same discipline as
the main `EXPERIMENTS.md`: gates pre-registered before launch, measured entries
after.

## q-shakeout-01 (Sprint Q6 gate) — launched 2026-07-11

First real-model run of the multi-turn loop. Branch `twentyq`, MLX backend,
Qwen3-8B-6bit, `configs/twentyq-tiny.yaml`: N=2 secrets, K=2 episodes, T=6
turns, 5 iterations, thinking ON, budgets 768/512/256 (secret/question/answer),
target band 0.9→0.1, swap_interval 20 (no swap inside 5 iters),
`--log-prompts`. Purpose: contract adherence + wall-clock, NOT learning.

Pre-registered gates (from DESIGN §4 Q6):
- secret parse rate ≥ 0.8 across the run
- validity rate ≥ 0.5
- no crash; all 5 iterations complete
- iteration wall-clock compatible with a 30-iteration q-01 run
- transcript sanity: guesser emits QUESTION:/GUESS: lines, answerer emits
  ANSWER: lines, episodes terminate for the right reasons

Watch items (not gates): format-ended episode share, answer_format_fails,
unauditable audits, phi parse failures (phi_mean = 0 with failed episodes
present would mean the closeness contract is broken), guesser think-share
truncation risk at the 512 question budget.

Result: PENDING
