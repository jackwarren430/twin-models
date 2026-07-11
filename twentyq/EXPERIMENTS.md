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

Result: **STOPPED after iter 0 (laptop crashed — unrelated: a second model was
loaded in another project; MLX backend confirmed).** But iter 0 already
returned the decisive shakeout finding:

    [0] A->create B->guess cat=food | Rc=+0.830 Rs=-0.425 guess%=0.00
        rgrad=0.48 | parse=1.00 valid=0.50 | eps 0/2 void=0 fmt=2 | phi=0.15

- parse 1.00, validity 0.50 (creator picked "water" for the easy slot; judge
  correctly ruled it INVALID as a *food*). **Both gates would pass.**
- **FAIL: every episode format-failed on turn 1** (`fmt=2`), guesser
  think-share **1.0**. Diagnosis: `question_max_tokens: 512` with thinking ON
  → Qwen3 ran the entire budget inside an unclosed `<think>`, never emitted a
  `QUESTION:` line. `strip_think` leaves unclosed blocks in, nothing matches
  the contract → format_fail. Exactly the failure Q6 exists to find.
- Judge contracts all fired correctly (validity INVALID on water, closeness
  0/3 parsed fine). The plumbing works; only the guesser budget was wrong.

Fix (committed): per-role thinking — creator ON (budget 1024), guesser &
answerer OFF (budgets 200 / 64). DESIGN §2.7 revised. Relaunched as
q-shakeout-02.

## q-shakeout-02 (Sprint Q6 gate, re-run) — launched 2026-07-11

Identical to q-shakeout-01 except the per-role thinking fix above. Same gates.

Result: PENDING
