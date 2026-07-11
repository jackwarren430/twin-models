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

## q-shakeout-02 (Sprint Q6 gate, re-run) — 2026-07-11

Identical to q-shakeout-01 except the per-role thinking fix (guesser/answerer
OFF, creator ON @ 1024). Same gates.

Result: **STOPPED after iter 1 — guesser fixed, creator broke.**

    [0] ... parse=0.50 valid=0.00 | eps 0/0 void=0 fmt=0 | phi=0.00
    [1] ... parse=0.00 valid=0.00 | eps 0/0 void=0 fmt=0 | phi=0.00

- **fmt=0 both iters — the guesser format-fail is fixed.** No episode ever
  format-failed again.
- **New FAIL: creator truncation.** With thinking ON @ 1024, the creator
  brainstormed candidates ("Truffle? Wasabi? Escargot?…" — 3958 chars, no
  JSON) past the budget: parse 0.50 then 0.00, zero valid secrets, **zero
  episodes**. Same truncation disease as the guesser, one role over.
- Also: creator picked "water" for easy-food again → judge INVALID (correct).

Fix (committed): creator thinking OFF too (all roles now emit contract output
directly; budget 256), + creator prompt nudge toward concrete in-category
picks and a single direct choice. The recurring lesson (mini-04a/05, now
twentyq ×2): a *thinking budget* on Qwen3 is a truncation trap; constrain by
turning thinking off, not by tightening the budget. Relaunched as
q-shakeout-03.

## q-shakeout-03 (Sprint Q6 gate, re-run) — 2026-07-11

All player roles thinking OFF (creator 256 / guesser 200 / answerer 64). Judge
still thinking. Same gates.

Result: **STOPPED after iter 0 — the loop finally plays real episodes; one last
truncation, this time on the judge.**

    [0] ... Rc=+1.100 | parse=1.00 valid=1.00 | eps 0/4 void=0 fmt=0 | phi=0.00

- **parse 1.00, valid 1.00, 4 episodes played, 0 format-fails, 0 voids.** The
  loop runs end-to-end. Transcript shows a *coherent* game: guesser narrows
  "fruit? vegetable? grain? legume? cereal grain?" then guesses Rice; answerer
  truthfully answers about the secret "bread" (grain YES, cereal-grain NO with
  the correct distinction). This is the milestone — self-play 20-questions
  works.
- **New (final) FAIL: phi=None on every episode.** The *closeness judge*
  truncated — thinking ON, it reasoned through all 6 questions and hit the
  budget before the `CLOSENESS:` line. Validity/audit survived (VALID/INVALID
  needs far less reasoning than a 0-10 score). Dead phi ⇒ the w_close partial-
  credit shaping is inert.
- Also slow: ~10 thinking judge calls/iteration dominated wall-clock.

Fix (committed): `twentyq.judge_thinking` default OFF — the judge's three
contracts are pure NL judgment (no arithmetic), so it emits its verdict line
directly. Faster *and* revives phi. (`_judge` gained an `enable_thinking`
override; SelfPlayTrainer's CAS math-judge is unchanged.)

## q-shakeout-04 (Sprint Q6 gate, re-run) — 2026-07-11

All players + judge thinking OFF. **STOPPED after iter 0 — phi STILL None.**
The closeness judge emitted `VERDICT: INCORRECT`, not a 0-10 score: `_judge`
fed it the self-play CAS math-grader system prompt (`twin.prompts.JUDGE_SYSTEM`
= "recompute with solve()… finish with VERDICT: CORRECT/INCORRECT"), which
hijacked the CLOSENESS contract. Validity/audit survived (their user prompts
overrode). Fix: a **neutral twentyq `JUDGE_SYSTEM`** + a `_judge(system=…)`
override. A real-model **probe** (scratchpad/probe_judge.py) then confirmed all
3 contracts parse (validity VALID, audit parses, closeness Φ=0.6 not None);
also tightened the audit prompt to flag F only for clear lies. Relaunched -05.

## q-shakeout-05 (Sprint Q6 gate) — 2026-07-11 — **PASS ✅**

All players + judge thinking OFF; neutral judge system; all four q-shakeout
fixes in. 5 iterations, N=2 K=2 T=6.

    it cat               parse valid   phi     Rs     Rc void guess%
     0 food                1.0   1.0  0.30  +0.150 +0.85    1   0.0
     1 food                1.0   1.0  0.28  +0.138 +1.10    0   0.0
     2 animal              1.0   1.0  0.50  +0.250 +0.85    1   0.0
     3 food                1.0   1.0  0.23  +0.113 +1.10    0   0.0
     4 household object    1.0   1.0  0.50  +0.250 +0.83    2   0.0

Gates (all met):
- **parse ≥ 0.8 → 1.00** (20/20 secrets parsed, every iter).
- **validity ≥ 0.5 → 1.00** (every secret VALID).
- **no crash → 5/5 complete, checkpointed at step 5, exit 0.**
- **wall-clock → 80 s/iter** (tiny config, excl. model load; ~6.7 min for 5).
  Projected q-01 (N=3 K=4 T=10, ~5× the gen+judge load) ≈ 3–3.5 h for 30 iters
  — feasible for a long run, but on the heavy side; trimming T or K is a lever.
- **transcript sanity → coherent games** (guesser narrows fruit→veg→grain→…;
  answerer truthful about the secret; contracts clean).
- **Φ live (the headline fix) → 0.23–0.50** every iter. `answer_format_fails=0`,
  `unauditable=0`, `fmt=0`.

The whole pipeline is proven: reliable contracts, real self-play episodes, live
shaping to BOTH sides — creator calibration reward `Rc` and solver phi
partial-credit `Rs>0` even with zero wins (the "partial credit on failure"
design working exactly as intended), rotation + checkpointing OK.

Watch items / q-01 follow-ups (not gate blockers):
1. **void 20 % (4/20).** The non-thinking audit false-flags some truthful
   answers as lies (probe: "is it a legume? NO" for bread); some voids may be
   genuine answerer mistakes. Improves via a better audit prompt and/or a
   stronger answerer. 80 % of episodes still train.
2. **guess% = 0 at T=6, untrained.** A weak non-thinking guesser never wins in
   6 turns, so the easy end of the creator's calibration ramp (target 0.9) has
   no positive signal yet. Expect wins once T=10 + training; if not, a
   guesser-thinking-ON arm (generous budget) is the Q8 lever.

Notably, **all four shakeout fixes were model-interaction/prompt issues
(thinking truncation ×3, wrong system prompt ×1) — zero were bugs in the RL
machinery** (trajectories, advantages, rewards, GRPO worked from iter 0). Good
signal for the architecture.
