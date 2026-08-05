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

## Model swap → gemma-4-E2B (2026-07-11)

User directive: run Q8 on **`google/gemma-4-E2B-it`** instead of Qwen3-8B (the
sibling `litert-community/...-litert-lm` is LiteRT/MediaPipe format, **cannot**
load in MLX — ruled out immediately). gemma-4-E2B is a `gemma4`
(Gemma4ForConditionalGeneration) multimodal checkpoint; mlx-lm 0.31.3 loads its
text tower (`gemma4`/`gemma4_text`), dropping vision/audio at load. ~2B-effective
(MatFormer), 10.2 GB bf16 on disk → **9.26 GB resident** (heavier than Qwen3-8B-
6bit's ~6.5 GB, still fine on 32 GB).

**Two load-time infra fixes** (production code, gated behind new `ModelConfig`
fields so the Qwen3/self-play path is untouched — `load()` verbatim there):
1. `load_strict: false` — the checkpoint ships per-layer k/v for its 20 KV-shared
   layers (15-34) that mlx-lm reuses from earlier layers; strict load raises on
   the 60 vestigial tensors. `strict=False` drops them; generation is coherent,
   so they really are vestigial.
2. `eos_token_ids: [1, 106, 50]` — gemma4's turn terminator `<turn|>`=106 lives
   ONLY in generation_config (mlx-lm's `load()` reads config.json's scalar eos=1),
   so without it generation never stops and pads to max_tokens.

**Thinking:** gemma4-E2B reasons by default via `<|channel>thought … <channel|>`
(different syntax from Qwen3's `<think>`). `enable_thinking=False` removes the
`<|think|>` injection and yields direct, parseable contract output — same
all-roles-thinking-OFF posture as Qwen3, for the same truncation reason.

**Judge quality (probes):** materially weaker than Qwen3-8B. Validity is
VALID-biased (3/6 — passes water/"happiness"/"the number seven" as valid); the
answerer and guesser are actually coherent, but the **audit false-flags truthful
answers as lies**. Closeness Φ is the most reliable contract (directional,
range-compressed) — good, since Φ is the solver's partial-credit and the headline
Q8 credit-granularity signal.

### q-gemma-shakeout-01/02/03 — **PASS ✅** (Sprint Q6 gate re-run on gemma4)

Three rounds, each fixing one weak-model failure the tiny config surfaced (N2 K2
T6, 5 iters, all thinking OFF):
- **-01:** `fmt=4` every iter — the guesser drifted from `QUESTION:` to `Q:`,
  mimicking the flattened `Q:`/`A:` history. Fix: **3-tier lenient parser**
  (`GUESS:` → `QUESTION:`/`Q:` → bare trailing `?` line) + history reworded to
  drop the `Q:` prefix cue.
- **-02:** `fmt=0` (format fixed) but `void≈55%` — the audit false-flagged
  all-truthful episodes (e.g. a perfect Apple game: vegetable NO / sweet YES /
  fruit YES / dessert NO / dairy NO, all correct) and one false-`F` voided the
  lot. Fix: **`audit_void_fraction`** — void only on a MAJORITY of flagged lies
  (gemma4 config 0.5; default 0.0 = historical "any F voids", Qwen3 unchanged).
- **-03 (PASS):**

      it cat               parse valid  phi     Rs     Rc void fmt guess%
       0 food               1.0   1.0  0.72  +0.362 +1.10   0   0   0.00
       1 food               1.0   1.0  0.68  +0.338 +1.10   0   0   0.00
       2 animal             1.0   1.0  0.80  +0.400 +0.85   1   0   0.00
       3 food               1.0   1.0  0.70  +0.270 +0.85   1   1   0.00
       4 household object   1.0   1.0  0.70  +0.512 +0.88   0   0   0.25

  Void ~10% (was ~55%), fmt ~0, **Φ live 0.68-0.80 (higher than Qwen3's
  0.23-0.50)**, Rs>0 from Φ partial-credit, a real guesser win at T=6 (household)
  — untrained Qwen3 never won at T=6. Coherent games; answerer truthful. Healthy
  as the Qwen3 q-shakeout-05 gate. Wall-clock ~3-4 min for 5 iters (gemma4 fast).

All fixes are principled weak-model adaptations, config-gated, no RL-machinery
bugs. Pipeline is **Q8-ready on gemma4**; residual caveat is noisier creator-side
signal (validity gate + audit) vs a cleaner solver-side Φ.

Pre-launch validation for Q8: **per-turn credit ran end-to-end for the first
time** (q-gemma-perturn-01, 5 iters: void ~0-10%, Φ 0.60-0.78, Rs>0, 2 wins) —
it had only ever been unit-tested; both credit paths are now real-model proven.
Q8 base config **`configs/twentyq-gemma-q8.yaml`** smoke-tested at N3 K4 T10
(16-layer LoRA = 50.9M params/adapter, 1 iter clean).

## Q8 — credit-granularity × rotation matrix (launched 2026-07-12)

The headline experiment: **broadcast vs per-turn credit** (does per-turn
reward-to-go with potential-based Φ shaping beat episode-broadcast?) crossed with
the **rotation control** (periodic A/B role swap vs none). Four arms, one base
config, differing only by CLI override:

    arm                    credit     swap_interval (rotation)
    q8-A1-broadcast-rot    broadcast  10  (ON)
    q8-A2-perturn-rot      per_turn   10  (ON)
    q8-A3-broadcast-norot  broadcast   0  (OFF, control)
    q8-A4-perturn-norot    per_turn    0  (OFF, control)

Each: N3 K4 T10, 30 iters, num_layers 16, lr 1e-5, kl_beta 0.02, all thinking
OFF, audit_void_fraction 0.5. Run sequentially (each loads a 9.26 GB base;
parallel would thrash 32 GB). Logs `runs/q8-A*.jsonl` + scratchpad `.log`,
checkpoints `checkpoints/q8-A*/` at steps 10/20/30.

Pre-registered read-outs (comparison, not pass/fail — this is the science):
- **solver learning curve** `Rs` and **guess-rate** trend over 30 iters, per arm.
- **broadcast vs per-turn:** does per-turn Φ-shaping lift guess-rate / Rs faster
  or higher? (per-turn telescopes to the same episode scalar at γ=1, so a
  difference is a *credit-assignment* effect, not a reward-magnitude one.)
- **rotation effect:** does A/B swapping change the creator↔solver co-adaptation
  (creator calibration `Rc`/`rgrad`, solver guess-rate) vs the frozen-role
  control?
- health throughout: parse/valid ~1.0, fmt ~0, void ≲15%, Φ live.

Watch (from the gemma4 shakeouts, not blockers): validity VALID-bias (weak
secret gate), category-dependent difficulty (food voids > animals).

### Q8 RESULTS (completed 2026-07-12, ~8 h wall)

Infra note first: the launch OOM-died 3× before running. Root cause was NOT in
the pre-registered plan — gemma4's **262k vocab** (1.7× Qwen3) made one broadcast
iteration's ~120 turn-trajectories build in a single GRPO graph, peaking
**103.58 GB** → SIGKILL on 32 GB. Fix (commit 7ada896): `grpo_microbatch: 1` +
`logit_chunk: 0` (the chunked LM-head path reaches `self.model.model`, which
gemma4's `Model→language_model` layout doesn't expose). Peak → **14.98 GB**;
all four arms then held 14.6–15.8 GB with zero OOM. Added per-iter `peak_mem_gb`
telemetry (it's what made the spike visible).

**Full 2×2 — overall guess-rate (mean over 30 iters):**

    |            | rotation-ON | rotation-OFF |
    | broadcast  | 0.083 (A1)  | 0.225 (A3)   |
    | per-turn   | 0.162 (A2)  | 0.213 (A4)   |

    Main effects:  ROTATION  off−on   = +0.097  (on 0.123 → off 0.219)
                   CREDIT    pt−bcast = +0.034  (bcast 0.154 → pt 0.188)

**Learning trajectories (guess-rate by third | overall Φ):**

    A1 broadcast+rot : 0.087 0.119 0.042  | Φ 0.58   declines (rotation damage late)
    A2 per-turn +rot : 0.154 0.000 0.333  | Φ 0.65   COLLAPSE@10-19, recovery@20-29
    A3 broadcast norot 0.193 0.275 0.208  | Φ 0.64   climbs then plateaus/dips
    A4 per-turn  norot 0.117 0.240 0.283  | Φ 0.62   ONLY monotonic climb; finishes top

**Findings (ranked by robustness):**

1. **ROTATION IS THE DOMINANT EFFECT AND IT HURTS (+0.097, ~78% rel).** Both
   credit types roughly double guess-rate with rotation OFF. The hard A/B swap at
   `swap_interval=10` **evicts the guess-trained adapter from the solver seat**;
   on a weak model with 30 iters there is no time to rebuild.
2. **A2's collapse is the cleanest proof of the mechanism.** guess-rate went
   0.62 (i9) → **0.00 for all of i10–19** (post-swap) → snapped back to 0.33+
   after the i20 swap-back. The competence is adapter-bound; rotation moves it in
   and out of the solver seat. The collapse coincided with a **void spike to
   0.35** (the guess-adapter, now the answerer, answers inconsistently → episodes
   void → no solver data) vs ≤0.12 elsewhere.
3. **CREDIT GRANULARITY IS SECOND-ORDER (+0.034, within noise).** Per-turn's big
   edge under rotation-ON (0.162 vs 0.083) was mostly **rotation-robustness**, not
   better credit. In the clean rotation-OFF test the two are ~tied overall
   (per-turn 0.213 vs broadcast 0.225) — BUT per-turn (A4) is the **only
   monotonic climber and finishes highest** (final third 0.283 vs 0.208), hinting
   at better *sustained* learning that a longer run might separate.
4. **Per-turn has a mild Φ-vs-guess tension even without rotation.** A4 shows
   recurring high-Φ (0.76–0.84) / zero-guess iters — potential-based shaping does
   slightly reward "get close, don't close." It doesn't run away when the guesser
   stays seated, but under rotation (A2) it deepened the post-swap collapse.

**CAVEAT — N=1, large variance.** A1 vs A3 differ 2× over iters 0–9 despite
*identical* pre-swap config (0.087 vs 0.193): temp-0.8/0.9 sampling gives a ~2×
run-to-run variance floor. The credit effect (+0.034) sits inside that noise; the
rotation effect (+0.097) is larger and consistent across both rows, so more
credible — but everything here is a single seed per arm. Treat as directional.

**Answer to the pre-registered question** ("does per-turn lift guess-rate faster/
higher?"): **not decisively.** ~tied in the clean comparison, with a late-climb
edge for per-turn. The experiment's real payload is the **rotation** result.

**Next (not launched — ask user):** (a) rotation redesign — longer interval /
soft blend / pre-swap warmup, since hard eviction is too costly at 30 iters;
(b) a longer per-turn no-rotation run to test whether A4's monotonic climb
continues; (c) N≥3 seeds to beat the variance floor before trusting the credit
effect. Checkpoints for all arms saved at steps 10/20/30 in `checkpoints/q8-A*/`.

## exp-ensemble-full v1 — ENSEMBLE dense reward, full-scale (2026-07-12/13)

First full-scale run of the **ensemble-logprob dense reward** (`credit:
ensemble`; see `twentyq-ensemble-reward` memory): the frozen 4-model ensemble's
mean log-prob of the secret at each Q/A prefix replaces the per-turn judge Φ.
Config `configs/twentyq-full.yaml` — gemma-4-E2B, torch/CUDA (Spark), N5·K8·T21,
LoRA num_layers=16 rank=64, 30 iters, w_ensemble=0.3, grpo_microbatch=1. Two
arms by `--swap-interval`: **rotation** (5) → `tq-runs/q-full-rot.jsonl`,
**control** (0) → `tq-runs/q-full-ctrl.jsonl`. Full per-iteration play-by-play
in `tq-runs/OBSERVATIONS.md`.

Result: **ARM A (rotation) completed 30/30 clean; ARM B (control) stopped by
user at iter 11/30** (enough to iterate on the design). Solver learned (guess%
0.187 → 0.279 over the run; Rs 0.66 → 0.81), creator calibration improved
(r_gradient 0.45 → 0.56), adapters drifted healthily (0.055 → 0.435, no
collapse), mem flat ~46–50 GB. The ensemble reward works end-to-end at scale and
Φ is correctly skipped (phi_mean=0 every iter). Rotation-swap cost reproduced
from Q8 (post-swap guess% dips) and then *faded* late as both adapters
cross-trained into competent guessers.

**KEY FINDING (drove the next change): the truthfulness audit is not working.**
Reading the transcripts, the frozen gemma4-E2B auditor voided a steady ~6
episodes/iter AND returned **unauditable on up to 13 of 40 episodes/iter** — it
false-flagged truthful games as lies and abstained on many more. Both outcomes
discard good on-policy data (voided episodes leave the solver GRPO group and the
creator's realized guess rate), shrinking effective batch size and adding noise
to creator calibration — all to police a lying incentive a cooperative-
calibration creator barely has. **Decision (2026-07-13): remove the audit
entirely; assume the creator answers truthfully.** Code + rationale in
`twentyq/DESIGN.md §2.8`. `judge_answer_audit` is retained but disconnected.

## exp-fullv2 — no-audit re-run of the ensemble experiment (QUEUED 2026-07-13)

Re-run of exp-ensemble-full with the audit removed (DESIGN §2.8), so the
rot-vs-control comparison is no longer confounded by audit voiding/abstention.
Config `configs/twentyq-full-v2.yaml` (identical to v1 minus `audit_void_fraction`
and with the truthful-creator answerer prompt). Same two arms:

    arm                 swap_interval   run-name         checkpoints
    q-fullv2-rot        5   (rotation)  q-fullv2-rot     checkpoints/twentyq-fullv2-rot
    q-fullv2-ctrl       0   (control)   q-fullv2-ctrl    checkpoints/twentyq-fullv2-ctrl

Each: N5·K8·T21, 30 iters, `credit: ensemble`, w_ensemble=0.3, all thinking OFF.
Launch commands are in the config header. Run the arms sequentially (each loads
the 9.26 GB base + 4 resident ensemble models; v1 held ~46–50 GB — safe on 128 GB).

Pre-registered read-outs (comparison, not pass/fail):
- **effective batch size / data yield:** with no voids, every played episode now
  trains — `episodes.total == episodes.guessed + failures`, no `void`/
  `unauditable` fields. Expect fuller GRPO groups than v1 and steadier creator
  calibration (`r_gradient`) iteration-to-iteration.
- **solver learning:** guess% and Rs trend over 30 iters, per arm — does removing
  the void noise lift the learning curve or its stability vs v1's rot arm
  (0.187 → 0.279)?
- **rotation vs control:** does the frozen-role control (B always guesses) reach
  a higher single-role guess% than rotation's oscillating mean, at the cost of a
  one-sided (A-can't-guess) specialization? (v1's control was only 11 iters.)
- health: parse/valid ~1.0, fmt low, phi_mean=0 (Φ skipped under ensemble), mem
  flat. **Watch:** does answerer honesty actually hold without the audit? Spot-
  check transcripts for the creator giving wrong answers about its own secret;
  if it becomes a real problem, the frozen-base-answerer control (DESIGN §5) is
  the principled fix, not re-adding the broken audit.

Stats: `python scripts/twentyq_run_stats.py tq-runs/q-fullv2-rot.jsonl tq-runs/q-fullv2-ctrl.jsonl --md tq-runs/REPORT-v2.md`.

**Mid-run finding — validity gate false-rejects valid secrets (2026-07-13).**
Spot-checking `q-fullv2-rot` transcripts, ~6.6% of secrets (6/91) were voided as
INVALID and **all 6 were valid, guessable** (Axolotl ×4, Salmon, Octopus). Cause:
gemma4-E2B ignores the vetting instruction, role-plays the guesser (`1. Is it a
mammal?`), and ends its turn before any `VERDICT:` line; the old **fail-closed**
gate then voided the secret (DESIGN §2.10). Same data-loss pathology as the audit
(§2.8), and it clusters on the hardest slot (creator_4) — biasing against the
difficult secrets the curriculum wants. **Fix:** `judge_secret_validity` gained a
`mode` (config `twentyq.secret_validity` / CLI `--secret-validity`); default is
now `fail_open` (only a clear INVALID voids), with `off` to disable the gate.
**This run was NOT stopped** — the rot arm had already run under fail-closed, so
to keep the control comparable the v2 config pins **both** arms to
`secret_validity: fail_closed`. Launch `q-fullv2-ctrl` from the config as-is
(fail_closed inherited). `fail_open` is the standing default for the *next*
experiment. When reading exp-fullv2 results, treat ~6% of hard secrets as voided
in both arms — a shared, matched handicap, not a rot-vs-ctrl confound.

## exp-fullv3 — rebalanced reward + larger GRPO groups (QUEUED 2026-07-13)

Follows exp-fullv2. Config `configs/twentyq-full-v3.yaml`. Two arms (rot/ctrl)
via the same `--swap-interval` overrides:

    arm                 swap_interval   run-name         checkpoints
    q-fullv3-rot        5   (rotation)  q-fullv3-rot     checkpoints/twentyq-fullv3-rot
    q-fullv3-ctrl       0   (control)   q-fullv3-ctrl    checkpoints/twentyq-fullv3-ctrl

**Motivating analysis — ensemble reward dominated the goal (from q-fullv2-rot,
712 episodes / 19 iters).** Decomposing each episode's return into the terminal
(goal) reward and the telescoped ensemble-shaping total (`R_0 = ensemble_total +
terminal`):

    outcome   n    terminal   ensemble   total
    WIN       92    1.086      3.046      4.132     ensemble ~2.8x the goal
    MISS      612   0.000      1.650      1.650     a LOSS still banks 1.65 ensemble
    FMTFAIL   8    -0.500      1.525      1.025

- Per successful episode the ensemble total (~3.05) was ~2.8x the terminal goal
  (~1.09). A *losing* episode still earned ~1.65 of ensemble reward just for
  narrowing questions — 1.5x a win's whole terminal reward, with no guess.
- GRPO keys on within-secret advantage, not raw magnitude. There the news was
  mixed: in mixed win/loss groups the win still took the top total return 31/32
  (97%) and the ensemble gap (+0.69) pointed the SAME way as terminal (+1.06) —
  so ensemble reinforced winning where wins existed. BUT its within-group SD
  (0.54) exceeded terminal's (0.44), and **64% of secret-groups (57/89) had zero
  wins**, so in the majority of groups 100% of the advantage was ensemble-driven
  — pure "raise the belief," no pressure to commit to a guess. `w_efficiency`
  (the guess-fast bonus) only fires on a win, so it rarely operated.
- (The apparent guess-rate decline over iters is heavily confounded by ROTATION
  — roles flip at iters 5/10/15 — and is noisy; the control arm is the clean read.)

**v3 changes (see config header):**
1. **w_ensemble 0.3 -> 0.1.** Puts ensemble within-group SD (~0.18) cleanly below
   terminal's (~0.44): the win/loss signal dominates the advantage again while the
   dense signal still guides questioning. Per-episode ensemble total falls to ~1.0,
   comparable to (not 2.8x) the terminal win. NOT zeroed — the dense shaping is the
   pipeline's point (sparse-reward credit assignment).
2. **N 5 -> 8, K 8 -> 12 (96 eps/iter).** Lower-variance GRPO baselines. Peak
   memory is DECOUPLED from N*K (sequential episodes + grpo_microbatch=1), so peak
   stays ~46-50 GB; cost is wall-clock only (~32 min/iter, ~16 h/arm).
3. **secret_validity fail_open** (the new default; v2 pinned fail_closed only to
   match its pre-fix rot arm).

Pre-registered read-outs:
- **does guess rate recover?** with the goal reward no longer swamped, expect a
  higher/steadier guess% than v2, esp. in the control arm's clean single-role curve.
- **structural check:** even at w_ensemble=0.1, all-miss groups still have zero
  guess pressure (terminal flat). If guess rate stays low, the next lever is a
  small reward for *making a guess at all* (or an efficiency term not gated on a
  win) — flagged for exp-fullv4, not done here.
- health: parse/valid ~1.0 (validity now fail-open), fmt low, phi_mean=0, mem flat
  ~48 GB despite 2.4x the episodes (confirms the N*K/memory decoupling).

Stats: `python scripts/twentyq_run_stats.py tq-runs/q-fullv3-rot.jsonl tq-runs/q-fullv3-ctrl.jsonl --md tq-runs/REPORT-v3.md`.

### exp-fullv3 RESULTS (both arms completed 2026-07-14/15, ~16 h/arm)

Both arms ran 30/30 clean: `result=success`, parse=valid=1.00 every iter (fail-open
gate — zero false voids), fmt low, mem flat ~47 GB across 96 eps/iter (N*K/memory
decoupling confirmed a second time). Control ran fixed roles (A creates, B guesses
every iter) — the clean single-role read.

    run                 mean g%  first10  last10  slope/it  meanRs
    v3-ROT  (swap=5)     0.118    0.059    0.131   +0.0028   0.275
    v3-CTRL (swap=0)     0.118    0.069    0.154   +0.0034   0.274

v3-CTRL guess% per iter:
  0.02 0.06 0.04 0.05 0.25 0.01 0.03 0.05 0.10 0.06 | 0.21 0.15 0.23 0.10 0.10
  0.12 0.06 0.16 0.06 0.11 | 0.16 0.17 0.15 0.08 0.25 0.18 0.17 0.18 0.10 0.11

**Findings:**
1. **Reward rebalance did what it was designed to.** For successful episodes the
   ensemble-total fell from ~2.8x the terminal goal (v2 @ w=0.3) to **0.93x**
   (v3 @ w=0.1, n=341 wins: terminal 1.059 vs ensemble 0.983) — parity, goal no
   longer swamped.
2. **Learning is NOT broken — but modest.** The clean control trends UP (slope
   +0.0034/it; first-10 0.069 -> last-10 0.154, roughly doubling), plateauing
   ~0.15. The earlier "guess rate decreasing" fear was rotation noise, not real.
3. **Rotation ≈ control at w=0.1.** Both arms 0.118 mean with near-identical
   slopes — the rot/ctrl distinction that mattered in Q8/v1 collapsed at this low
   ensemble weight. A finding in itself.
4. **Cross-version guess% (all rotation arms, CONFOUNDED — audit/weight/N-K all
   differ):** v1-rot 0.233 (audit, w=0.3, N5K8) > v2-rot 0.154 (25 iters, no-audit,
   w=0.3) > v3-rot 0.118 (no-audit, w=0.1, N8K12). Lowering the ensemble weight did
   NOT raise guess rate — consistent with the ensemble dense reward being net-
   HELPFUL for guessing (it teaches narrowing), not merely a distractor. w=0.1 is
   likely a touch LOW.

**Caveat on attribution:** v3 changed w_ensemble AND N/K together, so v2->v3 isn't
a clean weight ablation (flagged at launch). A one-variable read (w=0.2, N5K8) is
the exp-fullv3b candidate.

**Open structural issue (unaddressed by reweighting):** ~64% of secret-groups have
zero wins, so their entire GRPO advantage is ensemble-driven with no pressure to
commit to a guess — independent of w_ensemble. The next lever is a small reward for
making a guess at all (or a closeness/efficiency term not gated on winning), not
further reweighting. Candidate exp-fullv4.

**Measurement note:** these guess% are noisy TRAINING rollouts. The in-loop
`validation_every` stationary eval (greedy, frozen-base answerer, sparse/dense/
combined split) added after v3 launched is the intended clean metric going forward;
neither v3 arm has validation rows (both predate that code).

## exp-fullv4 — rolling diversity + clean reward/rotation matrix (QUEUED 2026-07-15)

Config: `configs/twentyq-full-v4.yaml`. Four 60-iteration arms form a clean 2x2
matrix over role rotation and ensemble reward:

    arm                       credit      swap_interval
    q-fullv4-ctrl-terminal    terminal    0
    q-fullv4-ctrl-ensemble    ensemble    0
    q-fullv4-rot-terminal     terminal    5
    q-fullv4-rot-ensemble     ensemble    5

Run the no-rotation terminal arm first. `credit: terminal` is deliberately not
legacy broadcast: it uses the same terminal reward-to-go and same-turn-index
GRPO baselines as `credit: ensemble`, but adds no dense reward. Thus the reward
factor changes only the ensemble potential term; the rotation factor changes
only role assignment. The ensemble arms retain v3's `w_ensemble: 0.1`.

All arms use the new shared `recent_secret_window: 128`, N10 creator groups and
K16 solver groups (160 episodes/iteration, +67% from v3's N8/K12), fail-open
validity, and `grpo_microbatch: 1`. Rollouts advance the K sibling episodes in
lockstep with `generation_batch_size: 16`; frozen history scoring uses
`ensemble_batch_size: 4`. A 2026-07-15 Spark smoke benchmark selected these
settings (details in `DESIGN.md` §6.6) and completed with all models resident.
V3 measured 47.3 GB mean / 49.77 GB max. Peak backward memory remains bounded by
one trajectory, but a full v4 iteration has not yet been timed; verify the first
control before any further N/K increase.

Stationary validation runs before training at step 0 and after completed steps
10, 20, 30, 40, 50, and 60. `reward_log: true` writes a compact sidecar for every
training iteration and validation pass with terminal, ensemble/dense, combined,
and turn-zero-return aggregates. Validation computes the ensemble signal as a
diagnostic in all four arms; it is not retroactively applied to terminal-arm
training. Primary comparisons are fixed-validation guess rate and terminal
outcome; secondary read-outs are training guess rate, ensemble gain, repeat
rates, creator calibration, format failures, and peak memory.

**STOPPED (2026-07-16): arm A (`q-fullv4-ctrl-terminal`) halted at iteration 18
on creator secret collapse.** The rolling exclusion list was working as coded
but the creator ignored it, increasingly so with training: attempt-0
exact-repeat rate 0.0 -> 0.9, Okapi at 8/10 animal ranks by iteration 11 with
Okapi listed in the prompt's exclusion block, cumulative creator advantage
+3.51 on Okapi (+1.19 Extension cord, +1.30 Black Garlic). Root causes and the
full advantage accounting are in DESIGN.md §6.5; the prompt-only intervention
loses to the reward gradient it is embedded in, and with solver guess rate ~0
repetition of any obscure entity is reward-optimal. The 18 iterations DID
verify the batching pipeline: 23.4 min/iteration at 160 episodes vs v3's 36.2
min at 96 (2.6x per-episode throughput), peak 46.7-50.0 GB, no OOM.

**RESTART plan (labeled v4.5):** all four arms restart FROM SCRATCH (the
collapse is baked into the step-10/18 checkpoints, and arms must share one
reward definition) under run names `q-fullv45-{ctrl,rot}-{terminal,ensemble}`
— same `configs/twentyq-full-v4.yaml`, now with `repeat_handling: retry`
(DESIGN §6.5b):

**v4.5 OUTCOME (stopped 2026-07-16 at iterations 0-13, superseded by v5):**
`q-fullv45-ctrl-terminal` validated the gate — attempt-0 repeat rate 0.1-0.3
at the iterations where v4 hit 0.9, zero repeats played, coverage visibly
broadened — and surfaced three live evasion channels (verbatim ban slips via
BPE resegmentation, one-letter misspellings that PAID, Unicode decoration).
Full shakeout record and fixes in DESIGN §6.7. The matrix reruns from scratch
as `q-fullv5-{ctrl,rot}-{terminal,ensemble}` on `configs/twentyq-full-v5.yaml`
(identical settings; the delta is the hardened code path: string-level
`BannedStringsProcessor` + `repeat_matches` edit-distance/ASCII-fold gate).
Success signatures unchanged from the v4.5 plan; additionally expect
`repeat_retry_playable == repeat_retries` (no more leaky retries) and zero
played secrets within edit-distance 1 of the exclusion list. attempt-0 repeats against the
in-prompt exclusion list are voided at the repeat gate (no episodes, reward
0.0 in the creator group) and re-sampled ONCE at creator_temp with the
excluded secrets masked to -inf at the logits level (string-level
`BannedStringsProcessor`, segmentation-proof; the gate matcher additionally
applies normalized edit-distance-1 for 5+ chars); a playable retry earns its
real game reward, a still-matching retry voids the rank. Attempt-0 repeat-rate telemetry keeps v4
semantics and stays directly comparable to the stopped run; new per-iteration
read-outs are `playable_rate`, `repeat_voided`, `repeat_retries`,
`repeat_retry_playable`, and `sampled_secrets` (which also makes resume
restoration exact). Expected signatures if the gate works: attempt-0 repeat
rate DECLINES over iterations instead of climbing, playable_rate recovers
toward 1.0, and no secret accumulates a large positive cumulative advantage.

**v5 OUTCOME (stopped 2026-07-16 at iterations 0-5, superseded by v6):**
`q-fullv5-ctrl-terminal` — the hardened repeat gate was healthy (novel
secrets, sensible difficulty ordering, parse/valid 1.00), but the run was
stopped for solver signal starvation: 31/912 episodes won (3.4%), 49/57 GRPO
groups all-loss ⇒ zero terminal advantage ⇒ the solver's only gradient was
format penalties while the creator trained one-sided. Root causes from the
transcript audit (DESIGN §6.8): the guesser was never told wrong guesses are
free (one forced guess per episode), prose-then-contract-line bleed-through
("QUESTION: Yes" polluting histories), and grammar-sensitive category
phrasing (83% degenerate first questions for "a animal" vs 2% for "a
household object"). A first false-alarm start of this arm (stopped at step-0
validation only) is archived in tq-runs/aborted-fullv5-falsealarm/.

## exp-fullv6 — tightened prompts + stationary ensemble, ensemble-first (RUNNING 2026-07-17)

Config `configs/twentyq-full-v6.yaml` (settings identical to v5; the delta is
prompt/reward-surface only, DESIGN §6.8): guesser contract line is the entire
reply, wrong-guess-is-free stated, article-free category phrasing,
constraint-consistent forced guess; ensemble scoring prompts renamed to
"20 Questions" and template dates pinned (SmolLM3/Llama were interpolating
today's date — reward non-stationary across midnight). Environment + reward
scale changed ⇒ v6 baselines are NOT comparable to earlier lineages; step-0
validation re-baselines.

Run order flipped: **arm B first** — `q-fullv6-ctrl-ensemble` (credit
ensemble, swap 0, full 4-model roster explicit in config). Rationale:
terminal-only credit starves the solver at the base win rate; the dense
per-turn channel is the arm that can bootstrap it. A/C/D follow only on
explicit decision.

What to watch (beyond the standing gate signatures, which carry over):
- Episode win rate and share of GRPO groups with terminal variance — the
  point of the prompt changes; v5 baselines were 3.4% and 8/57.
- Early-guess behavior: wins at turns < 19 (v5: only Toaster), wrong-guess
  turns appearing mid-episode in transcripts ("Is it X?" -> NO lines).
- Degenerate first questions by category — the 83%/21%/2% split should
  collapse if the phrasing fix is right.
- dense/terminal decomposition in the rewards sidecar: dense should give
  nonzero within-group variance even where terminal is all-loss.
- Ensemble memory/wall-time: policy + 4 frozen scorers resident; v3-era
  smoke peaked ~43-50 GB — expect the high end plus generation batching.

### exp-fullv6 RESULTS — arm B `q-fullv6-ctrl-ensemble` (completed 2026-07-19)

60/60 iterations clean: zero errors, zero restarts, ~31.4 h wall (~31 min/iter
including 7 validation passes), memory 46.6-50.3 GB peak against the 100 GB
cgroup cap. parse=valid=1.00 every iteration (fail-open gate, zero false
voids). Training episodes won: 688/8944 (7.7%), first-10 mean 0.040 ->
last-10 0.089. Artifacts: `tq-runs/q-fullv6-ctrl-ensemble.jsonl` (+ rewards
sidecar + transcript tree), checkpoints `checkpoints/twentyq-fullv6-ctrl-ensemble`.

Stationary validation (12 secrets, greedy, base answerer), B = solver:

    step        0      10     20     30     40     50     60
    wins/12     2      0      2      1      3      2      2
    dense     0.575  0.471  0.562  0.563  0.622  0.616  0.616

**Findings:**
1. **All three prompt fixes landed.** Degenerate first questions 83% (v5
   animal) -> 0%; mid-episode guessing appeared immediately (33-73 wrong-guess
   turns/iter) and converted (wins as early as turn 9 vs v5's single forced
   final guess); dense per-turn credit produced nonzero advantages inside
   all-loss GRPO groups (solver grad_norm ~0.04 every iteration) — the v5
   signal-starvation mechanism is gone.
2. **In-distribution learning, no measurable transfer.** Training win rate
   roughly doubled (last-10 0.089 vs first-10 0.040; category records
   household 46/160, animal 15/160) but fixed-validation wins ended exactly
   where they started (2/12, series above). Style DID transfer: greedy
   turns-on-success 16.5 vs 21.0 at step 0, the first-ever household
   validation win (step 40), final dense +7% over baseline. Reading: the
   solver learned to beat its co-adapting creator; whether a small real
   transfer exists is below the resolution of 12 greedy episodes.
   -> `data/twentyq-validation-v2.json` doubles the set to 24 secrets
   (superset; the 12 new entries are verified absent from this run's 308
   distinct training secrets), and a sampled-eval sweep is queued.
3. **Hardened gate held; the evasion arms race is documented end-to-end.**
   Zero played repeats. Rungs: duplicate -> misspelling ("Wasbi") -> accent
   ("Pangolín") -> space-split ("Ok API") -> foreign letter ("Axolotل") — all
   caught/voided — with 4 borderline breakthroughs that PLAYED (Sunchyon,
   Xerophyte, Pangol., Pinguin; all <= iter 16, none after). Two Okapi-style
   collapse episodes recurred (iters 13, 22) but the gate inverts the v4
   economics (repeats now take the 0.0 gate reward against a positive group
   baseline), producing oscillating recovery: from iter 37, diverse slates
   with zero voids.
4. **Creator calibration asymmetry (structural, unaddressed).** The
   calibration reward pays ~1.0 for a 0-to-1-win slate (iters 33, 57): while
   solver guess rate is low, hard-rank targets (0.1) are met by ANY obscure
   pick, so obscurity is reward-optimal at most ranks and the exclusion gate
   does not counter it. Candidate v7 lever.
5. **Format fails are purely sampling-tail.** Sampled rollouts 3-40%/iter
   (mean 14.4%); greedy validation 0% at all 7 steps. Two regimes:
   weird-secret concentration (iter 6) and stereotyped attractors on sane
   slates (iter 35+: bare "Towel" guesses, a Japanese prefix). Fails cluster
   at discrete turns because lockstep turn-indexed prompts synchronize any
   turn-sensitive failure. NOTE uncovered during closeout: solver sampling
   silently inherits **top_k=64** from gemma-4-E2B's generation_config.json —
   never set in any config, never recorded in run metadata, active in every
   sampled rollout of the lineage. Sampler sensitivity probe:
   `scripts/probe_twentyq_base_sampling.py`.
6. **Validation-set leakage note:** pangolin is both `val-v1-animal-03` and
   this run's most-sampled training secret (20 exact + 7 "Pangolín"). The
   description forbade using validation entries as training secrets but
   nothing enforces it against the creator's own sampling. v2 keeps the entry
   (series continuity); treat its cell with suspicion, and a creator-side
   exclusion of validation entries is a cheap v7 fix.

---

## exp-v7-solveronly — frozen creator + flat-rate bank (STOPPED 2026-07-25)

Diagnostic, not a matrix arm: v6's three confounds removed at once
(`freeze_creator: true`, `difficulty_mode: flat`, `credit: terminal`), everything
else held at v6 values. Config `configs/twentyq-v7-solver-only.yaml`, run
`q-v7-solveronly-flat90-terminal`. From a cold start the frozen creator's LoRA is
zero-init, so the opponent was the frozen base model throughout — a stationary
secret generator AND a stationary oracle.

Launched at `flat_target_rate: 0.5`, scrapped after validation step 0 (user
decision), relaunched at **0.9**. Stopped by the user at iteration 21/60.

    21 iterations, 7.95 h wall, 21.7 min/iter, mem 43.2-48.4 GB
    parse_ok 1.000 every iteration; zero errors, zero restarts
    training episodes won 119/3024 (3.9%); format_ended 615/3024 (20.3%)
    193 distinct secrets; repeat rate mean 0.281, max 0.70, 5 rounds >= 0.60

Stationary validation (v2, 24 secrets, greedy, base answerer), B = solver:

    step               0        10       20
    A wins/24          3         3        3      <- frozen control, byte-identical
    B wins/24          3         3        3
    B turns_on_succ  18.00    17.33    16.33
    B terminal_mean  0.1339   0.1349   0.1364

Pooled by category (the run's strongest result):

    household object   6 rounds   113/944  = 0.1197   mean_repeat 0.200
    food              10 rounds     4/1424 = 0.0028   mean_repeat 0.210
    animal             5 rounds     2/656  = 0.0030   mean_repeat 0.520

**Findings:**

1. **NULL on the stated criterion; a real secondary signal.** §7.7 said read
   success only as a rise in B's validation guess rate. It did not move (3/24 at
   all three steps). But adapter A — frozen, never updated — reproduced *byte-
   identically* at every checkpoint (same secrets, same 20.62 turns, same 18.00
   on success), making it an exact control, and against it B moved monotonically
   in three places: turns-on-success 18.00 -> 16.33, mean turns 20.62 -> 20.42,
   terminal_mean 0.1339 -> 0.1364. The solver learned to win the SAME games
   faster, not to win more. Thin (3 wins) but monotonic against a perfect
   control. **Keep the frozen-adapter control in future designs** — it is what
   made a small effect legible.
2. **Category effect dominates everything, 40x.** household 0.1197 vs food
   0.0028 vs animal 0.0030, on 3,024 episodes. Same prompt, same target: the
   model's obscurity ladder is shallow for household objects and near-vertical
   for animals/food. Any v7-lineage conclusion drawn without a per-category
   split is unsafe.
3. **The flat target INVERTS its own intent at extremes.** `prompts.py:81-87`
   states the percentage then immediately adds a hardcoded aim-for-the-middle
   clause ("not so obvious it is named in a few questions") that does not scale
   with `target_rate`. At 0.90 the two directly contradict, and the creator
   obeys the imperative over the number — in its own notes: *"known but not
   universally recognized **like a tiger or elephant**"*, *"isn't immediately
   obvious"*. It is explicitly rejecting the cow/penguin-class secrets
   (0.979/0.896 per DESIGN §7.6) that would deliver 90%. Compounding it,
   `prompts.py:104` bakes `"difficulty": 0.10` into the response template while
   the prose says 90%; the model's echo flip-flops between the two. Raising the
   target made the bank HARDER. This is v6 finding #4 (obscurity is
   reward-optimal) reappearing through a different mechanism.
4. **Frozen creator x string-level ban manufactures non-words.** The creator
   mode-collapses onto a favourite; the §6.5b ban blocks the literal string; the
   model emits a phonetic near-miss rather than a new concept:
   Okapi -> `Okoupee`/`Okoupes`/`Okoupy`/`Okoupi`, Spatula -> `Spatuloid`,
   Fenugreek -> `Fennugel`, Sunchoke -> `Sunchyon`. 8 of 193 distinct secrets
   were corruptions or category errors (`Cardiogram` declared as food).
   The v6 arms race (finding #3) is the precedent — `Sunchyon` PLAYED there too
   — but v6 recovered ("from iter 37, diverse slates with zero voids") because
   its creator was TRAINING. **A frozen creator cannot self-correct, so the
   collapse is permanent.** This interaction is not anticipated in DESIGN §7.3,
   which treats freezing as purely skipping updates.
5. **`fail_open` converts judge failures into valid secrets — and the failures
   correlate with the corruptions.** 21/206 validity calls (10.2%) produced no
   parseable `VERDICT:`, so `judge.py:112` passed them. Two modes: role-playing
   the guesser (`Axolotl -> "1. Is it a mammal?"` — the known flakiness
   `fail_open` was built for) and, on unrecognised words, free-associating a
   real one (`Okoupi -> "maraca"`, `Sunchoke -> "maritime"`, `Cardoon ->
   "maroon"`, `Spatuloid -> "marbles"`). The fail-open rationale is inverted
   here: it fails open exactly where the secret is invalid.
6. **Corrupted secrets play 16 episodes against an incoherent oracle.** With no
   referent the answerer self-contradicts inside one game —
   *"Okoupee is a type of antelope"* then *"'Okoupee' is a fictional creature"*
   — and leaks the secret (*"(The secret is Okoupee, an animal.)"*). This is
   label noise in the training signal, not a hard game.
   Evidence: `q-v7-solveronly-flat90-terminal.transcript.txt:90035-90731`.
7. **The i.i.d. category sampler fabricates trends.** `trainer.py:581` is
   `rng.choice(categories)` with no balancing. This run drew food 8x in
   iterations 0-9 and household 5x in 10-20, producing an apparent
   0.020 -> 0.058 first-half/second-half "improvement" that is **pure mix
   artifact**. Recorded because it looks like learning in the log and is not.
8. **Rank-0 exhaustion (benign).** Household rank 0 went 0.88 -> 0.81 -> 0.00 ->
   0.00 as obvious items entered the 128-secret window. Winnable secrets
   relocate to other ranks and round means held 0.02-0.22, so total winnability
   is preserved — consistent with §7.4's claim that rank no longer implies a
   ramp. Round-to-round `guess%` variance on a 10-secret bank is large enough
   (household spanned 0.02-0.22) that **no trend should be read from it**;
   judge on validation and pooled aggregates only.
9. **`credit: terminal` does NOT free the ensemble footprint.** `trainer.py:461`
   (validation path) calls `_ensemble_potentials()` with no
   `credit == "ensemble"` guard, unlike the guarded training path at
   `trainer.py:881`. All 4 models load and score every history prefix, then get
   multiplied by `w_ensemble=0.0`. Numerically inert (validation `dense=+0.000`)
   but costs load time and validation compute. The claim to the contrary in the
   v7 config header has been corrected in place.

**Recommended fixes (priority order):**

*Tier 1 — data integrity; without these a rerun trains on garbage*

- Stop retry emitting non-words: validate the masked-retry output and DROP the
  secret on failure, or set `repeat_handling: void` (skip the resample entirely).
- `secret_validity: fail_closed`, ideally with retry-once-then-reject on the
  judge call so genuine role-play flakiness is still recovered.
- Fix the judge prompt: it still reads *"Think briefly, then end with exactly
  one line: VERDICT:"* while `judge_thinking: false` bans reasoning at the
  decoder — the contradiction DESIGN §8.3 identified and resolved only on the
  decoder side. Note the evidence says unfamiliarity, not the thinking ban, is
  the primary trigger for finding #5, so treat this as removing a known
  contradiction rather than as the fix.

*Tier 2 — required for the experiment to mean anything*

- Make the flat dictation scale with `target_rate` (high target -> "common,
  immediately recognizable"; low -> "obscure"; middle language only near 0.5),
  and drop or correct the hardcoded `difficulty` in the response template.
- Until that lands, **0.50 is the safer target** — the middle-clause is
  accidentally correct there, which is why the original config specified it.
- Balance the category draw (round-robin or stratified) and always report
  per-category.

*Tier 3*

- Guard `trainer.py:461` on `credit == "ensemble"` to reclaim the 4-model
  footprint.
- Shorten `recent_secret_window` (128 -> ~32) for frozen-creator runs
  specifically: a fixed repertoire plus a long window guarantees the collisions
  that feed finding #4.
- Enforce creator-side exclusion of validation entries (v6 finding #6, still
  unfixed — `pangolin` and `penguin` recurred as training secrets here).

**Design note.** v7 removed three confounds at once and introduced a fourth
(frozen creator x repeat ban -> corrupted data). Suggested next step: rerun
**household-only**, the one category where the machinery demonstrably works, to
get a clean read on whether the solver improves at WINNING at all before
reintroducing animals/food.

Artifacts: `tq-runs/q-v7-solveronly-flat90-terminal.{jsonl,rewards.jsonl,console.log}`
+ transcript tree; checkpoints `checkpoints/twentyq-v7-solveronly/adapter_{A,B}_step{10,20}.safetensors`
(resume with `--resume-step 20`).

## exp-v8-diagnostics — what the solver was actually blocked on (2026-07-26)

No training. Four frozen-model probes on `data/twentyq-validation-v2.json`
(24 secrets, K=8 sampled episodes, gemma-4-E2B, no adapters) run to find out why
v5/v6/v7 all produced flat solvers. Harness: `scripts/probe_twentyq_headroom.py`,
analysis `scripts/analyze_headroom_probe.py`, raw
`tq-runs/probe-headroom-v1.json` (pre-parser-fix) and `-v2-*.json` (post).

### The finding

    arm (pre-fix parser)      win   early    fmt  distinctQ  usable_grp
    baseline               0.104   0.099  0.255      0.934       0.125
    dedup                  0.161   0.099  0.198      0.987       0.292
    oracle (Qwen3-8B)      0.068   0.062  0.146      0.935       0.208

    arm (fixed parser)        win   early    fmt  distinctQ  usable_grp
    baseline               0.115   0.089  0.005      0.928       0.250
    dedup                  0.146   0.115  0.016      0.975       0.167

`usable_group_rate` is the fraction of secrets whose K episodes contain both a
win and a loss. A solver GRPO group IS one secret's K episodes, so an all-win or
all-loss group has identically zero advantages and trains nothing. At 0.125,
**seven eighths of every iteration's compute produced no gradient.**

Per-secret win probability is bimodal — cow 8/8, penguin 7/8, then 0/8 for
twenty of twenty-four. The variance GRPO needs lives BETWEEN secrets, where the
per-secret baseline cannot reach it, and not WITHIN one, where it can.

This subsumes the earlier explanations. v5's "credit starvation" (49/57 all-loss
groups) is this. It is not a reward-design problem: `terminal`, `ensemble` and
`per_turn` all multiply an advantage that is already zero.

### `usable_group_rate` is not monotonic in win rate

The single most important operational consequence, and it was a surprise:

    parser fix only      win 0.115   usable_grp 0.250
    parser fix + dedup   win 0.146   usable_grp 0.167

Dedup pushed penguin and cow to 8/8 — out of the usable band. **Improving the
solver moves secrets out of the band.** A frontier bank is a moving target that
decays as the policy improves, which is the real argument that creator
calibration is not polish but the mechanism keeping solver training alive.

### Negative results

- **A stronger oracle does not help.** Qwen3-8B as answerer: win 0.104 -> 0.068,
  episodes running to full budget 0.641 -> 0.786. A stricter oracle starves the
  guesser of information rather than misleading it. It did raise usable_grp
  (0.125 -> 0.208) by pulling extremes toward the middle, but dedup beats it on
  both axes and costs no second model.
- **The oracle is not lying.** Literal self-contradiction rate across all 3,312
  v7 episodes / 57,901 answered questions: **1.39%**. The hypothesis came from
  one giraffe transcript ("hoofed mammal" YES / "hoofed ungulate" NO) and did
  not generalize.
- **Fixing the parser barely moves win rate.** Recovering 88.3% of v7's 632
  format failures took the format-kill rate 0.255 -> 0.005 and doubled
  usable_grp, but win rate only 0.104 -> 0.115. Those episodes were mostly
  doomed anyway. Worth having for reward hygiene (they were drawing spurious
  `-w_format`), not as a win-rate lever. This corrects an earlier claim in this
  session that parse loss was the largest lever available.
- **The "parrot lock" is a greedy-decoding artifact.** Prominent in greedy
  validation transcripts; ~6.6% of turns under the sampled decoding training
  uses.

### Caveats on the metric itself

- Answerer and guesser are the same base model, and validation uses that same
  base answerer, so part of "win rate" is *my twin answers the way I predict*
  rather than 21-questions skill. Inherent to self-play, but the number is not
  pure ability.
- ~~`early_win_rate` is ~0.10 and barely moves: the policy has essentially no
  "guess now" decision, and almost every win is the forced final guess.~~
  **CORRECTION (2026-07-26): this was backwards.** `early_win_rate` is
  `(wins - won_on_last_turn) / n`, so an early rate of 0.099 against a win rate
  of 0.104 means 95% of wins were *voluntary*, not forced. Counted directly:

      probe            wins   forced final   share
      v1 baseline        20        1          5%
      v2 gemma baseline  22        5         23%
      v2 gemma dedup     28        6         21%
      v2 smollm3          8        3         38%
      v2 llama32          6        5         83%

  gemma's voluntary wins are spread over turns 11-20, so it does have a
  "guess now" decision and exercises it. `w_efficiency` is therefore sharpening
  an existing behaviour rather than creating a missing one — still worth
  watching in v8, but not for the stated reason.
- v1..v7 validation (24 greedy episodes, ±13 points) could not have resolved
  these effects. See DESIGN §9.4 — "no transfer to validation" was never
  established by that data, so nothing here should be read as *explaining* v6.

### The creator's natural output distribution (sizes the bank build)

Measured on the v7 run record — 196 draws, 174 distinct secrets, dictated a
**90% guess rate on every one of them**:

    mean per-secret win rate      0.034
    secrets at exactly 0/K        148 of 174   (85%)
    secrets at 1.0                0
    secrets in [0.125, 0.875]     9.2%

Asked for 90%, the creator delivered 3.4%. This is the same failure as the
group-variance wall seen from the other side: the creator's output is not
merely miscalibrated, it is *concentrated on unwinnable*, so the band has to be
found by measurement rather than requested.

Three consequences, all acted on:

- The generation target sweep dropped 0.7 and 0.5 (they buy nothing but hours
  of calibration spent confirming 0/8); it is now 0.95/0.9/0.8.
- 9.2% is the yield **floor** for the bank build, not the estimate. It was
  measured under the flat-mode prompt bug — an unconditional "not so obvious
  ... not so obscure" clause that contradicted its own stated target, so the
  run asking for 90% was simultaneously told to aim for the middle. Fixed.
- For priority two this is the number to beat. A creator that has learned
  calibration is one whose dictated target predicts the measured rate; the
  baseline for that claim is a 0.87 gap.

### Sub-4B base-model bake-off

All arms: `dedup` (question_retries=4), validation-v2, K=8, n=192 episodes,
post-parser-fix so the comparison measures 21-questions ability rather than
markdown habits.

    model              win            early   fmt   budget  distinctQ  usable_grp  wall
    gemma-4-E2B  0.146 [0.10,0.20]    0.115  0.016   0.839    0.975      0.167     688s
    SmolLM3-3B   0.042 [0.02,0.08]    0.026  0.000   0.958    0.995      0.208    2389s
    Llama-3.2-3B 0.031 [0.01,0.07]    0.005  0.000   0.969    1.000      0.167     906s

**gemma-4-E2B wins and it is not close** — 3.5x the win rate of either
challenger and 3.5x faster than SmolLM3 (challengers lose more, losing episodes
run the full 21-turn budget, so weakness costs wall-clock too). The CIs of the
two challengers overlap each other but not gemma's.

Two traps in this table worth stating, because both would mislead a reader
scanning for the best number:

- **SmolLM3 has the highest `usable_group_rate` (0.208) and is the worst
  choice.** Four of its five nonzero secrets sit at exactly 1/8 — "barely
  winnable", not "well matched". And since the bank is calibrated per model,
  usable_group_rate on a fixed secret set is largely equalised by construction;
  it is a property of the set-model pairing, not of the model.
- **`fmt` favours the losers spuriously.** SmolLM3 and Llama score 0.000 format
  failures against gemma's 0.016, but an episode that never reaches a confident
  guess has fewer chances to break the output contract. Cleanliness here is a
  symptom of not playing, not of formatting discipline.

The qualitative split is sharper than the rate. Counting where wins land:
gemma's are spread over turns 11-20 with only 21% on the forced final turn,
whereas Llama-3.2 won 6 games total, 5 of them on the forced final guess, with
a single voluntary win at turn 6. gemma is the only candidate with a real
"guess now" behaviour for RL to amplify — which matters more than the headline
rate, since reinforcement sharpens existing behaviour rather than inventing it.

Decision: **gemma-4-E2B**, unchanged from v1..v7. Also the model the LoRA
text-decoder pinning is already tuned for (it is multimodal; the towers must
not be adapted).

### The v8 frontier bank (built 2026-07-26)

    750 creator rollouts  (0 parse failures)
    -> 118 unique         (15.7% distinct; dedup + validation holdout)
    -> 118 judge-vetted   (100% pass — see the caveat below)
    -> 61 in band         (51.7% yield, [0.125, 0.875] at K=8)

Balanced: animal 22, food 15, household object 24. Spread over the whole band
(wins/8 → 1:17 2:12 3:8 4:6 5:9 6:6 7:3), mean measured rate 0.391, no
duplicates, verified disjoint from validation-v2.

**What it buys — the entire premise of v8.** Expected `usable_group_rate` at
K=16 on this bank is **0.95**, against **0.125** measured on the v1..v7 secret
source: 7.6x more GRPO groups carrying a non-zero gradient per iteration. That
number is posterior-corrected — each secret's rate is only known to K=8, so the
estimate integrates `1 - E[p^16] - E[(1-p)^16]` over Beta(w+1, 8-w+1) rather
than trusting the point estimate. The correction is small (0.958 → 0.946)
because at K=16 the band is wide enough that K=8 noise barely matters.

Yield was 51.7% against the 9.2% floor measured on v7 output — the flat-mode
prompt fix plus dropping the 0.7/0.5 generation targets. Qualitatively visible:
at a dictated 0.9 v7 produced *saffron, truffle, black garlic, salsify,
habanero pepper, axolotl*; the fixed prompt produces *Lion, Elephant, Dog*.

**Two revisions to earlier beliefs:**

- Generation diversity, not yield, is the binding constraint on bank size. 750
  rollouts gave only 118 distinct secrets (15.7%), because targets at the easy
  end draw from a small vocabulary of instantly-recognizable entities. Yield
  per candidate is no longer the thing to optimise; candidate *count* is.
- **The "40x per-category win-rate spread" (household 0.120, food 0.003) was
  largely an artifact of what the creator picked, not of the categories.** On
  familiar-target output the per-category in-band rates are 56%/65%/60% and the
  mean rates 0.46/0.43/0.31. Food was not intrinsically hard; v7's food secrets
  were saffron and salsify.

### The validity judge is PARTIAL — do not cite the vet rate as evidence

The build vetted 118/118 VALID with zero rejections, which is exactly what a
rubber stamp looks like. Driving the same judge (Qwen3-8B, same prompt, same
terse-retry fallback, `fail_closed`) over deliberate negatives:

    REJECTED correctly   Okoupee, Flimberwocky, happiness, mammals
    FALSE ACCEPTS (4/8)  water (as food), "a dog or a cat", Pikachu,
                         Xenoturbella churro
    false rejects        0/6

So it catches non-words, abstractions and multi-entity *classes*, and misses
category mismatches, multi-entity *answers*, fictional-only entities and
invented obscure names. Consequences:

- The bank's quality guarantee is the **measured band**, not the judge. Two of
  the four miss-classes are caught by measurement anyway (obscure and
  multi-entity secrets score 0/8 and fall out); category mismatches and
  fictional entities are not.
- Hand-inspecting all 61 entries found exactly that: `Water` sits in the food
  category — the case the creator system prompt names as invalid and the judge
  false-accepted — alongside `Milk`/`Coffee`/`Milkshake`, an ambiguous
  `Chicken`, `Dinosaur` (a clade), and the confusable pairs
  `Toaster`/`Toaster oven` and `Toothbrush`/`Toothbrush holder`.
- Kept as built rather than hand-curated. Every one of these is measurably
  winnable-but-not-always, which is the actual selection criterion; `Water`
  scored 4/8. Curating on taste after measurement would make the bank
  unreproducible. Recorded here so the run stays interpretable.
- **This blocks a naive priority-two design.** A creator rewarded for producing
  "valid" secrets cannot be refereed by this judge alone — it would pass
  `a dog or a cat` and `Pikachu`.

### Actions taken

DESIGN §9. Parser recovers markup/quote/bare-entity turns; `question_retries`
+ local `w_repeat`; `secret_source: bank` with deck dealing and category
balance; `validation_episodes` + Wilson CIs; flat-mode difficulty steer now
tracks its own target; validation ensemble call guarded.

Hardening found while sizing the bank build (all pre-run, none of it measured
yet): the bank builder promised resumability and had none, so calibration now
checkpoints per candidate behind a signature gate; the judge model is dropped
before the calibration loop instead of held through it; and bank mode no longer
re-judges pre-vetted secrets with the small base model — that gate polices the
creator, and under `fail_closed` its ~10% unparseable verdicts were deleting
bank entries outright (confirmed: 0 of 4 episodes played). `usable_group_rate`
is now first-class iteration telemetry, since the run must not be altered on
telemetry and the metric moves independently of the win rate. 280 tests pass.

Next: build the calibrated bank, then run `configs/twentyq-v8-bank-solver.yaml`.

## exp-v8 — in-flight findings (run live, 2026-07-26)

Reported at iteration 18 of 60. The run is untouched; these are read-only.

### The wall is down, and the solver learns

`usable_group_rate` measured **0.938** across iterations (v1..v7 secret source:
0.125). Within-secret training win rate rose **+0.118 ±0.048** over 49 secrets
played in both halves of the run — same secrets on both sides, so neither the
draw nor regression to the mean applies. v5/v6/v7 produced nothing comparable.
The §9.1 diagnosis is therefore confirmed rather than merely assumed.

### Validation is flat, and three explanations were tested

    step        A (frozen)              B (solver)
       0    0.141 [0.098,0.197]    0.141 [0.098,0.197]
       5    0.109 [0.073,0.161]    0.141 [0.098,0.197]
      10    0.120 [0.081,0.173]    0.172 [0.125,0.232]
      15    0.130 [0.090,0.185]    0.141 [0.098,0.197]

B's spread (0.031) equals the frozen control's (0.031). The step-10 rise did
not survive to step 15.

- *Instance memorization* — **ruled out**. Regressing each secret-play's delta
  on iteration + exposures-to-that-secret + calibrated rate gives exposures
  −0.082 ±0.085 (n.s.) against iteration +0.021 ±0.012 (t=+3.5). A matched
  within-iteration contrast (same training time, secrets on 1st vs 2nd
  exposure) gives −0.117 ±0.121. Neither shows a positive exposure effect, so
  improvement is not secret-specific.
- *Distribution shift* — plausible, untested. The bank is familiar entities
  (Lion, Dog, Cheese) because generation targets were pushed to the easy end;
  validation-v2 is deliberately obscure (pangolin, rambutan, couscous,
  shoehorn, trivet). "Guess common things" would raise bank rate without
  touching validation.
- *The instrument cannot see it* — **confirmed, and this is the big one.**

### validation-v2 is underpowered by COMPOSITION, not sample size

Adapter B's per-secret wins/8 across all four checkpoints: **15 of 24 secrets
are 0/8 at every checkpoint** (platypus, pangolin, kangaroo, hedgehog, tapir,
pizza, hummus, rambutan, popcorn, couscous, persimmon, stapler, smoke detector,
shoehorn, corkscrew), and cow is pinned at 8/8. The entire dynamic range lives
in ~7 secrets.

**This is the group-variance wall applied to the measuring instrument.** A
secret that is unwinnable at every checkpoint contributes no information about
whether the policy improved, exactly as an all-loss GRPO group contributes no
gradient. Raising `validation_episodes` from 1 to 8 fixed the sample size and
could not fix this, because the limit is which secrets are in the set.

~~Caveat against over-claiming: validation's *reachable* subset (television,
penguin, banana, bread, refrigerator, scissors) is also roughly flat, so
resolution alone does not explain the null.~~ **RETRACTED — see the MDE section
at the end of this file.** The live subset's minimum detectable effect is
+0.18; its being flat is precisely what resolution predicts, not evidence
against it. The defensible statement is simply that training improved and
validation-v2 cannot resolve whether it transferred.

Action: `validation-v3`, built like the bank (play candidates, keep those in a
resolvable band), held out and fixed in advance; then re-score the saved v8
checkpoints against it. NOT swapped mid-run — v8 finishes on the instrument it
started with, for the pre-registered criterion and v1..v7 comparability.

### Bank decay is starting

40% of recently-played secrets (22 of 55) now sit at a band edge — 11 at/above
0.875 (Tiger, Shark, Dog, Monkey, Chocolate, Toaster, ...), 11 at/below 0.125.
`usable_group_rate` has not yet moved, because at K=16 a secret at p=0.875
still yields a mixed group 88% of the time. It is a leading indicator, not yet
a cost.

### UPDATE at iteration 25: the null is uninformative, and one retraction

Six checkpoints in, with `scripts/validation_power.py`:

    step        A (frozen control)      B (solver)             B-A
       0    0.141 [0.098,0.197]    0.141 [0.098,0.197]      +0.000
       5    0.109 [0.073,0.161]    0.141 [0.098,0.197]      +0.031
      10    0.120 [0.081,0.173]    0.172 [0.125,0.232]      +0.052
      15    0.130 [0.090,0.185]    0.141 [0.098,0.197]      +0.010
      20    0.151 [0.107,0.209]    0.146 [0.103,0.203]      -0.005
      25    0.172 [0.125,0.232]    0.188 [0.139,0.249]      +0.016

B posted its best value at step 25 — and so did the FROZEN control. A's weights
are byte-identical at every checkpoint, so its entire 0.062 swing is sampled-
decoding noise, and it now exceeds B's total rise from step 0 (+0.047). That is
the cleanest possible demonstration that this instrument cannot see an effect
of the size in play.

Quantified, paired because both adapters are scored on the same secrets:

    paired sd of per-secret (B - A)   ALL 0.1159    LIVE 0.1946
    minimum detectable effect         ALL +0.0685   LIVE +0.1817

    vs the +0.148 within-secret training gain:
      pooled   0.148 x (9/24 live) = +0.0555  vs +0.0685  -> INVISIBLE
      live     0.148                          vs +0.1817  -> INVISIBLE

**Even perfect transfer would have produced this series.** The v8 null is not a
negative result; it is an absent measurement. The pre-registered criterion was
correct as a guard against reading trends out of noise — v6 and v7 both did
exactly that — but it could never have fired here.

**Retraction.** The "reachable subset is also flat, so resolution alone does not
explain the null" caveat above is wrong and is struck through. The live
subset's MDE is +0.18, so its flatness is what resolution *predicts*. Checked
for hidden signal and there is none either: on the 9 live secrets B ends below
its own control (0.389 vs 0.403 at step 20), mean turns-on-success falls
equally for both (B 16.77->16.40, A 17.56->17.13), and off-the-floor events are
2 for B against 1 for A. The instrument is silent, not the policy.

**This is the third time in this project that a conclusion was read off an
instrument that could not support it** — v6's 2,0,2,1,3,2,2 "trend", v7's
per-iteration `guess%` (which mostly measured which category came up), and now
v8's null. The generalisable lesson is to compute the minimum detectable effect
BEFORE interpreting any result, not after: `validation_power.py RUN.jsonl
--effect <the effect in play>`.

Consequence: validation-v3 is sized rather than hoped for. At paired sd 0.19 an
all-live set needs ~27 secrets to resolve +0.10 and ~56 for +0.07, so
`run_validation_v3_build.sh` hard-fails below 30 (MIN_V3). The sd is estimated
from v2's live secrets, which sit mid-range where per-secret variance is
highest, so that planning number is mildly optimistic — overshoot on size.

## exp-v9 — bank-v2 solver, CRITERION MET (2026-07-28 → 08-02)

Config `configs/twentyq-v9-bank2-solver.yaml` (its header is the document of
record), records `tq-runs/q-v9-bank2-solver.jsonl`, checkpoints
`checkpoints/twentyq-v9-bank2/`. 60 iterations, 39.4h, exit success.

Inputs: `data/twentyq-bank-v2.json` (138 secrets, calibrated by playing
adapter_B_step60, hinted generation) and `data/twentyq-validation-v5.json`
(82 secrets, base-calibrated, composed — see below). Verified disjoint under
edit-distance-1.

### Result

    step        A (frozen)   B (solver)    B - A
       0          0.3350       0.3350     +0.0000
      15          0.3350       0.3516     +0.0166
      30          0.3350       0.3733     +0.0383
      45          0.3350       0.3872     +0.0522
      60          0.3350       0.3945     +0.0595

    PRIMARY (pre-registered): slope of B-A on step
      +0.113 pts/step, t=+4.505, dof 11, crit 2.201          MET
    SECONDARY: B-A at step 60 > 0
      +0.0595, paired per-secret t=+2.21, crit 1.99, n=82     MET

    within-secret training, first 10 iters vs last 10: +0.148 ±0.060
    transfer ratio: ~46%  (v8 was ~34% and not significant)

Robustness: leave-one-out slope t stays in [+3.82, +5.48]; effect by source is
+0.056 (v3) / +0.099 (v4) / +0.037 (bank-v1 leftovers), so it is SMALLEST in
the only component carrying contamination risk; frozen control identical to
four decimals at all 13 checkpoints.

Not established: sign test p=0.058, 27 of 82 secrets unchanged, absolute win
rate 0.395. See DESIGN §10.4.

### Gradient supply reversed direction

    first 10 iterations -> last 10
      v8, base-calibrated bank     0.938 -> 0.713   (-0.225)
      v9, forward-calibrated bank  0.688 -> 0.812   (+0.125)

Calibrating the bank against a STRONGER policy than the one that will train on
it makes secrets drift INTO the band instead of out of it. Discovered by
accident — the mismatch was found at iteration 0 and the run deliberately not
restarted, because resuming from step60 would have contaminated 27 of
validation-v5's secrets and dropped criterion power below threshold.

### Supporting work in this block

- **Common random numbers** (DESIGN §9.10). A and B never faced the same draws:
  at step 0, where the adapters are provably identical, 160 of 192 episodes
  differed on turns while the totals coincidentally tied at 27/192. Fixed by
  reseeding per adapter and per step; control spread went 0.062 → 0.000.
- **Hinted generation** (DESIGN §9.12). Serial generation gave 7 distinct
  animals from 60 draws; rotating subcategory hints gave 37. Measured 3.7x in
  production. Bank size 61 → 138. The ceiling is PROMPTING-limited, not
  knowledge-limited — which is the enabling fact for training the creator.
- **validation-v5 composed, not generated.** validation-v4 drew 2400 rollouts
  against 259 reserved names and 88% of the distinct novel candidates were 0/8
  for the base model; only 19 landed in band. v5 = v3 (36) + v4 (19) + bank-v1
  minus the bank-v2 carryover (27) = 82. Valid for v9 ONLY: it contains
  bank-v1 secrets v8 trained on.
- **MDE discipline** (`scripts/validation_power.py`, DESIGN §9.8). v8's null
  was uninformative rather than negative; v9's criterion was sized in advance
  to t=2.56 at the effect v8 actually produced.

### Next

Solver transfer is established. The open items, in the order they now matter:

1. **Train the creator (dual-RL).** Reward on measured band membership, not the
   validity judge (which false-accepts 4/8). The diversity ceiling being
   prompting-limited means there is real headroom for a policy to find.
2. **Bank-v3 forward-calibrated from v9's final adapter**, per §10.3.
3. **New categories.** Three categories are mined out; this bounds both bank
   and validation growth independently of anything else.
