# 21 Questions — multi-turn twin-models pipeline (DESIGN, 2026-07-11)

Canonical spec for the 21-questions ("twentyq") mode. Lives in `twentyq/` to keep
it distinct from the single-step self-play pipeline documented in `DESIGN_V2.md`.
Run history for this mode goes in `twentyq/EXPERIMENTS.md` (created at Sprint Q6).

Branch: `twentyq` (off `dgx-spark`, so the MLX⇄torch backend abstraction is
available). Code lands under `src/twin/games/twentyq/` + `src/twin/train/base.py`;
tests under `tests/twentyq/`.

## 1. The game

Two players, same frozen Qwen3-8B base, two LoRA adapters (A/B) — identical to
the existing twin setup.

- **Creator** (secret-setter + answerer): per iteration emits **N secrets** with
  dictated difficulty ranks (mirrors `creator_mode: per_problem`). During each
  episode it answers the guesser's questions about its secret.
- **Solver** (guesser): plays **K episodes per secret**. Each episode: up to
  **T turns** of (question → answer), terminated early by a correct
  `GUESS: <x>` or by budget exhaustion.
- **Judge** = frozen base (zeroed adapter), never trained, never persona'd —
  same trust anchor as the existing pipeline.

Not zero-sum by design: the creator is rewarded for **calibrated** difficulty
(realized guess rates matching a target ramp), not for stumping. This is the
same anti-collapse choice as the main pipeline (and mini-03a's "stump an
expert" think-spiral is the cautionary tale).

## 2. RL design decisions (locked 2026-07-11)

### 2.1 Algorithm: GRPO, one optimizer step per iteration

PPO's critic is rejected: it doubles memory on hardware that already swaps at
4096-token worst case, a critic trained on ~30 iterations of data would be
noise, and the base judge is a better value proxy for free. The existing
`grpo.py` correctness argument (single inner step ⇒ π_new == π_old ⇒ ratio ≡ 1,
clip inactive) is preserved untouched.

**Update cadence: once per iteration**, never per question. The full iteration
runs to completion (N secrets × K episodes each), rewards and group advantages
are computed, then one GRPO step on the solver adapter and one on the creator
adapter. Reasons: (a) rewards don't exist until an episode ends; (b) the group
baseline needs the K sibling episodes; (c) mid-collection updates would break
the ratio≡1 assumption.

Alternatives considered and rejected: RLOO/REINFORCE++ (≡ what we have),
GiGPO step-grouping (needs repeated identical states; question histories
diverge), VinePPO (re-rollout value estimates multiply generation cost by T),
offline preference methods (don't fit the online loop).

### 2.2 Trajectory granularity: per-turn, always (forced)

The Qwen3 chat template strips `<think>` from prior turns, so the tokens a
guesser conditioned on at turn t are NOT the concatenation of its raw earlier
outputs. A single concatenated per-episode trajectory (opponent turns
loss-masked) would score a token stream the policy never produced-in-context.

Therefore: **one `Trajectory` per guesser turn** — `prompt_ids` = the exact
rendered history the model saw at that turn, `completion_ids` = that turn's
thinking + question/guess. Every completion token is policy-sampled; no
`loss_mask` needed. `rl/core.py`, both backends' `grpo_update`, and
microbatching consume this unchanged.

### 2.3 Credit granularity: broadcast, per-turn, terminal, and ensemble

- **v1 (broadcast):** episode k earns one scalar R_k; advantage
  A_k = R_k − mean(R over the K sibling episodes on that secret); every turn
  trajectory of episode k carries A_k. Same pattern as Sprint-7 broadcast
  credit. Judge cost: one closeness call per episode (final state only) + one
  batched truthfulness audit per episode.
- **v2 (per-turn, config `twentyq.credit: per_turn`):** turn t of episode k
  gets its own return R_{k,t} = Σ_{t'≥t} γ^{t'−t} r_{t'}, with per-turn
  r_t = w_close·(γ·Φ(h_{t+1}) − Φ(h_t)) (judge closeness deltas,
  potential-based shaping, policy-invariant per Ng et al.), plus the FULL v1
  episode scalar as the terminal reward at the last turn. Terminal- and
  start-state potentials are pinned to 0 (the Ng condition), so at γ=1 the
  shaping telescopes to zero and R_{k,0} equals the v1 episode scalar exactly
  — per-turn credit redistributes the total without changing it, making the
  broadcast-vs-per-turn ablation a pure credit-scheme comparison.
  Baseline: same-turn-index returns across the K sibling episodes
  (group_advantages per index); a turn index only one episode reached has no
  counterfactual and gets advantage 0 (an episode-mean fallback would compare
  reward-to-go against full-episode returns — wrong scale — so it was
  dropped at implementation). Judge cost: one closeness call per intermediate
  state (≈K·N·T per iteration) — this is why it is not v1.
- **terminal-only (`twentyq.credit: terminal`):** sparse terminal reward-to-go
  with the same turn-index sibling baselines as ensemble credit, but no dense
  scorer. At `gamma: 1`, every turn in an episode receives the terminal scalar.
  This is the matched no-ensemble control: it changes only the dense reward term,
  unlike legacy broadcast credit, which would also change baseline grouping.
- **ensemble (`twentyq.credit: ensemble`):** terminal reward-to-go plus frozen-
  ensemble potential differences, baselined across sibling episodes at the same
  turn index. This is the dense-reward treatment paired with terminal-only.

Because shaping telescopes, v1's episode scalar still inherits Φ(final) as
**partial credit on failed episodes** — direct mitigation of the mini-02
advantage-degeneracy failure (all-fail groups → zero variance → skipped
updates). "Does per-turn credit beat broadcast" is the headline experiment of
this mode (mirrors the `credit: per_problem` ablation).

### 2.4 Judge contracts: discrete anchored scales, never raw floats

Extends the `VERDICT:` strict-parse pattern (`verifiers/judge.py`). Three
rubrics, all parsed with strict regex + graceful degradation:

1. **Secret validity** (per secret, before any episode): is the secret a real,
   unambiguous, guessable entity of the declared category? → `VERDICT:
   VALID | INVALID`. Invalid ⇒ secret voided: no episodes, drags the creator's
   R_consistency exactly like an inconsistent problem. Degradation posture is
   configurable via `twentyq.secret_validity` — **default `fail_open`** (a reply
   with no parseable verdict counts VALID; only a clear INVALID voids), because
   fail-closed was false-voiding valid secrets on gemma4's format flakiness (see
   §2.10). `fail_closed` restores the strict posture; `off` skips the gate.
2. **Answer truthfulness audit** (per episode, one batched call): judge sees
   the secret + all Q/A pairs, returns per-pair verdicts (`AUDIT: T T F T…`).
   Any lie ⇒ episode voided (excluded from solver reward AND from the secret's
   realized guess rate; drags creator consistency). A creator rewarded for
   difficulty has an incentive to answer misleadingly — voiding is the same
   anti-cheat structure as the existing consistency check.
   **⚠️ REMOVED 2026-07-13 (see §2.8) — no longer wired into the loop; the
   creator is now assumed to answer truthfully.**
3. **Closeness Φ** (v1: final state only; v2: every turn): `CLOSENESS: n`,
   integer 0–10 with anchor descriptions in the prompt (0 = category not even
   constrained … 5 = category + several key attributes pinned … 10 = uniquely
   determined). Mapped to Φ ∈ [0,1]. Parse failure ⇒ Φ contribution 0 (no
   shaping), never a crash.

**Implementation (learned in q-shakeout-03/04):** the grader must run with a
**neutral twentyq system prompt** (`twentyq.prompts.JUDGE_SYSTEM`), NOT the
self-play CAS math-grader `twin.prompts.JUDGE_SYSTEM` — the math system's
"recompute with solve()… finish with VERDICT: CORRECT/INCORRECT" instruction
overrode the CLOSENESS contract and the judge scored VERDICT instead of a
number. And it runs **thinking OFF** (`twentyq.judge_thinking`): a thinking
budget truncated the closeness trace before its verdict line. `_judge` takes
`system=` and `enable_thinking=` overrides; the audit prompt is tuned
fail-open (mark F only for clear lies) so a sloppy non-thinking audit doesn't
false-void honest episodes and drag creator consistency.

### 2.5 Rewards

Solver, per episode (v1 scalar):

    R = w_guess·guessed
      + w_efficiency·(1 − turns_used/T_max)·guessed      # only if guessed
      + w_close·Φ(final)·(1 − guessed)                   # partial credit on failure
      − format penalties (unparseable turn ⇒ episode ends as failure)

Efficiency is gated on success (w_brevity discipline: efficiency can never
beat correctness). w_close < w_guess so a near-miss never outearns a win.

Creator, per iteration: realized "solve rate" of secret i = fraction of its K
non-void episodes guessed within budget. **`RewardEngine.creator_reward` is
reused verbatim** — exp(−β·MSE) fit against the target ramp, scored-fraction
scaling (voided secrets unprofitable), expected_n semantics, target_by_problem
pinning. Consistency flag per secret = valid (the "no voided episodes from
lying" term was dropped when the audit was removed — §2.8).

Creator answering turns are **not trained in v1** (reward attribution for an
individual "yes" is murky; the creator is assumed truthful — §2.8). The creator
trains on its N secret-emission rollouts. Training answer turns = future
ablation (§7).

**Creator credit is per-secret, not broadcast (decided at Q5 implementation):**
the self-play loop gets creator advantage variance from G_c candidate suites
per iteration, but an episode-based iteration can afford only ONE set of N
secrets — a broadcast suite reward would make every creator advantage
identically zero (all-tied group; the creator would train on nothing but
parse failures). `RewardEngine.creator_problem_rewards` gives each secret its
own calibration fit + consistency, making the N rollouts a real GRPO group.

### 2.6 Roles, rotation, control

`RoleManager` reused unchanged. Rotation flips which adapter creates vs
guesses every `roles.swap_interval` iterations; **`swap_interval: 0` is the
no-rotation control arm** — already implemented, free.

### 2.7 Token budgets and per-role thinking (revised after q-shakeout-01)

History grows linearly in turns (turn T's prompt carries T−1 Q/A pairs).
Mitigations, all config: T_max default **10** (not 21) for the first runs;
answers constrained to `ANSWER: YES | NO | SOMETIMES | UNKNOWN`, so history
stays compact.

**Thinking is per-role (`twentyq.creator_thinking` / `guesser_thinking` /
`answerer_thinking`), all OFF for v1.** Two shakeouts killed the original
"thinking ON where it reasons" plan — a thinking *budget* is the worst case
(full cost, zero usable output), and prompt constraints don't bind inside
`<think>`:
- q-shakeout-01: **guesser** at 512 tokens ran the whole budget inside an
  unclosed `<think>`, never wrote a `QUESTION:` line — think-share 1.0,
  **every** episode a turn-1 format fail.
- q-shakeout-02: with the guesser fixed, the **creator** at 1024 tokens
  brainstormed candidates ("Truffle? Wasabi? Escargot?…") past the budget and
  emitted no JSON — 0/2 then 2/2 parse failures across two iterations.

So v1 emits contract output directly from every role (creator JSON ~256 tok,
guesser `QUESTION:`/`GUESS:` ~200, answerer `ANSWER:` ~64) and relies on the
**reward**, not visible reasoning, to train calibration. Turning any role's
thinking back ON is a Sprint-Q8 ablation and only safe with a *generous*
budget — and for the creator it carries the mini-04b caution that reward alone
may under-ratchet difficulty without a reasoning trace (an open q-01 risk: if
creator calibration stalls, the lever is creator-thinking-ON at a 2048+
budget). Judge keeps `model.enable_thinking` (validation reasoning; its budget
is the untaxed oracle budget, not a per-turn one). Format-failed episodes now
dump the raw guesser completion to the transcript (the summary alone read
`[format_fail] -> None`, undebuggable).

### 2.8 Answer audit REMOVED — creator assumed truthful (2026-07-13)

The per-episode truthfulness audit (§2.4 item 2) is **removed from the training
loop.** The creator is now assumed to answer its own secret truthfully: no
episode is voided for "lying," every played episode trains the solver, and a
secret's consistency flag = validity alone (the §2.5 "no voided episodes" term
is dropped).

**Why:** on gemma4-E2B the audit was net-harmful, not protective. Reading the
`q-full-rot` / `q-full-ctrl` transcripts, the frozen-base auditor (a) false-
flagged *truthful* episodes as lies and (b) abstained as **unauditable** on a
large fraction — ~6 voids **plus up to 13 of 40 episodes unauditable per
iteration**. Both outcomes throw away good on-policy data: a voided episode is
excluded from the solver's GRPO group and from the creator's realized guess
rate, shrinking effective batch size and adding noise to creator calibration,
all to defend against a lying incentive that a *cooperative-calibration* creator
(rewarded for hitting a target guess rate, not for stumping — §1) barely has.
The audit's own graceful-degradation posture already treated unauditable
episodes as "not proven lying," so removing it is close to the fail-open limit
the design was trending toward anyway. `audit_void_fraction` (the gemma4
majority-threshold band, q-gemma-shakeout-02) is deleted with it.

**Kept but disconnected:** `judge.judge_answer_audit()`, the `AUDIT:` prompt,
and `Episode.audited_pairs` remain (with their unit tests) so a stronger judge
model could re-enable the mechanism later; nothing in `run_iteration` calls
them. Record schema drops `episodes.void` / `episodes.unauditable`
(`twentyq_run_stats.py` reads them with `.get(..., 0)`, so old and new logs both
parse). ANSWERER_SYSTEM no longer threatens an audit; it just asks for truthful,
accurate answers. This is a *for-now* simplification, not a claim that answerer
honesty is a solved problem — the frozen-base-answerer control arm (§5) remains
the principled long-term fix.

### 2.9 Transcript logging: flat file + GRPO-structured tree (2026-07-13)

Two sinks run side by side (both gated on `--no-transcript`):

1. **Flat** `tq-runs/<run>.transcript.txt` — the existing
   `TranscriptLogger`, everything in write-order (grep-friendly, unchanged).
2. **Tree** `tq-runs/<run>.transcript/` — `twin.log.transcript_tree.
   TwentyQTranscriptTree`, a folder hierarchy mirroring the two GRPO groups an
   iteration trains:

       <run>.transcript/
         _run.md                       run header (config, arm, model, N/K/T)
         iter_NN/
           _iter.md                    per-iter header + aggregate metrics
           creator_<rank>__<secret>/   CREATOR GRPO member (one secret rollout)
             _creator.md               the rollout + its reward/advantage
             episode_<k>__<outcome>.md SOLVER GRPO member (one game) + per-turn credit

   **Creator-over-solver is forced by the reward math, not chosen:** a solver
   GRPO group is baselined *per secret* (`rewards.secret_turn_trajectories` /
   `per_turn_secret_trajectories` each take one secret's K episodes), so every
   solver group nests inside exactly one creator member. Parse-failed and
   invalid secrets are still creator members (gate/void reward) with no episode
   files (`…__PARSE-FAIL/`, `…__INVALID/`). Each `episode_k` file carries the
   per-turn potential Φ_t, reward-to-go R_t, and per-turn advantage (the credit
   the trajectories actually train on); outcome tag = `win_t<n>` / `miss` /
   `fmtfail`.

   **Full visibility:** unlike the flat log (gated on `--log-prompts`), the tree
   always writes *every* input prompt and model output — the creator rollout
   (system+user+completion) and, per turn, the guesser and answerer system+user
   prompts and raw completions. Prompts sit in collapsed `<details>` so the file
   stays scannable. The per-turn player prompts are **reconstructed** from the
   finished episode (`TwentyQTrainer._episode_prompts`): the guesser prompt is
   deterministic in the turn's history slice and the answerer prompt depends only
   on the question text, so no model is re-run. A wrong GUESS shows an engine-
   referee ground-truth NO (no answerer call). The tree is a passive sink — the
   trainer feeds it already-computed values (solver trajectories gained an
   `ep_in_secret` meta key so turns regroup into episodes for the advantage
   table).

### 2.10 Secret validity gate → fail-open by default (2026-07-13)

The validity gate (§2.4 item 1) was **fail-closed**: `judge_secret_validity`
voided any secret whose judge reply had no parseable `VERDICT:` line. On
gemma4-E2B that default false-rejected *valid* secrets. Reading `q-fullv2-rot`,
the judge routinely ignores the "think briefly, then `VERDICT:`" instruction and
**role-plays the guesser** — emitting a numbered 20-questions list (`1. Is it a
mammal?`) and hitting `<end_of_turn>` before any verdict. Whether a verdict
survives is pure sampling luck: "Dog" got 7 question lines *then* `VERDICT: VALID`
and passed; "Axolotl"/"Salmon"/"Octopus" stopped after line 1 and voided. Scale:
**6 of 91 vetted secrets (~6.6%) voided, 100% of them false rejections**, and
they cluster on the hardest slot (creator_4, target 0.10 → obscure-but-valid
picks like Axolotl, hit 4×) — so the gate biased *against* exactly the difficult
secrets the curriculum wants. Same "throws away good on-policy data" pathology as
the removed audit (§2.8).

**Fix:** `judge_secret_validity(secret, oracle, mode=…)` now takes a posture,
default **`fail_open`** — an unparseable reply or judge error counts VALID; only
a clear `VERDICT: INVALID` voids. A *clear* verdict is always honoured, so real
mashups/category-mismatches still get caught when the judge actually emits one.
`fail_closed` keeps the pre-fix posture; `off` skips the judge call entirely.
Wired via `twentyq.secret_validity` (config) + `--secret-validity` (CLI). This is
not a budget/truncation issue — the judge stops ~6 tokens in, well under the cap;
raising the budget wouldn't help. A stronger judge (or a verdict-first prompt /
constrained decode) could tighten it later; fail-open is the low-risk fix now.

**exp-fullv2 caveat:** the rotation arm (`q-fullv2-rot`) ran under the old
fail-closed code, so the config pins **both** v2 arms to `secret_validity:
fail_closed` — the control must match the arm it's compared against. `fail_open`
becomes the standing default for all runs *after* exp-fullv2.

## 3. Reuse map

| Reused unchanged | New |
|---|---|
| `rl/core.py` (Trajectory, group_advantages, all_zero_advantages) | `games/twentyq/schema.py` — Secret spec + parse |
| `backends/*` `grpo_update` (MLX + torch) | `games/twentyq/judge.py` — 3 rubrics + parsers |
| `RoleManager` (incl. no-rotation control) | `games/twentyq/episode.py` — turn engine |
| `RewardEngine.creator_reward` / `creator_problem_rewards` | `games/twentyq/rewards.py` — episode reward math |
| `verifiers/judge.py` strict-parse pattern | `games/twentyq/trainer.py` — `TwentyQTrainer` |
| adapters, config loader, JsonlLogger, TranscriptLogger | `train/base.py` — extracted `BaseTrainer` (Q1) |
| checkpoint save/load, seeding, bench harness patterns | `configs/twentyq-tiny.yaml`, `scripts/train_twentyq.py` |

`SelfPlayTrainer.run_iteration` is domain-entangled — **not** subclassed.
Shared plumbing is extracted to `BaseTrainer`; both trainers extend it.

Config: new `twentyq:` section (dataclass `TwentyQConfig`) — n_secrets,
episodes_per_secret, max_turns, categories, credit
(`broadcast|per_turn|terminal|ensemble`),
gamma, w_guess/w_efficiency/w_close, closeness_scale, answer budgets. Existing
sections (model/lora/gen/roles/train/compute/paths/rewards) reused as-is.

## 4. Sprint plan (start → v1 → v2)

Every sprint lands with its tests green (`pytest tests/ -x -q -m "not slow"`)
and zero regressions in the existing suite (408 fast + 4 e2e at branch time).

### Sprint Q1 — BaseTrainer extraction (pure refactor)
Extract from `SelfPlayTrainer` into `twin/train/base.py::BaseTrainer`:
backend/optimizer/adapter/roles wiring, `_score`, `_grpo`, `_judge`,
`_build_tool_runner`, `_generate`, `_tr`/`_tr_section`, `save_checkpoints`,
`train` driver (abstract `run_iteration`). `SelfPlayTrainer(BaseTrainer)` keeps
every attribute/method name and signature (tests build via `__new__` and poke
internals — inheritance preserves that).
**Tests:** entire existing suite passes unchanged; no new behavior. Gate: 0
regressions, `git diff` shows loop.py shrink + base.py add only.

### Sprint Q2 — Secret schema + judge contracts
`schema.py`: `Secret` dataclass (text, category, difficulty rank, JSON block
parse with the same strictness philosophy as `parse_problem`). `judge.py`:
the three rubric prompts + parsers (§2.4), pure functions over a
`Callable[[str], str]` oracle (stub-testable, same shape as `verifiers/judge`).
**Tests (new `tests/twentyq/`):** parse good/malformed secrets; verdict/audit/
closeness parsing incl. rambling text, missing lines, out-of-range ints
(clamp), empty replies → graceful degraded results, never exceptions; Φ
mapping monotone.

### Sprint Q3 — Episode engine (scripted players, no model)
`episode.py`: `run_episode(guesser_fn, answerer_fn, secret, cfg) -> Episode`.
Alternates turns; detects `QUESTION:` / `GUESS:` lines; terminates on correct
guess (normalized match — reuse `_normalize_named_value` lessons) or T_max;
records per-turn `(prompt_render, completion, tokens)` for both players;
unparseable guesser turn ⇒ episode ends as failure with a format flag.
Players are injected callables, so tests script them deterministically.
**Tests:** termination on correct guess mid-budget; budget exhaustion; guess
normalization (case/articles/whitespace); per-turn prompt fidelity (prompt at
turn t == rendered history through turn t−1, thinking stripped); format-fail
path; turn records complete and ordered.

### Sprint Q4 — Reward + advantage wiring (pure math)
`rewards.py`: episode reward (§2.5); per-secret realized guess rate over
non-void episodes; assembly of solver turn-trajectories into per-secret groups
with broadcast episode advantages; creator side mapped onto
`RewardEngine.creator_reward` (validity/void → flags/scored_mask/expected_n).
**Tests:** monotonicity (fewer turns ⇒ ≥ reward when guessed; guessed always >
not-guessed regardless of Φ and turns); Φ partial credit orders failed episodes;
voided episodes excluded from rates but drag consistency; group advantage
shapes (K episodes × variable turns); all-tied group ⇒ `all_zero_advantages`
skip; per-secret grouping matches `solver_attempts` semantics.

### Sprint Q5 — TwentyQTrainer end-to-end (v1)
`trainer.py`: `TwentyQTrainer(BaseTrainer).run_iteration` — creator emits N
secrets (per-secret rollouts, dictated difficulty + target rate in prompt),
judge validity gate, K episodes per valid secret (guesser = solver adapter,
answerer = creator adapter, both via injected generate closures), truthfulness
audit, rewards, two GRPO updates, JSONL record + transcript + checkpoints.
Config section + `configs/twentyq-tiny.yaml` + `scripts/train_twentyq.py`
(mirrors `scripts/train.py` incl. `--resume-step`).
**Tests:** scripted e2e iteration with a fake base (pattern from
`tests/sprint4/test_e2e_model.py` + `__new__` trainers): correct trajectory
counts/groups; parse-gate on unparseable secret rollouts; void paths; JSONL
record schema; rotation flips creator/solver; `swap_interval: 0` control never
flips; checkpoint files written; resume reseeds.

### Sprint Q6 — v1 shakeout (first real run)
Tiny MLX run: N=2 secrets, K=2 episodes, T=6 turns, ~5 iterations,
`log_prompts` on. Measure: wall-clock per iteration, judge-call count vs
budget, secret parse rate, validity rate, audit lie rate, episode termination
mix, transcript sanity (contract adherence: QUESTION/GUESS/ANSWER formats).
**Gate:** ≥80% secret parse, ≥50% validity, no crash, iteration wall-clock
compatible with a 30-iteration run. Fix contract failures found here (this is
where prompt templates get real). Then **q-01**: first full run (~30 iters,
N=3, K=4, T=10), logged in `twentyq/EXPERIMENTS.md` with pre-registered
success/abort criteria (mini-04b discipline).

### Sprint Q7 — v2 per-turn credit
Per-turn judge Φ after every guesser turn; per-turn shaped rewards
r_t = γΦ_{t+1} − Φ_t; reward-to-go returns; same-turn-index sibling baseline
with episode-mean fallback; config `twentyq.credit: per_turn` + `gamma`.
**Tests:** telescoping identity (γ=1 ⇒ Σ_t r_t == Φ_final − Φ_0, so per-turn
and broadcast agree on the episode total); reward-to-go recursion; baseline
fallback on ragged episode lengths; judge-call accounting (exactly K·N·T
closeness calls); broadcast path bit-identical when flag off.

### Sprint Q8 — the experiment matrix
- **q-v1 vs q-v2:** broadcast vs per-turn credit, same seed/config otherwise —
  the headline question.
- **rotation control:** best credit arm with `swap_interval: 0`.
- Curves: realized guess-rate ramp vs target (creator calibration), questions-
  used distribution, Φ trajectories, think-share, adapter drift — reuse
  `analysis/curves.py` where fields line up.

## 5. Open questions (deferred, not blockers)

- Train creator answering turns (broadcast episode credit)? — ablation after v2.
- Answerer = frozen base conditioned on secret (removes lying entirely) as a
  control arm — cheap to add once the engine takes injected answerer closures.
- Secret categories: start with concrete nouns (animal/object/place/person);
  grounded category weighting (à la Sprint-9 themes) later.
- Cross-secret diversity penalty (creator emitting near-duplicate secrets) —
  `analysis/diversity` reuses directly if needed.

## 6. RL/reward audit — concerns logged 2026-07-14

This section records the post-`exp-fullv3` code-and-log audit. The core update
plumbing appears internally consistent: exact per-turn prompts are rescored,
advantages are assigned before flattening, `grpo_microbatch: 1` accumulates
gradients without changing weights between trajectories, the KL reference is the
frozen zero-adapter base, and the solver and creator use separate AdamW state.
The concerns below are chiefly about what the experiment identifies, reward
validity, and a few places where the loss does not quite match the environment.

1. **The no-rotation arm is not a no-RL or reward ablation.** Both adapters still
   train, and the creator's secret distribution moves throughout the run. Online
   guess rate therefore mixes solver learning, creator difficulty changes,
   category effects, and repeated-secret memorization. `swap_interval: 0` cleanly
   isolates hard role rotation, but cannot by itself establish that GRPO works or
   that the ensemble reward helps. Required read-outs: evaluate every checkpoint
   on a fixed held-out secret set with a fixed answerer; compare `w_ensemble: 0`
   against v3's `0.1` with the terminal reward held fixed; optionally include a
   zero-LR/no-update sampling baseline.

2. **Most solver groups still have no terminal win signal.** In the completed
   v3 rotation arm, 147/240 secret groups (61.3%) had zero wins. In the first
   seven completed v3 control iterations, 41/56 (73.2%) had zero wins. Every
   advantage in those groups is ensemble-driven: the policy is rewarded for
   raising frozen-model belief with no pressure to commit to a guess. Lowering
   `w_ensemble` fixed the scale imbalance found in v2, but not this structural
   absence. Log the all-miss-group fraction as a first-class metric. Candidate
   future levers are a small guess-at-all reward, an ungated attempt/commitment
   term, or a separately tested terminal-reward curriculum; do not silently add
   one to the current matched experiment.

3. **The learned difficulty ramp is still badly miscalibrated.** Across the
   completed v3 rotation arm, mean guess rates by rank were approximately
   `0.217, 0.408, 0.147, 0.033, 0.028, 0.047, 0.050, 0.017` against targets
   `0.900, 0.786, 0.671, 0.557, 0.443, 0.329, 0.214, 0.100`. The absolute rates
   are far too low and rank 1 is easier than nominal rank 0. A rising
   `r_gradient` is therefore only movement in the reward's aggregate fit, not
   evidence that the requested monotone curriculum has been learned. Report
   per-rank held-out rates, monotonicity violations, and target MSE alongside the
   scalar reward.

4. **The answerer is neither trained nor currently verified.** The creator
   adapter is updated from its secret-emission trajectories, while its answer
   turns receive no policy gradient. Since the failed truthfulness audit was
   removed, semantically wrong answers now silently corrupt the solver state,
   terminal outcome, and ensemble shaping signal. Spot-checking transcripts is
   necessary but not a scalable contract. The cleanest control is a frozen,
   stronger answerer conditioned on the secret; lighter alternatives are a
   sampled audit by a stronger independent judge or deterministic fact handling
   where the category permits it.

5. **Lenient parsing gives credit to text the environment did not execute.** The
   guesser parser accepts the last labelled question, `Q:`, or bare question, but
   GRPO trains every token in the raw completion. Live transcripts contain
   duplicated questions and cases where an unused preamble asks a different
   question from the final parsed line. The ensemble and answerer react only to
   the parsed question, yet its advantage reinforces or suppresses the entire
   completion. Either enforce one canonical line with constrained/strict decode,
   mask non-executed completion tokens out of PG/KL, or explicitly penalize extra
   visible content. Format telemetry must distinguish "parseable" from "exact
   contract" rather than reporting both as clean.

6. **The loss has a deliberate-but-unresolved length bias.** The torch/MLX GRPO
   losses sum token log-probabilities per trajectory and divide by total trained
   tokens. A longer turn therefore receives more gradient mass for the same
   advantage, and a longer episode contributes more turn trajectories. This is
   the Dr.GRPO-family length issue already noted in `RELATED_WORK.md`; here it
   interacts with efficiency pressure and the lenient-preamble problem. Measure
   completion length versus advantage/gradient contribution and decide explicitly
   between token-average, per-turn-average, and per-episode-average objectives.

7. **The ensemble path lacks proportionate tests and calibration telemetry.** The
   older judge-Φ shaping path has arithmetic tests, but the central v3 path lacks
   direct automated coverage for `ensemble_shaped_returns`, answer-span location,
   per-model aggregation, and end-to-end advantage assignment under ragged episode
   lengths. Raw mean log-probability deltas from different tokenizers/models need
   not have the same scale; an equal arithmetic mean can be dominated by the
   highest-variance scorer. Log per-model deltas and disagreement, add synthetic
   invariants (constant potential, increasing/decreasing potential, terminal and
   format cases, γ != 1), and run leave-one-member-out reward analyses.

8. **`solver_reward_mean` is not mean episode reward under ensemble credit.** It
   averages per-turn reward-to-go values across flattened solver trajectories.
   Long failed episodes contribute more entries than early wins, so the console's
   `Rs` is useful health telemetry but not a clean learning metric. Log separate
   episode terminal mean, ensemble gain `w*(Phi_final-Phi_0)`, total episode
   return, per-turn return by index, and the loss's token-weighted mean.

9. **Resume is not experiment-identical and the documentation is inconsistent.**
   Checkpoints currently save adapter tensors only. A resumed run constructs new
   optimizers and reseeds Python/backend RNGs; Adam moments, RNG progress, and the
   original drift baseline are lost. `scripts/train_twentyq.py --help` correctly
   notes the optimizer/drift reset, while `CONFIG.md` says optimizer state is
   saved. Until full state checkpointing lands, treat a resumed arm as a marked
   discontinuity and correct the configuration documentation.

10. **Creator diversity/generalization remains unmeasured.** Prompts prevent
    duplicates inside one iteration, but there is no cross-iteration diversity
    reward in twentyq and common secrets recur. The adaptive solver can memorize
    this distribution, making online gains look like general Twenty Questions
    skill. Log cross-iteration secret repetition/similarity and make the fixed
    held-out-secret evaluation the primary capability read-out.

### 6.1 Terminology clarified

Three different meanings of "member" must not be conflated:

- an **ensemble member** is one frozen scoring LLM;
- a **creator GRPO member** is one of the N secret-emission rollouts;
- a human-readable **solver member** is one complete episode, but under
  `credit: per_turn|ensemble` the actual comparison group at `(secret, turn t)`
  is the turn-t trajectory from every sibling episode that reached t.

The `Trajectory` object is the exact prompt/completion chunk sent through the
loss. Membership describes which samples are compared to form an advantage; it
does not imply an optimizer update per member.

### 6.2 Proposed within-game/per-turn optimizer updates

Two superficially similar proposals have very different meanings:

**A. Accumulate gradients by turn, then step once.** After all episodes finish,
process turn-0 trajectories, then turn-1, etc., call `backward()` on each chunk,
and call `optimizer.step()` only after the final chunk. If it uses the same
reward-to-go, advantages, token normalizer, and no parameter changes between
chunks, this is mathematically the current algorithm in a different batch order
(up to floating-point summation). `grpo_microbatch: 1` already accumulates every
trajectory's gradient before the single solver step. It may improve organization
or memory locality, but it does not create faster within-game learning.

**B. Actually step the optimizer after each turn index.** This is feasible only
after restructuring episode collection into lockstep rounds: generate turn t for
all active K episodes (ideally all N*K episodes), observe their answers and
ensemble deltas, compute same-secret group advantages at t, then take one solver
step before generating turn t+1. All actions at t were sampled by the same
pre-step policy, so a one-step update on those fresh transitions is locally
on-policy. The run would make up to T solver optimizer steps per iteration, and
later turns would intentionally come from newer policies.

It is **not the current single-inner-step GRPO objective**, and it loses the main
benefit of delayed reward-to-go. At turn t the dense ensemble delta and any
terminal event occurring at t are known, but future success is not. Updating on
only immediate group-relative ensemble reward is myopic and is especially likely
to worsen the observed "raise belief, never guess" failure. If the eventual
terminal reward is later attached to already-trained early turns, those stored
actions are off-policy because the weights changed in between; the ratio=1
argument no longer holds. Correct reuse would require stored behavior log-probs
plus importance ratios/clipping (real PPO/GRPO), or a learned value function / GAE
or another eligibility-trace mechanism for backward credit.

Ragged termination adds another complication: the same-turn group shrinks as
episodes win or format-end, so late updates are conditioned on the survivor
population and singleton groups have zero relative advantage. Optimizer state and
KL behavior also advance T times faster unless LR/KL schedules are retuned.

**Recommendation:** keep the deferred one-step update as the primary design for
the current experiment. It assigns the known terminal outcome to every preceding
action without off-policy correction, and it already accumulates gradients across
all turns. If online updating is tested, pre-register it as a separate
`online_immediate` algorithmic arm, collect all K episodes synchronously, take at
most one update per global turn across all per-secret groups, log the changing
behavior-policy version, and judge it on the fixed held-out evaluation. A safer
intermediate ablation is horizon-chunked collection (hold weights fixed for H
turns, then update on an H-step return), but without a value bootstrap it still
weakens terminal credit for chunks that end before the outcome.

### 6.3 Stationary validation and reward observability (implemented 2026-07-14)

`twentyq.validation_every` now schedules a pre-training baseline at step 0 and
then a post-update evaluation every X completed iterations (`0` disables it).
Both adapters are evaluated as guessers
on the versioned `data/twentyq-validation-v1.json` set; the answerer is always the
frozen base adapter. Both sides decode greedily, so validation neither updates
weights nor advances the stochastic sampling stream used by training. Validation
is logged as a distinct JSONL `type: validation` record with aggregate,
per-adapter, per-category, and per-secret results.

Training and validation reports now keep the sparse terminal signal, immediate
dense shaping gain, combined immediate reward, and discounted combined return at
turn zero separate. Every episode's JSON summary contains its component trace.
The flat transcript writes a per-turn reward table only after all K sibling
episodes are available (so the GRPO advantage is known), and every hierarchical
episode file shows the same table plus an inline reward line at each turn. Under
legacy broadcast credit, the table explicitly distinguishes the terminal event
on the last turn from the episode scalar assigned to every turn trajectory.
Periodic evaluations also get a hierarchical
`validation_step_N/adapter_{a,b}/secret_*.md` tree with the same per-turn fields.

### 6.4 Deferred ticket: optional open-category play

Keep category-constrained secret generation as the current default, but add an
open-category mode in a future change. In that mode the creator may choose any
real, commonly known, concrete entity and must still declare a broad category
as metadata. Whether that declared category is revealed to the solver should be
a separate option: creator-domain restriction and solver category hint are two
different experimental variables.

Evaluate open-category play independently from creator-diversity work. Removing
the category constraint may broaden the output distribution, but it does not
provide cross-iteration novelty pressure and therefore is not expected to solve
secret repetition by itself. The near-term diversity direction is a bounded
recent-secret exclusion list shared across iterations; open-category behavior
remains a separate future ticket.

### 6.5 Rolling recent-secret exclusions (implemented 2026-07-15)

`twentyq.recent_secret_window` controls a single bounded deque of parsed creator
secrets shared across adapters, iterations, and role rotations (`128` for the v4
matrix; `0` disables cross-iteration history). Each entry retains the scheduled
category. A creator prompt receives only deque entries from its current category,
plus secrets already parsed earlier in the current iteration, and shows compact
secret names rather than prior JSON objects. The prompt asks for neither exact
repeats nor obvious variants.

This is initially a prompt-and-measurement intervention only. A repeated secret
is not rejected, regenerated, voided, or penalized. Each iteration logs raw-exact
and normalized (case/article/punctuation/whitespace canonicalized) repeat rates
over parsed outputs, along with the retained history size. Parsed outputs enter
the deque whether or not the later validity judge accepts them, because the deque
tracks creator sampling rather than playable episodes.

On `--resume-step`, `scripts/train_twentyq.py` reconstructs the deque from the
run JSONL before generation resumes, retaining only rows before the checkpoint
step and only the newest configured number of secrets. If iterations were
retried, the last JSONL row for an iteration wins; a crash-truncated final line
is ignored. Reusing `--run-name` selects the same log automatically, while
`--resume-log` permits resuming into a differently named output run.

**Finding (v4 arm A, stopped at iteration 18, 2026-07-16): the prompt-only
intervention loses to the reward gradient.** The exclusion list was plumbed
correctly — iteration 11's transcripts show "Okapi" printed in the exclusion
block and the creator picking Okapi anyway, 8 of 10 ranks. The attempt-0
exact-repeat rate climbed 0.0 -> 0.9 over 18 iterations, and every repeat was
byte-identical (normalized rate == exact rate throughout). Three causes:

1. The base model's per-(category, rank) secret distribution is nearly a point
   mass: the v3 sampling probe measured "Apple" 64/64 at temp 0.9, 97% at 1.5,
   and 66% at 2.5 where parsing starts failing — sampler-side entropy cannot
   create diversity that is not there.
2. With the solver's guess rate ~0 at nearly every rank, the calibration
   reward degenerates: any never-guessed obscure entity earns
   `exp(-β·target²) + consistency + validity`, near-maximal at low targets, so
   duplicates parked on the low-target half of the ramp systematically beat
   the group mean. Repetition is reward-optimal under this regime.
3. The exclusion block is part of the creator's training prompt, so every
   positive-advantage repeat literally trains "see the exclusion list, ignore
   it". Summing per-secret creator advantages over the run: Okapi +3.51
   (accelerating: +0.75/+0.68/+0.68/+1.06 in iterations 11/13/15/16),
   Black Garlic +1.30, Sichuan peppercorn +1.26, Extension cord +1.19. At
   iteration 11, 112/160 episodes (70%) were rollout compute spent on
   duplicate Okapi games.

### 6.5b Repeat gate + masked resample (implemented 2026-07-16)

`twentyq.repeat_handling` turns the exclusion list from advice into a
constraint (`off` preserves the v4 arm-A prompt-and-measurement behaviour):

- **Attempt 0 stays unconstrained.** It measures the policy's true repeat
  propensity (the logged exact/normalized rates keep their v4 semantics) and
  gives the gate an on-policy trajectory to train against.
- **`void`**: a parsed attempt-0 secret matching the exclusion list it was
  shown (`repeat_matches`: normalized + bare-plural tolerance + normalized
  edit-distance-1 for strings of 5+ characters — deliberately broader than
  both the exact/normalized telemetry and episode win judging, which keeps
  `guess_matches` untouched) is voided. It plays NO
  episodes, enters the creator GRPO group at the fixed
  `twentyq.repeat_gate_reward` (default 0.0: well-formed but disallowed, so
  below every honest secret at ~1.3-1.5 while a parse failure stays strictly
  worse at -1.0), and its absence from the suite scales `r_gradient` down
  exactly like a parse drop. Group-relative advantage then pushes probability
  mass off the attractor every time it fires.
- **`retry`**: `void` plus ONE masked resample. The SAME prompt is re-decoded
  at `creator_temp` with every excluded secret banned at the logits level by
  a STRING-level processor (`BannedStringsProcessor`): at each step the
  generated text is decoded, and whenever it ends with any prefix of a banned
  phrase (casefolded; plural variants included), every vocab token whose
  surface would complete that phrase is masked to -inf — so the decoded
  completion can never contain an excluded secret under ANY tokenization,
  and the decoder continues with the next-most-likely non-excluded
  continuation. A pure token-sequence ban (`bad_words_ids`) is NOT sufficient
  here and was replaced after the first v4.5 attempt: masking the final token
  of the banned sequence just reroutes the sampler onto an alternate BPE
  segmentation of the same surface string ('Black'+' Card'+'am'+'om' after
  'amom' was banned — 3 of the first 15 masked retries emitted their banned
  secret verbatim this way). The sample is drawn from the
  renormalized masked distribution, NOT greedy: greedy would deterministically
  anoint the policy's #2 candidate as the next attractor, and once the
  dominant mode is masked the renormalized tail is where the usable entropy
  lives. A retry that parses non-repeat plays the rank's episodes and trains
  the creator with its real game reward — the "do say Ocelot" half of the
  signal alongside the repeat gate's "don't say Okapi". A retry that still
  matches (a surface variant the token ban missed) voids the rank outright: a
  repeat never plays episodes.

Off-policy caveat (pre-registered): the retry trajectory is drawn from the
masked distribution, not the raw policy, so training it under the ratio≡1
single-step GRPO objective carries the same class of deliberate sampling bias
as top-p truncation. This is accepted: the masked sample is exactly the
alternative we want credited. The repeat-gate trajectory itself is fully
on-policy.

Because reward now depends only on the prompt (the exclusion list is shown
in-prompt) and the completion, the reward stays a function of the
conditioning — this trains instruction-following rather than chasing hidden
state. Telemetry adds `playable_rate`, `repeat_voided`, `repeat_retries`, and
`repeat_retry_playable` per iteration; `parse_ok_rate` keeps counting
attempt-0 parses. Every parsed output — voided repeats and masked retries
included — still enters the rolling deque in sample order, recorded as the
iteration's `sampled_secrets` list, which `restore_recent_secrets` prefers on
resume (older logs without the field fall back to the playable summaries,
which were their complete sample stream). Transcript trees mark these members
`repeat_void` / `__RETRY`.

The residual risk is serial collapse (the policy re-collapses onto each freed
next-best candidate in turn). The rolling window plus the gate keep forcing
novelty mechanically; if post-fix runs still show weak diversity pressure,
the next levers are conditioning-side diversity (random subcategory/letter
seeds per rank) and the target ramp itself (0.9 is unreachable for a solver
at guess rate ~0, which is what makes "any obscure entity" optimal).

### 6.6 Lockstep rollout and ensemble batching (implemented 2026-07-15)

Two independent knobs now remove the largest serial-forward bottlenecks:

- `generation_batch_size` advances the K sibling episodes for one secret in
  lockstep. At each turn all active guessers decode together, then all questions
  that need oracle answers decode together. Correct guesses and format failures
  leave the active set, so ragged termination is preserved. The creator's N
  secret samples remain serial because each prompt excludes earlier samples from
  the same iteration. Validation episode generation also remains serial; its
  relatively small history-scoring diagnostic uses the ensemble batch setting.
- `ensemble_batch_size` flattens all history prefixes across a secret's K
  episodes and scores them in chunks for each frozen member. Inputs are
  right-padded with an attention mask. Only the answer-position logits are
  promoted to fp32 for log-softmax, avoiding a full fp32
  `[batch, sequence, vocabulary]` allocation.

Both defaults are `1`, which retains the prior scalar code paths. Every run's
main JSONL, reward-sidecar metadata, transcript metadata, and iteration rows log
the configured sizes. Batched stochastic decoding changes the RNG stream, so it
is distribution-equivalent rather than transcript/seed-identical to serial
decoding. Deterministic engine tests verify identical ordering, termination,
turn contents, and active-batch shrinkage between the scalar and lockstep paths.

The reusable benchmark is `scripts/smoke_test_twentyq_batching.py`. On the DGX
Spark's NVIDIA GB10, with the v4 Gemma-4 policy/LoRA surface, 16 representative
guesser prompts, 64-token caps, and BF16 CUDA inference, generation measured:

| generation batch | prompts/s | tokens/s | peak CUDA allocation |
| ---: | ---: | ---: | ---: |
| 1 | 2.55 | 21.66 | 10.709 GB |
| 2 | 4.08 | 36.94 | 10.778 GB |
| 4 | 5.59 | 51.39 | 10.917 GB |
| 8 | 6.33 | 58.19 | 11.201 GB |
| 16 | 7.78 | 71.44 | 11.729 GB |

Batch 16 is the K16 rollout optimum: it is 3.05x the scalar prompt throughput,
uses about 1.02 GB more allocated CUDA memory, and avoids a second decoder chunk
while all siblings are active. A repeat run measured 72.29 tokens/s. With the
policy and all four frozen ensemble models resident simultaneously, batch 16
also completed without OOM at 73.45 tokens/s and 42.753 GB peak allocation.

For 16 representative variable-length histories with the policy retained and
all four BF16 ensemble scorers resident, aggregate history scoring measured:

| ensemble batch | histories/s | speedup | peak CUDA allocation | max raw score delta vs batch 1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.88 | 1.00x | 42.094 GB | 0.000 |
| 2 | 7.67 | 1.97x | 42.096 GB | 0.167 |
| 4 | 8.84 | 2.28x | 42.520 GB | 0.164 |
| 8 | 7.43 | 1.91x | 43.374 GB | 0.161 |
| 16 | 6.87 | 1.77x | 45.082 GB | 0.161 |

Batch 4 is selected because larger batches lose throughput on this workload
while consuming more memory. BF16 model matmuls are batch-shape-sensitive, so
batched scores are not bitwise equal to scalar scores even though span selection
and fp32 log-softmax are unchanged. The worst observed aggregate-potential delta
was 0.167 raw log-probability units. At v4's `w_ensemble: 0.1`, that is 0.0167 at
one potential and bounds an adversarial adjacent-potential shaping difference at
about 0.0334. This drift does not affect terminal reward, and all ensemble v4
arms use the same batch setting; it is accepted for the measured 2.28x scoring
gain and is explicitly retained as benchmark telemetry.

The v4 matrix therefore uses `generation_batch_size: 16` and
`ensemble_batch_size: 4`. These smoke tests exercise inference and joint model
residency, not GRPO backward or a complete 160-episode iteration. Keep
`grpo_microbatch: 1`, and treat the first terminal-control iteration's wall time
and peak memory as the final launch check before increasing N or K further.

### 6.7 v4.5 shakeout: the repeat gate under live fire, and its hardening (2026-07-16)

The first repeat-handling run (`q-fullv45-ctrl-terminal`, arm A restarted with
`repeat_handling: retry` per §6.5b) was deliberately stopped after iterations
0-13 to fold in what fourteen iterations of live adversarial pressure taught.
This section is the consolidated record of that shakeout: what worked, the
three evasion channels the creator found, and the code that closes them. v5
(`q-fullv5-*` run names, same `twentyq-full-v4.yaml` settings otherwise)
restarts the matrix from scratch with the hardened stack.

**The mechanism works.** At the two iterations where the original v4 arm A had
collapsed to 8/10 "Okapi" (iterations 11 and 13, attempt-0 exact-repeat rate
0.9), v4.5 measured 0.1 and 0.3, with visibly broadened coverage (Gerenuk,
Tarsier, Kakapo, Toco Toucan, Mantis Shrimp — entities the collapsed run never
sampled). Every attempt-0 repeat was voided; ZERO repeats played episodes; most
voided ranks were recovered by the masked resample (playable_rate 1.0 in 10 of
14 iterations). Step-0 validation reproduced the stopped run's baseline
bit-for-bit, confirming the gate changes nothing upstream of enforcement.

**Live finding 1 — a token-sequence ban defeats itself.** 3 of the first 15
masked retries emitted their banned secret VERBATIM. Diagnosis from the stored
raw completions plus the tokenizer: the ban list correctly contained the
canonical path `['Black', ' Card', 'amom']`, so when the sampler reached
`'Black', ' Card'` the processor masked `'amom'` — and the next-most-likely
token was `'am'`, after which `'om'` completed the SAME surface string through
a segmentation no banned sequence covers. `bad_words_ids` masking is
per-token-path; masking one path reroutes a determined distribution onto
another spelling of the same string. Fix: `BannedStringsProcessor`
(torch_backend), a LogitsProcessor that decodes each row's generated tail and,
whenever it ends with any prefix of a banned phrase (casefolded, plural
variants included, EVERY matched prefix length — periodic prefixes can match
at several lengths at once), masks every vocab token whose surface completes
the phrase, via binary search over a sorted casefolded vocab-surface table.
Invariant: the decoded completion can never contain a banned phrase under ANY
tokenization. Verified against the live leak state on the real tokenizer;
cost ≈1.9 s one-time table build + ~1 ms/step with 50 phrases (retry-only).

**Live finding 2 — misspelling evasion, and it PAYS.** With clean completions
blocked, mass squeezed into one-letter deletions: 'Okpi' (iteration 2, caught
by the validity judge) and 'Wasbi' (iteration 6, PASSED the fail-open judge,
played 16 episodes, and earned 1.47 against a 0.65 creator mean — a strongly
positive advantage for evading the exclusion list, i.e. the exact incentive
the gate exists to remove, re-created one edit away). Fix: the gate matcher
`repeat_matches` (schema) extends `guess_matches` with normalized
edit-distance-1 for strings of 5+ characters. Episode win judging keeps
`guess_matches` untouched.

**Live finding 3 — Unicode decoration.** Iteration 13's retry emitted
'Axolotல்' ("Axolot" + a Tamil grapheme) for a banned 'Axolotl' — a genuinely
different string, so the string-level ban cannot and should not block it, and
the raw edit distance is 2 (the grapheme is two code points). Fix:
`repeat_matches` also compares NFKD ASCII-folded normalized forms (fold both
sides, drop combining marks and non-ASCII), so 'Axolotல்' -> 'axolot' lands at
distance 1 from 'axolotl'. Accented but legitimately distinct secrets ('Açai
berry', 'Jalapeño') are unaffected.

The escalation ladder itself is the lesson: exact repeat -> resegmented exact
string -> one-letter misspelling -> Unicode substitution, each rung appearing
within iterations of the previous one closing. GRPO reliably finds any evasion
channel that still pays; a repeat gate must therefore be layered — decode-time
string ban (exact strings impossible), gate-time fuzzy match (near-misses
voided before episodes), judge validity (semantic garbage) — with the
telemetry (`repeat_voided`, `repeat_retries`, `repeat_retry_playable`,
`sampled_secrets`) to notice the next rung.

Known residual gaps, deliberately deferred: synonym evasion (iteration 13
sampled 'Kombu' with 'Kelp' already in-round — string-distinct, semantically
near-identical; would need embedding or judge-based novelty), validity-judge
noise now more visible under diversity pressure ('Calamine', 'Sajou',
'Ylang-Ylang' played as foods; 'Chervil' falsely voided), and serial next-best
collapse (the policy re-collapsing onto each freed candidate in turn — the
rolling window keeps forcing novelty mechanically; conditioning-side seeds and
the unreachable 0.9 target band remain the next levers per §6.5b).

### 6.8 Prompt audit: solver signal starvation and gemma prompt sensitivity (2026-07-17)

**Why v5 arm A was stopped (iters 0–5).** 31/912 episodes won (3.4%). At
K=16, 49 of 57 GRPO groups were all-loss — identical rewards, zero terminal
advantage, no solver gradient. The only variance in those groups came from
format penalties, so the solver was learning line formatting, not deduction,
while the creator collected full calibration signal every iteration (Rc
+0.47…+1.14, novel secrets, sensible difficulty ordering). Self-play was
running one-sided.

**Transcript audit findings (v4 + v4.5 + v5, ~3,900 episodes):**

1. *One guess per episode.* The engine has always treated a wrong GUESS as
   free (referee answers NO, play continues — `episode.py`), but the guesser
   was never told. The base policy volunteered a guess only at the forced
   last turn: 24 of 31 wins landed at turn ≥ 19. The lone exception (Toaster,
   12/16 wins, guesses from turn 13) shows the ceiling when the modal
   candidate happens to be the secret.
2. *Prose-then-contract-line bleed-through.* The dominant malformed output is
   the model writing its question as prose, then "completing the pattern" on
   the contract line — `Is it a mammal?\nQUESTION: Yes` — because the
   flattened history it reads is `"question" -> YES`. The parsed "question"
   becomes `Yes`, which pollutes every later turn of that episode. The old
   prompt invited this: "END your reply with exactly ONE line".
3. *Grammar-correlated destabilization.* Fraction of episodes whose first
   parsed question was degenerate (`Yes`/`No`/…), by category phrasing of
   the old `"The secret is a {category}."` line:
   animal ("a animal", ungrammatical) **83.4%**; food ("a food", odd mass
   noun) **21.5%**; household object (natural) **1.8%**. Category semantics
   and grammaticality are confounded (probe skipped by decision), but the
   ordering tracks grammaticality exactly and the fix costs nothing.
4. *Forced last guess ignores constraints.* E.g. fruit/sweet/whole/uncooked
   established, final guess "Tofu". The last-turn instruction never asked for
   consistency with the accumulated answers.

**Ensemble scoring-path audit** (all four members rendered with their own
tokenizers, span alignment checked):

- Span location correct on all members; Qwen3/SmolLM3 no-think rendering
  correct (Qwen3's empty `<think>` block is its native no-think format).
- *Non-stationary reward (fixed):* SmolLM3 and Llama-3.2 templates
  interpolate today's date into the system header, so identical
  (history, answer) pairs scored differently across midnight — mid-run, and
  across runs. `_apply_template` now pins `date_string` and shadows the
  `strftime_now` Jinja global with a constant.
- *Naming miscue (fixed):* the scoring prompts said "the game 21 questions";
  the deduction game is "20 Questions" in the scorers' pretraining data ("21
  questions" names a different party game). DEFAULT_SYSTEM/INSTRUCTION now
  say "20 Questions". The 21-turn budget is a trainer setting, not the
  game's name.
- Policy-side rendering was audited clean: the gemma-4 template accepts the
  `system` role natively, `enable_thinking=False` injects no think tokens,
  and `add_special_tokens=False` avoids double-BOS.

**Changes (v6, all prompt/reward-surface — no algorithm change):** guesser
system prompt makes the contract line the ENTIRE reply and states that wrong
guesses cost a turn but never end the game; user prompt uses article-free
"The secret is in the category: X."; the last turn demands the guess most
consistent with ALL answers. Ensemble prompts renamed + date-pinned as above.
These change the environment and the reward scale: v6 numbers are not
comparable to earlier lineages, and step-0 validation re-baselines.

**Run-order decision:** v6 starts with arm B (ensemble dense credit,
no rotation). Terminal-only credit demonstrably starves the solver at the
base win rate; the dense per-turn channel is the arm that can bootstrap it.
Wrong-guess rendering in history ("Is it X?" -> NO) also feeds the dense
potential exactly like a question, so guess-probing and dense credit compose.

### 6.9 Decoding-policy audit: the silent top_k=64, and sampler sensitivity (2026-07-19)

**Discovery.** While reconstructing solver inference after the v6 run, we
found that every sampled rollout in the lineage has been running under
**top_k=64** — a setting that appears nowhere in this repo. The backend
passes only `temperature` and `top_p` to HF `model.generate`
(`torch_backend.py`); HF merges every unset knob from the model's own
`generation_config.json`, and the pinned gemma-4-E2B snapshot ships
`{"do_sample": true, "temperature": 1.0, "top_k": 64, "top_p": 0.95}`. So
the effective sampler was temp -> top_k=64 -> top_p=0.95 (HF warper order)
for the solver (0.8), creator (0.9), and answerer (0.2) in every run,
unrecorded in any config or run metadata. Greedy validation is unaffected
(`do_sample=False` builds no warpers), which is also why the format-fail
asymmetry of §6.8/EXPERIMENTS v6 finding 5 (sampled 3–40%/iter, greedy 0%
at all 7 checkpoints) cleanly brackets the fails as sampling-tail events.
Portability trap for reproduction attempts: an unstated top_k means three
different samplers in three stacks (llama.cpp defaults 40, vLLM disables
top_k, HF's library fallback is 50).

**Code change (commit be35a62).** `generate`/`generate_batch` now accept
optional `top_k`/`min_p`. `None` means UNSET — the kwarg is omitted so HF
still falls back to the model config (i.e. exactly the historical behavior;
passing `None` through would instead override-and-disable the warper), and
`top_k=0` requests no truncation explicitly. No caller changed; configs
should start pinning the sampler explicitly from v7 (whatever the value) so
runs stop depending on a file HF could revise.

**Validation set v2 (same commit).** The sensitivity probe needed more
resolution than 12 greedy episodes (±1 win ≈ ±8.3pp), so
`data/twentyq-validation-v2.json` doubles the set to 24: the 12 v1 entries
verbatim (ids preserved — the v1 series stays computable as a subset) plus
12 new entries mirroring the 0.15/0.35/0.65/0.85 ladder per category. New
entries were screened against the v6 run's 308 distinct training secrets
("apple" was rejected for this — the creator had used it). Known blemish
carried over: pangolin is both `val-v1-animal-03` and the v6 run's
most-sampled training secret (20 exact + 7 "Pangolín"); nothing stops the
creator from sampling validation entries. Cheap v7 fix: seed the creator's
exclusion list with the validation secrets.

**Probe design** (`scripts/probe_twentyq_base_sampling.py`). BASE model, no
adapters — this isolates the decoding policy from anything training did.
Sampled episodes on the v2 set, 8 per secret (192/arm, 1152 total), guesser
at solver temp 0.8, answerer pinned at oracle settings (temp 0.2,
config-default truncation) so only the guesser's sampler varies; 21-turn
budget, per-arm seeds (20260719+i). Six arms; min_p arms disable top_k and
top_p so min_p is read alone, not stacked under two other warpers:

    arm                       win%   fmt%  | easy(.15) med(.35) hard(>=.65)
    top_k=64  (historical)    13.5   19.8  |   37.5     16.7      0.0
    top_k=40                  13.0   13.5  |   37.5     14.6      0.0
    top_k=20                  10.9   18.8  |   27.1     16.7      0.0
    top_k=0   (top_p .95)     14.1   17.7  |   39.6     14.6      1.0
    min_p=.05 (top_p 1.0)     13.5   15.6  |   39.6     14.6      0.0
    min_p=.10 (top_p 1.0)     12.5   20.3  |   33.3     16.7      0.0

    n=192/arm -> SE ~2.4pp (win), ~2.7pp (fmt). Wall: ~9 min/arm, 53 min total.

**Findings** (raw per-episode rows: `tq-runs/probe-base-sampling-v1.json`,
`settings[i].episodes[]`, each row `{secret, category, guessed, ended,
turns}`; per-secret console lines in the `.log` sibling):

1. **Win rate is sampler-flat.** 10.9–14.1% spans ~1.3 SE. top_k=64 was
   buying nothing; nominal best is NO truncation beyond top_p 0.95. The
   decoding policy is not a lever on wins.
2. **Sampler sensitivity is real but localized to easy secrets** —
   confirming the observation that motivated the probe. The easy tier
   spreads 27.1–39.6% across arms while medium is flat (~15–17%) and hard
   is floored. The variance is carried by a few volatile secrets: banana
   3/0/0/4/5/1 wins-of-8 across arms (k64/k40/k20/k0/mp05/mp10),
   television 3/5/2/3/5/4, bread 2/3/3/4/1/3 — while robust secrets ignore
   the sampler entirely (cow 8/8/8/7/8/8, penguin 6/7/8/7/7/8). By
   category: animal is stable (23.4–25.0%) across all six arms; food
   (4.7–14.1%) and household (3.1–10.9%) carry all the sampler variance.
3. **The difficulty cliff is capability, not decoding.** Difficulty >=0.65:
   1 win in 576 episodes pooled across all six arms (hummus, 1/8, top_k=0
   arm); the 0.85 tier is 0 for every arm. No sampler rescues what the
   model cannot deduce — training, not decoding, owns the medium/hard tiers.
4. **Format fails are v-shaped in BOTH truncation families.** Moderate
   truncation trims the tail (top_k=40: 19.8 -> 13.5%; min_p=.05: ->
   15.6%) but harder truncation gives it back (top_k=20: 18.8%; min_p=.10:
   20.3% — the worst arm). Consistent with §6.8/EXPERIMENTS: the fail modes
   are partly HEAD-of-distribution attractors (bare-guess/stereotyped
   lines), which aggressive truncation concentrates rather than removes.
   Caveat: the key 64-vs-40 fmt gap (6.3pp) is ~1.7 SE — suggestive, not
   proven.
5. **The difficulty labels do not track winnability.** giraffe (0.15) wins
   1/0/0/1/0/0 across arms; penguin (0.35) is near-automatic. Rank-based
   analyses should use measured winnability, not the authored difficulty.

**Decisions/recommendations.** (a) Pin the sampler explicitly in every v7
config — independent of value chosen. (b) Candidate solver setting is
top_k=40 (best fmt arm at zero win cost), but it should not be baked in on
a 1.7-SE read: the cheap firm-up is a 64-vs-40 head-to-head at 16
eps/secret (~35 min) before v7. (c) Expect nothing from decoding on
medium/hard secrets. (d) The creator/answerer samplers were NOT probed
here; creator diversity under truncation interacts with the §6.5b collapse
dynamics and needs its own read before touching creator_temp's sampler.

**Verify:** re-run the full grid with
`.venv/bin/python -u scripts/probe_twentyq_base_sampling.py` (defaults =
this probe exactly: config `twentyq-full-v6.yaml`, v2 set, seed 20260719);
summaries via `jq '.settings[] | {label, guess_rate, format_rate,
by_category}' tq-runs/probe-base-sampling-v1.json`. The snapshot's shipped
sampler: `cat ~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/
snapshots/9dbdf8a.../generation_config.json`. Closeout context:
EXPERIMENTS.md "exp-fullv6 RESULTS" findings 5–6 (commit 506ade1).

---

## 7. Frozen roles and flat-rate difficulty (2026-07-25)

Three config levers added so a run can hold parts of the self-play system
still and ask a narrower question than "does the whole loop work". Motivated
by the v6 closeout: the solver improved in-distribution (first-10 win rate
0.040 → last-10 0.089) with no measurable transfer to stationary validation
(2,0,2,1,3,2,2 wins at steps 0..60) while the dense ensemble proxy rose
monotonically (0.575 → 0.616) — a result with at least three live
explanations (co-adapting creator, a ramp that spends most ranks on
unwinnable games, a proxy whose fixed point is not winning) that the v6
design cannot separate.

### 7.1 Config surface

| key | default | meaning |
|---|---|---|
| `twentyq.freeze_creator` | `false` | creator role plays and is scored, takes no GRPO step |
| `twentyq.freeze_solver` | `false` | solver role plays and is scored, takes no GRPO step |
| `twentyq.difficulty_mode` | `"gradient"` | `gradient` \| `flat` — how per-rank difficulty/target is dictated |
| `twentyq.flat_target_rate` | `0.5` | flat mode only: the one target guess rate every rank is given |

CLI overrides on `scripts/train_twentyq.py`: `--freeze-creator/--no-freeze-creator`,
`--freeze-solver/--no-freeze-solver`, `--difficulty-mode`, `--flat-target-rate`.
All four land in the run-log meta, the transcript-tree meta, and every
iteration record (`difficulty_mode`, `flat_target_rate`, `frozen: {creator,
solver}`), so a log identifies its own regime without the config beside it.

### 7.2 Defaults are exactly the v6 behaviour

`difficulty_mode="gradient"` reproduces the historical prompt **byte for
byte** (verified across 108 category × N × rank × target × exclusion
combinations against `HEAD:src/twin/games/twentyq/prompts.py`), and both
freeze flags default off. No existing config changes behaviour.

### 7.3 What freezing skips — and what it does not

A frozen role still generates, is still scored, and still appears in the
logs. What is skipped is only the work that exists to feed its update:

| | frozen creator | frozen solver |
|---|---|---|
| generation | runs | runs |
| reward computation | runs (pure arithmetic) | runs (pure arithmetic) |
| trajectory construction | skipped | skipped |
| base-reference KL pass + policy pass | skipped | skipped |
| **frozen-ensemble dense scoring** | unaffected | **skipped entirely** |

The asymmetry is the point. Freezing the creator saves ~10 short rollouts of
GRPO out of an iteration dominated by ~160 episodes of generation and ~3000
solver turn-trajectories at `grpo_microbatch=1` — its value is
**experimental** (a stationary opponent *and* a stationary answerer, since
the creator adapter is also the oracle), not compute. Freezing the solver is
the real compute lever: the dense potentials exist only to build solver
returns, so a frozen solver skips K·(T+1) scored histories per secret across
every ensemble member and never loads the ensemble at all.

Episodes still run under a frozen solver because the creator's calibration
reward is measured *from* their guess rates. The terminal component trace is
still built (zero potentials, pure arithmetic) so `reward_signals` keeps the
shape every log reader expects; `dense_*` fields read 0.

`creator_reward_mean` is now measured over the per-rollout rewards rather
than the trajectories, so a frozen creator reports its real calibration mean
instead of 0. Skipped updates log `{"n_traj": 0, ..., "frozen": true}` — a
skipped step is never confusable with a step that ran to a zero loss.

**Freezing is by ROLE, not by adapter name.** Whichever adapter plays the
frozen role that iteration is not updated, so under rotation both adapters
still train (each while it plays the unfrozen role) — almost never what a
freeze is meant to express. The launch script warns when a freeze flag is
combined with `roles.swap_interval > 0`, and warns separately when both roles
are frozen (a play-and-measure run that trains nothing — a legitimate mode).

### 7.4 Flat-rate difficulty

Gradient mode dictates rank *i* difficulty `i/(N−1)` and a target guess rate
from the `rewards.target_hi → target_lo` ramp. Flat mode dictates the same
target (`flat_target_rate`) and the same difficulty (`1 − rate`) to every
rank, so an iteration asks for a uniformly-pitched *bank* instead of a
ladder.

The reward needs no new machinery. `creator_problem_rewards` already scores
each secret against the target its own rollout was **prompted** with
(`target_by_problem`, the "prompt targets == reward targets" invariant from
§6.1), so flat mode only makes that vector constant:
`R_i = w_grad·exp(−β·(p_i − t)²) + w_cons·consistent_i + w_valid·valid`. The
per-secret GRPO group, its mean baseline, the exclusion list, the repeat
gate, and the parse/validity gates are all untouched. Advantage variance
survives — with one shared target the group ranks secrets by *how close to
t* they landed, which is a cleaner comparison than the ramp's (each rank
judged against a different target).

Round framing and the exclusion block are unchanged in flat mode, so the
creator still sees its recent and current-round picks and repeat handling
behaves identically. The rank number now orders the round without implying a
ramp.

### 7.5 The difficulty-spread trap (found during implementation, fixed)

`ProblemSuite.validate()` carries an anti-collapse rule: a suite whose
hardest and easiest dictated difficulty differ by < 0.3 is INVALID. Flat mode
dictates one difficulty to every rank by construction, so every flat suite
would have failed it — silently zeroing the `w_valid` term on every secret
(a constant shift, so advantages under `adv_mode=mean` are unaffected, but
reward *levels* would have looked like a regression against v6 for a reason
found nowhere in the diff).

Fix: `validate`/`is_valid` take `min_spread` (default `0.3`, unchanged), and
the trainer passes `min_spread=0.0` in flat mode and the explicit result to
both `creator_problem_rewards(valid=...)` and `creator_reward(valid=...)`.
Gradient mode is unaffected. Regression-tested by
`test_flat_mode_keeps_the_validity_term_despite_zero_difficulty_spread`.

### 7.6 Is a flat 50% target reachable? (raw data)

A flat target above the solver's ceiling would turn calibration into a
one-sided "as easy as possible" gradient. It is not: base-model per-secret
win rates, 48 episodes each (pooled over the 6 sampler arms of the §6.9
probe, `tq-runs/probe-base-sampling-v1.json`):

| secret | label | win rate | best arm |
|---|---|---|---|
| cow | 0.15 | **0.979** | 1.000 |
| penguin | 0.35 | **0.896** | 1.000 |
| television | 0.15 | 0.458 | 0.625 |
| bread | 0.15 | 0.333 | 0.500 |
| banana | 0.15 | 0.271 | 0.625 |
| refrigerator | 0.15 | 0.062 | 0.250 |
| giraffe | 0.15 | 0.042 | 0.125 |
| scissors, pizza, hummus | — | 0.021 | 0.125 |
| the other 15 | — | 0.000 | 0.000 |

0.50 is comfortably interior, so the calibration gradient is two-sided.
Two real caveats: (a) the achievable distribution is **bimodal** — 12 of 24
secrets sit at exactly 0.000 and two above 0.89, with only
television/bread/banana in between — so a flat 0.50 target selects precisely
for that narrow frontier band, which is the desired curriculum but a coarse
(flat-with-cliffs) learning signal for the creator; (b) at K=16 a true-0.50
secret is measured with SE ±0.125, and `exp(−4(p−t)²)` pays ~0.94 for pure
sampling noise, so flat-mode creator credit is noisy at the target. Neither
matters while the creator is frozen. Note also that the §6.9 finding stands:
the authored difficulty labels do not track winnability (cow 0.15 → 0.979 vs
giraffe 0.15 → 0.042), so this table, not the label, is the reachability
reference.

**Reproduce:** the per-secret table is a pooled regroup of the probe's raw
rows — `jq '[.settings[].episodes[]] | group_by(.secret) | map({secret:
.[0].secret, n: length, wins: map(select(.guessed)) | length})'
tq-runs/probe-base-sampling-v1.json`.

### 7.7 First run using this: `configs/twentyq-v7-solver-only.yaml`

Solver-only, no rotation, flat @ 0.50, `credit: terminal` (no ensemble) —
the three v6 confounds removed at once, deliberately, as a diagnostic rather
than a matrix arm. From a cold start the frozen creator's LoRA is zero-init,
so the opponent *is* the frozen base model for the whole run. Everything
else is held at v6 values (N=10, K=16, T=21, lr, kl_beta, adv_mode,
microbatch); validation moves to the v2 24-secret set, a strict superset of
v1 with ids preserved, so the v1 subset stays comparable to the v6 series.
Sampling still inherits `top_k=64` (§6.9) — there is no config surface for
it, and pinning it here would silently diverge from v6.

**Read it as a success only if adapter B's VALIDATION guess rate rises.** A
rising training win rate alone reproduces v6 and means nothing. `credit:
terminal` is also the scheme that starved in v5 (3.4% win rate, 49/57
all-loss GRPO groups); whether the flat-0.50 bank restores group outcome
variance is the second thing to watch, visible directly as
`guess_rates_by_rank` sitting off 0.0/1.0.

**Verify the implementation:** `.venv/bin/python -m pytest tests/twentyq -q`
(covers frozen-role skips, ensemble-skip under a frozen solver, flat targets
and prompts, the validity-spread fix, and gradient-mode prompt invariance).

---

## 8. Thinking control: config-consistent, decode-enforced (2026-07-25)

`enable_thinking` was a *request*, not a guarantee. A chat template's
`enable_thinking=False` only declines to INVITE reasoning — nothing stops the
model opening a reasoning span anyway, and gemma-4-E2B sometimes did. The
flag is now enforced in two halves, resolved in one place
(`BaseTrainer._generate` / `_generate_batch` / `_judge`):

1. **template** — `render(..., enable_thinking=think)`, as before. In gemma-4's
   template this injects `<|think|>` at the top of the system turn; in Qwen3's
   it switches the `<think>` convention.
2. **decoder** — when thinking is off, `suppress_thinking=True` bans the
   reasoning-open markers (`twin.think.THINK_OPEN_MARKERS`) at the logits
   level via `BannedStringsProcessor`, the same string-level mechanism the
   §6.5b repeat gate uses. The model *cannot* begin a trace under any BPE
   segmentation.

`twin.think.THINK_SPANS` is now the single definition of "thinking": the same
literal markers drive `strip_think`, `think_share`, `think_text`, and the
decode-level ban, so suppression and telemetry cannot drift apart. Adding a
model family = adding one `(open, close)` pair.

### 8.1 What is and is not guaranteed

**Guaranteed:** no marker-delimited reasoning span (`<think>…</think>`,
`<|channel>…<channel|>`) when thinking is off. **Not guaranteed:** that the
model never deliberates in *plain prose* before its contract line — banning
control tokens cannot forbid ordinary text, and shouldn't. Prose rambling is
a prompt/contract problem (§6.8), not a thinking-flag problem.

The ON direction is weaker by nature: you can invite reasoning, you cannot
compel it. The template flag is the model's designed switch; the probe below
verifies it actually fires rather than assuming it.

**Backends:** enforcement is torch-only. MLX generation goes through
`make_sampler`, which has no logits-processor hook, so `suppress_thinking`
there warns once per process (`RuntimeWarning`) and applies the template flag
alone. It warns rather than raises because thinking is off by default and
raising would break every MLX run. The Spark runs torch, so twentyq gets the
hard guarantee.

### 8.2 Measured on the real model (Spark, gemma-4-E2B)

`scripts/probe_thinking_suppression.py`, 8 samples/arm, raw results in
`tq-runs/probe-thinking-suppression.json`:

| role | arm | markers | think_share | completion tokens |
|---|---|---|---|---|
| answerer | off (suppressed) | 0/8 | 0.000 | 4 |
| answerer | on | 8/8 | 0.970 | 96 |
| answerer | historical (v1..v6) | 0/8 | 0.000 | 4 |
| guesser | off (suppressed) | 0/8 | 0.000 | 9 |
| guesser | on | 8/8 | 0.968 | 279 |
| guesser | historical (v1..v6) | 0/8 | 0.000 | 9 |

Both directions hold: OFF is clean, ON thinks. Note the cost of thinking when
it IS wanted — a guesser turn goes 9 → 279 tokens, so `question_max_tokens`
would have to grow before `guesser_thinking: true` is survivable (the
q-shakeout-01 truncation spiral).

### 8.3 How often did this actually bite? (honest answer: rarely)

The `historical` arm is identical to the suppressed one, so on twentyq's
prompt paths the base model does **not** think unbidden at meaningful rates —
this change is a guarantee, not a rescue, and it does not move the v6
baseline. The live rate over the whole 60-iteration v6 run
(`tq-runs/q-fullv6-ctrl-ensemble.transcript.txt`, 482,060 lines):

- `guesser_think_share` / `creator_think_share` logged **0.000** every
  iteration (max 0.001).
- **10** `<|channel>` occurrences total. **7 were the JUDGE** — whose prompt
  literally says *"Think briefly, then end with exactly one line: VERDICT:"*,
  i.e. the prompt invited exactly what `judge_thinking: false` forbade. That
  contradiction is now resolved in favour of the config.
- **2 were guesser turns, and both became `[format_fail]`** — out of 1,283
  format fails, so ~0.2% of them. A real but minor cause, now removed.

Verify: `grep -c '<|channel>' tq-runs/q-fullv6-ctrl-ensemble.transcript.txt`,
and `grep -B3 '<|channel>'` for the role context.

### 8.4 Transcript: an "answerer thoughts" block

The episode file renders, in order: `answerer SYSTEM prompt` (collapsed),
`answerer USER prompt` (collapsed), `**answerer output**` with the raw
completion, then a collapsed **`answerer thoughts`** block holding the
reasoning content extracted by `twin.think.think_text`. The block is *absent*,
not empty, when the model did not think — so its presence is itself the
signal. An unclosed span (a truncation spiral) is shown to its end rather than
dropped for lacking a close tag, and gemma's `thought` channel label is
stripped so the block starts at real reasoning.

**Verify:** `.venv/bin/python -m pytest tests/test_think_control.py
tests/backends/test_banned_decoding.py tests/twentyq/test_q_transcript_tree.py -q`
and, on the Spark, `.venv/bin/python -u scripts/probe_thinking_suppression.py`.

## 9. Turn waste, parse loss, and the group-variance wall (2026-07-26)

Sections 6-8 tuned the reward, the difficulty schedule, and the decode policy.
This section records what an audit of the v7 run's *transcripts* — rather than
its metrics — found underneath all of that, and the four changes it forced.

### 9.1 The wall: a GRPO group that cannot vary

A solver GRPO group is the K episodes played on ONE secret (`rewards.
per_turn_secret_trajectories`, baselined per turn index). Its advantages are
identically zero when the group is all-win or all-loss. Measured on the v7 base
model over validation-v2 at K=8 (`scripts/probe_twentyq_headroom.py`):

    usable_group_rate  0.125       3 of 24 secrets had both a win and a loss
    per-secret wins    cow 8/8, penguin 7/8, television 4/8, bread 1/8,
                       0/8 for the other twenty

Per-secret win probability is **bimodal**. The variance GRPO needs therefore
lives *between* secrets, where the per-secret baseline cannot reach it, and not
*within* one, where it can. Seven eighths of every iteration's compute produced
no gradient at all.

This subsumes several earlier explanations. v5's "credit starvation" (49/57
all-loss groups) is this. v6's and v7's flat validation is consistent with this.
It is not a reward-design problem, and no credit scheme fixes it: `terminal`,
`ensemble` and `per_turn` all multiply an advantage that is already zero.

§7 proposed difficulty calibration as the fix, and it cannot be, for two
reasons. The creator is asked to hit a target guess rate for an opponent whose
competence it cannot observe; and it cannot *learn* calibration while the solver
it calibrates against is itself untrained. The dependency is circular.

**Resolution — `secret_source: bank`.** Cut the circle on the solver's side
first, since that is priority one. `scripts/build_twentyq_bank.py` generates
creator candidates, dedups them with the repeat-gate matcher, vets them
fail-closed, then *plays* K episodes per candidate and keeps only those whose
measured win rate lands in a band. At p ∈ [0.125, 0.875] with K=16, ~88% of
groups carry signal instead of 12%.

The creator adapter still answers, so the twin structure is intact — only
authorship moves. This is scaffolding with an explicit exit condition: the
bank's `measured_guess_rate` is exactly the supervision target the creator has
to learn to hit, and the rates drift as the solver improves, so they must be
re-measured. Handing authorship back is §10's job.

### 9.2 Parse loss: a fifth of all episodes were deleted, not lost

Sampled v7 episodes ended in `format` 20-25% of the time — an instant `-w_format`
and a dead episode. Re-parsing all 632 recorded format-failed completions shows
88.3% carried well-formed intent that the line-anchored contract regexes could
not see:

    <h3>QUESTION: Is the food animal-based?</h3>      200   markup-wrapped
    "Is it a nut?"                                    159   quoted question
    Key lime pie / Taco / Truffle                     238   bare guess
    Closure: GUESS: clear broth                         -   not line-initial

`parse_guesser_turn` now strips presentation markup (HTML, markdown, LaTeX,
braces, edge quotes) *before* the contract regexes, accepts non-line-initial
contract words and U+FF1F, and as a final rung reads a lone short entity as a
GUESS. That last rung is the *safe* error: a wrong guess costs one turn and play
continues, whereas a format failure ends the episode. 74 of 632 still fail,
correctly (prose, empty completions, meta-commentary).

Note this interacts with §9.1: a format failure is always a loss, so parse loss
was actively pushing groups toward all-loss.

### 9.3 Turn waste: the lock, and why discounting is the wrong fix

Greedy v7 validation transcripts show the guesser locking:

    giraffe  turns 16-19  "Is the animal a hoofed ungulate?" x4   lost
    penguin  turns 10-19  "Is the animal a parrot?"          x10  WON on turn 20

Two independent mechanisms, deliberately separable:

- `question_retries` (engine): resample a turn whose content-word key
  duplicates an earlier question or guess in the same episode. Changes the DATA.
- `w_repeat` (reward): a **local, non-propagating** penalty on a turn that
  repeated anyway. Changes the GRADIENT.

`w_repeat` is needed because retries are a scaffold, and validation deliberately
never uses them — measuring under retries would score the scaffold rather than
the policy.

The locality matters. Under `terminal` credit at γ=1 every turn of a winning
episode receives the same return, so GRPO reinforced those ten parrot turns
exactly as hard as the informative ones. Discounting is **not** the fix: the
lock occupies the LATE turns and the informative questions the early ones, so
γ<1 moves credit *toward* the repeats. Hence a term that lands on the offending
turn and nowhere else — winning is a shared property of an episode, but asking a
question the transcript already answered is a defect of exactly one turn.

Caveat on scope: the lock is prominent under GREEDY decoding. Under the sampled
decoding training actually uses, measured repeat rate is ~6.6% of turns, and
§9.2 is the larger effect. Both are fixed; only §9.2 should be expected to move
the headline number much.

### 9.4 The instrument could not have seen the effect

Validation was 24 GREEDY episodes — one per secret. That metric moves in steps
of 1/24 = 4.2%, and a 24-flip binomial near p=0.125 has a 95% Wilson interval of
roughly ±13 points. The v6 series (2,0,2,1,3,2,2 wins across steps 0..60) was
discussed as a trend, and every point of it is inside every other point's
interval. **"No transfer to validation" was never established by that data.**

`validation_episodes: K` now samples K games per secret (greedy would replay one
game K times) through the batched engine, and `guess_rate_ci95` is reported at
every level. At K=8, n=192 per adapter and the interval is ~±4.5 points.

This is why no result in §§9.1-9.3 should be read as explaining v6's outcome:
v6's outcome was not measured tightly enough to need explaining.

### 9.5 Confound removed: per-iteration category sampling

v7 drew ONE category per iteration i.i.d. while per-category win rates differ by
40x (household 0.120, food 0.003, animal 0.003 pooled). The per-iteration
win-rate series therefore mostly measured which category came up; three separate
"trends" read off it during that run were sampling artifacts, and each had to be
retracted. Bank mode draws category-balanced within an iteration
(`bank_balance_categories`), and the episode loop now uses each SECRET's
category rather than the iteration's. Per-secret summaries record it, so the
confound is conditionable-on even where it still exists.

### 9.6 Resolution: the measured frontier bank (built 2026-07-26)

§9.1 states the problem; this is what was actually built and what it delivers.

`scripts/build_twentyq_bank.py` generates creator candidates, dedups them
(including against the validation holdout), vets them, then **plays** each one K
times and keeps only those whose measured win rate lands in [0.125, 0.875].
Difficulty stops being dictated and becomes measured: a bank entry's
`difficulty` is `1 - measured_guess_rate`, which is also the supervision target
that makes creator calibration trainable later.

    750 creator rollouts -> 118 unique -> 118 vetted -> 61 in band (51.7%)
    animal 22, food 15, household object 24;  mean measured rate 0.391

Delivered, measured at iteration 0 of the v8 run:

    usable_group_rate   0.875      (v1..v7 secret source: 0.125)

That is the wall removed — 7 of 8 groups now carry a gradient where 7 of 8
previously carried none.

Three things learned in the build that change how the next one should be run:

- **Diversity, not yield, now bounds bank size.** 750 rollouts produced only 118
  distinct secrets. Targets at the easy end of the scale draw from a small
  vocabulary of instantly-recognizable entities, so more rollouts saturate.
  Raising bank size means raising creator diversity, not sampling harder.
- **Selecting on a noisy measurement causes winner's curse.** Entries near the
  top of the band were partly lucky at K=8 and regress when replayed: measured
  drift after one update was -0.070, concentrated entirely in the high-rate
  secrets (Tiger .88->.62, Whale .75->.31). Correcting for measurement noise
  (posterior over p) does NOT correct for selection on it. Either calibrate at
  larger K, or expect the delivered bank to sit below its nominal band.
- **The validity judge is partial and the vet rate hid it.** 118/118 passed;
  driven over deliberate negatives the same judge false-accepts 4 of 8 (water
  as a food, "a dog or a cat", Pikachu, an invented obscure name) at 0/6 false
  rejects. The bank's guarantee is the measured band, not the judge. This
  blocks any priority-two design that rewards the creator for "valid" secrets
  refereed by this judge alone.

Success criterion for the run this bank feeds, fixed in advance: adapter B's
VALIDATION guess rate must rise beyond its own confidence interval AND beyond
the spread of the frozen adapter A control measured on the same schedule.
Training win rate on a bank selected to be winnable proves nothing by
construction.

### 9.7 The instrument, not the policy: validation-v2 is composition-limited (2026-07-26)

The v8 run does what §9.6 predicted on the axis it targeted, and nothing on the
axis that was pre-registered. Both facts are solid, and they are not in
tension — but reading them requires knowing what the metric can and cannot
register.

What moved:

    usable_group_rate     0.125  ->  0.896 mean over 24 iterations   (7.2x)
    within-secret change  +0.148 +-0.048 (95%) over 59 secrets played in both halves
    bank drift            +0.074 mean, 0 of 61 secrets gone all-win or all-loss

The within-secret panel is the one to trust: the same secrets appear on both
sides, so neither the category draw (§9.5) nor regression to the mean applies.
Instance memorisation is ruled out separately — regressing each play's delta on
iteration + prior exposures to that secret gives exposures -0.082 +-0.085 (n.s.)
against iteration +0.021 (t = +3.5), and a matched within-iteration contrast of
first vs second exposure gives -0.117 +-0.121. The solver is getting better at
the game, not at these secrets.

What did not move, on the pre-registered criterion:

    step        A (frozen control)    B (solver)     B-A
       0      0.141                 0.141          +0.000
       5      0.109                 0.141          +0.031
      10      0.120                 0.172          +0.052
      15      0.130                 0.141          +0.010
      20      0.151                 0.146          -0.005

The frozen control's own spread across those checkpoints is 0.042, which
exceeds every B-A in the table. The step-10 "+0.052" was noise, and the
criterion in §9.6 exists precisely so that it gets read as noise.

**Why this run cannot resolve the disagreement.** Per-secret, adapter B scores
0/8 on fifteen of validation-v2's twenty-four secrets at EVERY checkpoint
(platypus, pangolin, kangaroo, hedgehog, tapir, pizza, hummus, rambutan,
popcorn, couscous, persimmon, stapler, smoke detector, shoehorn, corkscrew),
while cow is pinned at 8/8. The metric's entire dynamic range is about seven
secrets.

This is §9.1's wall applied to the MEASURING INSTRUMENT. A secret that no
checkpoint ever wins carries no information about whether the policy improved,
in exactly the way an all-loss group carries no gradient — the same
zero-variance argument, one level up. Raising K from 1 to 8 (§9.4) fixed the
sample size and could not fix this, because the binding limit is which secrets
are in the set, not how many times each is played.

Two explanations remain live and this run distinguishes neither: real skill the
instrument cannot see, versus bank-distribution fitting that genuinely does not
transfer. Honesty requires keeping a counter-caveat attached: validation-v2's
*reachable* subset (television, penguin, banana, bread, refrigerator, scissors)
is also roughly flat, so resolution alone does not explain the null.

**The protocol, and why it is shaped this way.** The instrument is never
swapped mid-run. The pre-registered number stands on validation-v2, for that
criterion and for comparability with v1..v7. What is legitimate is to replay
the SAVED checkpoints against a better set afterwards
(`scripts/rescore_checkpoints.py`), reported as a clearly labelled secondary
analysis. Choosing a new instrument after seeing the first one's answer is the
goalpost move a pre-registered criterion exists to prevent; the two defences
are that the primary number still stands, and that validation-v3
(`scripts/run_validation_v3_build.sh`) is selected by playing the BASE policy
alone and never consults a trained adapter. Selecting test items on which the
model under test does well would be contamination; selecting them on being
resolvable at all is not.

Two deliberate differences from the bank build, everything else held fixed so
that exactly one property of the instrument changes:

- **K=16 rather than 8.** Band membership is decided on a noisy measurement, so
  entries near an edge are partly lucky and regress on replay (the winner's
  curse in §9.6). A training bank tolerates that; for a measuring instrument
  selection error is the dominant risk, and K is the knob that reduces it.
- **400 rollouts per category rather than 250.** 85 names are already spoken
  for by the bank and validation-v2, and this creator yielded only 118 distinct
  secrets in 750 draws. Diversity, not calibration yield, bounds set size.

Note what band-selected validation does and does not cost. Selection noise
shifts the absolute level of the set (selected entries regress toward the
population mean on replay), but it does not bias the CONTRAST between two
adapters scored on the same fixed set — which is what the criterion is stated
in. The re-scorer also reseeds identically before every step, so all
checkpoints face the same sampled games; in-run validation cannot do that
without disturbing the training stream, which makes the replay strictly more
sensitive than the series it re-scores.

### 9.8 The v8 null is uninformative: minimum detectable effect (2026-07-26)

§9.7 argued from composition that validation-v2 could not resolve the question
it was asked. That argument can be made quantitative, and the number is
decisive. `scripts/validation_power.py` computes it from the run's own data.

The comparison is PAIRED — adapters A and B are scored on the same secrets at
every checkpoint — so the noise estimate is the spread of the per-secret
difference B-A, not the spread of either rate. This matters: per-secret rates
run 0/8 to 8/8, and in an unpaired calculation all of that between-secret
spread counts as noise even though both adapters face the same items. Unpaired,
v8 would appear to need ~188 secrets to resolve +0.05; paired it needs ~136.

Measured on v8 (5 checkpoints, 24 secrets, 9 live, 15 dead):

    paired sd of per-secret (B - A)     ALL  0.1159      LIVE  0.1867
    minimum detectable effect           ALL +0.0663      LIVE +0.1744
    (80% power, two-sided p<0.05)

Against the +0.148 within-secret training gain, checked both ways because the
two views can only mislead separately:

    pooled   0.148 x (9/24 live) = +0.0555  vs MDE +0.0663  -> INVISIBLE
    live     0.148                          vs MDE +0.1744  -> INVISIBLE

**Both framings agree.** Even PERFECT transfer of the measured training gain
would have produced a validation series indistinguishable from flat. The v8
null therefore carries no evidence about transfer in either direction, and the
run's pre-registered criterion — correct as a guard against reading trends out
of noise — could never have fired. This is worth stating plainly because for
most of the run the flat series was treated as being in tension with the rising
training curve. It never was.

A finer-grained look confirms there is no hidden signal to rescue either. On
the 9 live secrets the pooled series is B 0.375, 0.375, 0.458, 0.375, 0.389
against control A 0.375, 0.292, 0.319, 0.347, 0.403 — B is not above A at the
last checkpoint. Mean turns-on-success fell 16.77 -> 16.40 for B and 17.56 ->
17.13 for A, so B is not winning faster either. Off-the-floor events (a secret
at 0 in step 0 later reaching >=1) are 2 for B and 1 for A. None of this is
evidence of anything; it is confirmation that the instrument is silent rather
than that the policy is.

**Consequence for validation-v3, which is now a sized instrument rather than a
hopeful one.** A band-selected set is live by construction, so the LIVE sd of
0.1867 is the right planning number:

    effect   +0.15   +0.10   +0.07   +0.05   +0.03
    secrets     12      27      56     109     304

`scripts/run_validation_v3_build.sh` therefore hard-fails below 30 secrets
(MIN_V3). Below that it would reproduce v8's failure in a new costume — a null
that cannot distinguish "no transfer" from "could not have seen it" — after
spending hours to build and hours more to score against.

Note the honest limit of this analysis: the sd is estimated from validation-v2,
whose live secrets sit mid-range, and a band-selected set concentrates
everything mid-range where per-secret binomial variance is highest. The
planning number is therefore mildly optimistic, which is a further argument for
overshooting on size rather than hitting 30 exactly.

### 9.9 v8 final: what the completed run says, and a correction to §9.8

The run finished all 60 iterations cleanly (`Result=success`). Final numbers,
with two corrections to what §9.7/§9.8 recorded mid-flight.

**Correction 1: the effect size.** §9.7 quotes a within-secret gain of
+0.148 ±0.048. That was measured mid-run, splitting the ~28 iterations that had
then completed at their midpoint. On the full run the same statistic is
**+0.084 ±0.032** over 61 secrets — smaller, because a half-vs-half split of a
longer run averages two halves that each already contain most of the learning.
The end-to-end contrast is the larger number: comparing the FIRST TEN
iterations with the LAST TEN, within-secret win rate rose **+0.204 ±0.066**
over the 55 secrets appearing in both. Both statistics are correct and they
answer different questions; the honest headline is +0.204 end-to-end, and any
comparison against a detection threshold must say which one it means.

**Correction 2, and it matters: §9.8's central claim no longer holds.** That
section concluded "even PERFECT transfer would have produced a flat series."
That was true of the instrument as it stood at nine checkpoints, where the MDE
was +0.174 on the live subset and +0.066 pooled. The finished instrument is
better on both counts, for two reasons: thirteen checkpoints instead of nine,
and — less obviously — the live subset GREW from 9 secrets to 12 as the solver
started occasionally winning giraffe, trivet, scissors and refrigerator. The
instrument improved because the policy did.

    13 checkpoints, 24 secrets, 12 live, 12 dead
    paired sd (B - A)      ALL 0.1306     LIVE 0.1813
    minimum detectable     ALL +0.0747    LIVE +0.1467

    effect on live secrets   pooled equivalent   verdict
      +0.204 (end-to-end)         +0.102         DETECTABLE
      +0.150                      +0.075         DETECTABLE (marginal)
      +0.100                      +0.050         INVISIBLE
      +0.084 (half-vs-half)       +0.042         INVISIBLE

So the correct reading is sharper than "uninformative", and it cuts both ways:

- **Complete transfer is disfavoured.** If the +0.204 end-to-end training gain
  had transferred in full, this instrument had roughly 80% power to see it. It
  saw nothing: B rose +0.041 from step 0 against a frozen-control spread of
  0.062, failing the pre-registered criterion.
- **Partial transfer cannot be excluded.** Anything at or below about half the
  training gain sits under the detection threshold. The honest conclusion is
  that transfer is bounded above, not that it is zero.

The full validation series, for the record:

    step     0     5    10    15    20    25    30    35    40    45    50    55    60
    B    .141  .141  .172  .141  .146  .188  .177  .146  .203  .193  .177  .151  .182
    A    .141  .109  .120  .130  .151  .172  .130  .135  .151  .120  .146  .125  .167

B peaked at step 40 and declined for three checkpoints afterwards. A post-hoc
mean B-A of +0.029 reaches t=+4.2, but that test rose in significance purely
because n grew while the estimate stood still, and it asks the weakest
question — whether B sits above a frozen base on average, not whether B
improved. Its slope on step never reached significance at any point in the run
(peak |t| = 1.66 at step 40, falling to 1.30 by step 55). Treat it as an
artifact until the replay says otherwise; part of it may simply be the sampler
asymmetry documented in §9.10.

**Bank decay, quantified over a full run.** This is the cleanest result of v8
after the wall removal itself:

    window      0-9   10-19  20-29  30-39  40-49  50-59
    usable     .938    .912   .787   .812   .775   .713
    train win  .395    .516   .559   .537   .610   .584

`usable_group_rate` fell 24% relative while training win rate rose, exactly the
non-monotonicity of §9.1 — improving the solver pushes secrets out of the band
from the top. Mean bank drift is +0.138 and no secret has yet gone all-win or
all-loss at K=16, so the bank was still teaching at iteration 60, but it was
teaching measurably less than at iteration 0. A static bank has a shelf life,
and this measures it: roughly 60 iterations to lose a quarter of the gradient.

### 9.10 The adapters were never playing the same games (2026-07-27)

Found while checking whether B's step-40 rise was real, and it is a defect in
the measurement rather than in any run.

`run_validation` scores the adapters in a loop, A then B, off a single
advancing sampler stream. So the two arms never faced the same random draws.
The evidence is unambiguous because step 0 provides a perfect control: both
adapters are zero-init there, so they ARE the same policy and must score
identically up to sampling noise. They did not:

    step 0, identical weights:  160 of 192 episodes matched on turns
                                178 of 192 matched on outcome
                                totals tied at 27/192 by coincidence

That coincidence in the totals is why this went unnoticed through v1..v8: the
headline numbers agreed exactly while a third of the underlying games differed.

This matters because every criterion in this project is stated in B - A, and
independent sampling noise on the two arms does not cancel in a difference. It
also produces a specific, recognisable artifact — a roughly CONSTANT offset
between two adapters with no trend, which is exactly the pattern the post-hoc
mean-difference test picked up in v8 (§9.9).

`twentyq.validation_common_random_numbers` reseeds the sampler identically
before each adapter's pass. Two properties are deliberate:

- The seed does not depend on the adapter, so identical policies score
  identically by construction — which is what a frozen control is supposed to
  demonstrate, and now does.
- The seed does not depend on the STEP either, so B-at-step-60 versus
  B-at-step-0 is also a common-draw comparison. Varying it per step would
  restore the noise this removes from every across-checkpoint comparison.

Training gets a moving stream handed back afterwards (`seed + 1 + step`), so
no training phase replays the same sampler sequence after each validation.

Default is OFF. It changes which games are played, so enabling it by default
would silently break comparability with the v1..v8 series. It is forced on in
`scripts/rescore_checkpoints.py` — a replay has no training stream to protect —
and enabled for v9, where it is a precision change and not a third
experimental variable: it alters the variance of the estimate, not its
expectation.
