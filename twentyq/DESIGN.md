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

### 2.3 Credit granularity: episode-level broadcast (v1), per-turn reward-to-go (v2)

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
   R_consistency exactly like an inconsistent problem.
2. **Answer truthfulness audit** (per episode, one batched call): judge sees
   the secret + all Q/A pairs, returns per-pair verdicts (`AUDIT: T T F T…`).
   Any lie ⇒ episode voided (excluded from solver reward AND from the secret's
   realized guess rate; drags creator consistency). A creator rewarded for
   difficulty has an incentive to answer misleadingly — voiding is the same
   anti-cheat structure as the existing consistency check.
3. **Closeness Φ** (v1: final state only; v2: every turn): `CLOSENESS: n`,
   integer 0–10 with anchor descriptions in the prompt (0 = category not even
   constrained … 5 = category + several key attributes pinned … 10 = uniquely
   determined). Mapped to Φ ∈ [0,1]. Parse failure ⇒ Φ contribution 0 (no
   shaping), never a crash.

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
pinning. Consistency flag per secret = valid AND no voided episodes from lying.

Creator answering turns are **not trained in v1** (reward attribution for an
individual "yes" is murky; lying already handled by voiding). The creator
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

**Thinking is per-role, not global** (`twentyq.creator_thinking` /
`guesser_thinking` / `answerer_thinking`). q-shakeout-01 killed the original
"tight per-turn guesser thinking budget" idea: at 512 tokens with thinking ON,
Qwen3 ran the whole budget inside an unclosed `<think>` and never emitted a
`QUESTION:` line — think-share 1.0, **every** episode a turn-1 format fail.
A tight thinking budget is the worst case (full cost, zero usable output), so
v1 splits by role:
- **creator ON** — secret calibration is genuine reasoning; `secret_max_tokens`
  must clear the trace *and* the JSON (768 was borderline at 0.955 share → use
  1024).
- **guesser OFF** — a direct `QUESTION:`/`GUESS:` line is reliable and ~10×
  cheaper across T·K·N generations. `question_max_tokens` drops to ~200.
- **answerer OFF** — a truthfulness lookup; the audit voids drift anyway.
  `answer_max_tokens` ~64.

Guesser thinking ON (with a *generous* budget, never a tight one) is a
Sprint-Q8 ablation, not the v1 default. Judge keeps `model.enable_thinking`
(validation reasoning). Format-failed episodes now always dump the raw guesser
completion to the transcript (the summary alone read `[format_fail] -> None`,
undebuggable).

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
episodes_per_secret, max_turns, categories, credit (`broadcast|per_turn`),
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
