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
