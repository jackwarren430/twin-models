# Twin-Models — Self-Play RLVR with Two LoRA Adapters on One Frozen Base (v2)

**Status:** living design spec, current as of **2026-07-03** (post-Sprint 7, pre-mini-04).
Supersedes **DESIGN.md**, which is kept for sprint-by-sprint build history (its §11 documents how each
mechanism landed and why). **EXPERIMENTS.md** is the lab notebook (run entries + backlogs). This
document describes the system *as it exists* — every claim below is implemented and tested unless
explicitly marked as a known gap (§14) or a plan (§12).

---

## 1. Core idea

Two model instances **A** and **B** — two LoRA adapter trees over ONE shared frozen base — play a
self-play curriculum game. Each iteration one is the **creator** (proposes a suite of N problems
spanning easy→hard, with its own solutions and machine-checkable certificates) and the other is the
**solver** (attempts each problem K times). Both adapters are updated every iteration with GRPO on
orthogonal signals; roles swap every `swap_interval` iterations.

Claimed contributions over AZR / SSP-style single-weight self-play:

1. **Independent weights + role rotation.** A model that has been solving carries those weights into
   its next stint as creator — solver heuristics are "injected" into the creator role by rotation
   alone, no weight surgery. (Explicit LoRA blending exists but is off; rotation is the committed
   mechanism.)
2. **Difficulty-gradient reward.** The creator is scored on how closely the suite's *realized*
   solve-rate curve fits a target ramp. This is a **calibration** game, not a stumping game: to score
   well the creator must predict what the solver can and cannot do at several difficulty levels at
   once — the main anti-collapse pressure. (Sprint 7's persona prompts state this win condition to the
   model explicitly.)
3. **Knowledge oracle as an epistemic floor.** The frozen base (which never trains, so never forgets)
   is reachable as a taxed tool. *Design intent; the tax is counted but the tool is not yet wired into
   rollouts — see §14.*

Plus a **consistency check**: the creator's own stated solution/answer is verified mechanically; a
problem whose answer fails is *void* — excluded from the solver's reward (the solver is never punished
for an incoherent problem) but still drags the creator's consistency reward. Voids are strictly
unprofitable (the gradient reward is scaled by the scored fraction).

---

## 2. System reality (measured, not estimated)

Hardware: MacBook **M5, 32 GB**. Base: **Qwen3-8B 6-bit MLX** (~6.2 GB resident), thinking mode **ON**
for real runs. Stack: **mlx / mlx-lm** (the only stack that trains on Apple Silicon), GRPO implemented
from scratch (§7), no torch/TRL.

Measured constants that shape everything:

- **Decode ~16.4 tok/s** (thinking on/off identical — thinking just emits 3-5× more tokens).
  Generation, not training, dominates wall-clock; the cost driver is `G_c · N · K` solver generations
  per iteration plus creator/judge rollouts. mini-03b measured **~28 min/iter** at N=3, K=4, G_c=4
  with high consistency; mini-04 (N=5, K=8) worst-cases 160 solver generations/iter.
- **The GRPO backward is the memory constraint**, not the weights: one 4096-token completion
  materializes a ~2.5 GB [T,V] fp32 logits tensor. Probe-measured worst case: **55 GB @ 4096 budgets /
  132 GB @ 8192** for a full batch in one graph. Consequences (all landed): `train.grpo_microbatch: 1`
  (one trajectory per backward chunk, grads accumulated — identical math, verified ~1e-7), the
  reference-logprob pass evals per-trajectory, **token budgets stay at 4096** until a chunked
  `completion_logprobs` backward exists (§12). Per-problem creator generation (Sprint 7) also shortens
  creator trajectories.
- Token budgets **4096 for all roles**: 2048 truncated the thinking creator mid-`<think>` (mini-01
  shakeout); 8192 is memory-infeasible (above).

Operational gotchas: run under `caffeinate -i` (suspension inflates wall-clock); launch with
`conda run --no-capture-output -n twin-models python -u …` (plain `conda run` buffers stdout until
exit); keep the **transcript on** — it is the only reason mini-02's consistency mystery and mini-03b's
fake tool calls were diagnosable.

---

## 3. Model layer: one base, two adapters, one oracle

```
                      ┌────────────────────────────────┐
                      │   Frozen Qwen3-8B 6-bit (MLX)  │   loaded once, never trained
                      │   = also oracle / KL reference │
                      └───────────────┬────────────────┘
                                      │ shared QuantizedLinear weights
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                        ▼
       adapter θ_A (LoRA)     adapter θ_B (LoRA)       zeroed adapter (= base exactly)
       Model A ("alpha")      Model B ("omega")        oracle / KL ref / judge
```

- `linear_to_lora_layers` wraps the target modules once; the two trainable parameter trees are swapped
  in with `model.update(tree)` (`Adapters.activate`). The 6.2 GB base is never duplicated.
- LoRA: **rank 64, α 128, all 7 projections × last 16 blocks** (~78 M params/adapter). GOTCHA: mlx-lm's
  `scale` is the raw delta multiplier = PEFT's α/rank (here 2.0), not α.
- **Zeroed adapter ≡ base** (LoRA `B` init = 0), verified exact (maxdiff 0). It serves as oracle, KL
  reference, and consistency judge — no extra weights.
- `TwinBase` provides `render` (chat template; `enable_thinking`; `tools=` for native function
  calling), `generate` (captures the TRUE sampled token ids via `stream_generate`),
  `generate_react` (segmented generation with tool splicing + loss mask, §8/§9), and
  `completion_logprobs` (grad-enabled per-token log-probs, the GRPO primitive).
- Per-adapter AdamW (wd=0) so A/B moments don't bleed; both Python and MLX RNGs seeded from
  `train.seed` (reproducible sampling).

---

## 4. Roles, rotation, personas

Adapters belong to **models, not roles**: A always uses θ_A. The role manager assigns creator/solver
each iteration and flips every `swap_interval` (after `warmup_iterations`). Rotation IS the injection
mechanism (§1); `roles.injection_rate = 0` committed (explicit blend exists only as a future ablation).

**Personas (Sprint 7, `game.personas`).** A is **"alpha"**, B is **"omega"** — names glued to
adapters, never to roles, so each model keeps a stable identity across role flips. Both system prompts
open with a fair-competition frame; the creator's is deliberately worded as *calibration* ("you win by
predicting exactly what omega can and cannot solve"), because the gradient reward is a calibration game
and mini-03a proved that pure difficulty-aspiration language ("stump an expert") sends a thinking model
into deliberation spirals that truncate the output. The hardest rank still gets the full competitive
push ("make the hardest problem you can — one you are confident omega will almost never crack;
structurally hard, never bigger numbers"). The judge and the held-out benchmark never see personas;
with personas off the prompts are byte-identical to v1 (tested).

---

## 5. The game — one iteration end-to-end

Current run configuration (mini4.yaml): `creator_mode: per_problem`, `tools.protocol: native`,
personas on, domains `["math"]` (coding dead until its pipeline sprint, §14).

1. **Creator generation — N separate rollouts per candidate suite** (`per_problem` mode, Sprint 7).
   For each of `G_c` candidate suites, the creator writes problems one at a time. Rank i's prompt
   dictates the difficulty value (i/(N−1), overwriting whatever the model claims — ranks cannot
   scramble) and the **target solve rate straight from the reward's own ramp** ("omega should solve
   this about 30% of the time" — prompt and reward cannot disagree). With
   `game.condition_on_previous` the prompt also carries the *JSONs* of the problems already written
   (never their thinking) plus a strictly-harder / genuinely-distinct instruction. Each rollout gets
   the full 4096 budget for its own thinking and emits ONE problem JSON: `statement, difficulty,`
   **`solution` before `answer`** (autoregressive CoT: derive, then state), and for math a
   `verification` certificate (§6). Mid-rollout the creator calls the CAS `solve`/`calc` tools through
   **native Qwen3 function calling** (§9). The legacy whole-suite mode (`creator_mode: suite`) remains
   the code default for back-compat with mini-01..03 configs.
2. **Consistency check** on each problem: certificate-first (mechanical CAS check), judge fallback
   (frozen base + tools) for cert-less problems; optional **strict tool gate**
   (`game.require_tool_use`): a problem whose rollout made no successful tool call is voided outright.
3. **Solver rollouts**: K single-turn attempts per *consistent* problem (across all G_c suites — the
   combinatorial cost). Attempts are graded against the creator's answer by the same verifier.
4. **Realized solve rate** per problem `p_i = solved/K`; ordering by rank gives the suite's curve.
5. **Rewards → advantages → two sequential GRPO updates** (solver's adapter, then creator's).
6. **Log** (JSONL record + raw transcript), checkpoint every `checkpoint_every` iters.

**GRPO group structures.** Solver: the K attempts at one problem = one group (many groups/iter).
Creator: all creator trajectories this iteration = one group. In suite mode that is the G_c suites
(v1 semantics). In per-problem mode it is G_c·N trajectories under **broadcast credit**: every parsed
rank of suite g carries the suite-level reward R_g (identical math to v1 whenever every rank parses —
the trajectory mean equals the suite mean); a rank that failed to parse gets the parse-gate reward
individually, so garbage is penalized at the trajectory that produced it. Per-problem credit
*decomposition* is a queued Sprint-8 ablation (§12).

---

## 6. Rewards

Problems `i = 1..N` by rank; realized solve rate `p_i`; target `t_i` = linear ramp `target_hi → target_lo`
(run configs use the **interior band 0.9 → 0.1** so every rank keeps outcome variance at
creator-optimum; the code default stays 1→0, band changes are per-run ablations).

**Creator (per candidate suite):**

```
R_creator = w_grad · r_gradient · (n_scored/N) + w_cons · R_consistency
            − w_oracle · n_oracle_calls + w_valid · R_valid
r_gradient = exp(−β · mean_scored (p_i − t_i)²)
```

- The **scored-fraction scaling** makes voided/unparsed problems strictly unprofitable (Sprint 5 fix:
  before it, a mostly-void suite could out-earn an honest one by fitting a tiny re-stretched target).
- `R_consistency` = fraction of the suite's problems that passed verification (voids drag it).
- Parse gate: a rollout yielding no parseable problem gets a fixed −1.0 and feeds nothing to the solver.

**Solver (per attempt):**

```
R_solver = w_solve · solved + w_brevity · (1 − L/L_max) · solved − w_oracle_s · n_oracle_calls
```

The **brevity bonus** is gated on `solved` (truncating into wrongness earns nothing) and capped below
`w_solve` (the worst correct answer still beats the best wrong one). Validated in mini-03a: the
token-length spread breaks all-solved K-group ties — the solver took real updates on 4/4 eligible
iters vs mini-02's 4/30. mini-04 runs `w_brevity: 0.25`.

**Solver sampling is part of the measurement.** `p_i` is literally "does the solver, sampled at
`solver_temp` 0.8, solve this" — the quantity the creator is graded on. Decision (2026-07-03): outcome
variance comes from *harder problems* + brevity, **not** hotter sampling, which would silently
recalibrate the game and break cross-run comparability.

---

## 7. GRPO (minimal, single-inner-step, as implemented)

For a group of trajectories with rewards `R_1..R_G`:

```
A_g   = R_g − mean(R)                        (Dr.GRPO mean baseline, default `train.adv_mode: mean`)
      = (R_g − mean) / (std + eps)           (legacy "std" form, kept as ablation arm)
L_pg  = − mean over TRAINED tokens [ A_g · logπ_θ ]      (single step ⇒ ratio≈1, no clip needed)
kl    = k3 estimator vs the zeroed-adapter base;  loss = L_pg + β·kl
```

- **Mean baseline** because ÷std is degenerate at our group sizes (G_c 2-4, K 4-8): any non-tie maps to
  ±1 (a 0.015 solve-rate-noise gap trains as hard as a 2.6 one) and ties give 0. Mean-centering
  preserves magnitude.
- **Zero-advantage skip**: a fully tied batch returns `skipped_zero_adv` *before* the reference pass —
  the PG term vanishes exactly and the pure-KL gradient is ≈0; mini-01 paid minutes per saturated iter
  for that no-op.
- **Loss masking**: injected tool-observation tokens condition the forward pass but are index-selected
  out of the PG/KL sums and the normalizer (differentiable gather; avoids the `inf·0 → nan` of a
  mask-multiply).
- **Microbatching** (`grpo_microbatch`): chunked backward with grad accumulation (∇Σ = Σ∇), each chunk
  realized before the next — the fix for mini-01 run0's OOM (§2).
- Reference pass = same weights with the zeroed adapter; per-adapter AdamW; global-norm grad clip.

---

## 8. Verification & consistency

**Certificate-first (math).** Every math problem must carry `verification.check` + `symbol`: equations
in the symbol that hold exactly when the symbol equals the answer, and that **recompute the answer from
the problem's quantities** ("Tickets cost $4; how many for $20?" → `check: "4*x = 20"`).
`check_predicate` accepts what models actually write (`a = b`, `a == b`, `Eq`, bare vanish-at-answer
expressions, comma-separated systems with tuple answers) and **rejects self-certifying checks**
(`"x = <answer>"`) as trivial — otherwise a wrong-answer problem could bless itself, poison the
solver's grades, and farm fake hardness. `_normalize_named_value` (mini-03a audit) rewrites
named-assignment answers (`"x = 2, y = 1"`) into symbol order before parsing — that answer-format
mismatch had been voiding *correct* problems and was the largest bucket of mini-02's ~38% cert-fail
rate. All parsers that eval model text share one locked-down SymPy namespace (`__builtins__` emptied).

**Judge fallback** (cert-less problems): the frozen base runs a tool-augmented grading loop — it must
recompute the value with the CAS tool before ruling `VERDICT: CORRECT/INCORRECT` (eyeballing worked
solutions is exactly where an LLM slips). Untaxed, unscored, still on the legacy ReAct protocol (§9,
migration queued). The prompt tells the creator cert-less problems are discarded, pushing toward always
certifying; per-suite `n_cert` coverage is logged as the watchdog.

**Coding** (`verify_code`, sandboxed subprocess: no network, CPU/wall limits, temp cwd) exists and is
tested, but the **domain is dead in training** — 0/141 creator coding problems passed their own tests
in mini-02 (the creator contract never asks for `verification.tests`/`solution_code`). Coding is out of
`game.domains` until its pipeline sprint (§12).

---

## 9. Tools

- `solve(expr[, var[, max|min|real|first]])` — creator-only untaxed SymPy CAS: equations, systems,
  evaluation, calculus via `.doit()`. Parses with the locked namespace. Normalizes Python-style `==` to
  `=` (models write it; the bare parser silently returned `[]`). **Stateless**: each call is an
  isolated expression — no intermediates across calls (session state is a Sprint-8 item).
- `calc(expr)` — cheap untaxed evaluator (also the judge's).
- `run_python(code)` — sandboxed exec; powers the code verifier (not exposed to rollouts).
- `oracle(question)` — base-only query, counted and taxed in the rewards. **Not yet instantiated in the
  loop** — the tax currently taxes only what the policy *writes* as oracle-call text (§14).

**Protocol (Sprint 7).** Two implementations behind `tools.protocol`:

- `react` (legacy): `<tool>name(arg)</tool>` → `<obs>…</obs>` splice. **Empirically broken with
  thinking ON**: Qwen3 was never trained on the foreign tags — mini-03b's creator made *zero* real
  calls in 4 iterations while writing "The tool would return x=3, y=2" inside `<think>`, i.e. it
  hallucinated the observations and burned its thinking budget re-solving its own problems.
- `native` (current for the creator): tools are declared through the chat template
  (`render(tools=tool_schemas(...))` → `<tools>` JSON signatures in the system block) and the model
  emits `<tool_call>{"name": …, "arguments": …}</tool_call>` — the format Qwen3 was post-trained on.
  The same segmented-generation machinery is reused: stop on `</tool_call>`, execute, splice the chat
  template's own tool-response turn (`<|im_end|> … <tool_response>…</tool_response> …
  <|im_start|>assistant`) as **masked** tokens, resume. Malformed call JSON gets a corrective
  format-error observation (`ToolHarness` has a pluggable parser). The hardcoded glue is verified
  against `apply_chat_template` ground truth by a tokenizer-only test, so chat-template drift fails CI,
  not a run. The judge stays on `react` until the native path is validated by mini-04.

Adoption telemetry logged per iteration: `creator_tool_ok` (successful calls),
`creator_answer_in_obs` (stated answers that literally appeared in a real observation),
`creator_tool_gated` (ranks voided by the strict gate). If mini-04 shows high consistency with low
`answer_in_obs`, flip `game.require_tool_use: true` next run.

---

## 10. Instrumentation & held-out benchmark

**Per-iteration JSONL record**: rewards, per-rank solve rates, parse rate, `r_gradient`, KL(c/s),
adapter norm/drift (L2 vs init snapshot), oracle/CAS usage, the Sprint-7 tool watchdogs, per-suite
summaries (`n_cert`, `n_consistent`, `n_parse_failed`, `n_tool_gated`, …), and update metrics incl.
`skipped_zero_adv`. **Raw transcript** captures everything the models write. `twin.analysis.curves` /
`scripts/analyze_run.py` turn a run into solve-rate-vs-difficulty curves, linearity fits, time series,
and a terminal report. NOTE: `ramp_mse` is fit against the fixed 1→0 reference ramp for cross-run
comparability — with an interior band, read the curve values, not ramp_mse.

**Held-out benchmark** (`twin.bench`, `scripts/benchmark.py`): the in-training curves are
creator-relative and can look healthy while absolute skill stalls. The benchmark is absolute: fixed
item sets graded exactly as training grades (same verifiers), greedy for reproducibility. Two tiers:
hand-authored **core** (base = 100%, catastrophic-collapse floor) and **hard**
(MATH-500 / MBPP / MMLU-Pro / BBH; **base = 66%**: math 70 / coding 50 / knowledge 60 / reasoning 85).
Workflow: re-run at each checkpoint, watch accuracy **relative to base**. mini-01 step30:
A 70% (+4), B 71% (+5) — real transferable capability from 30 iterations.

---

## 11. Run history (details in EXPERIMENTS.md)

| Run | Config | Outcome |
|---|---|---|
| mini-01 (2026-06-30) | mini.yaml: G_c2 N3 K4, thinking ON | **SUCCESS**, 30/30, ~9h15m. Curve pearson −0.979; bench A+4/B+5 vs base. Found: coding dead, advantage degeneracy, KL creep late-run |
| mini-02 (2026-07-01) | mini2.yaml: +G_c4, mean baseline, band 0.9/0.1 | **DONE**, 30/30, ~10.7h. Creator signal fixed (pg==0: 7/30 vs 15/30) but solver starved (4 real updates; 88% K-groups saturated); coding measured 0/141 consistent; curve ceiling-compressed |
| mini-03a (2026-07-02) | mini3.yaml: math-only, w_brevity 0.15, "stump an expert" prompt | **ABORTED** iter 10: prompt-induced think-spiral truncation (parse 0.5→0.15). w_brevity validated (solver 4/4 eligible updates). Transcript decomposed mini-02 cert-fails → normalizer fix |
| mini-03b (2026-07-02) | mini3.yaml + decisive-design prompt + normalizer | **STOPPED** by user at iter 4: parse recovered (1.0) but solve% re-saturated AND `creator_tool_calls = 0` every iter — the creator simulated the CAS inside `<think>`. Trigger for Sprint 7 |
| mini-04 (queued) | mini4.yaml: native tools, per-problem creator, personas, N5 K8, w_brevity 0.25 | Hypotheses & watch-list in EXPERIMENTS.md. Shakeout (2 iters, transcript on) before the full 30 |

---

## 12. Sprint 8 — planned

Queued 2026-07-03 (from the Sprint-7 discussion), roughly in order:

1. **Batched solver generation.** K=8 attempts are identical prompts with no tools — batch them
   through mlx-lm batch generation (~2-3× wall-clock from the earlier probe; per-row sampling is
   independent, so outputs differ automatically). This is the lever that funds mini-04's 160
   generations/iter and any further K/N growth. Acceptance: identical reward semantics, measured
   speedup on a 2-iter shakeout.
2. **Per-problem creator credit decomposition** (ablation vs mini-04's broadcast). Trajectory i gets
   its own rank fit (|p_i − t_i|²) + own consistency + a shared suite-level term (distinctness/ordering
   don't decompose). Sharper credit for bad ranks; more moving parts — measured, not default.
3. **CAS session state.** Per-rollout namespace so calls can define intermediates
   (`a = solve(...)`, then use `a`), or multi-statement calls. Ergonomics for creative multi-step
   problems; pointless until mini-04 proves the model calls the tool at all.
4. **Judge → native protocol.** The judge is the same base model with the same legacy-ReAct failure
   mode; migrate once mini-04 validates native on the creator path.
5. **Strict tool gate decision.** Flip `game.require_tool_use: true` if mini-04's `answer_in_obs`
   shows the creator still faking tool grounding.

Next after Sprint 8 (standing queue): **coding pipeline sprint** (creator contract for
`verification.tests`/`solution_code`, solver code branch, transcript-on shakeout — brings the second
domain back); **wire the oracle** (instantiate OracleTool in rollouts so the tax taxes something);
**kl_beta/LR probe** (mini-01's late-run KL creep); **chunked `completion_logprobs` backward** (needed
for >4096 budgets and worst-case-4096 robustness); rank-adaptive K; thinking on/off ablation
(pilot.yaml); the Sprint-4 ablation backlog in EXPERIMENTS.md.

---

## 13. Failure modes & mitigations (updated with what actually happened)

| Failure | Status / mitigation |
|---|---|
| Proposer collapse (all-easy / all-impossible) | Gradient reward targets a *spread*; observed failure is the opposite — **ceiling compression** (hard problems not hard enough); attacked via difficulty targets in-prompt + personas + N=5 (mini-04) |
| Fake-answer reward hacking | Certificates + triviality rejection + scored-fraction scaling (voids strictly unprofitable) |
| **Tool simulation** (model hallucinates observations instead of calling) | OBSERVED (mini-03b, zero real calls). Native protocol + `answer_in_obs` telemetry + optional strict gate |
| **Think-spiral truncation** (thinking model deliberates past its budget) | OBSERVED (mini-03a). Prompt language must pair difficulty pressure with "design decisively"; output-budget warning in system prompt; per-problem generation gives each problem its own budget |
| **Answer-format voids** (correct answers rejected on format) | OBSERVED (mini-02, largest cert-fail bucket). `_normalize_named_value` + canonical solver answer format in prompts |
| Advantage degeneracy (tied groups, ±1 quantization) | Mean baseline + interior band + G_c=4 + K=8 + w_brevity + zero-adv skip |
| Oracle over-reliance | Tax + call cap (oracle itself not yet wired — §14) |
| Knowledge degradation under RL | KL-to-frozen-base term; held-out benchmark vs base bar at every checkpoint |
| Format collapse early in training | Parse gates + validity shaping; parse_ok is a first-class curve |
| Sandbox escape (model code) | Subprocess isolation, no network, rlimits/wall timeout; locked SymPy parse namespaces everywhere model text is evaluated |
| OOM at scale | grpo_microbatch + per-trajectory ref eval + 4096 budget ceiling (measured limits, §2) |
| Throughput death-by-tokens | Small G_c/N/K, zero-adv skip, brevity pressure; batched solver generation queued |

---

## 14. Known gaps (deliberately honest)

- **Oracle is not wired.** `OracleTool` is never instantiated in the loop; `n_oracle_calls` counts
  only oracle-call *text* the policy writes (always 0 so far). The "epistemic floor" contribution is
  design intent, not yet an experiment.
- **Coding domain is dead** and out of training until its pipeline sprint (§12).
- **Judge still on legacy ReAct** — same protocol the creator just abandoned for cause.
- **Solver is single-turn, tool-less** — inline tools/oracle for the solver remain future work.
- A structured-but-colluding certificate (`"x = 12 + 0"`) passes the triviality tripwire; cert
  coverage + consistency-vs-solve curves are the watchdogs.
- `analysis.curves` fits vs the 1→0 ramp regardless of the configured band (intentional, §10).
- Sprint-7 native protocol and per-problem mode are **fast-test-verified but not yet
  real-token-verified** — that is what the mini-04 shakeout is for.
- ~~Four audit-2026-07-03 bugs~~ **FIXED pre-mini-04** (tests/sprint8/): (1) draft JSON inside
  `<think>` no longer beats the final JSON (`_extract_json_object` searches post-think text first,
  raw-text fallback for truncations); (2) `verify_math` now normalizes named-assignment answers
  (`"x = 2, y = 1"`) on EITHER side, closing the cert-passes/solver-graded-wrong reward-hack
  channel; (3) per-problem parse-fails scale `r_gradient` by parsed/intended-N and grade each rank
  against its PROMPTED target (`expected_n`/`target_by_problem` engine params) — dropping hard
  ranks is now strictly unprofitable; (4) `extract_final_answer` ignores `\boxed{}` inside
  `<think>` when post-think text exists. Also `creator_tool_rounds` bumped to calls+1 in
  mini4.yaml (rounds==calls made the last allowed call unusable).

---

*End of spec. DESIGN.md holds the v1 spec + Sprints 1-7 build history; EXPERIMENTS.md holds run
entries, the mini-04 plan, and the Sprint-8 backlog.*
