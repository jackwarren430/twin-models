# Configuration reference

Every run is driven by a YAML file in `configs/` loaded via `Config.from_yaml`
(`src/twin/config.py`). The schema is a nested set of dataclasses, one per
top-level section. Two loader properties worth knowing:

- **Partial YAMLs are fine.** Any key you omit takes the default listed below,
  so a config only needs to state what it changes.
- **Unknown keys raise.** A typo like `creater_temp` fails at load time rather
  than silently using the default.

Sections: [`model`](#model) · [`lora`](#lora) · [`gen`](#gen) ·
[`game`](#game) · [`rewards`](#rewards) · [`oracle`](#oracle) ·
[`tools`](#tools) · [`roles`](#roles) · [`train`](#train) · [`paths`](#paths)

---

## `model`

The base model both twins share (LoRA adapters sit on top of it).

- **`path`** (default `~/.cache/lm-studio/models/mlx-community/Qwen3-8B-6bit`)
  Filesystem path to the MLX model. `~` is expanded.

- **`enable_thinking`** (default `false`)
  Qwen3 thinking mode (`<think>...</think>`) for all rollouts. Off by default
  for throughput; mini-05 runs with it on.

- **`creator_enable_thinking`** (default `null`)
  Per-role override of thinking for the **creator** rollout only. `null` means
  "follow `enable_thinking`"; `true`/`false` forces it. Exists because
  mini-04a showed a thinking creator simulating the CAS inside `<think>`
  instead of calling it — denying the creator thinking (while the solver keeps
  it) is the structural fallback and an ablation lever.

- **`max_kv_size`** (default `null`)
  Cap on the KV cache size passed to generation. `null` = unbounded.

## `lora`

LoRA adapter shape. Each twin (A and B) owns one adapter of this shape.

- **`num_layers`** (default `16`)
  Convert the **last N** transformer blocks to LoRA.

- **`rank`** (default `64`)
  LoRA rank *r*.

- **`alpha`** (default `128.0`)
  LoRA alpha. The effective mlx-lm scale is `alpha / rank` (PEFT convention),
  so the defaults give scale 2.0.

- **`dropout`** (default `0.0`)
  LoRA dropout.

- **`keys`** (default: all 7 projections)
  Which linear projections in each block get adapters. The default names all
  seven explicitly (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`).
  Set to `null` in YAML to fall back to mlx-lm auto-discovery (which selects
  the same 7 for Qwen3 — the explicit list just makes the target set visible).

- **`scale`** (default `null`)
  Raw mlx-lm scale override. Leave `null` to use `alpha / rank`; set a number
  only to bypass alpha and pin the scale directly.

## `gen`

Sampling settings, per call-site.

- **`creator_temp`** (default `0.9`) / **`solver_temp`** (default `0.8`) /
  **`oracle_temp`** (default `0.2`)
  Sampling temperatures for the three roles. Solver temp is locked at 0.8 per
  the Sprint-7 decision; the oracle runs cool because it's a reference answer,
  not an explorer.

- **`top_p`** (default `0.95`)
  Nucleus sampling, shared by all call-sites.

- **`creator_max_tokens`** (default `1024`) / **`solver_max_tokens`**
  (default `768`) / **`oracle_max_tokens`** (default `512`)
  Per-rollout completion budgets. `creator_max_tokens` doubles as the
  deliberation hard cap when the creator thinks (mini-05 uses 2048: a
  deliberation spiral costs a bounded 2048 tokens and lands as a parse-fail
  instead of burning 4096). `solver_max_tokens` is also the `L_max` in the
  brevity bonus formula (see `rewards.w_brevity`).

- **`solver_batch`** (default `1`)
  Sprint 8 batched solver generation: sample the K solver attempts at a
  problem in continuous-batching chunks of this size (they share one prompt —
  the ideal batching case). `1` = the sequential v1 path, unchanged. `>1`
  changes the RNG *stream* (not the sampling distribution) and trades KV-cache
  memory (~0.6 GB/sequence worst-case at a 4096 budget) for decode
  throughput. `8` (matching K=8) is the validated mini-04b/05 setting,
  ~30–35 min/iter.

## `game`

The self-play game structure.

- **`n_problems`** (default `5`)
  **N** — problems per suite. The suite is a difficulty ramp: rank 1 easiest,
  rank N hardest (target solve rates interpolate `rewards.target_hi` →
  `rewards.target_lo` across ranks).

- **`creator_group`** (default `4`)
  **G_c** — candidate suites sampled per creator prompt; this is the creator's
  GRPO group (advantages are computed across the G_c suites).

- **`solver_attempts`** (default `4`)
  **K** — solver attempts per problem; the solver's GRPO group. Also the
  denominator of the empirical solve rate the calibration reward is fit
  against.

- **`domains`** (default `["math", "coding"]`)
  Pool the per-iteration domain draw samples from (uniform choice per
  iteration). Recognized names (case-insensitive, per
  `verifiers/dispatch.py`):
  - math: `math`, `arithmetic`, `algebra`
  - coding: `coding`, `code`, `python` (dropped from training until the
    coding pipeline sprint — its verifier path was 0/141 consistent in
    mini-02)
  - logic: `logic`, `puzzle`, `puzzles`, `knights-and-knaves` (Knights &
    Knaves, built 2026-07-05, shakeout via `configs/logic-shakeout.yaml`)

- **`creator_mode`** (default `"suite"`)
  Creator generation architecture (Sprint 7):
  - `"suite"` — one rollout emits the whole N-problem suite (v1 behaviour).
  - `"per_problem"` — N separate rollouts, one problem each. Each problem
    gets its own thinking budget, trajectories are shorter (smaller GRPO
    backward), and the suite reward is broadcast to all N trajectories
    (unless `credit: per_problem`).

- **`condition_on_previous`** (default `true`)
  `per_problem` mode only: show problem *k*'s rollout the JSONs of problems
  1..k−1 (never their thinking) so the suite stays coherent — distinct
  problems, a real ramp. Prompt-injection only, so cheap to ablate.

- **`personas`** (default `false`)
  Inject named personas (alpha/omega, glued to adapters A/B) plus
  fair-competition framing into creator AND solver prompts. The judge and the
  held-out benchmark always stay neutral.

- **`require_tool_use`** (default `false`)
  Strict tool gate: void a problem (`consistency=False`) when its creator
  rollout contained no successful tool call — the anti-"guess what the tool
  would return" measure. `false` = log-only (per-problem tool usage and
  answer-appears-in-observation are always recorded in the suite summary
  either way). Validated in mini-04b: 100% adoption, 1/300 problems gated —
  it selects, it doesn't starve.

- **`credit`** (default `"broadcast"`)
  Creator credit assignment, `per_problem` creator_mode only (ignored in
  `suite` mode, which has one trajectory per suite):
  - `"broadcast"` — all N rank trajectories share the suite reward (one
    advantage per suite, as in v1).
  - `"per_problem"` — rank *i* earns its own calibration fit + consistency
    flag + oracle tax (suite validity stays shared). The TRR++ direction
    (RELATED_WORK.md §1): a rank that nailed its target isn't dragged down by
    a sibling.

- **`theme_weights`** (default `null`)
  Sprint 9 SPICE-style target grounding: path to a theme-weights JSON
  (`{domain: [[theme, weight], ...]}`), usually built by
  `scripts/build_grounded_themes.py` from a held-out bench result. Biases the
  per-iteration theme draw toward measured failure topics. `null` = uniform
  over the static `prompts.THEMES` pool (v1 behaviour). mini-05 uses
  `data/themes-grounded.json`.

## `rewards`

Weights and shape parameters for the reward engine
(`src/twin/rewards/engine.py`). Creator reward ≈ calibration gradient +
consistency + validity − oracle tax − diversity penalty; solver reward ≈
solve + brevity − oracle tax.

- **`w_gradient`** (default `1.0`)
  Weight of the creator's calibration reward `exp(-mse_beta · MSE)`, where
  MSE is between the suite's empirical per-rank solve rates and the target
  ramp. The main "make a real difficulty ramp" incentive. All-solved suites
  floor at ~0.267 with the default ramp — `r_gradient` lifting off that floor
  is the difficulty-ratchet success signal.

- **`w_consistency`** (default `0.5`)
  Weight of the fraction of the suite's problems that pass verification
  (verifier or judge agrees the stated answer is right). Voided problems
  (including tool-gate voids) count against this.

- **`w_oracle`** (default `0.1`)
  Creator oracle tax, per oracle call (multiplied by
  `oracle.cost_per_query`).

- **`w_valid`** (default `0.1`)
  Weight of suite validity (parses, N problems, required fields). Parse-fails
  land here as an own-rank penalty in per_problem mode.

- **`w_solve`** (default `1.0`)
  Solver reward for a correct answer.

- **`w_oracle_solver`** (default `0.05`)
  Solver oracle tax, per call.

- **`w_brevity`** (default `0.0`, off)
  Brevity bonus for **correct** solver attempts only:
  `w_brevity · (1 − L/L_max)` with `L` = completion tokens and `L_max` =
  `gen.solver_max_tokens`. Gated on solved so it can never reward truncating
  into a wrong answer, and capped at `w_brevity` so a correct answer always
  out-scores an incorrect one. Measurably fixed solver starvation in mini-03
  (4/5 solver updates vs mini-02's 4/30); mini-05 runs 0.25.

- **`mse_beta`** (default `4.0`)
  Sharpness of the `exp(-beta·MSE)` calibration reward.

- **`clip`** (default `10.0`)
  Absolute clamp on total per-trajectory reward.

- **`w_diversity`** (default `0.0`, off)
  Sprint 9 cross-suite repetition penalty (the R-Zero-style diversity
  mechanism, RELATED_WORK.md §2/§5). Per suite: mean nearest-neighbour
  word-bigram Jaccard of its problems vs the OTHER suites in the same
  iteration (`twin.analysis.diversity`; digit-normalized, so a
  coefficient-swapped template reads ~1.0 similar), subtracted from the suite
  reward scaled by this weight. `0` = off — telemetry is still always logged
  as `problem_similarity` / per-suite repetition. GRPO feels only the
  suite-vs-suite contrast. mini-05 is the first run with it on (0.3).

- **`target_hi`** / **`target_lo`** (defaults `1.0` / `0.0`)
  Target solve-rate ramp endpoints, easy rank → hard rank. The default
  1.0 → 0.0 ramp makes the endpoint ranks' solver K-groups zero-variance
  (all-solved / all-failed) at creator-optimum; an interior band like
  0.9 → 0.1 keeps outcome variance at every rank. Changing the band changes
  the game's incentives — treat as an ablation (DESIGN §11 S6). mini-05 runs
  0.9 → 0.1.

## `oracle`

The frozen base model as an answer oracle both roles may query (at a reward
tax).

- **`enabled`** (default `true`)
  Master switch for oracle availability.

- **`cost_per_query`** (default `1.0`)
  Cost counted per call; the reward engine multiplies it by
  `w_oracle` / `w_oracle_solver`.

- **`max_calls_per_turn`** (default `4`)
  Hard budget per rollout.

## `tools`

Inline tool use for the **creator** rollout (ReAct-style: results are spliced
back as observations and masked out of the GRPO loss) and for the **judge**
(verifier fallback on the frozen base — its tool use is untaxed and never
scored). The solver has no inline tools in v1.

Available tool names (registered in `tools/native.py`): `solve` (SymPy CAS),
`calc` (arithmetic evaluator), `run_python` (sandboxed subprocess), and
`logic_solve` (Knights & Knaves claim checker).

- **`creator_tools`** (default `["solve", "calc"]`)
  Tools exposed to the creator rollout.

- **`creator_max_tool_calls`** (default `4`)
  ToolHarness budget per creator rollout.

- **`creator_tool_rounds`** (default `4`)
  Max ReAct rounds (generate → tool → continue) per creator rollout.

- **`protocol`** (default `"react"`)
  Tool-call protocol for the creator rollout:
  - `"react"` — legacy text protocol (`<tool>name(arg)</tool>` /
    `<obs>...</obs>`). Qwen3 was never trained on it; with thinking ON it
    plans calls inside `<think>` and hallucinates the results (mini-03b: zero
    real calls).
  - `"native"` — Qwen3 function calling: tools declared via the chat
    template, model emits `<tool_call>{json}</tool_call>`, harness splices a
    `<tool_response>` user turn back (masked from the loss). The setting all
    runs since mini-04 use.
  The judge stays on `"react"` either way (native migration is a Sprint-8
  backlog item).

- **`cas_timeout_s`** (default `3.0`)
  Best-effort wall-clock guard for the `solve` CAS tool.

- **`run_python_timeout_s`** (default `5.0`)
  Wall-clock limit for the sandboxed `run_python` tool (the sandbox also
  enforces CPU/memory/file limits).

- **`judge_tools`** (default `["solve", "calc"]`) /
  **`judge_max_tool_calls`** (default `4`) /
  **`judge_tool_rounds`** (default `4`)
  Same three knobs for the judge, so it can recompute answers instead of
  eyeballing them.

## `roles`

Role rotation between the twins (`src/twin/roles/manager.py`). Adapters are
owned by models (A keeps θ_A, B keeps θ_B); the role manager only decides who
**creates** and who **solves** each iteration. Rotation IS the injection
mechanism: a model that has been solving carries its just-trained adapter into
the creating role — no weight surgery (locked decision, DESIGN §12 Q1).

- **`swap_interval`** (default `100`)
  Iterations between role flips. `<= 0` disables swapping entirely.

- **`warmup_iterations`** (default `0`)
  No swaps before this iteration; the interval counts from here.

- **`injection_rate`** (default `0.0`, off)
  Explicit cross-model LoRA blending at a swap: the incoming creator receives
  this fraction of the outgoing solver's adapter. Kept as an ablation only —
  the locked design is rotation-only (`0.0`).

## `train`

GRPO optimization loop.

- **`iters`** (default `1000`)
  Total training iterations.

- **`learning_rate`** (default `1.0e-5`)
  Adam learning rate.

- **`kl_beta`** (default `0.02`)
  KL penalty toward the frozen base policy.

- **`grad_clip`** (default `1.0`)
  Global-norm gradient clip per GRPO step. `0` = off.

- **`grad_accumulation_steps`** (default `1`)
  Optimizer-level gradient accumulation.

- **`grpo_microbatch`** (default `0`)
  Max trajectories scored per GRPO backward chunk; gradients are accumulated
  across chunks for one optimizer step (identical math, bounded peak memory).
  `0` = whole batch in one graph — fine for tiny configs / short budgets, but
  a single 4096-token completion materializes a ~2.5 GB `[T, V]` logits
  tensor, so a dozen at once blows past 32 GB. Long-budget runs set `1`.

- **`logit_chunk`** (default `0`)
  Sprint 8 chunked+checkpointed LM-head in `completion_logprobs`: the
  transformer runs once for the `[T, H]` hidden states, then the head +
  log-softmax run over the completion in `mx.checkpoint`-ed chunks of this
  many positions — the backward rematerializes one `[chunk, V]` at a time
  instead of pinning two fp32 `[T, V]` tensors (the mini-02 probe's 55 GB
  worst case at 4096 budgets). Gradients are numerically identical (the head
  carries no LoRA params). `0`/`null` = plain v1 path; `256` is a sane
  setting. Not yet probe_memory-certified, so no run has enabled it.

- **`adv_mode`** (default `"mean"`)
  Group-advantage normalization (`twin.rl.group_advantages`):
  - `"mean"` — Dr.GRPO-style `A = R − mean(R)`. For this project's small
    groups (G_c 2–4, K=4–8) this preserves reward magnitude: near-ties give
    near-zero gradients.
  - `"std"` — legacy standardized form (÷ std), kept for ablation. With small
    groups it maps ANY non-tie to ±1, so a 0.015 reward gap (solve-rate
    noise) trains as hard as a 2.6 one.

- **`seed`** (default `0`)
  RNG seed for the run (domain/theme draws, sampling streams).

- **`checkpoint_every`** (default `100`)
  Save adapters + optimizer state every N iterations.

- **`log_every`** (default `1`)
  Iteration-log frequency.

- **`log_prompts`** (default `false`)
  Also write the INPUT prompts to the raw-text transcript: system prompts
  once per iteration (they change at rotation when personas are on), each
  creator rollout's user prompt, and each problem's solver prompt (the K
  attempts share one prompt, logged once). Off by default — prompts are
  large and mostly templated, so this roughly doubles transcript size. No
  effect on the JSONL log; `scripts/train.py --log-prompts` sets it from the
  CLI.

## `compute`

Which compute backend runs the model, plus that backend's native performance
levers. The default reproduces the historical Apple-silicon behaviour exactly;
`backend: torch` selects the NVIDIA DGX Spark path. See **DGX_SPARK.md** for
setup and the acceptance smoke. All `torch`-prefixed fields below are read only
by the torch backend (the mlx backend ignores them, so one config can carry
both).

- **`backend`** (default `"mlx"`)
  `"mlx"` = Apple silicon / mlx-lm (default). `"torch"` = DGX Spark / CUDA
  (PyTorch + transformers + PEFT, base loaded bf16).

- **`device`** (default `"cuda"`) *(torch)*
  torch device: `"cuda"` | `"cpu"` | `"mps"`. Falls back to CPU if CUDA is
  unavailable.

- **`dtype`** (default `"bfloat16"`) *(torch)*
  Base-weight compute dtype: `"bfloat16"` (Blackwell-native) | `"float16"` |
  `"float32"`.

- **`attn_impl`** (default `"sdpa"`) *(torch)*
  Attention kernel: `"sdpa"` (torch fused, always available), `"flash_attention_2"`
  (needs the flash-attn wheel; fastest on Blackwell), or `"eager"` (reference).

- **`compile`** (default `false`) *(torch)*
  `torch.compile` the scoring forward. Experimental — may recompile on adapter
  switch; opt-in.

- **`tf32`** (default `true`) *(torch)*
  Allow TF32 matmul/cuDNN. Large speedup on Ampere+ for a tiny precision cost.

- **`matmul_precision`** (default `"high"`) *(torch)*
  `torch.set_float32_matmul_precision`: `"highest"` | `"high"` | `"medium"`.

- **`grad_checkpointing`** (default `false`) *(torch)*
  HF gradient checkpointing on the scoring backward. Unnecessary at 128GB;
  enable only if you OOM the backward. (Requires LoRA/base dropout 0 — the
  default — because it briefly flips the model to train() mode.)

Related, already-existing memory levers that apply to **both** backends:
`gen.solver_batch` (batch the K solver attempts), `train.grpo_microbatch`
(trajectories per backward chunk), `train.logit_chunk` (chunked LM-head
scoring), `model.max_kv_size` (mlx-only). On the Spark's 128GB the last three
relax toward off/large — see `configs/spark.yaml`.

## `paths`

- **`checkpoints`** (default `"checkpoints"`)
  Directory for saved adapter/optimizer checkpoints.

- **`runs`** (default `"runs"`)
  Directory for run logs and telemetry.

---

## Minimal example

Only override what you change; everything else defaults:

```yaml
# configs/my-run.yaml
game:
  domains: ["math"]
  solver_attempts: 8
rewards:
  w_brevity: 0.25
train:
  iters: 30
  grpo_microbatch: 1
```

For a fully-specified, battle-tested reference config with rationale in the
comments, see `configs/mini5.yaml`.
