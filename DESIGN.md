# Twin-Models — Self-Play RLVR with Two LoRA Adapters on One Frozen Base

**Status:** living design spec (supersedes `2026-06-26_183000-selfplay-rlvr-design.md`, which targeted a
14B GGUF/llama.cpp/PEFT stack). This revision is written for the actual hardware, model, and framework
we are using and folds in the decisions made in the kickoff conversation.

---

## 0. What changed from the original spec, and why

| Topic | Original spec | This spec | Reason |
|---|---|---|---|
| Base model | Qwen2.5-14B / Llama-3.1-13B (GGUF) | **Qwen3-8B (6-bit MLX)** already on disk | Fits comfortably in 32 GB with room for training; newer; user-selected |
| Framework | llama.cpp + HF Transformers + PEFT | **Apple MLX + `mlx-lm`** | The model is in MLX format; MLX is the only stack that does fast *training* on Apple Silicon. llama.cpp is inference-only |
| Models in memory | Base + separate Q8 oracle (~22 GB of weights) | **One** frozen base shared by A, B, and the oracle | User's explicit memory-saving design; ~6.2 GB total weights |
| A / B weights | Two full LoRA stacks | **Two LoRA adapter trees over one shared frozen base** | Inheritance/wrapping design; base loaded once |
| Oracle | Separate larger/Q8 model | **The frozen base with no adapter** | Decided: reuse base for v1 |
| RL algorithm | GRPO or PPO | **GRPO** (single-inner-step form) | Decided |
| Dual training | Two-stage fallback suggested | **Both adapters trained every iteration, sequentially** | Decided; memory allows it because only one adapter trains at a time |
| Role-swap injection | Explicit LoRA weight blending | **Implicit injection via role rotation** (primary); explicit blending optional/off by default | Cleaner, matches the "specialize per phase" narrative, avoids destructive LoRA averaging — see §4 |
| Replay buffer, priority sampling, search tool, W&B | Central to the design | **Deferred / optional**; v1 is on-policy with local JSONL logging | GRPO is on-policy; a priority replay buffer is a research add-on, not needed to get a loop running |

Everything below is the plan we actually build against.

---

## 1. Core idea (unchanged)

Two model instances **A** and **B** play a self-play curriculum game. Each iteration one is the
**creator** (proposes a *suite* of problems spanning easy→hard, plus a reference solution for each) and
the other is the **solver** (attempts them). They are rewarded on orthogonal signals and both are updated
every iteration. Periodically they swap roles.

Three claimed contributions over AZR / SSP-style single-weight self-play:

1. **Independent weights + role rotation.** Two adapters instead of one shared policy. A model that has
   been solving carries those weights into its next stint as creator, "injecting" solver heuristics into
   the creator role. (See §4 for why this is the clean version of "injection.")
2. **Difficulty-gradient reward.** The creator is scored on how close the suite's *realized* solve-rate
   curve is to a target linear ramp from ~100% (easiest) to ~0% (hardest). To score well it must model the
   solver's ability at several levels at once — this is the main anti-collapse pressure.
3. **Knowledge oracle as an epistemic floor.** A tool that routes factual queries to the frozen base
   (which never trains, so never forgets), taxed per call so the policies don't lean on it.

Plus a **consistency check**: the creator's own stated solution is verified; if it's wrong/incoherent the
creator is penalized. This is an orthogonal anti-degeneration pressure (you can't farm gradient reward by
shipping nonsense with fake answers).

---

## 2. Hardware & model reality check (M5, 32 GB)

```
Weights (resident once):
  Qwen3-8B 6-bit base .................. ~6.2 GB
  LoRA adapter A (rank 64, α=128, all 7 proj × 16 blocks ≈ 78 M params) ~0.15–0.3 GB
  LoRA adapter B ....................... same
  AdamW state for the adapter training . ~2× adapter
Transient:
  KV cache @ 4k ctx .................... ~0.5–0.7 GB
  Fwd/bwd activations (1 seq, LoRA) .... low hundreds of MB
-----------------------------------------------------------------
Peak well under 32 GB. Memory is NOT the constraint.
```

**The real constraint is throughput.** Every iteration generates a lot of tokens (creator suites +
many solver attempts). Budget on the order of **minutes per iteration**. Implications baked into the
design:

- Start with a *tiny* config (small group sizes, few problems, short `max_tokens`, thinking mode off) and
  prove the loop end-to-end before scaling.
- **Qwen3 thinking mode is OFF by default** for rollouts (`enable_thinking=False` / `/no_think`). Thinking
  traces multiply token counts; it comes back **on at scale** (Sprint 4) once the loop is stable — improving
  reasoning is the whole point, so we pay for the extra tokens only when the loop can afford them (§12 Q4).
- Everything is sequential (generate → score → update A → update B). That's fine; it keeps peak memory low.

---

## 3. Model layer: one base, two adapters, one oracle

```
                      ┌────────────────────────────────┐
                      │   Frozen Qwen3-8B 6-bit (MLX)  │   loaded once, never trained
                      │   = also the Knowledge Oracle  │
                      └───────────────┬────────────────┘
                                      │ same QuantizedLinear weights, shared
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                        ▼
       adapter θ_A (LoRA)     adapter θ_B (LoRA)       (no adapter = base)
       owned by Model A       owned by Model B          = oracle / KL reference
```

**Mechanism (`mlx-lm`):**

- `mlx_lm.load(path)` loads the quantized base + tokenizer.
- `mlx_lm.tuner.utils.linear_to_lora_layers(model, n_layers, lora_cfg)` wraps target `QuantizedLinear`
  modules with trainable LoRA (QLoRA over a 6-bit base is supported). We do this **once**.
- The trainable parameters form a small **parameter tree**. We keep **two** such trees in memory
  (`θ_A`, `θ_B`) and swap the active one in with `model.update(tree)` before any forward/generate.
- The base quantized weights are *shared* — switching adapters does not touch them, so we never pay for a
  second copy of the 6.2 GB base.
- **Oracle / KL reference = adapter with `lora_b = 0`.** Because LoRA output is
  `base(x) + scale · (x·A)·B` and `B` initializes to zeros, a zeroed adapter reproduces the base exactly.
  We keep a frozen zero-tree for clean reference/oracle passes.

This is the whole "inheritance that wraps the base with LoRA adapters" idea, realized as adapter-tree
swapping over a shared frozen base. (Module subclassing is unnecessary and would duplicate base weights.)

Concrete object model:

```python
class TwinBase:
    """Loads the shared frozen base once. Provides generate() and logprobs()
    against a supplied adapter tree, plus an oracle() = base-only generate."""
    model, tokenizer
    def set_adapter(self, tree): self.model.update(tree)
    def generate(self, messages, adapter, **kw) -> Rollout      # swaps adapter in, samples
    def token_logprobs(self, tokens, adapter) -> mx.array       # for GRPO (grad-enabled)
    def oracle(self, question) -> str                           # base-only, no adapter

class Adapters:
    """Owns θ_A, θ_B (and the zero reference tree). Handles save/load and swap."""
    theta_a, theta_b, zero_ref
```

---

## 4. Roles, rotation, and "injection" (a clarification of the original spec)

Adapters are owned by **models**, not by **roles**:

- Model A always uses θ_A; Model B always uses θ_B.
- Each iteration, the **role manager** assigns one of them to *create* and the other to *solve*.
- The creator's reward updates *its* adapter; the solver's reward updates *its* adapter.
- Every `swap_interval` iterations, the role assignment flips.

**Why this already gives "injection of solver heuristics into the creator role" for free:** when A finishes
a stint as solver and the roles flip, A keeps θ_A — the very weights that just learned to solve are now the
weights doing the creating. No weight surgery needed. Each model becomes a generalist that is *temporally*
specialized to its current role. This matches the user's framing ("specialize during its assigned phase,
while role swaps inject the solver's learned heuristics into the creator role") without the fragility of
averaging two independently-trained low-rank adapters.

**Optional experimental knob — explicit cross-model blending** (default **off**, `injection_rate = 0`):
at a swap, mix a fraction of the outgoing solver's adapter into the incoming creator's adapter
`θ_creator ← (1−η)·θ_creator + η·θ_solver`. Dimensionally valid (same base, same rank) but LoRA averaging
can be destructive; treat as an ablation, not the default. **Decided (§12 Q1): rotation-only is the
committed mechanism; this blend stays off and exists only for the Sprint-4 ablation.**

---

## 5. The game, one iteration end-to-end

Let the current assignment be **creator = C**, **solver = S** (each is A or B).

1. **Creator rollout (inline ReAct).** C is prompted with a theme/domain and the suite contract. It emits
   `G_c` candidate suites (a GRPO group). Each suite = `N` problems, each with `{statement,
   reference_solution, stated_answer, claimed_difficulty, domain}`. C may call the untaxed `solve` CAS
   mid-rollout to compute exact answers (it generates → `<tool>solve(…)</tool>` → harness runs it →
   `<obs>…</obs>` is spliced back → C continues); the injected obs tokens are masked out of the GRPO loss
   (§8, §9). Any oracle calls the policy *writes* are still counted/taxed.
2. **Consistency check (creator's own solutions).** For each problem, an **external verifier** checks the
   creator's `reference_solution`/`stated_answer` against ground truth it can compute itself
   (run the code, evaluate the math). Output per problem: `consistent ∈ {0,1}` (+ a continuous score).
   Verifier-first; LLM-judge fallback for non-executable problems (§7).
3. **Solver rollout.** For every problem in every candidate suite, S makes `K` attempts (a GRPO group per
   problem). Each attempt is graded by the same verifier against the *creator's* `stated_answer`
   **only if that answer passed the consistency check** (otherwise the problem is void and excluded from
   solver reward, so the solver isn't punished for an incoherent problem).
4. **Realized solve rate.** For each problem, `p_solved = (#solved attempts)/K`. Ordering problems by
   `claimed_difficulty` gives the suite's realized solve-rate curve.
5. **Rewards** (§6) → **advantages** → **two sequential GRPO updates**: update S's adapter on solver
   reward, then C's adapter on creator reward. (Order doesn't matter; they touch different trees.)
6. **Log** suite examples, curve, rewards, oracle usage, KL to JSONL. Checkpoint adapters periodically.
7. Every `swap_interval` iters, flip roles (and optionally blend, §4).

**Why the GRPO group structures differ (this was glossed in the original spec):**

- **Solver GRPO:** the group is the `K` attempts at *one* problem; advantage is normalized within that
  group. Many groups per iteration (one per problem).
- **Creator GRPO:** the group is the `G_c` candidate *suites* for one creator prompt; advantage is
  normalized across suites. The solver's realized solve rates are the measurement the creator reward is
  built from — so the solver must be rolled out on *all* candidate suites. This is the combinatorial cost
  driver (`G_c · N · K` solver generations/iter); keep all three small at first.

---

## 6. Rewards

Notation: suite has problems `i = 1..N` ordered by `claimed_difficulty`; realized solve rate `p_i ∈ [0,1]`;
target curve `t_i` = linear ramp from `target_hi` down to `target_lo` across the `N` problems
(config `rewards.target_hi/lo`, default `1.0 → 0.0`). The default endpoints make the extreme ranks'
solver K-groups zero-variance by design at creator-optimum (all-solved / all-failed); an **interior band**
like `0.9 → 0.1` keeps outcome variance at every rank. The band changes the game's incentives, so
non-default bands run as measured ablations (§11 Sprint 6; mini-02 is the first).

### 6.1 Creator reward (per candidate suite)

```
R_creator = w_grad · R_gradient + w_cons · R_consistency − w_oracle · n_oracle_calls + w_valid · R_valid
```

- **R_gradient** — fit of realized curve to target. Use negative mean-squared error mapped to (0,1]:
  `R_gradient = exp(−β · mean_i (p_i − t_i)^2)`. Maximized when the suite produces a clean easy→hard ramp.
  *Note:* this rewards **spread and ordering**, which is exactly the anti-collapse pressure — an all-easy
  or all-impossible suite scores poorly.
- **R_consistency** — fraction of the suite's problems whose creator-supplied solution passed the verifier
  (`mean_i consistent_i`). Kills the "ship unsolvable problems with fake answers" exploit.
- **n_oracle_calls** — count of oracle tool calls in the creator rollout (the oracle tax).
- **R_valid** — small shaping bonus for producing a well-formed suite (parses, `N` problems, monotone
  non-decreasing claimed difficulty, distinct problems). Prevents the policy from collapsing to malformed
  output early in training. Can be folded into a parse-gate instead (see §6.3).

### 6.2 Solver reward (per attempt)

```
R_solver = w_solve · solved + w_brevity · (1 − L/L_max) · solved − w_oracle_s · n_oracle_calls
           (solved ∈ {0,1} from the verifier; L = completion tokens, L_max = gen.solver_max_tokens)
```

Keep it blunt for v1: reward correctness, lightly tax oracle use. The original spec's "efficiency" bonus
— deferred until the base loop was validated (mini-01) — is now the **brevity term**: correct attempts
earn up to `w_brevity` extra, linearly more the shorter the completion. Two guards against the
reward-hacky-truncation worry that deferred it: it is **gated on `solved`** (a wrong short answer earns
nothing, so truncating into wrongness is strictly unprofitable), and it is **capped at `w_brevity`**
(keep `w_brevity < w_solve` so the worst correct answer still out-scores the best incorrect one).
`w_brevity` defaults to **0.0** (off) — like the target band, it changes the game's incentives (with
thinking ON it also pressures the length of the `<think>` block), so it runs as an explicit per-run
ablation, never a silent default.

### 6.3 Practical guards

- **Parse gate.** A creator rollout that doesn't yield a parseable suite gets a fixed low reward and is not
  fed to the solver. Same for unparseable solver attempts (reward 0). This keeps the loop alive while the
  policies learn the format.
- **Void problems.** Problems failing the consistency check are excluded from the *solver's* reward
  computation but still drag the *creator's* `R_consistency` down. Asymmetry is intentional.
- **Reward clipping / normalization.** GRPO normalizes within-group, so raw scales matter less, but clip
  outliers to keep advantages sane.

---

## 7. Verifier & consistency

Start with **deterministic, executable domains** so the verifier is ground truth, not opinion:

- **Math (closed-form numeric/symbolic answer).** Parse the stated answer; check with SymPy
  (exact/symbolic equality, with a numeric fallback within tolerance).
- **Python coding (function + tests).** The problem ships hidden tests (or the creator's
  `reference_solution` is executed against asserts). Run in a **subprocess sandbox** (separate process,
  no network, CPU+wall-time limit, restricted cwd). This is the single most security-sensitive component
  — it executes model-generated code — so it gets hard isolation from day one.

**Consistency check** = run the verifier on the *creator's own* solution/answer. Verifier-first.
For problems the verifier can't execute, fall back to an **LLM judge**. Judge options the user raised:
(a) the frozen **base/oracle** as judge, (b) a separate model, (c) the **solver (A)** as judge. v1 default:
**verifier-first, oracle-base as the judge fallback** (no extra weights, no role conflict-of-interest). The
solver-as-judge variant is interesting (it ties consistency to the same model being trained) but risks
collusion; keep as an ablation. **See §12 Q3.**

```python
@dataclass
class VerificationResult:
    correct: bool
    score: float          # 0..1
    detail: str
    method: str           # "sympy" | "exec" | "judge"
```

---

## 8. GRPO (minimal, single-inner-step)

We implement the simplest correct GRPO directly in MLX (no TRL/torch). For a group of responses to one
prompt with scalar rewards `R_1..R_G`:

```
advantage A_g = R_g − mean(R)                    (Dr.GRPO-style mean baseline, the default)
              = (R_g − mean(R)) / (std(R) + eps) (legacy standardized form, train.adv_mode="std")
per-token policy loss (one gradient step, so ratio≈1, no clipping needed):
    L_pg = − mean over tokens [ A_g · logπ_θ(token) ]
KL-to-reference penalty (k3 estimator, ref = zeroed-adapter base):
    kl  = mean over tokens [ exp(logπ_ref − logπ_θ) − (logπ_ref − logπ_θ) − 1 ]
loss = L_pg + β · kl
```

**Why the mean baseline (Sprint 6).** For the small groups this project runs (G_c 2-4, K=4) the classic
÷std is degenerate: any non-tied group maps to ±1, so a 0.015 reward gap (solve-rate noise) trains exactly
as hard as a 2.6 one, and ties give 0. Mean-centering preserves magnitude — near-tie groups yield
near-zero gradients, real gaps proportionally strong ones. The standardized form is kept as an ablation
arm (`train.adv_mode: std`). Relatedly, a **fully tied batch (every advantage zero) skips the GRPO update
entirely** — reference pass, policy pass and backward — since the PG term vanishes exactly and the pure-KL
gradient is ≈0 right after the reference pass; mini-01 paid minutes per saturated 4096-token iteration for
that no-op (logged as `skipped_zero_adv`).

- `logπ_θ` from a grad-enabled forward over `prompt+response` (mask out prompt tokens), via
  `mx.value_and_grad` w.r.t. the active adapter tree only.
- **Loss masking for inline tool use:** a trajectory may carry a per-token `loss_mask` marking injected
  tool-observation (`<obs>`) tokens with 0. Those positions still condition the forward pass but are
  index-selected *out* of both the PG and KL sums (and out of the token-count normalizer) before reducing
  — a clean differentiable gather that also dodges the `inf·0 → nan` a `k3·mask` multiply would hit.
- `logπ_ref` from a no-grad forward with the zero adapter (the frozen base) — same weights, no extra RAM.
- One optimizer step per group-batch ⇒ `π_new == π_old` at update time ⇒ the PPO ratio is 1 and the clip is
  inactive. This is a standard, stable simplification for the on-policy single-epoch regime and removes a
  lot of bookkeeping. We can add multi-epoch PPO clipping later if we want sample reuse.
- Optimizer: `mlx.optimizers.AdamW`, separate optimizer state per adapter. LR ~1e-5..5e-5, β (KL) ~0.01–0.1.

Group sizes are the cost knobs: `G_c` (creator suites), `K` (solver attempts/problem), `N` (problems/suite).

---

## 9. Tools

All tools are plain Python callables exposed to the policy via a **ReAct-style text protocol** for v1
(model emits `<tool>name(args)</tool>`, harness executes, returns `<obs>...</obs>`). Rationale: robust
across chat templates and trivial to parse/sandbox; we can switch to Qwen3 native function-calling later.

- `oracle(question) -> str` — base-only generation; **counted and taxed**.
- `python(code) -> str` — sandboxed subprocess execution (also powers the code verifier).
- `calc(expr) -> str` — SymPy evaluate (cheap, untaxed; reduces oracle temptation for arithmetic).
- `solve(expr_or_eq[, var[, sel]]) -> str` — **creator-only**, untaxed SymPy computer-algebra
  ("Wolfram-Alpha-like") tool: solve equations/systems, evaluate, or run calculus (`integrate`/`diff`/
  `factor`/… via `.doit()`). It lets the creator compute *exact* answers while building a suite, so it
  one-shots correct `(problem, answer, solution)` tuples instead of hallucinating answers that become
  void problems. Distinct from the verifier/judge, which only *checks* a proposed answer. Both `solve`
  and `calc` parse with a **locked-down namespace** (`__builtins__` emptied) so `parse_expr` cannot be
  used for code execution (e.g. `__import__`). *(The older `calc`/`math_verifier` parsers still use the
  raw namespace — flagged for a follow-up hardening.)*
- (`search` from the original spec is **dropped** for v1 — no network in the loop, and it muddies the
  "self-reliance vs oracle" story.)

Each tool call has a per-turn cap (`max_tool_calls`) to bound rollout length. **Inline execution is live
for the creator**: it generates ReAct-style, the harness runs `solve`/`calc` mid-rollout, and the injected
`<obs>...</obs>` tokens are spliced into the completion (so the forward pass conditions on them) but
carry `loss_mask=0` so GRPO ignores them (§8). The solver path stays single-turn in v1.

---

## 10. Repository layout (v1)

```
twin-models/
├── DESIGN.md                      # this file
├── 2026-06-26_183000-...-design.md  # original (kept for history)
├── README.md
├── pyproject.toml                 # package `twin`, editable install
├── configs/
│   ├── tiny.yaml                  # smoke/dev: G_c=2,K=1,N=3, short max_tokens
│   └── base.yaml                  # default research config
├── src/twin/
│   ├── config.py                  # dataclasses + YAML loader        [Sprint 1]
│   ├── models/
│   │   ├── base.py                # TwinBase: load, generate, logprobs, oracle  [Sprint 1]
│   │   └── adapters.py            # Adapters: θ_A/θ_B/zero, swap, save/load     [Sprint 1]
│   ├── problems/
│   │   └── schema.py              # Problem, ProblemSuite, parse/validate       [Sprint 1]
│   ├── tools/                     # oracle, python sandbox, calc, cas(solve)    [Sprint 2/3.5]
│   ├── verifiers/                 # sympy math, exec code, judge fallback       [Sprint 2]
│   ├── rewards/engine.py          # creator/solver rewards                      [Sprint 2]
│   ├── roles/manager.py           # role assignment, swap, optional blend       [Sprint 3]
│   ├── rl/grpo.py                 # advantages, logprobs, loss, step            [Sprint 3]
│   ├── train/loop.py              # SelfPlayTrainer                             [Sprint 3]
│   ├── prompts/                   # creator/solver/judge templates             [Sprint 3]
│   ├── log/jsonl.py               # local run logging                          [Sprint 3]
│   └── analysis/curves.py         # run curves: solve-rate/linearity/KL/drift  [Sprint 4]
├── scripts/
│   ├── smoke_test.py              # load base + attach LoRA + swap + generate   [Sprint 1]
│   └── analyze_run.py             # JSONL -> curves (CSV/JSON/report/plots)     [Sprint 4]
├── EXPERIMENTS.md                 # run log: launch/analyze + entry template + backlog
├── tests/                         # pytest                                      [each sprint]
├── checkpoints/                   # adapter trees (git-ignored)
└── runs/                          # JSONL logs, suite samples (git-ignored)
```

---

## 11. Build plan (re-scoped sprints)

**Sprint 1 — Foundation. ✅ DONE & VERIFIED.** Env = conda env `twin-models` (mlx 0.31.2, mlx-lm 0.31.3,
sympy, numpy, pyyaml; editable install). `config.py`, `TwinBase` (load/generate/oracle/logprobs),
`Adapters` (θ_A/θ_B/zero, swap, blend, save/load), `schema.py` (`Problem`/`ProblemSuite` + parse +
validate). 25 fast tests pass; `scripts/smoke_test.py` confirms on the real model: base loads & generates;
two adapters of **2.42 M params each** (rank 8 × last 4 blocks) sit over one shared base; zeroed adapter
reproduces base **exactly** (maxdiff 0); perturbing A moves A's logits while B stays == base (independence);
`using('base')` restores; `save(A)`→`load(B)` round-trips exactly.

**Sprint 2 — Scoring. ✅ DONE & VERIFIED.** `tools/` (`calc` sympy eval; `run_python` subprocess sandbox
with rlimits/timeout/isolated-session/temp-cwd/minimal-env; `OracleTool` counted+taxed; `ToolHarness`
ReAct `<tool>…</tool>`/`<obs>…</obs>` protocol with per-turn budget). `verifiers/` (`verify_math`
sympy symbolic+numeric; `check_predicate` certificate; `verify_code` sandboxed exec; `judge_*` oracle
fallback; `verify_answer`/`check_consistency` dispatch by type→domain). `rewards/engine.py`
(`RewardEngine`: creator gradient/consistency/oracle-tax/valid + parse-gate, solver solve/oracle-tax,
clip). 47 new fast tests (72 total) pass with synthetic suites; address-space rlimit is Linux-only (macOS
bounds memory via CPU+wall instead).

**Sprint 3 — RL loop. ✅ DONE & VERIFIED.** `TwinBase.completion_logprobs` (grad-enabled per-token
log-probs; `generate` now captures the *true* sampled ids via `stream_generate`, fixing the Sprint-1
re-encode approximation), `rl/grpo.py` (group advantages, k3 KL to the zeroed-adapter base, single-step
`grpo_update` with global-norm clip; per-adapter AdamW so A/B moments don't bleed), `roles/manager.py`
(rotation + warmup, off-by-default cross-model blend), `prompts/` (creator/solver templates + themes),
`train/extract.py` (final-answer + oracle-call count), `log/jsonl.py`, and `train/loop.py`
(`SelfPlayTrainer`: creator group → consistency → solver groups → realized solve-rate curve → rewards →
group-relative advantages → two sequential GRPO updates, each with a base-KL reference pass). Void
problems are dropped from the creator's gradient curve (`RewardEngine.creator_reward(scored_mask=…)`) so
they can't be farmed as "hard", but still drag `R_consistency`. 27 new fast tests (99 total) pass,
including a real `grpo_update` step on a toy LM that confirms the `value_and_grad`→`optimizer.update` path
moves log-probs in the advantage direction. `scripts/smoke_test_sprint_3.py` runs the loop end-to-end on
the real base for two iters: rewards finite and moving (parse-gate −1.0 → +0.74 once a suite parsed with a
clean ramp; solver +1.0), KL to base bounded (~2e-4), and the zeroed ('base') adapter still reproduces
base log-probs exactly (maxdiff 0) after training A/B. The solver rollout is single-turn; the **creator
rollout is inline ReAct** (see below).

**Sprint 3.5 — Creator CAS tool (inline ReAct). ✅ DONE & VERIFIED.** `tools/cas.py` (`solve`, the
creator-only untaxed SymPy CAS, with a locked-down parse namespace that closes the `parse_expr` code-exec
hole), `TwinBase.generate_react` (segmented generation that pauses on `</tool>`, splices the harness's
`<obs>…</obs>`, and resumes — with a `loss_mask` over the spliced completion), `Trajectory.loss_mask` +
mask-aware `grpo_update` (index-selects trained tokens out of PG/KL and the normalizer; nan-safe),
`ToolsConfig`, and the creator prompt's ReAct protocol. The trainer builds a fresh `ToolHarness` per
creator rollout. 28 new fast tests (127 total) — including a toy-LM test proving injected `<obs>` positions
contribute zero to PG/KL — plus `scripts/smoke_test_creator_cas.py` (real base): a tool call fires, the
result splices in, mask 0/1 partitions correctly, and the frozen base is unchanged by the masked update.
This is the masking machinery the Sprint-4 inline-oracle work will reuse.

**Sprint 4 — Scale & study (base loop). ✅ LANDED; ablations deferred.** The scale config (`base.yaml`)
now runs the **bigger loop with thinking ON** (16 LoRA layers; N=5, G_c=4, K=4; `enable_thinking: true`
with token budgets sized for the `<think>` block — §12 Q4). The trainer logs per-iteration **curve
instrumentation**: oracle usage (`creator_oracle_calls`/`solver_oracle_calls`), CAS tool usage
(`creator_tool_calls`), curve-fit quality (`r_gradient_mean`), and **adapter magnitude/drift**
(`adapter_norm`/`adapter_drift` = L2 of each LoRA tree and its distance from the init snapshot, via
`Adapters.global_norm`/`snapshot`/`drift_from`). The new **`twin.analysis.curves`** module turns a run's
JSONL into the study curves — `solve_rate_curve` + `linearity` (slope / Pearson r / R² / MSE-to-ramp),
`time_series`, `summarize_run`, ASCII `sparkline`/`report_text`, and optional matplotlib `render_plots`
(no-op without matplotlib). `scripts/analyze_run.py` drives it (CSV + JSON + terminal report). The output
parsers were confirmed thinking-safe (`extract_final_answer` takes the last `ANSWER:`; `parse_suite` scans
for the JSON object — a `<think>` prefix is skipped). `tests/sprint4/` adds fast coverage (analysis curves,
adapter-drift tree maths, scale-config, thinking-aware parsing) plus a **model-gated short-e2e suite**
(`test_e2e_model.py`, `--only-model`): a 2-iteration real-base loop that checks the new metrics, feeds the
produced log through the analysis module, and verifies the frozen base is untouched and thinking mode runs
with bounded KL. Runs are recorded in **`EXPERIMENTS.md`** (lab notebook + template + ablation backlog).

**Held-out benchmark (absolute capability).** The curves above are *creator-relative* — solve-rate is
measured against the creator's own moving distribution, so they can look healthy while absolute skill
stalls or both policies collude. The **`twin.bench`** package + **`scripts/benchmark.py`** add the absolute
counterpart: a fixed, hand-verified item set spanning **math** (SymPy `verify_math`), **coding** (sandboxed
`verify_code`), **knowledge** (multiple-choice + normalized short-answer) and **reasoning**, scored against
the frozen base and/or the trained A/B adapters. Prompting and grading are keyed on each item's
`verification.type` and reuse the trainer's own solver path + verifiers, so the benchmark grades exactly as
training does. Generation is greedy (`temp=0`) for reproducibility; the CLI prints a base-vs-A-vs-B accuracy
table (with per-category deltas vs base) and writes a JSON report. The core is a pure `solve_fn`/grader
seam, unit-tested with a fake solver (no weights). **Run the base bar once, re-run at each checkpoint:**
rising accuracy *relative to base* is the proof that self-play produced real capability rather than
curve-fitting. Two tiers: a hand-authored **core** set (`data/bench/*.json`) the 8B base already aces
(100% — a cheap regression floor), and a **hard** tier (`data/bench/hard/`) that is the real progress
signal. `scripts/build_hard_bench.py` adapts open benchmarks into the hard tier — **MATH-500** (numeric,
level≥3), **MBPP** (assert tests), **MMLU-Pro** (10-way MCQ), **BIG-Bench-Hard** (logical deduction /
dates / boolean / counting) — pulled dependency-free via the HF datasets-server JSON API and converted to
the same schema/graders (used to evaluate our model, not train on; GPQA excluded for its gating/canary).
The frozen base lands at **66%** on the hard tier (math 70 / coding 50 / knowledge 60 / reasoning 85),
leaving headroom in both directions.

**Sprint 4 — remaining (deferred).** Ablations: oracle on/off, consistency on/off, rotation vs explicit
blend, solver-as-judge vs oracle-judge, thinking on/off. Scaling levers: gradient accumulation / batched
scoring in `grpo_update` (`grad_accumulation_steps` field exists, unused). Remaining inline-ReAct work:
extend execution to the **solver/oracle** path (the creator path and its `<obs>` masking already exist).
These are queued in `EXPERIMENTS.md`; the inaugural `base.yaml` baseline run goes first.

**Sprint 5 — mini-01 audit fixes. ✅ DONE & VERIFIED.** An audit of the inaugural `mini-01` run
(EXPERIMENTS.md) found three exploitable/broken mechanisms; this sprint closes them.

1. **Gradient reward scaled by scored fraction** (`rewards/engine.py`). The curve fit is measured over
   the *consistent-only* subset against a re-stretched target ramp, so a suite with ONE consistent easy
   problem fit its `[1.0]` target perfectly (`r_gradient=1.0`, total 1.27) and a `[1.0, 0.0]`
   easy+impossible pair scored 1.43 — nearly out-earning the honest 3-problem ideal (1.6). mini-01's
   own reward table showed mostly-void suites beating fully-consistent ones. `r_gradient` is now
   multiplied by `n_scored/n`, making voided problems strictly unprofitable (§6.3 amendment).
2. **Math verification certificates** (`verification.check` + `verification.symbol`) replace the LLM
   judge as the *primary* consistency check for math. The creator contract (`prompts.creator_user`, math
   domains only) now requires each problem to carry a certificate that **recomputes the answer from the
   problem's quantities** — e.g. statement "Tickets cost $4; how many for $20?" → `check: "4*x = 20"`.
   `check_predicate` (math_verifier) was hardened to accept what models actually write: `"a = b"`,
   Python-style `"a == b"`, `"Eq(a, b)"`, bare vanish-at-answer expressions, comma-separated relation
   systems with multi-symbol tuples (`symbol: "x, y"`, answer `"(6, 4)"`), and tolerance on `Eq`
   (compared via `lhs−rhs` so floats don't hard-fail). Self-certifying checks (`"x = <answer>"`,
   a literal number/fraction against a bare symbol) are **rejected as trivial** — otherwise a
   wrong-answer problem could bless itself, poison the solver's grades (sympy grades against
   `problem.answer`), and farm fake "hardness". `judge_consistency` remains the fallback for
   cert-less problems (a starvation safety net while the creator learns the format — the prompt claims
   cert-less problems are discarded, pushing the policy toward always certifying). Known limitations:
   a structured-but-colluding cert (`"x = 12 + 0"`) still passes the triviality tripwire, and a creator
   could *omit* certs to reach the softer judge fallback; the per-suite `n_cert` coverage the trainer
   now logs plus the consistency-vs-solve curves are the watchdogs — if cert coverage stalls or drops,
   harden the fallback to void.
   All parses of model text (`check_predicate`, `verify_math`, `calc`) now share `cas.py`'s locked
   `SAFE_GLOBAL_DICT` namespace — sympy's parser `eval`s transformed source, so builtins must be
   unreachable everywhere, not just in `solve`.
3. **Reproducible sampling.** `train.seed` seeded only the Python RNG (domain/theme schedule); MLX's
   sampler RNG was never seeded, so no two "identical" runs generated the same tokens.
   `SelfPlayTrainer.__init__` now seeds `mx.random` too. (A `--resume-step` run reseeds from the start
   — its stream matches a fresh run's schedule, not the interrupted run's mid-stream state.)

**Advantage degeneracy — diagnosis (fixes landed in Sprint 6 below).** mini-01's silent iterations
(creator `pg==0` in 15/30, solver `pg==0` in 9/15 solver-active iters, ~60% of solver rollout compute
gradient-free) trace to three stacked causes, in order of weight:

* **Minimal group sizes × coarse rewards.** With G_c=2 a std-normalized advantage is ±1 or 0 — 0
  whenever the two suites tie. 12/15 creator ties were the (structural, constant-0.1) coding iters,
  but 3 were math ties from reward quantization: K=4 quantizes solve rates to quarters and
  r_consistency to thirds, so distinct suites often map to the same scalar. When suites *don't* tie,
  ±1 discards magnitude: a 0.7889-vs-0.7738 gap (solve-rate noise) trained as hard as 1.6-vs-−1.0.
* **Target-curve endpoints.** The ramp asks for p=1.0 at rank 0 and p=0.0 at rank N−1; realized
  groups at those rates are all-identical (zero variance), so at creator-optimum 2/3 of solver
  K-groups are gradient-free **by design**. Bigger K only helps marginally (P(all solve | p=0.9) is
  0.66 at K=4, 0.43 at K=8); the band placement is the real lever.
* **Sampling is a minor factor.** Parse failures were budget truncation mid-`<think>` (fixed at
  4096); creator temp 0.9 beat 0.6 empirically; solver temp 0.8 is fine. Manufacturing outcome
  variance via hotter solver sampling would corrupt the solve-rate measurement the creator reward
  depends on — rejected.

**Sprint 6 — advantage-degeneracy fixes. ✅ CODE DONE — the `mini-02` run (configs/mini2.yaml) is the
measured ablation vs mini-01.** The four queued fixes, as landed:

1. **G_c 2→4** (`configs/mini2.yaml` only, no code). Creator rollouts are the cheap generation term
   (+2 ReAct generations ≈ +5–9 min/iter); 4-point groups sharply cut the all-tie probability and give
   the creator advantage real gradation. *Cost caveat:* the solver term also scales with consistent
   problems across all 4 suites (`G_c·N·K`), so consistency-heavy math iters can approach 2× mini-01
   late-run times — budget the run's wall-clock accordingly.
2. **Mean-baseline advantages** (`rl/grpo.py::group_advantages`, config `train.adv_mode`, default
   `"mean"`). A = R − mean(R), Dr.GRPO-style, no ÷std — near-tie suites get near-zero gradients instead
   of ±1 and real gaps proportionally strong ones (§8 amendment). `"std"` kept as the ablation arm.
   This is a fix (new default everywhere), not an opt-in.
3. **Interior target curve** (`schema.py::target_curve(n, hi, lo)`, config `rewards.target_hi/lo`,
   default `1.0/0.0`). mini2.yaml sets `0.9/0.1` so every rank keeps outcome variance and solver groups
   stay informative at creator-optimum (§6.1 amendment). Because it changes the game's incentives the
   default stays `1→0` — the band is an explicit per-run ablation knob, never a silent swap.
4. **Zero-advantage skip** (`loop.py::_grpo`). A fully tied batch returns a `skipped_zero_adv` no-op
   *before* the reference pass — mini-01 paid reference+policy+backward minutes per saturated math iter
   for a ≈0 pure-KL gradient. Longer-term idea recorded: rank-adaptive K (fewer attempts on ranks
   expected to saturate).

Tests: `tests/sprint6/` (+ updated `tests/sprint3/test_grpo.py`), 254 fast tests green. Analysis note:
`twin.analysis.curves` still plots/fits against the default `1→0` ramp — pearson/slope are unaffected by
a linear re-band, and keeping one fixed reference ramp makes `ramp_mse` comparable across runs.

Deferred from the same audit (tracked in EXPERIMENTS.md): the **coding domain is structurally dead**
(the creator contract never asks for `verification.tests`/`solution_code`, so `check_consistency` fails
every coding problem with "no tests supplied", and the solver path has no code branch) — next sprint;
and **wiring the oracle** (OracleTool is never instantiated in the loop; the tax currently taxes
nothing). Tests: `tests/sprint5/` + updated `tests/sprint3/test_rewards_scored_mask.py` (236 fast
tests green).

---

## 12. Open decisions for the user — RESOLVED

All four forks are now decided (user-confirmed 2026-06-28). Recorded here so the build has a single
source of truth.

**Q1 — "Injection" mechanism. → ROTATION-ONLY.** Implicit injection via role rotation (each model keeps
its adapter across role flips; §4). `roles.injection_rate = 0` is the committed default. Explicit
cross-model LoRA blending (`Adapters.blend_into`) stays implemented but **off**, available purely as a
Sprint-4 ablation. Rotation is the headline mechanism.

**Q2 — Initial problem domain. → VERIFIABLE-FIRST, broaden later.** Bring the loop up on the
deterministically-verifiable domains (**math + Python-coding**) so the reward is ground truth, not
LLM-judge opinion, while we debug end-to-end. Logic/reasoning (which leans on the judge path) is added
once the loop is stable. The judge path is still **built now** in Sprint 2 so broadening is a config
change, not new code.

**Q3 — Consistency-judge fallback. → ORACLE/BASE AS JUDGE.** For non-executable problems the frozen
base (zeroed adapter) is the judge — no extra weights, no role conflict-of-interest. Solver-as-judge
(collusion risk) is kept only as an ablation.

**Q4 — Thinking mode. → OFF for v1, ON at scale.** `model.enable_thinking = false` for all rollouts
while we get the loop running end-to-end (throughput). Re-enable thinking when we move to the larger
configs in Sprint 4 — improving reasoning is the point, so it comes back on once the loop can afford the
extra tokens.

---

## 13. Failure modes & mitigations (carried forward, trimmed)

| Failure | Mitigation in this design |
|---|---|
| Proposer collapse (all-easy / all-impossible) | Difficulty-gradient reward rewards a *spread* easy→hard curve, not raw hardness |
| Fake-answer reward hacking | Consistency check on the creator's own solution via deterministic verifier |
| Oracle over-reliance | Per-call tax in both rewards; per-turn call cap |
| Knowledge degradation under RL | Oracle = frozen base (never trains); KL-to-base term in GRPO |
| Format collapse early in training | Parse gates with fixed low reward; small validity shaping bonus |
| Destructive adapter averaging | Injection-by-rotation is the default; explicit blending is opt-in and rate-limited |
| Sandbox escape (model-written code) | Subprocess isolation, no network, CPU/wall limits, restricted cwd, from day one |
| Throughput death-by-tokens | Tiny default config; thinking off; small G_c/K/N; everything sequential |

---

*End of spec. Sprint 1 implementation accompanies this document.*
