# Experiments log

A running log of self-play training runs and what we learned. One entry per run
(or per small batch of related runs). Keep it append-only; don't rewrite history,
add a follow-up entry instead. The point is that six months from now we can see
*why* a config was chosen, not just what it was.

See **[DESIGN.md](DESIGN.md)** for the system; this file is the lab notebook.

## How to run

```bash
# launch a run (logs one JSON record per iter to runs/<run-name>.jsonl)
conda run -n twin-models python scripts/train.py --config configs/base.yaml \
    --iters 1000 --run-name 2026-06-29-scale-01

# tiny/dev loop (fast, thinking off)
conda run -n twin-models python scripts/train.py --config configs/tiny.yaml --iters 20
```

## How to analyze

```bash
# terminal report (sparklines + solve-rate-vs-difficulty curve + linearity),
# plus <run>_series.csv and <run>_summary.json next to the log
conda run -n twin-models python scripts/analyze_run.py runs/<run-name>.jsonl

conda run -n twin-models python scripts/analyze_run.py --latest          # newest run
conda run -n twin-models python scripts/analyze_run.py runs/<run>.jsonl --plots  # PNGs (needs matplotlib)
```

The curves to watch (all produced by `twin.analysis.curves`):

- **solve-rate vs difficulty** — should be a clean descending line (`linearity`:
  `slope` < 0, `pearson_r` → −1, low `ramp_mse`). A flat/noisy curve means the
  creator isn't producing a real difficulty gradient.
- **rewards over time** — `creator_reward` and `solver_reward` trending up;
  `parse_ok_rate` → 1 early (format learned).
- **oracle / tool usage** — `creator_oracle_calls` / `solver_oracle_calls` should
  stay low (the tax is working); `creator_tool_calls` (CAS) is untaxed and may
  rise as the creator learns to one-shot correct answers.
- **KL to base** — `creator_kl` / `solver_kl` bounded, not exploding (the policy
  isn't running away from the frozen base).
- **adapter drift** — `adapter_drift_A` / `_B` rising smoothly; a sudden jump
  often precedes a KL spike or a reward collapse.

## Held-out benchmark (absolute capability)

The analysis curves above are *creator-relative*: solve-rate is measured against
the creator's own moving problem distribution, so they can look healthy while
absolute skill stalls or both policies drift into a private regime. The held-out
benchmark is the absolute counterpart — a fixed, hand-verified set spanning math
(SymPy), coding (sandboxed tests), knowledge (MCQ + short answer) and reasoning.

```bash
# frozen base only — the reference bar every checkpoint is compared against
conda run -n twin-models python scripts/benchmark.py --config configs/base.yaml

# base vs a trained checkpoint (greedy, reproducible); prints a delta table +
# writes runs/bench-<ts>.json
conda run -n twin-models python scripts/benchmark.py --config configs/base.yaml \
    --adapters base,A --checkpoint-a checkpoints/adapter_A_step1000.safetensors
```

Record the base bar once, then re-run at each checkpoint and watch the overall
and per-category accuracy move **relative to base** — that's the signal that
self-play produced real capability, not just curve-fitting against the creator.

**Two tiers.** The hand-authored *core* set (`data/bench/*.json`) is a cheap
regression floor — the 8B base already scores 100%, so it only catches
catastrophic collapse. The *hard* tier (`data/bench/hard/`) is the real
progress signal; it's adapted from open benchmarks by `scripts/build_hard_bench.py`
(MATH-500 / MBPP / MMLU-Pro / BIG-Bench-Hard, pulled dependency-free via the HF
datasets-server API) and lands the base in a measurable band. Add items to the
core set by editing JSON; rebuild/extend the hard set via the builder.

```bash
conda run -n twin-models python scripts/build_hard_bench.py --per-category 20   # build data/bench/hard
caffeinate -i conda run -n twin-models python scripts/benchmark.py \
    --config configs/base.yaml --data data/bench/hard --adapters base
```

### Recorded base bars (frozen Qwen3-8B-6bit, greedy, thinking OFF)

| Tier | math | coding | knowledge | reasoning | overall |
|------|------|--------|-----------|-----------|---------|
| core (`data/bench/`)       | 100% | 100% | 100% | 100% | **100%** (54 items) |
| hard (`data/bench/hard/`)  | 70%  | 50%  | 60%  | 85%  | **66%** (80 items)  |

Sources: math←MATH-500 (level≥3, numeric), coding←MBPP test split, knowledge←MMLU-Pro
(10-way MCQ), reasoning←BBH (logical deduction / dates / boolean / counting). The hard
tier is the baseline every trained checkpoint is compared against; reasoning is the
highest at 85% (its boolean/navigate items are easy for the base) but still has headroom.
Note: the benchmark loads the full base and generates per item, so run it with the machine
awake (`caffeinate`) — a suspended laptop inflates wall-clock enormously.

## Entry template

Copy this block for each run.

```
### <date> — <run-name>

- **Config:** configs/<file>.yaml (note any CLI overrides)
- **Commit / seed:** <git sha or "uncommitted"> / seed=<n>
- **Iters / wall time:** <n> / <hh:mm>
- **Hypothesis:** <what this run is meant to show or test>
- **Headline metrics** (from analyze_run.py summary):
  - solve-rate curve: <list> — linearity slope=<>, pearson_r=<>, ramp_mse=<>
  - final Rc / Rs: <> / <>   parse_ok_rate: <>
  - KL(c)/KL(s) final: <> / <>   adapter drift A/B: <> / <>
  - oracle calls (c/s) trend: <>   CAS tool calls trend: <>
- **Observations:** <what happened — collapses, plateaus, surprises>
- **Decision / next:** <keep config? change what? next run id>
```

---

## Runs

### 2026-06-29 — (queued) inaugural scale run

- **Config:** configs/base.yaml (thinking ON; 16 LoRA layers; N=5, G_c=4, K=4)
- **Commit / seed:** TBD / seed=0
- **Iters / wall time:** 1000 / TBD
- **Hypothesis:** the loop is stable at scale with thinking on — rewards rise,
  the solve-rate-vs-difficulty curve becomes linear-descending, KL stays bounded,
  and adapter drift is smooth (no collapse). This is the baseline every ablation
  below is measured against.
- **Status:** SUPERSEDED by `mini-01` below. Measured throughput (see next entry)
  makes a 1000-iter thinking-on run on this M5 ~2 months of wall-clock, so the
  full base.yaml scale run is deferred until grad-accumulation/batched scoring
  lands. The inaugural run is the smaller `mini-01` instead.

### 2026-06-30 — mini-01 (inaugural real run)

- **Config:** configs/mini.yaml (thinking ON; 16 LoRA layers; **G_c=2, N=3, K=4**;
  warmup=5/swap_interval=8 so rotation fires at iters 13/21/29; checkpoint_every=5;
  token budgets all roles = **4096**, see shakeout below)
- **Commit / seed:** uncommitted / seed=0
- **Iters / wall time:** 30 / **~5-8h** (revised down from a 12-14h guess after the
  shakeout measured 6-9 min/iter; early consistency is near-zero so the dominant
  solver term barely fires — wall time will rise as consistency climbs)
- **Why this and not base.yaml:** a timing probe on the real model measured
  **~16.4 tok/s decode** for Qwen3-8B-6bit on this M5 (thinking on/off identical —
  same model, thinking just emits 3-5x more tokens). At that rate a base.yaml math
  iteration (~75k generated tok: 4 creator + 20 LLM-judge consistency checks +
  ~56 solver) is ~80-95 min, so 1000 iters ≈ 2 months. mini-01 shrinks the
  combinatorial knobs (G_c·N·K) to fit a meaningful run in one overnight window
  while keeping thinking ON and every mechanism (oracle tax, CAS tools, ReAct
  judge, rotation) faithful to base.yaml.
- **Hypothesis:** the loop is stable with thinking on — rewards rise, parse_ok→1
  early, the solve-rate-vs-difficulty curve trends to a descending line, KL stays
  bounded, adapter drift is smooth, and rotation at iters 13/21/29 doesn't cause a
  collapse. Establishes that the machinery learns before spending days on scale.
- **Pre-flight done:** 210/210 fast tests green; `--resume-step N` added to
  train.py (loads adapter_{A,B}_step{N}, continues at iter N; optimizer moments +
  drift baseline reset at the resume point); configs parse + rotation verified.
- **Run notes (gotchas):** must run under `caffeinate -i` (a suspended laptop
  inflates wall-clock enormously) and `nohup … &` so it survives the terminal;
  the LLM-judge consistency check fires once per *math* problem (no check+symbol
  certificate yet), so math iters cost ~1.5x coding iters.
- **Shakeout (2 iters, 2026-06-30):** clean exit; iter0 coding 6.2min (0/3 consistent,
  no solver rollouts), iter1 math 9.0min (1/3 consistent, solved 1/1, Rs=+1.0); KL
  bounded (~1e-3), drift smooth (B drifts only once it has trajectories). Caught a
  real bug: **parse_ok=0.50** — the creator hit the 2048 token cap mid-`<think>` and
  never emitted JSON. Direct probe confirmed (3/4 truncated). Raised creator budget
  2048→3072 (parse→~87%), then set all roles to 4096 to leave headroom for longer CoT
  on harder problems. Lowering creator temp 0.9→0.6 made parse WORSE, so temp stays 0.9. Coding is the rambly
  domain (the creator over-thinks tool applicability); watch its parse rate.
- **run0 (OOM, 2026-06-30):** launched at 4096 budgets; **SIGKILLed at iter 5** (OOM
  on 32GB). Root cause: GRPO scored the *whole* trajectory batch in one autograd
  graph, and a heavy math iter has K·n_consistent ≈ 12 solver trajectories of up to
  4096 tokens — a single 4096-token completion already materializes a ~2.5GB [T,V]
  logits tensor, so ~12 at once (plus the reference pass evaling all 12 forwards
  together) blew past RAM. Coding iters survived only because they had 0 consistent
  problems → 0 solver rollouts. Observed iters 0-4: parse coding 0.5 / math 1.0 (as
  expected), Rc rose to +0.92 on the math iter, solver solved 1/1, KL ~1e-3, drift
  smooth — i.e. the loop was healthy; it was purely a memory ceiling.
- **Fix:** implemented the deferred microbatching lever — `grpo_update` now takes
  `microbatch_size` (config `train.grpo_microbatch`), splitting the batch into
  chunks, accumulating grads (identical math: ∇Σ = Σ∇, verified equivalent to ~1e-7
  vs the single-graph path incl. KL + loss masks), realizing each chunk before the
  next so its graph frees. Also made the trainer's reference-logprob pass eval each
  trajectory immediately instead of batching the eval. mini.yaml set to
  `grpo_microbatch: 1` (one 4096-token rollout in the backward at a time, ~15GB
  peak). 4096 budgets kept. +2 equivalence tests → 212 fast tests pass.
- **run1 (relaunched 2026-06-30):** fresh from iter 0 with the fix (run0 artifacts
  saved as runs/2026-06-30-mini-01.oom-run0.*). **Completed all 30 iters cleanly, no
  OOM, ~9h15m wall.**
- **Headline metrics (run1, from analyze_run.py):**
  - **solve-rate vs difficulty: [0.94, 0.69, 0.58] — slope=-0.18, pearson_r=-0.979,
    r2=0.96, ramp_mse=0.125.** Clean monotonic descending curve = the creator makes a
    real difficulty gradient and the solver degrades smoothly easy->hard. (This is the
    result mini-01 existed to prove.)
  - parse_ok_rate -> 1.0 (creator learned the JSON format; a few 0.5 dips, no sustained
    failure). oracle calls (c/s): 0 throughout (tax works, no reliance). CAS tool calls:
    sparse (iters 0/9/15).
  - KL(c): flat ~5e-4 for ~20 iters then crept up to a 0.025 peak ~iter28 (ended 0.005);
    KL(s) bounded, spiky on math iters. drift A 0.013->2.28, drift B 0->1.92 — both
    smooth, no jumps (no collapse). Rotation fired at iters 13/21/29 without disruption.
  - Rs/solve% per-iter look bimodal (1.0 on math iters, 0 on coding) — NOT a collapse,
    it's the coding-consistency-zero artifact below; the aggregate difficulty curve is
    the real read.
- **Held-out hard-tier benchmark (step30, greedy, thinking OFF, vs base bar):**
  overall base 66% -> **A 70% (+4), B 71% (+5)** — both beat the frozen base on held-out
  problems (real transferable capability, not creator-relative). By category:
  math 70->A80(+10)/B70; coding 50->50/50 (never trained); knowledge 60->A70(+10)/B80(+20);
  reasoning 85->A80(-5)/B85. CAVEAT: 20 items/cat, so per-category deltas are directional,
  not significant; trust the 80-item aggregate + the direction (math, the only trained
  domain, moved most). Report: runs/bench-hard-mini01-step30.json.
- **Observations / weaknesses to fix:**
  1. **Coding is a dead domain** — coding problems never reach consistency (creator's
     solution_code fails its own tests) -> 0 solver rollouts on coding iters -> no coding
     training signal, and coding bench flat. Highest-leverage next fix.
  2. **Late-run creator KL/drift creep** — KL(c) trended up in the back half; LR (1e-5)
     / kl_beta (0.02) will need attention before a much longer run.
- **Decision / next:** loop is validated. Next: (a) diagnose+fix coding consistency
  (inspect why creator code fails its own tests — sandbox/format/test-harness), (b)
  probe kl_beta/LR for the KL creep, then (c) a longer run and/or the thinking-off
  pilot.yaml ablation. mini-01 is the baseline all of these compare against.

### 2026-07-01 — (queued) mini-02 — advantage-degeneracy ablation

- **Config:** configs/mini2.yaml (= mini.yaml + Sprint-6 fixes: **G_c 2→4**,
  **rewards.target_hi/lo 1,0→0.9,0.1** (interior target ramp), **train.adv_mode:
  mean** (Dr.GRPO A = R − mean(R), new code default); zero-advantage batches now
  skip the whole GRPO update, logged as `skipped_zero_adv` in
  solver_update/creator_update)
- **Commit / seed:** TBD / seed=0
- **Iters / wall time:** 30 / TBD — expect **longer than mini-01's 9h15m**: G_c=4
  adds 2 creator ReAct generations (~5-9 min/iter) AND doubles the solver term on
  consistent problems (solver rolls out on all 4 suites), partially offset by the
  zero-adv skip reclaiming the ~60% of update compute mini-01 wasted. Budget a
  full day; `caffeinate -i` + `nohup` as before.
- **Hypothesis:** the degeneracy fixes convert mini-01's silent iterations into
  learning signal — creator/solver `pg==0` rate drops well below mini-01's 15/30
  and 9/15; `skipped_zero_adv` counts the (now-free) remaining ties; the
  solve-rate curve settles toward the interior band ([0.9, 0.5, 0.1] target
  instead of [1, 0.5, 0]) while staying monotonic (pearson_r ≤ −0.9); no
  KL/drift regression vs mini-01. NOTE the target-band change means `ramp_mse`
  (measured vs the fixed 1→0 ramp in analyze_run) should *rise* toward ~0.007
  ((0.1)² at the endpoints) even when the creator is perfectly on-band — read
  the curve values, not just ramp_mse, when comparing to mini-01.
- **Watch:** whether the interior band invites hacking (creator engineering
  ~always-solvable rank-2 problems to hit 0.1 "hardness" cheaply); coding domain
  is STILL dead (pipeline unbuilt) — coding iters remain constant-reward, but
  now they cost nothing at update time (skipped) instead of a pure-KL backward.
- **Status:** config + code ready (254 fast tests green), not yet launched.

---

## Backlog — planned experiments (deferred from Sprint 4)

Run the inaugural baseline first, then work down this list (each is a single
config change vs the baseline — the code paths already exist):

- [ ] **Thinking on vs off** — base.yaml (on) vs a thinking-off copy; does the
      extra reasoning budget improve solve rate / curve linearity enough to
      justify the token cost?
- [ ] **Oracle on/off** — `oracle.enabled` / set `w_oracle`=0; does the tax
      meaningfully suppress oracle reliance, and at what reward cost?
- [ ] **Consistency on/off** — drop the consistency gate; confirm it's what
      prevents fake-answer reward hacking (expect curve to degrade without it).
- [ ] **Rotation vs explicit blend** — `roles.injection_rate` 0 vs >0; is naive
      LoRA blending destructive as predicted (DESIGN §4)?
- [ ] **Solver-as-judge vs oracle-judge** — swap the consistency-judge fallback;
      watch for collusion (consistency stays high while real solve rate drops).
- [ ] **Grad accumulation / batched scoring** — the `grad_accumulation_steps`
      config field exists but is unused; needed before pushing G_c·N·K higher.
- [ ] **Inline ReAct for the solver/oracle** — extend `generate_react` + `<obs>`
      masking (already built for the creator) to the solver path so the oracle is
      actually *used* mid-rollout, not just counted.
- [ ] **Brevity bonus** (added 2026-07-01, code landed during mini-02) —
      `rewards.w_brevity > 0` (try 0.1–0.2, keep < `w_solve`): correct solver
      attempts earn `w_brevity·(1 − L/L_max)` extra, so among solved attempts in
      a K-group the shorter one gets the higher advantage. Gated on `solved`
      (no reward for truncating into a wrong answer). Watch: solve rate must not
      drop (thinking-block compression is the intended effect, skipping thinking
      that was load-bearing is the failure mode); mean solver `n_tokens` (now
      logged per attempt) should fall.
