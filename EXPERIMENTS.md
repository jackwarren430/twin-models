# Experiments log

A running log of self-play training runs and what we learned. One entry per run
(or per small batch of related runs). Keep it append-only; don't rewrite history,
add a follow-up entry instead. The point is that six months from now we can see
*why* a config was chosen, not just what it was.

See **[DESIGN_V2.md](DESIGN_V2.md)** for the current system spec
([DESIGN.md](DESIGN.md) is kept for sprint-by-sprint history); this file is
the lab notebook.

## How to run

```bash
# launch a run (logs one JSON record per iter to runs/<run-name>.jsonl).
# --no-capture-output + python -u are REQUIRED for live logs (plain `conda
# run` buffers stdout until exit); caffeinate -i because a suspended laptop
# inflates wall-clock enormously; transcript stays ON (no --no-transcript).
caffeinate -i conda run --no-capture-output -n twin-models python -u \
    scripts/train.py --config configs/mini4.yaml --run-name <date>-mini-04 \
    > runs/<date>-mini-04.out 2>&1 &

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

### 2026-07-01 — mini-02 — advantage-degeneracy ablation — DONE, analyzed 2026-07-02

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
- **RESULT (30/30 iters, ~10.7h wall, 2026-07-01 18:42 → 07-02 05:23; artifacts
  runs/2026-07-01-mini-02.{jsonl,_series.csv,_summary.json}):**
  - solve-rate curve [0.952, 0.908, 0.821] — pearson_r=−0.982, r2=0.965,
    slope=−0.065. Monotonic BUT **ceiling-compressed**: the band asked for
    [0.9, 0.5, 0.1] and the creator delivered ~[0.95, 0.91, 0.82] — its
    "hard" problems barely dent the solver. (ramp_mse=0.281 is vs the fixed
    1→0 reference ramp; not comparable to the band, read the curve values.)
  - **Coding domain measured DEAD: 0/141 parsed coding problems consistent**
    (math: 57/150) → 15 of 30 iterations produced zero solver rollouts —
    half the wall-clock bought parse-gate-only creator signal.
  - **Solver starved: 4 real solver updates in 30 iters.** 88% of scored math
    problems had K-group solve rate exactly 0 or 1 (7/57 interior), so groups
    tied and the zero-adv skip (correctly) fired — the fixes moved waste from
    the update step to the generation step, but saturation is a *problem
    difficulty* issue the Sprint-6 knobs can't fix. Creator side better:
    pg==0 in 7/30 (vs mini-01's 15/30) — G_c=4 + mean baseline worked.
  - Math cert-pass ~38% noisy-flat across the run (NOT degrading — decomposed
    by the mini-03a transcript, see next entry). KL bounded (mean c 0.005 /
    s 0.0006), drift smooth (A 2.04 / B 0.97), oracle calls 0.
- **Decision / next:** (a) drop coding from `game.domains` until its pipeline
  sprint (user-confirmed 2026-07-02); (b) attack solver starvation from the
  problem-difficulty side (w_brevity tie-break + a recalibrated difficulty
  prompt) → mini-03; (c) memory probe (probe_memory.py) measured the GRPO
  backward worst case at **55GB@4096 / 132GB@8192 budgets** → budgets stay
  4096, grpo_microbatch stays 1, longer contexts need a chunked
  completion_logprobs backward first.

### 2026-07-02 — mini-03a — ABORTED at iter 10/30 (prompt-induced think-spiral)

- **Config:** configs/mini3.yaml (= mini2.yaml + **domains ["math"]** only,
  **w_brevity 0.15**, transcript ON, and a code-side recalibrated creator
  difficulty prompt: "stump a strong expert solver").
- **Commit / seed:** uncommitted / seed=0
- **Iters / wall time:** 10 of 30 (killed) / ~4.7h
- **What happened:** the new difficulty language sent the thinking creator
  into an in-`<think>` design-deliberation spiral (draft → find flaw →
  backtrack → repeat) that blew the whole 4096 budget before any JSON:
  parse_ok [0.5, 0, 0.5, 1.0, 0.5] iters 0-4 → [0.25, 0, 0.25, 0.25, 0]
  iters 5-9; Rc mean −0.53 overall, **−0.84 over iters 5-9**; the solver got
  nothing in the back half. Killed at the iter-10 checkpoint.
- **What worked anyway:** **w_brevity fixed solver starvation** — the solver
  took a real update on 4/4 eligible iterations (mini-02: 4/30 overall), the
  token-length spread breaking all-solved ties exactly as intended. And the
  first-ever transcript **paid for itself immediately**: it decomposed
  mini-02's ~38% cert-pass mystery into (1) correct problems VOIDED by answer
  format (`"x = 2, y = 1"` vs spec'd `(2, 1)` — a predicate parse error, NOT
  a wrong answer), (2) literal-restatement certs on probability problems
  (trivial-rejected by design), (3) genuinely wrong answers.
- **Fixes out of the audit:** `_normalize_named_value` in
  `src/twin/verifiers/math_verifier.py` (rewrites named-assignment answers
  into symbol order before parsing; +5 tests) closes bucket 1; the difficulty
  prompt rewritten to keep the structural-hardness push but add "design
  decisively / commit to the first workable idea / deliberation gets you
  truncated" (the mini-03a lesson: aspirational difficulty language makes a
  thinking model deliberate, and deliberation IS truncation at 4096).
- **Artifacts:** runs/2026-07-02-mini-03a-aborted.{jsonl,out,transcript.txt}

### 2026-07-02 — mini-03b — STOPPED by user at iter 4/30 (superseded by Sprint 7)

- **Config:** configs/mini3.yaml + the decisive-design prompt rewrite +
  CREATOR_SYSTEM output-budget warning + the answer-format normalizer.
- **Commit / seed:** uncommitted / seed=0
- **Iters / wall time:** 4 of 30 (stopped) / ~1.9h (~28 min/iter — heavier
  than mini-02's average because every iter is math and consistency was high)
- **Early signals (all 4 iters):** parse_ok 1.0 / 0.25 / 1.0 / 1.0 — the
  decisive-design rewrite fixed the truncation spiral. Rc +0.64 / −0.64 /
  +0.57 / +0.46; Rs ≈ +1.13 on solver-active iters. BUT solve% saturated at
  1.0 on 3 of 4 iters — the softened prompt gave back the difficulty push
  (ceiling compression again).
- **Why stopped: `creator_tool_calls` = 0 on every iteration.** The
  transcript shows the creator *simulating* the CAS tool inside `<think>`
  ("The tool would return x=3, y=2") — planning calls, hallucinating the
  observations, never emitting a single real `<tool>` call. The legacy ReAct
  text protocol is a format Qwen3 was never trained on; with thinking ON the
  model reasons *about* the tool instead of calling it. That (plus the
  wasted-thinking and difficulty problems above) motivated **Sprint 7**
  (DESIGN_V2.md): native Qwen3 function calling, per-problem creator
  generation, personas, and the strict tool gate — run as mini-04 below.
  No checkpoints were saved (first save at iter 5; the `checkpoints/`
  step5-30 files predate this run).
- **Artifacts:** runs/2026-07-02-mini-03b.{jsonl,out,transcript.txt}

### 2026-07-03 — (STOPPED at iter 2/30, by design) mini-04(a) — Sprint-7 shakeout (native tools, per-problem creator, personas)

- **Config:** configs/mini4.yaml (= mini3.yaml + Sprint 7: **tools.protocol
  native**, **game.creator_mode per_problem** (+ condition_on_previous),
  **game.personas true**, **N 3→5**, **K 4→8**, **w_brevity 0.15→0.25**;
  require_tool_use stays FALSE — log-first). Solver temp stays 0.8 (decision
  2026-07-03: outcome variance comes from harder problems + brevity, not
  hotter sampling, which would redefine the solve-rate measurement).
- **Commit / seed:** TBD / seed=0
- **Iters / wall time:** 30 / TBD — **budget generously**: max solver
  generations/iter is G_c·N·K = 160 vs mini-03's 48 (~3.3×), partially offset
  by per-problem creator rollouts finishing early (EOS after one problem's
  JSON) and w_brevity-shortened solver attempts. Batched solver generation is
  the Sprint-8 lever if this is too slow. `caffeinate -i` +
  `conda run --no-capture-output … python -u`, transcript ON.
- **Hypothesis:** (1) native protocol makes the creator actually CALL the CAS
  tool — `creator_tool_ok` > 0 from iter 0 and `creator_answer_in_obs`
  climbing toward N per suite (mini-03b: zero real calls, hallucinated obs);
  (2) with real tool answers, consistency (cert-pass) climbs well above
  mini-02's ~38% flatline; (3) per-problem generation holds parse_ok ≈ 1.0
  while allowing real thinking per problem (no truncation spiral);
  (4) N=5 + K=8 + interior band yields interior solve rates (K-groups no
  longer 88% saturated → solver updates most iterations);
  (5) personas/competition framing doesn't reintroduce the mini-03a
  deliberation spiral (watch creator token counts).
- **Watch:** `creator_answer_in_obs` low while consistency high = the creator
  still fakes tool grounding → flip `game.require_tool_use: true` for the
  next run. Per-problem conditioning blocks growing too long (prompt carries
  k−1 JSONs). Wall-clock per iter vs mini-03b's measured ~28 min/iter (at
  N=3/K=4 with high consistency — mini-04's 160-gen worst case will be
  slower still; consider trimming iters if >1h/iter).
- **Status:** LAUNCHED 2026-07-03 23:07 AFTER the four audit-2026-07-03 bug
  fixes (commit 599e35d); **STOPPED deliberately 2026-07-04 ~01:10 after 2
  complete iterations** (~50-70 min/iter) — the shakeout had answered its
  questions and the answers demanded a prompt fix, not 28 more iterations.
  Artifacts: `runs/2026-07-03-mini-04.{jsonl,out,transcript.txt}` (2 records).
- **RESULTS (2 iters, 40 creator rollouts):**
  - **Machinery: WORKS.** Per-problem creation, dictated difficulties,
    conditioning, broadcast credit, personas, cert coverage (n_cert ==
    n_problems on every suite), parse-gates, and the audit fix all behaved:
    partially-parsed suites earned r_gradient 0.14–0.33 (scaled by
    parsed/5 against prompted targets) where pre-fix code would have paid
    ~0.9+ for a perfect re-stretched fit. Solver got REAL updates both iters
    (64 trajectories iter 1 — vs mini-02's 4/30 iterations). KL ~0.001,
    drift tiny, no OOM.
  - **Hypothesis (1) REFUTED: creator_tool_calls = 0 / 40 rollouts.** The
    transcript names the mechanism: the model decides mid-<think> that
    "I can't actually run the function here, I need to simulate", and the
    system prompt's "output ONLY a single JSON object" reads as forbidding
    a tool call before the JSON. (My earlier "~3/11 adoption" read was wrong
    — those n_tool_calls lines were the JUDGE's, which calls solve fine.)
  - **Hypothesis (4) REFUTED so far: solve rates saturated** (iter 0 0.93,
    iter 1 0.98; 63/64 attempts solved iter 1) — difficulty push lost again.
  - **New: per-RANK parse failures 5/20 then 9/20**, concentrated at the
    hard ranks (x.3/x.4): the difficulty demand triggers mini-03a-style
    deliberation spirals that truncate mid-<think>. Numbers: iter 0
    Rc=+0.398 Rs=+1.127 rgrad=0.334; iter 1 Rc=−0.075 Rs=+1.183 rgrad=0.228.
- **Pivot (2026-07-04):** creator prompts rewritten to a two-phase
  VERIFY (emit a real tool call, wait) / DELIVER (then the JSON) protocol,
  stating explicitly that tools cannot run inside private reasoning; hard-rank
  brief gains a one-pass-commit instruction ("a merely-good hard problem
  beats a discarded perfect one"). `scripts/probe_tool_adoption.py` (6 real
  rollouts, no training) gated the relaunch — pre-registered rule: adoption
  ≥2/6 → relaunch with the strict gate; 0 → iterate the prompt again.

### 2026-07-04 — probes 1 & 2 — tool adoption under the rewritten prompt

- **Probe 1 (creator thinking ON):** adoption 2/6 — both rank-2 rollouts
  called; both rank-4 rollouts burned the FULL 4096 budget in <think> without
  ever leaving it (the mini-03a spiral, surviving the prompt rewrite); rank-0
  answers directly. Post-call "parse failures" turned out to be OUR bug, not
  the model's (below).
- **Probe 2 (creator thinking OFF, `model.creator_enable_thinking: false`):**
  **adoption 5/6 including both hard ranks**, ~215-token rollouts (~20×
  shorter), every delivered JSON valid. The 1/6 no-call rollout parsed fine —
  exactly what the gate should select against.
- **Extractor bug found by the probe dump:** a native rollout's own
  `<tool_call>` JSON (`{"name": "solve", ...}`) sits earlier in the text than
  the final problem JSON and won first-balanced-object extraction — every
  tool-USING rollout "parse-failed" while doing everything right (this also
  poisoned probe 1's post-call reads and would have voided mini-04b).
  `_extract_json_object` now strips `<tool_call>`/`<tool_response>` markup in
  its primary pass. Probes: `runs/probe-tool-adoption-{1,2}.out` + rollout
  texts in `runs/probe-tool-adoption.rollouts.txt`.

### 2026-07-04 — (RUNNING) mini-04b — probe-validated relaunch

- **Config:** configs/mini4b.yaml = mini4.yaml + `creator_enable_thinking:
  false` (solver keeps thinking) + `require_tool_use: TRUE` (gate on — at 5/6
  baseline adoption it selects rather than starves) + `solver_batch: 8`.
  Same seed (0), same rewards/targets, 30 iters, checkpoints every 5.
- **Launched:** 2026-07-04 ~02:00, `runs/2026-07-04-mini-04b.*`.
- **Hypotheses:** (1) creator_tool_ok ≈ N·G_c from iter 0 and answer_in_obs
  high (tool-grounded answers); (2) consistency (cert-pass) well above
  mini-02's ~38%; (3) per-rank parse ok ≈ 1.0 (no think spirals to truncate);
  (4) iterations several× faster (short creator rollouts + batched solver);
  (5) the open question inherited from 04(a): can the REWARD (not
  deliberation) push difficulty — watch solve_rate_mean vs the 0.9→0.1
  targets and problem_similarity for trivial-template collapse.
- **Watch:** creator_tool_gated (how often the gate bites); think-share
  telemetry (solver only now); repetition/similarity; wall-clock/iter.
- **Iter-0 record (~40 min):** hypotheses 1-4 CONFIRMED — 44 tool calls/38 ok
  (04a: 0), **consistency 14/14 = 100% cert-pass** (mini-02: ~38%), suite
  parse 1.0 (6/20 rank parse-fails, call-budget overruns), 112 solver trajs,
  tool_gated 0. Hypothesis 5 = the live experiment: solve_rate EXACTLY 1.0,
  per-suite repetition 0.72-0.96 (near-duplicate trivial systems, as probe 2
  predicted for a non-thinking creator) → r_gradient 0.26. The run now
  measures whether the calibration reward alone ratchets difficulty;
  decision point at checkpoint 5-10: repetition >0.8 + solve ~1.0 →
  mini-05 = w_diversity ablation. (answer_in_obs 0 is format sensitivity —
  "(6, 4)" vs the tool's "{x: 6, y: 4}" — not fake grounding; log-only.)
- **Checkpoint-5 deep-dive (iters 0-4, adapters saved 10:51):** machinery
  stays healthy — suite parse 1.0 every iter, 0-2 rank parse-fails/suite,
  tool_gated 0/100 (gate never bit: every scored problem made a real call),
  cert coverage 100% of parsed, ~30-35 min/iter, KL ≤0.004, drift A 0.625.
  **Difficulty ratchet: NO traction yet.** Of consistent(scored) problems,
  the solver goes 8/8 on essentially everything — solve_rates_by_rank is
  1.0 across ranks with exactly two 0.0 exceptions in 5 iters; r_gradient
  sits at 0.22-0.36 ≈ the all-solved floor exp(−4·mean((1−t)²)) ≈ 0.267.
  Rc's climb (+0.30 → +0.87) is consistency+validity+tool terms, not
  gradient fit. Consistency oscillates (14/14, 11/19, 20/20, 9/19, 18/19)
  — the dips are wrong-creator-answer buckets, penalized as designed.
  Repetition is volatile, not locked: iter 2 hit 0.88-1.0 (near-duplicate
  linear systems) but iter 4 read 0.25-0.62. **Decision rule half-met**
  (solve ~1.0 YES; repetition >0.8 persistent NO) → let it run; re-evaluate
  at checkpoint 10. If rgrad is still floor-pinned there with rotation
  having swapped roles (warmup 5, swap 8), that's evidence the calibration
  reward alone cannot ratchet difficulty without deliberation space →
  mini-05 candidates: w_diversity > 0, or creator thinking back ON with a
  hard token cap instead of OFF.
- **Checkpoint-10 review (iters 5-10):** pre-registered stop rule literally
  NOT met — repetition metric reads 0.22-0.85, mostly 0.3-0.7 — but the
  transcript shows it's being DODGED, not beaten: the last ~12 problems are
  one template ("solve x^2-ax+b=0, select the maximum root") with
  coefficient swaps, which break enough word-bigrams to lower Jaccard while
  structural repetition is ~total. Solve stays saturated (rates 1.0 with
  isolated exceptions), rgrad floor-pinned (0.13-0.36, mean ≈ the 0.267
  all-solved floor) through 11 iterations of A creating. Measurement
  lesson for mini-05: normalize digits/numbers out of the bigram metric
  (or hash statement templates) so coefficient-swap collapse is visible.
  **Decision: continue** (rule not met; rotation at iter 13 is the headline
  mechanism and imminent). Re-evaluate at checkpoint 15, which covers
  B-as-creator iters 13-14. Pre-registered: if B also collapses to a
  trivial template by 15, the live question resolves NO — the calibration
  reward alone cannot ratchet difficulty in a non-thinking creator — and
  mini-05 restores creator deliberation (thinking ON + hard token cap)
  and/or turns on w_diversity with a digit-normalized similarity.
  Corollary already visible: consistency is the binding constraint the
  creator is optimizing around — writing hard problems risks
  self-inconsistency, so the safe-template strategy dominates; iter dips
  to 2/5 consistent are wrong-creator-answer buckets on the rare
  non-template attempts.

---

## Backlog — Sprint 8 (queued 2026-07-03; most items LANDED same day, during the mini-04 run)

- [x] **Batched solver generation** — LANDED: `TwinBase.generate_batch`
      (continuous batching, exact sampled ids incl. EOS) + `gen.solver_batch`
      (default 1 = v1 path) + `_generate_solver_group` chunking. Measure with
      `scripts/probe_batch_generate.py` (NOT while a run is live) to pick
      future batch sizes; mini-04b uses 8.
- [x] **Per-problem creator credit decomposition** — LANDED as
      `game.credit: per_problem` (default broadcast);
      `RewardEngine.creator_problem_rewards`; suite summaries log
      `problem_rewards`. Run as a measured ablation against mini-04.
- [x] **Chunked LM-head backward** (pulled from the standing queue) — LANDED
      as `train.logit_chunk` (default 0): kills the mini-02-measured 55GB
      [T,V] worst case; equivalence proven on the CPU stream; certify with
      probe_memory before making it a run default.
- [x] **Coding pipeline** (pulled from the standing queue) — root cause of
      mini-02's 0/141 found by code audit (the creator prompt never asked for
      verification.tests; verify_code fails closed "no tests supplied"):
      executable contract in both creator prompts, sandboxed `run_python`
      creator tool, solver code path (SOLVER_CODE_SYSTEM + extract_code_block
      scoring). Validate with `configs/coding-shakeout.yaml` (3 iters,
      transcript on) post-mini-04; coding stays out of game.domains until
      it passes and is re-admitted.
- [x] **Sprint 9 countermeasures** (DESIGN_V2 §12): diversity telemetry
      (`problem_similarity`, per-suite `repetition`) + `rewards.w_diversity`
      penalty (default 0); grounded themes (`game.theme_weights` +
      scripts/build_grounded_themes.py; data/themes-grounded.json built from
      the base hard-bench bar); think-share telemetry
      (`creator/solver_think_share`).
- [ ] **CAS session state** — per-rollout namespace so tool calls can define
      intermediates (`a = solve(...)` then reuse `a`), or multi-statement
      calls. Ergonomics for creative multi-step problems; pointless until the
      model reliably calls the tool at all (verify in mini-04 first).
- [ ] **Judge → native protocol** — the judge shares the legacy-ReAct failure
      mode (it's the same base model); migrate it to native tool calling once
      mini-04 validates the protocol on the creator path.
- [ ] **Strict tool gate on** (`game.require_tool_use: true`) — if mini-04's
      `answer_in_obs` shows the creator still faking tool grounding. (Early
      iter-0 transcript: most rollouts still simulate the CAS in <think>, but
      ~3/11 make REAL native calls — the gate would have something to select
      on. Decide on the full-record numbers.)

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
