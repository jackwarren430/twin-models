# twin-models

Self-play RLVR with **two LoRA adapters over one frozen MLX base** (Qwen3-8B, 6-bit), on a 32 GB
Apple-Silicon Mac. Two model instances A and B alternate as problem **creator** and **solver**, trained
with **GRPO** on a difficulty-gradient reward + a consistency check + an oracle tax. See
**[DESIGN_V2.md](DESIGN_V2.md)** for the current specification and rationale
([DESIGN.md](DESIGN.md) keeps the sprint-by-sprint build history).

The memory trick that makes this fit in 32 GB: the 6.2 GB quantized base is loaded **once** and shared by
A, B, and the knowledge oracle. A and B are just two small LoRA parameter trees swapped in over the same
frozen weights; the oracle/KL-reference is the same base with a zeroed adapter.

## Status

- **Sprint 1 — Foundation: done & verified.** Config loader, `TwinBase` (load/generate/oracle/logprobs),
  `Adapters` (two LoRA trees + zero reference, swap, blend, save/load), problem schema + robust parsing.
  `scripts/smoke_test.py` confirms all core invariants on the real model.
- **Sprint 2 — Scoring: done & verified.** `tools/` (`calc`, sandboxed `python`, taxed `oracle`, ReAct
  tool harness), `verifiers/` (sympy math, sandboxed code exec, oracle-judge fallback, consistency
  dispatch), `rewards/` (`RewardEngine`: creator gradient/consistency/oracle-tax/valid, solver
  solve/oracle-tax).
- **Sprint 3 — RL loop: done & verified.** Grad-enabled `completion_logprobs` + true-id sampling,
  `rl/` (GRPO: group advantages, k3 KL to the frozen base, single-step update + grad clip), `roles/`
  (rotation/warmup/optional blend), `prompts/`, `log/` (JSONL), and `train/` (`SelfPlayTrainer`).
  `scripts/smoke_test_sprint_3.py` runs the loop end-to-end on the base — rewards move, KL stays bounded,
  the base adapter is untouched.
- **Sprint 3.5 — Creator CAS tool (inline ReAct): done & verified.** `tools/cas.py` (`solve`, a
  creator-only untaxed SymPy "Wolfram-Alpha-like" tool), `TwinBase.generate_react` (segmented generation
  that splices `<obs>` tool results mid-rollout), and a per-token `loss_mask` so GRPO ignores injected
  tokens — the creator now one-shots correct problem+answer pairs instead of hallucinating them. 127 fast
  tests pass (incl. a toy-LM proof that masked obs tokens contribute zero to PG/KL);
  `scripts/smoke_test_creator_cas.py` verifies the splice/mask and a clean masked update on the real base.
- **Sprint 4 — Scale & study (base loop): done.** `base.yaml` runs the bigger loop with **thinking ON**;
  the trainer logs per-iteration curve instrumentation (oracle/tool usage, adapter norm + drift); a new
  `twin.analysis.curves` module + `scripts/analyze_run.py` turn a run's JSONL into the study curves
  (solve-rate-vs-difficulty linearity, KL, adapter drift) as CSV/JSON + a terminal report (PNGs if
  matplotlib is installed). 156 fast tests pass; a model-gated short-e2e suite runs a 2-iteration real-base
  loop end-to-end. Runs are recorded in [EXPERIMENTS.md](EXPERIMENTS.md). Ablations are queued (see §11).
- **Held-out benchmark — done & verified.** `twin.bench` + `scripts/benchmark.py` score the frozen base
  and/or trained A/B adapters against a fixed item set spanning **math** (SymPy), **coding** (sandboxed
  tests), **knowledge** (multiple-choice + short-answer), and **reasoning** — the *absolute* capability
  metric the in-training curves can't give (the loop's solve-rate is only relative to the creator's own
  moving distribution). Greedy/reproducible; prints a base-vs-A-vs-B table with deltas and writes a JSON
  report. Two tiers: a hand-authored **core** set (`data/bench/*.json`, a cheap regression floor — base
  scores 100%) and a **hard** tier (`data/bench/hard/`) adapted from open benchmarks via
  `scripts/build_hard_bench.py` — **MATH-500**, **MBPP**, **MMLU-Pro**, **BIG-Bench-Hard** — where the
  frozen base lands at **66%** (math 70 / coding 50 / knowledge 60 / reasoning 85), i.e. real headroom to
  measure training. 43 fast tests cover loading, every grader path, the source converters, and the runner
  via a fake solver (210 fast tests total).

## Setup

The MLX env is already created as the conda env **`twin-models`** (mlx 0.31.2, mlx-lm 0.31.3). To recreate:

```bash
conda create -y -n twin-models python=3.12
conda run -n twin-models python -m pip install mlx mlx-lm sympy pytest numpy pyyaml
conda run -n twin-models python -m pip install -e .   # from repo root
```

The base model is expected at `~/.cache/lm-studio/models/mlx-community/Qwen3-8B-6bit` (MLX format).

## Run

Tests live one folder per sprint under `tests/` and are driven by a single
runner, `scripts/run_tests.py` (see `tests/README.md` for the full layout):

```bash
# all fast tests (no model load) — the default
conda run -n twin-models python scripts/run_tests.py

conda run -n twin-models python scripts/run_tests.py --sprint 2 -v   # one sprint, verbose
conda run -n twin-models python scripts/run_tests.py -k consistency  # filter by name
conda run -n twin-models python scripts/run_tests.py --model         # include heavy model tests
conda run -n twin-models python scripts/run_tests.py --help          # all options

# plain pytest still works if you prefer it
conda run -n twin-models python -m pytest

# end-to-end smoke tests (load ~6 GB, print PASS/FAIL invariants)
conda run -n twin-models python scripts/smoke_test_sprint_1.py     # adapter mechanics
conda run -n twin-models python scripts/smoke_test_sprint_3.py     # full RL loop, 2 iters
conda run -n twin-models python scripts/smoke_test_creator_cas.py  # creator inline-CAS splice + mask

# run the self-play loop from a config
conda run -n twin-models python scripts/train.py --config configs/tiny.yaml --iters 5

# the bigger Sprint-4 scale loop (thinking ON; minutes per iteration)
conda run -n twin-models python scripts/train.py --config configs/base.yaml --iters 1000

# analyze a finished run into curves (CSV + JSON + terminal report; --plots for PNGs)
conda run -n twin-models python scripts/analyze_run.py runs/<run-name>.jsonl
conda run -n twin-models python scripts/analyze_run.py --latest

# score capability on the held-out benchmark (absolute, not creator-relative)
conda run -n twin-models python scripts/benchmark.py --config configs/base.yaml          # core set, base only
conda run -n twin-models python scripts/benchmark.py --config configs/base.yaml \
    --data data/bench/hard --adapters base,A \
    --checkpoint-a checkpoints/adapter_A_step1000.safetensors                            # HARD tier, base vs trained
conda run -n twin-models python scripts/benchmark.py --config configs/tiny.yaml --limit 2  # quick smoke

# (re)build the hard tier from open benchmarks (MATH-500/MBPP/MMLU-Pro/BBH; no model)
conda run -n twin-models python scripts/build_hard_bench.py --per-category 20
```

The benchmark loads the full ~6 GB base and generates per item, so a full hard-tier pass is minutes of
compute — run it with the machine awake (e.g. wrap in `caffeinate -i`) so it isn't suspended mid-run.

Record runs and what you learned in [EXPERIMENTS.md](EXPERIMENTS.md).

## Layout

```
DESIGN_V2.md         current spec (read this first)
DESIGN.md            v1 spec + Sprints 1-7 build history
configs/             tiny.yaml (dev), base.yaml (scale), mini*.yaml (real runs; mini4 = current)
src/twin/
  config.py          typed config + YAML loader
  models/base.py     TwinBase: shared frozen base, generate, oracle, logprobs
  models/adapters.py Adapters: two LoRA trees over one base, swap/save/load
  problems/schema.py Problem / ProblemSuite + parsing & validation
  tools/             calc, solve (creator CAS), sandboxed python, taxed oracle, ReAct harness
  verifiers/         sympy math, sandboxed code exec, oracle-judge, dispatch
  rewards/engine.py  RewardEngine: creator/solver rewards (DESIGN §6)
  rl/grpo.py         GRPO: group advantages, k3 KL, single-step update
  roles/manager.py   role assignment, rotation/warmup, optional blend
  prompts/           creator/solver templates + themes
  train/loop.py      SelfPlayTrainer: one self-play iteration end-to-end
  train/extract.py   final-answer + oracle-call parsing
  log/jsonl.py       local JSONL run logging
  analysis/curves.py run analysis: solve-rate/linearity/KL/adapter-drift curves
  bench/             held-out benchmark: dataset loader, type-keyed grading, runner, report
data/bench/          core benchmark items (one JSON per category); hard/ = adapted open-benchmark tier
scripts/
  smoke_test_sprint_1.py  adapter mechanics on the real model
  smoke_test_sprint_3.py  full RL loop end-to-end (2 iters) on the real model
  train.py           run the self-play loop from a config
  analyze_run.py     turn a run's JSONL into curves (CSV/JSON/report/plots)
  benchmark.py       score base/A/B on the held-out benchmark (base-vs-trained table + JSON)
  build_hard_bench.py  adapt MATH-500/MBPP/MMLU-Pro/BBH into the data/bench/hard tier
  run_tests.py       central test runner (per-sprint selection, verbosity, ...)
tests/               one folder per sprint (sprint1..4) + tests/bench + conftest auto-markers
EXPERIMENTS.md       run log: how to launch/analyze + entry template + backlog
```
