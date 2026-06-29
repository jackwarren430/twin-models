# twin-models

Self-play RLVR with **two LoRA adapters over one frozen MLX base** (Qwen3-8B, 6-bit), on a 32 GB
Apple-Silicon Mac. Two model instances A and B alternate as problem **creator** and **solver**, trained
with **GRPO** on a difficulty-gradient reward + a consistency check + an oracle tax. See **[DESIGN.md](DESIGN.md)**
for the full specification and rationale.

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
  solve/oracle-tax). 72 fast tests pass (no model needed).
- Sprints 3–4 (GRPO loop, scale) — see DESIGN.md §11.

## Setup

The MLX env is already created as the conda env **`twin-models`** (mlx 0.31.2, mlx-lm 0.31.3). To recreate:

```bash
conda create -y -n twin-models python=3.12
conda run -n twin-models python -m pip install mlx mlx-lm sympy pytest numpy pyyaml
conda run -n twin-models python -m pip install -e .   # from repo root
```

The base model is expected at `~/.cache/lm-studio/models/mlx-community/Qwen3-8B-6bit` (MLX format).

## Run

```bash
# fast unit tests (no model load)
conda run -n twin-models python -m pytest -q

# heavy tests that load the real base
TWIN_RUN_MODEL_TESTS=1 conda run -n twin-models python -m pytest -q -m model

# end-to-end Sprint-1 smoke test (loads ~6 GB, prints PASS/FAIL invariants)
conda run -n twin-models python scripts/smoke_test.py
```

## Layout

```
DESIGN.md            full spec (read this first)
configs/             tiny.yaml (dev) + base.yaml (research)
src/twin/
  config.py          typed config + YAML loader
  models/base.py     TwinBase: shared frozen base, generate, oracle, logprobs
  models/adapters.py Adapters: two LoRA trees over one base, swap/save/load
  problems/schema.py Problem / ProblemSuite + parsing & validation
  tools/             calc, sandboxed python, taxed oracle, ReAct harness
  verifiers/         sympy math, sandboxed code exec, oracle-judge, dispatch
  rewards/engine.py  RewardEngine: creator/solver rewards (DESIGN §6)
scripts/smoke_test.py
tests/
```
