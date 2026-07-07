# Running twin-models on the NVIDIA DGX Spark

The training loop was built on MLX / Apple silicon. This branch adds a **native
PyTorch/CUDA backend** so the *same* self-play RLVR loop runs on the DGX Spark,
selectable with one config field. Nothing about the Mac workflow changes unless
you set `compute.backend: torch`.

## The hardware (why the native path looks the way it does)

The DGX Spark is a **GB10 Grace Blackwell** desktop system:

- **GPU:** Blackwell, compute capability **`sm_121`**, 5th-gen Tensor Cores,
  ~1 PFLOP FP4. **CPU:** 20-core Arm (10× Cortex-X925 + 10× Cortex-A725),
  **aarch64**, Ubuntu 24.04.
- **Memory:** **128 GB unified LPDDR5x** (CPU+GPU share one address space).
  ~273 GB/s bandwidth — high capacity, modest bandwidth, so decode is
  bandwidth-bound (batching still pays), but the 32 GB memory pressure that
  dominates the mlx config simply isn't a concern here.
- **Software:** CUDA 13 (aarch64), PyTorch (needs an **`sm_121`** build —
  nightly cu13/cu12.8 at time of writing), Python 3.12.

Design consequences (locked in this implementation):

- **bf16 full-precision base (~16 GB) + bf16 LoRA** — no quantization. Trivially
  fits 128 GB and gives the cleanest gradients for autograd-through-LoRA.
- **Generation via native `transformers.generate`** (batched, SDPA /
  FlashAttention-2). The policy's LoRA weights change every step, and PEFT
  `set_adapter` hot-swaps them for free — so there's no separate inference
  engine to re-sync (the reason a vLLM path was deferred; it's noted as a future
  lever below).
- FP4 / NVFP4 is an **inference** format; it is **not** used for training here.

## Setup

```bash
# 1) create the env (conda or venv), Python 3.12
conda create -n twin-spark python=3.12 && conda activate twin-spark

# 2) install a PyTorch build that supports sm_121 FIRST (this is the one wheel
#    that must match the GPU; the exact index/tag drifts — check the PyTorch
#    "Get Started" page for the current aarch64 + CUDA 13 build).
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130

# 3) the rest of the torch extra (transformers drives the model, peft the LoRA)
pip install -e ".[torch]"        # transformers, peft, accelerate, safetensors + core deps
#    (optional, fastest attention) pip install flash-attn --no-build-isolation

# 4) pre-download the base model in HF format (bf16)
huggingface-cli download Qwen/Qwen3-8B      # or set model.path to a local HF dir

# sanity: torch sees the GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Note: mlx is **not** required on the Spark. The codebase is arranged so
importing the loop, the torch backend, and the shared RL/model value types pulls
in no mlx; only the mlx backend imports it (and only when selected).

## Running

```bash
# native DGX Spark run
python scripts/train.py --config configs/spark.yaml --iters 30

# the SAME command on a Mac, mlx backend, just points at a different config
python scripts/train.py --config configs/mini.yaml --iters 30
```

`configs/spark.yaml` is `configs/mini.yaml` with a `compute:` block
(`backend: torch`), an HF `model.path`, and the 32 GB memory levers relaxed for
throughput (`grpo_microbatch: 0`, `logit_chunk: 0`, `solver_batch: 8`). Every
config knob is documented in **CONFIG.md** (`## compute`).

### Acceptance smoke (run this first on real hardware)

```bash
python scripts/train.py --config configs/spark.yaml --iters 2 --run-name spark-smoke
```

This exercises the full path end-to-end on the GB10: batched generation, inline
ReAct tool use, per-token scoring, both GRPO updates (solver then creator),
checkpoint save, and the JSONL + transcript logs. Confirm the per-iter line
prints finite `Rc/Rs`, `KL`, and `drift`, and that
`runs/spark-smoke.transcript.txt` contains real creator/solver text. Then let a
full run go.

## Performance levers (all in `compute:` unless noted)

| Lever | Default | Effect |
|---|---|---|
| `attn_impl` | `sdpa` | `flash_attention_2` is fastest on Blackwell (needs the wheel); `eager` for debugging. |
| `dtype` | `bfloat16` | Blackwell-native; keep it unless you have a reason. |
| `tf32` | `true` | TF32 matmul/cuDNN — free speedup. |
| `matmul_precision` | `high` | `torch.set_float32_matmul_precision`. |
| `compile` | `false` | `torch.compile` the scoring forward; experimental (adapter-swap recompiles). |
| `grad_checkpointing` | `false` | Activation checkpointing on the backward; only if you OOM. |
| `gen.solver_batch` | (spark: 8) | Batch the K solver attempts — the ideal batch (shared prompt). |
| `train.grpo_microbatch` | (spark: 0) | Trajectories per backward chunk; 0 = whole batch in one graph (128 GB has room). |
| `train.logit_chunk` | (spark: 0) | Chunked LM-head scoring; a 32 GB memory saver, off here. |

Peak GPU memory is available programmatically via
`backend.peak_memory_gb()` / `reset_peak_memory()` (torch: `cuda.max_memory_allocated`).
The `scripts/probe_memory.py` probe remains **mlx-specific**; on the Spark use
the backend's CUDA memory stats instead.

## Checkpoints

Adapter checkpoints keep the same filename pattern
(`checkpoints/adapter_{A,B}_step<N>.safetensors`) and the `--resume-step` flow is
unchanged. The **contents differ by backend** (torch state_dict vs mlx tree), so
a torch run resumes torch checkpoints and an mlx run resumes mlx ones — they are
not interchangeable.

## Not done / future levers

- **vLLM generation.** Would raise rollout throughput substantially, but the
  policy LoRA changes every iteration, so it needs per-step adapter hot-sync
  into the engine. Deferred; native `transformers.generate` is the correct,
  simple v1.
- **NVFP4 inference** of the frozen base for faster rollouts (training stays
  bf16). Possible later optimization; out of scope here.
- **Multi-Spark** (2× via ConnectX) for larger bases — not needed for Qwen3-8B.

## Sources

- NVIDIA DGX Spark product page — https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- DGX Spark User Guide, Hardware Overview — https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- "DGX Spark GB10 CUDA 13.0 Python 3.12 SM_121" (PyTorch Forums) — https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744
- natolambert/dgx-spark-setup (ML training on GB10, CUDA 13, aarch64) — https://github.com/natolambert/dgx-spark-setup
- Kubesimplify, "GB10, Unified Memory, sm_121, and NVFP4" — https://blog.kubesimplify.com/day-3-the-dgx-spark-unpacked-gb10-unified-memory-sm-121-and-the-one-reason-this-hardware-exists
