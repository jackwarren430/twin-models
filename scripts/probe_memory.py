#!/usr/bin/env python
"""Measure the real peak (Metal) memory of a GRPO update at a given rollout
length — BEFORE committing a run to bigger token budgets.

    conda run -n twin-models python scripts/probe_memory.py \\
        --config configs/mini2.yaml --completion-tokens 8192

Why: the backward pass is the memory ceiling, not generation. mini-01 run0
OOMed (SIGKILL) at 4096-token budgets until `grpo_microbatch: 1` bounded the
backward to one rollout (~15GB peak, EXPERIMENTS.md). The dominant transients
(the [T,V] logits and the per-layer activations) scale ~linearly with realized
completion length, which puts an 8192 budget at ~23GB by arithmetic — close
enough to Metal's working-set ceiling on a 32GB box that it must be measured,
not assumed.

This mirrors the trainer's `_grpo` exactly — no-grad reference pass on the
zeroed adapter (per-trajectory mx.eval, as in loop.py), then `grpo_update`
with the config's microbatch — on synthetic token ids, so there is no
generation wait and nothing is saved (the throwaway update trains adapter A
on noise, in-process only).

REFUSES to run while another train/probe process is alive (the probe's peak
plus a live run would OOM the box, and macOS jetsam kills the biggest
process — usually the *training run*). Override with --force at your peril.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx                        # noqa: E402
import mlx.optimizers as optim               # noqa: E402

from twin.config import Config               # noqa: E402
from twin.models import Adapters, TwinBase   # noqa: E402
from twin.rl.grpo import Trajectory, grpo_update  # noqa: E402


def _peak_gb() -> float:
    get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return get_peak() / 2**30


def _other_runs_alive() -> list[str]:
    out = subprocess.run(
        ["pgrep", "-fl", "scripts/(train|probe_memory)\\.py"],
        capture_output=True, text=True,
    ).stdout.strip()
    me = str(os.getpid())
    return [line for line in out.splitlines()
            if line and line.split(None, 1)[0] != me]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "mini2.yaml"))
    ap.add_argument("--completion-tokens", type=int, default=8192,
                    help="realized completion length to probe (the budget you "
                         "are considering; worst case = a rollout that hits it)")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--n-traj", type=int, default=2,
                    help=">=2 exercises the chunk-free-between-microbatches path")
    ap.add_argument("--force", action="store_true",
                    help="run even if another train/probe process is alive")
    args = ap.parse_args()

    others = _other_runs_alive()
    if others and not args.force:
        ap.error("refusing to probe while these are running (the probe's peak "
                 "would OOM the box and macOS may SIGKILL the training run):\n  "
                 + "\n  ".join(others))

    cfg = Config.from_yaml(args.config)
    print(f"Loading base: {cfg.model.path}")
    base = TwinBase(cfg.model.path)
    adapters = Adapters.from_config(base.model, cfg.lora)
    mx.eval(base.model.parameters())
    print(f"  after load: peak {_peak_gb():.2f} GB")

    # Synthetic trajectories: random benign token ids, alternating advantages
    # so the batch is not zero-advantage. Memory does not care about content.
    T, P = args.completion_tokens, args.prompt_tokens
    vocab_lo, vocab_hi = 1000, 20000
    trajs = [
        Trajectory(
            prompt_ids=[int(x) for x in mx.random.randint(vocab_lo, vocab_hi, (P,))],
            completion_ids=[int(x) for x in mx.random.randint(vocab_lo, vocab_hi, (T,))],
            advantage=1.0 if i % 2 == 0 else -1.0,
        )
        for i in range(args.n_traj)
    ]
    print(f"Probing: {args.n_traj} trajectories of {P}+{T} tokens, "
          f"microbatch={cfg.train.grpo_microbatch}, kl_beta={cfg.train.kl_beta}")

    # Reference pass (frozen base = zeroed adapter), exactly as loop.py::_grpo.
    adapters.activate("base")
    for t in trajs:
        t.ref_logprobs = base.completion_logprobs(t.prompt_ids, t.completion_ids)
        mx.eval(t.ref_logprobs)
    print(f"  after reference pass: peak {_peak_gb():.2f} GB")

    # Policy update on adapter A (throwaway — nothing is saved).
    adapters.activate("A")
    optimizer = optim.AdamW(learning_rate=cfg.train.learning_rate)
    metrics = grpo_update(
        base.model, optimizer, trajs,
        lambda model, p, c: base.completion_logprobs(p, c),
        kl_beta=cfg.train.kl_beta,
        grad_clip=cfg.train.grad_clip,
        microbatch_size=cfg.train.grpo_microbatch,
    )
    peak = _peak_gb()
    print(f"  after grpo_update: peak {peak:.2f} GB "
          f"(loss={metrics['loss']:.4f} n_tokens={metrics['n_tokens']})")
    print(f"\nPEAK {peak:.2f} GB at completion length {T} "
          f"(budget verdict is yours: leave several GB for macOS + generation KV).")


if __name__ == "__main__":
    main()
