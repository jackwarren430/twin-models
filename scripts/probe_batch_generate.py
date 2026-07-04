"""Measure batched-decode throughput of TwinBase.generate_batch (Sprint 8).

Loads the real base and samples the same solver-style prompt at batch sizes
1/2/4/8, reporting tokens/sec and peak memory per setting. Decode on Apple
silicon is memory-bandwidth-bound, so a batch of B should approach B× the
sequential rate until compute saturates — this probe tells us what
``gen.solver_batch`` to set in mini-05.

DO NOT run while a training run is live (it loads a second 6.2GB base copy).

    conda run --no-capture-output -n twin-models python -u scripts/probe_batch_generate.py \
        --config configs/mini4.yaml --max-tokens 512
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from twin.config import Config  # noqa: E402
from twin.models import TwinBase  # noqa: E402
from twin.prompts import solver_system, solver_user  # noqa: E402
from twin.problems.schema import Problem  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "mini4.yaml"))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--batches", default="1,2,4,8")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    print(f"Loading base: {cfg.model.path}")
    base = TwinBase(cfg.model.path)

    problem = Problem(
        statement="Solve the system: 3a + 2b = 22 and 2a + 3b = 21.",
        difficulty=0.5,
    )
    prompt = base.render(
        solver_user(problem), system=solver_system(),
        enable_thinking=cfg.model.enable_thinking,
    )

    print(f"{'batch':>6} {'wall s':>8} {'tokens':>8} {'tok/s':>8} {'peak GB':>8}")
    for b in [int(x) for x in args.batches.split(",")]:
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        results = base.generate_batch(
            [prompt] * b,
            max_tokens=args.max_tokens,
            temp=cfg.gen.solver_temp,
            top_p=cfg.gen.top_p,
            completion_batch_size=b,
            seed=0,
        )
        wall = time.perf_counter() - t0
        n_tok = sum(len(r.completion_tokens) for r in results)
        peak = mx.get_peak_memory() / 1e9
        print(f"{b:>6} {wall:>8.1f} {n_tok:>8} {n_tok / wall:>8.1f} {peak:>8.1f}")
        # sanity: every completion decodes and ends sensibly
        for r in results:
            assert r.completion_tokens, "empty completion"


if __name__ == "__main__":
    main()
