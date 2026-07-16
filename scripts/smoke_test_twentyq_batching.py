#!/usr/bin/env python
"""Benchmark safe TwentyQ generation/ensemble batch sizes on the real backend.

This is a throughput smoke test, not training: it loads the v4 policy + LoRA
surface, exercises representative guesser prompts, then (optionally) loads the
frozen ensemble and checks batched history scores against the scalar baseline.
OOMs are reported and stop larger-size trials instead of continuing in a
potentially poisoned CUDA process.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.backends import get_backend  # noqa: E402
from twin.config import Config  # noqa: E402
from twin.games.twentyq.ensemble_reward import EnsembleReward  # noqa: E402
from twin.games.twentyq.prompts import GUESSER_SYSTEM, guesser_user  # noqa: E402


def _sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _memory(torch) -> float:
    return (torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else 0.0)


def _histories(n: int):
    questions = [
        "Is it alive?", "Is it commonly kept by people?", "Does it live in water?",
        "Is it larger than a cat?", "Does it have fur?", "Can it fly?",
        "Is it native to Africa?", "Does it eat meat?", "Is it nocturnal?",
        "Does it have four legs?", "Is it endangered?", "Is it a mammal?",
    ]
    out = []
    for i in range(n):
        length = 3 + (i % 10)
        out.append([(q, "YES" if (i + j) % 3 else "NO")
                    for j, q in enumerate(questions[:length])])
    return out


def generation_benchmark(cfg, sizes, n_prompts):
    import torch

    backend = get_backend(cfg.compute.backend)
    base = backend.load_base(cfg.model, cfg.compute)
    adapters = backend.build_adapters(base, cfg.lora)
    adapters.activate("B")
    histories = _histories(n_prompts)
    prompts = [base.render(
        guesser_user("animal", history, len(history), cfg.twentyq.max_turns),
        system=GUESSER_SYSTEM,
        enable_thinking=cfg.twentyq.guesser_thinking,
    ) for history in histories]

    # One short warm-up absorbs lazy CUDA/kernel setup before timed trials.
    base.generate_batch(
        prompts[:1], max_tokens=8, temp=0.0, top_p=cfg.gen.top_p,
        completion_batch_size=1)
    _sync(torch)
    results = []
    for size in sizes:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            generations = base.generate_batch(
                prompts,
                max_tokens=64,
                temp=cfg.gen.solver_temp,
                top_p=cfg.gen.top_p,
                seed=cfg.train.seed,
                completion_batch_size=size,
            )
            _sync(torch)
        except torch.OutOfMemoryError as exc:
            row = {"component": "generation", "batch_size": size,
                   "status": "oom", "error": str(exc)[:200]}
            print(json.dumps(row), flush=True)
            results.append(row)
            break
        elapsed = time.perf_counter() - started
        tokens = sum(len(g.completion_tokens) for g in generations)
        row = {
            "component": "generation", "batch_size": size, "status": "ok",
            "prompts": len(prompts), "tokens": tokens,
            "seconds": round(elapsed, 4),
            "prompts_per_second": round(len(prompts) / elapsed, 4),
            "tokens_per_second": round(tokens / elapsed, 4),
            "peak_memory_gb": round(_memory(torch), 3),
        }
        print(json.dumps(row), flush=True)
        results.append(row)
    return base, adapters, results


def full_resident_generation_smoke(base, cfg, batch_size: int, n_prompts: int):
    """One policy-decode trial while all ensemble models remain resident."""
    import torch

    histories = _histories(n_prompts)
    prompts = [base.render(
        guesser_user("animal", history, len(history), cfg.twentyq.max_turns),
        system=GUESSER_SYSTEM,
        enable_thinking=cfg.twentyq.guesser_thinking,
    ) for history in histories]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        generations = base.generate_batch(
            prompts, max_tokens=64, temp=cfg.gen.solver_temp,
            top_p=cfg.gen.top_p, seed=cfg.train.seed,
            completion_batch_size=batch_size)
        _sync(torch)
    except torch.OutOfMemoryError as exc:
        row = {
            "component": "generation_full_resident",
            "batch_size": batch_size, "status": "oom",
            "error": str(exc)[:200],
        }
        print(json.dumps(row), flush=True)
        return row
    elapsed = time.perf_counter() - started
    tokens = sum(len(g.completion_tokens) for g in generations)
    row = {
        "component": "generation_full_resident", "batch_size": batch_size,
        "status": "ok", "prompts": len(prompts), "tokens": tokens,
        "seconds": round(elapsed, 4),
        "tokens_per_second": round(tokens / elapsed, 4),
        "peak_memory_gb": round(_memory(torch), 3),
    }
    print(json.dumps(row), flush=True)
    return row


def ensemble_benchmark(cfg, sizes, n_histories):
    import torch

    ensemble = EnsembleReward.load(
        cfg.twentyq.ensemble_models or None,
        device=cfg.twentyq.ensemble_device,
        dtype=cfg.twentyq.ensemble_dtype,
    )
    histories = _histories(n_histories)
    names = ["octopus", "elephant", "snow leopard", "honeybee"]
    answers = [names[i % len(names)] for i in range(n_histories)]

    baseline = None
    results = []
    for size in sizes:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            scores = ensemble.score_histories(
                histories, answers, batch_size=size)
            _sync(torch)
        except torch.OutOfMemoryError as exc:
            row = {"component": "ensemble", "batch_size": size,
                   "status": "oom", "error": str(exc)[:200]}
            print(json.dumps(row), flush=True)
            results.append(row)
            break
        elapsed = time.perf_counter() - started
        values = [score.score for score in scores]
        if baseline is None:
            baseline = values
        max_delta = max(abs(a - b) for a, b in zip(values, baseline))
        row = {
            "component": "ensemble", "batch_size": size, "status": "ok",
            "histories": len(histories), "seconds": round(elapsed, 4),
            "histories_per_second": round(len(histories) / elapsed, 4),
            "speedup_vs_batch1": round(
                results[0]["seconds"] / elapsed, 4) if results else 1.0,
            "max_abs_score_delta": round(max_delta, 7),
            "peak_memory_gb": round(_memory(torch), 3),
        }
        print(json.dumps(row), flush=True)
        results.append(row)
    return ensemble, results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/twentyq-full-v4.yaml"))
    ap.add_argument("--component", choices=["generation", "ensemble", "all"],
                    default="all")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--generation-prompts", type=int, default=16)
    ap.add_argument("--ensemble-histories", type=int, default=16)
    ap.add_argument("--json", default=None, help="optional combined result path")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    payload = {"config": args.config, "batch_sizes": sizes}
    keepalive = []
    if args.component in ("generation", "all"):
        base, adapters, rows = generation_benchmark(
            cfg, sizes, args.generation_prompts)
        keepalive.extend([base, adapters])
        payload["generation"] = rows
    if args.component in ("ensemble", "all"):
        ensemble, rows = ensemble_benchmark(
            cfg, sizes, args.ensemble_histories)
        keepalive.append(ensemble)
        payload["ensemble"] = rows
        if args.component == "all":
            payload["generation_full_resident"] = full_resident_generation_smoke(
                base, cfg, max(sizes), args.generation_prompts)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
