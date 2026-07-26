#!/usr/bin/env python3
"""Re-score saved adapter checkpoints against a validation set, offline.

Why this exists. A run's validation series is measured against whatever set was
configured when it launched, and that set must never be swapped mid-run: the
pre-registered criterion and comparability with the v1..v7 series both depend
on finishing on the instrument you started with. But when the instrument turns
out to be the limiting factor — validation-v2 has 15 of 24 secrets at 0/8 for
every v8 checkpoint, so its dynamic range is ~7 secrets — the run's null result
is uninterpretable, and the only honest way to learn anything more is to replay
the SAVED checkpoints against a better set afterwards.

That replay is a SECONDARY analysis and must be labelled as one wherever it is
reported. Selecting a new instrument after seeing the first one's answer is the
goalpost move a pre-registered criterion exists to prevent; the defence is that
the primary number still stands on validation-v2, and that the replacement set
is built by a procedure (measured band, base policy only) that never consults
the trained adapters.

Design notes:

* Measurement is ``TwentyQTrainer.run_validation`` itself, not a reimplemention.
  A private copy of the episode loop here would silently drift from the one
  that produced the in-run numbers, and the whole value of a re-score is that
  the two are the same measurement.
* Adapter A in the v8 lineage is FROZEN, so its re-scored series is a control:
  it shows what this instrument's noise looks like across steps with no policy
  change at all. Any B movement smaller than A's spread is not a result.
* The RNG is reseeded identically before every step, so all checkpoints face
  the same sampled games (common random numbers). In-run validation cannot do
  this — it must not disturb the training stream — which makes this replay
  strictly more sensitive than the series it is re-scoring.
* Step 0 needs no file: adapters are zero-init, so a freshly built adapter IS
  the base model, which is exactly the step-0 policy.

    python scripts/rescore_checkpoints.py \\
        --config      configs/twentyq-v8-bank-solver.yaml \\
        --checkpoints checkpoints/twentyq-v8-bank \\
        --set         data/twentyq-validation-v3.json \\
        --steps       all \\
        --out         tq-runs/v8-rescore-v3.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.backends import get_backend  # noqa: E402
from twin.config import Config  # noqa: E402
from twin.games.twentyq.trainer import (  # noqa: E402
    TwentyQTrainer,
    load_validation_secret_set,
)
from twin.log.jsonl import JsonlLogger  # noqa: E402


def discover_steps(ckpt_dir: Path) -> list[int]:
    """Steps with a checkpoint for EVERY adapter, plus the implicit step 0."""
    per_name: dict[str, set[int]] = {}
    for p in ckpt_dir.glob("adapter_*_step*.safetensors"):
        name, _, tail = p.stem[len("adapter_"):].partition("_step")
        try:
            per_name.setdefault(name, set()).add(int(tail))
        except ValueError:
            continue
    if not per_name:
        return [0]
    common = set.intersection(*per_name.values())
    return sorted({0} | common)


def gpu_is_busy() -> str | None:
    """Name of a live training unit, if one is holding the GPU.

    Re-scoring is heavy enough to roughly double a concurrent run's wall time,
    and on this box memory is unified, so a second model is not just slow but a
    genuine OOM risk for the run already in flight."""
    try:
        out = subprocess.run(
            ["systemctl", "--user", "list-units", "--type=service",
             "--state=running", "--no-legend", "--plain"],
            capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        unit = line.split()[0] if line.split() else ""
        if unit.startswith("q-") and unit.endswith(".service"):
            return unit
    return None


def already_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") == "validation" and "step" in row:
            done.add(int(row["step"]))
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="secret_set", required=True,
                    help="validation secret set to score against")
    # Deliberately required rather than defaulted to cfg.paths.checkpoints.
    # Runs override the checkpoint directory on the command line (v8 used
    # --checkpoints-dir checkpoints/twentyq-v8-bank against a config that says
    # checkpoints/twentyq) and nothing writes that override back into the
    # config, so the "obvious" default points at a DIFFERENT run's adapters and
    # would produce a complete, plausible, wrong series.
    ap.add_argument("--checkpoints", type=Path, required=True,
                    help="directory holding adapter_{A,B}_step<N>.safetensors")
    ap.add_argument("--steps", default="all",
                    help="'all' or a comma-separated list, e.g. 0,20,40,60")
    ap.add_argument("--episodes", type=int, default=None,
                    help="override twentyq.validation_episodes (K per secret)")
    ap.add_argument("--out", type=Path, required=True,
                    help="JSONL output; existing steps are skipped (resume)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="score even while a q-* training unit is running")
    args = ap.parse_args()

    busy = None if args.allow_concurrent else gpu_is_busy()
    if busy:
        print(f"REFUSING: {busy} is running. Re-scoring would contend for "
              f"unified memory and roughly double that run's wall time. "
              f"Wait for it, or pass --allow-concurrent.", file=sys.stderr)
        return 1

    cfg = Config.from_yaml(args.config)
    cfg.twentyq.validation_secret_set = args.secret_set
    if args.episodes is not None:
        cfg.twentyq.validation_episodes = args.episodes
    ckpt_dir = args.checkpoints

    meta, secrets = load_validation_secret_set(args.secret_set)
    available = discover_steps(ckpt_dir)
    if args.steps == "all":
        steps = available
    else:
        steps = sorted({int(s) for s in args.steps.split(",") if s.strip()})
        missing = [s for s in steps if s not in available]
        if missing:
            print(f"no checkpoint for steps {missing} in {ckpt_dir} "
                  f"(have {available})", file=sys.stderr)
            return 1

    done = already_done(args.out)
    todo = [s for s in steps if s not in done]
    if done:
        print(f"resuming: {sorted(done)} already scored in {args.out}")
    if not todo:
        print("nothing to do")
        return 0

    k = max(1, int(cfg.twentyq.validation_episodes))
    n_adapters = 2
    print(f"re-scoring {len(todo)} steps {todo} on "
          f"{meta.get('name', args.secret_set)} "
          f"({len(secrets)} secrets x {k} episodes x {n_adapters} adapters "
          f"= {len(todo) * len(secrets) * k * n_adapters} episodes)")

    backend = get_backend(cfg.compute.backend)
    print(f"loading base: {cfg.model.path}", flush=True)
    base = backend.load_base(cfg.model, cfg.compute)
    adapters = backend.build_adapters(base, cfg.lora)

    logger = JsonlLogger(args.out, meta={
        "mode": "twentyq_rescore",
        "config": args.config,
        "checkpoints": str(ckpt_dir),
        "validation_secret_set": args.secret_set,
        "validation_episodes": k,
        "model": cfg.model.path,
        "seed": args.seed,
        # The provenance that makes this readable a month from now: which run's
        # checkpoints, scored on which set, and that it is a replay rather than
        # the run's own pre-registered series.
        "secondary_analysis": True,
    })
    trainer = TwentyQTrainer(base, adapters, cfg, logger=logger,
                             transcript=None, backend=backend)
    trainer.transcript_tree = None

    started = time.time()
    for i, step in enumerate(todo):
        for name in adapters.NAMES:
            p = ckpt_dir / f"adapter_{name}_step{step}.safetensors"
            if step == 0 and not p.exists():
                continue  # zero-init adapter == base == the step-0 policy
            adapters.load(name, str(p))
        # Common random numbers: every step faces the same sampled games, so a
        # step-to-step difference is policy, not draw.
        backend.seed(args.seed)
        record = trainer.run_validation(step)
        rates = {n: m["guess_rate"] for n, m in record["adapters"].items()}
        elapsed = time.time() - started
        eta = elapsed / (i + 1) * (len(todo) - i - 1)
        print(f"[{i + 1}/{len(todo)}] step {step}: "
              + " ".join(f"{n}={r:.3f}" for n, r in rates.items())
              + f" | {elapsed / 60:.1f}m elapsed, ~{eta / 60:.0f}m left",
              flush=True)
    logger.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
