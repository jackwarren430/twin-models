#!/usr/bin/env python
"""Run the self-play RLVR loop from a config (DESIGN.md §5/§11).

    conda run -n twin-models python scripts/train.py --config configs/tiny.yaml --iters 5

Logs one JSON record per iteration to ``runs/<run-name>.jsonl`` and prints a
one-line-per-iter summary (rewards, solve rate, parse rate, KL, grad norm).
Everything the models actually write also goes to a raw-text transcript at
``runs/<run-name>.transcript.txt`` (tail-able, crash-safe; ``--no-transcript``
to disable) — mini-02's "12 certified problems, all inconsistent" iteration
was undebuggable without one.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.config import Config              # noqa: E402
from twin.log import JsonlLogger, TranscriptLogger  # noqa: E402
from twin.models import Adapters, TwinBase  # noqa: E402
from twin.train import SelfPlayTrainer      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "tiny.yaml"))
    ap.add_argument("--iters", type=int, default=None,
                    help="override train.iters (this is the ABSOLUTE target "
                         "iteration count, not a count of additional iters)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume-step", type=int, default=None,
                    help="resume from checkpoints/adapter_{A,B}_step<N>.safetensors "
                         "and continue training at iteration N (so the role-swap / "
                         "warmup schedule stays consistent). NOTE: optimizer (AdamW) "
                         "moments and the drift baseline reset at the resume point.")
    ap.add_argument("--no-transcript", action="store_true",
                    help="skip the raw-text transcript (runs/<run-name>.transcript.txt)")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    print(f"Loading base: {cfg.model.path}")
    base = TwinBase(cfg.model.path)
    adapters = Adapters.from_config(base.model, cfg.lora)
    print(f"  loaded; rank={cfg.lora.rank} alpha={cfg.lora.alpha} "
          f"scale={cfg.lora.effective_scale:g}; "
          f"{adapters.num_params('A'):,} params/adapter.")

    # Resume: load A/B checkpoints BEFORE constructing the trainer, so its init
    # drift snapshot is taken from the resumed weights (drift then measures
    # movement from the resume point, not from a fresh adapter).
    start_iter = 0
    if args.resume_step is not None:
        start_iter = args.resume_step
        ckpt_dir = Path(cfg.paths.checkpoints)
        for name in adapters.NAMES:
            p = ckpt_dir / f"adapter_{name}_step{start_iter}.safetensors"
            if not p.exists():
                ap.error(f"resume checkpoint not found: {p}")
            adapters.load(name, str(p))
        print(f"  resumed adapters from step {start_iter} "
              f"({ckpt_dir}/adapter_{{A,B}}_step{start_iter}.safetensors)")

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    log_path = Path(cfg.paths.runs) / f"{run_name}.jsonl"
    logger = JsonlLogger(log_path, meta={"config": args.config, "run": run_name,
                                         "resume_step": start_iter})
    transcript = None
    if not args.no_transcript:
        transcript_path = Path(cfg.paths.runs) / f"{run_name}.transcript.txt"
        transcript = TranscriptLogger(
            transcript_path,
            meta={"config": args.config, "run": run_name, "resume_step": start_iter})
        print(f"Transcript: {transcript_path}")
    trainer = SelfPlayTrainer(base, adapters, cfg, logger=logger,
                              transcript=transcript)

    iters = args.iters if args.iters is not None else cfg.train.iters
    if start_iter >= iters:
        ap.error(f"--resume-step {start_iter} >= target iters {iters}: nothing to do")
    print(f"Running iterations {start_iter}..{iters - 1}; logging to {log_path}\n")
    try:
        for it in range(start_iter, iters):
            rec = trainer.run_iteration(it)
            cu, su = rec["creator_update"], rec["solver_update"]
            drift = rec.get("adapter_drift", {})
            print(
                f"[{it:>4}] {rec['creator']}->create {rec['solver']}->solve | "
                f"Rc={rec['creator_reward_mean']:+.3f} Rs={rec['solver_reward_mean']:+.3f} "
                f"solve%={rec['solve_rate_mean']:.2f} parseOK={rec['parse_ok_rate']:.2f} | "
                f"KL c={cu['kl']:.3f} s={su['kl']:.3f} | "
                f"drift A={drift.get('A', 0.0):.3f} B={drift.get('B', 0.0):.3f}"
            )
            ce = cfg.train.checkpoint_every
            if ce and (it + 1) % ce == 0:
                trainer.save_checkpoints(it + 1)
                print(f"       checkpointed at step {it + 1}")
    finally:
        logger.close()
        if transcript is not None:
            transcript.close()
    print(f"\nDone. Log: {log_path}")


if __name__ == "__main__":
    main()
