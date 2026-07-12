#!/usr/bin/env python
"""Run the 21-questions multi-turn loop from a config (twentyq/DESIGN.md).

    conda run --no-capture-output -n twin-models python -u scripts/train_twentyq.py \
        --config configs/twentyq-tiny.yaml --iters 5

Same harness contract as scripts/train.py: one JSON record per iteration to
``runs/<run-name>.jsonl``, a raw-text transcript beside it, checkpoints via
``train.checkpoint_every``, ``--resume-step`` to continue a paused run.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.backends import get_backend              # noqa: E402
from twin.config import Config                      # noqa: E402
from twin.games.twentyq.trainer import TwentyQTrainer  # noqa: E402
from twin.log import JsonlLogger, TranscriptLogger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "twentyq-tiny.yaml"))
    ap.add_argument("--iters", type=int, default=None,
                    help="override train.iters (ABSOLUTE target, not additional)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume-step", type=int, default=None,
                    help="resume from checkpoints/adapter_{A,B}_step<N>.safetensors "
                         "and continue at iteration N (optimizer moments and the "
                         "drift baseline reset at the resume point)")
    ap.add_argument("--no-transcript", action="store_true")
    ap.add_argument("--log-prompts", action="store_true",
                    help="also write input prompts to the transcript")
    # Q8 matrix knobs: drive all arms from ONE base config so they differ only in
    # the varied factors (no config drift), with per-arm checkpoint dirs.
    ap.add_argument("--credit", choices=["broadcast", "per_turn"], default=None,
                    help="override twentyq.credit (Q8 credit-granularity arm)")
    ap.add_argument("--swap-interval", type=int, default=None,
                    help="override roles.swap_interval (0 = no-rotation control)")
    ap.add_argument("--checkpoints-dir", default=None,
                    help="override paths.checkpoints (keep arms from clobbering)")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.log_prompts:
        cfg.train.log_prompts = True
    if args.credit is not None:
        cfg.twentyq.credit = args.credit
    if args.swap_interval is not None:
        cfg.roles.swap_interval = args.swap_interval
    if args.checkpoints_dir is not None:
        cfg.paths.checkpoints = args.checkpoints_dir
    Path(cfg.paths.checkpoints).mkdir(parents=True, exist_ok=True)
    backend = get_backend(cfg.compute.backend)
    print(f"Backend: {backend.name} | loading base: {cfg.model.path}")
    base = backend.load_base(cfg.model, cfg.compute)
    adapters = backend.build_adapters(base, cfg.lora)
    print(f"  loaded; rank={cfg.lora.rank} alpha={cfg.lora.alpha} "
          f"{adapters.num_params('A'):,} params/adapter.")

    start_iter = 0
    if args.resume_step is not None:
        start_iter = args.resume_step
        ckpt_dir = Path(cfg.paths.checkpoints)
        for name in adapters.NAMES:
            p = ckpt_dir / f"adapter_{name}_step{start_iter}.safetensors"
            if not p.exists():
                ap.error(f"resume checkpoint not found: {p}")
            adapters.load(name, str(p))
        print(f"  resumed adapters from step {start_iter}")

    run_name = args.run_name or ("q-" + time.strftime("%Y%m%d-%H%M%S"))
    log_path = Path(cfg.paths.runs) / f"{run_name}.jsonl"
    logger = JsonlLogger(log_path, meta={"config": args.config, "run": run_name,
                                         "mode": "twentyq",
                                         "credit": cfg.twentyq.credit,
                                         "swap_interval": cfg.roles.swap_interval,
                                         "resume_step": start_iter})
    transcript = None
    if not args.no_transcript:
        transcript_path = Path(cfg.paths.runs) / f"{run_name}.transcript.txt"
        transcript = TranscriptLogger(
            transcript_path,
            meta={"config": args.config, "run": run_name, "resume_step": start_iter})
        print(f"Transcript: {transcript_path}")
    trainer = TwentyQTrainer(base, adapters, cfg, logger=logger,
                             transcript=transcript, backend=backend)

    iters = args.iters if args.iters is not None else cfg.train.iters
    if start_iter >= iters:
        ap.error(f"--resume-step {start_iter} >= target iters {iters}: nothing to do")
    print(f"Running iterations {start_iter}..{iters - 1}; logging to {log_path}\n")
    try:
        for it in range(start_iter, iters):
            rec = trainer.run_iteration(it)
            ep = rec["episodes"]
            print(
                f"[{it:>4}] {rec['creator']}->create {rec['solver']}->guess "
                f"cat={rec['category']} | "
                f"Rc={rec['creator_reward_mean']:+.3f} Rs={rec['solver_reward_mean']:+.3f} "
                f"guess%={rec['guess_rate_mean']:.2f} rgrad={rec['r_gradient']:.2f} | "
                f"parse={rec['parse_ok_rate']:.2f} valid={rec['validity_rate']:.2f} | "
                f"eps {ep['guessed']}/{ep['total']} void={ep['void']} "
                f"fmt={ep['format_ended']} | phi={rec['phi_mean']:.2f}"
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
