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
from twin.games.twentyq.schedule import validation_steps  # noqa: E402
from twin.games.twentyq.trainer import TwentyQTrainer  # noqa: E402
from twin.log import JsonlLogger, TranscriptLogger  # noqa: E402
from twin.log import TwentyQTranscriptTree  # noqa: E402


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
    ap.add_argument("--resume-log", default=None,
                    help="JSONL used to restore recent-secret history (defaults "
                         "to runs/<run-name>.jsonl; useful when resuming under a "
                         "new run name)")
    ap.add_argument("--no-transcript", action="store_true")
    ap.add_argument("--log-prompts", action="store_true",
                    help="also write input prompts to the transcript")
    # Q8 matrix knobs: drive all arms from ONE base config so they differ only in
    # the varied factors (no config drift), with per-arm checkpoint dirs.
    ap.add_argument("--credit",
                    choices=["broadcast", "per_turn", "terminal", "ensemble"],
                    default=None,
                    help="override twentyq.credit (terminal is the clean "
                         "no-ensemble control for ensemble)")
    ap.add_argument("--swap-interval", type=int, default=None,
                    help="override roles.swap_interval (0 = no-rotation control)")
    ap.add_argument("--secret-validity", choices=["fail_open", "fail_closed", "off"],
                    default=None,
                    help="override twentyq.secret_validity (fail_open = new default; "
                         "fail_closed = pre-fix posture for the v2 control arm)")
    ap.add_argument("--repeat-handling", choices=["off", "void", "retry"],
                    default=None,
                    help="override twentyq.repeat_handling (off = prompt-and-"
                         "measurement only; void = repeat rollouts get no episodes "
                         "and the repeat-gate reward; retry = void + one masked "
                         "resample with excluded secrets banned at the logits level)")
    # Frozen roles + difficulty dictation (DESIGN §7). Both freeze flags accept
    # an explicit --no- form so a config default can be turned off per-arm.
    ap.add_argument("--freeze-creator", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="override twentyq.freeze_creator: the creator role "
                         "plays and is scored but takes no GRPO update")
    ap.add_argument("--freeze-solver", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="override twentyq.freeze_solver: the solver role plays "
                         "and is scored but takes no GRPO update (also skips "
                         "the ensemble's dense scoring entirely)")
    ap.add_argument("--difficulty-mode", choices=["gradient", "flat"],
                    default=None,
                    help="override twentyq.difficulty_mode (gradient = the "
                         "easy->hard ramp across ranks; flat = one target "
                         "guess rate for every rank)")
    ap.add_argument("--flat-target-rate", type=float, default=None,
                    help="override twentyq.flat_target_rate (flat mode only)")
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
    if args.secret_validity is not None:
        cfg.twentyq.secret_validity = args.secret_validity
    if args.repeat_handling is not None:
        cfg.twentyq.repeat_handling = args.repeat_handling
    if args.freeze_creator is not None:
        cfg.twentyq.freeze_creator = args.freeze_creator
    if args.freeze_solver is not None:
        cfg.twentyq.freeze_solver = args.freeze_solver
    if args.difficulty_mode is not None:
        cfg.twentyq.difficulty_mode = args.difficulty_mode
    if args.flat_target_rate is not None:
        cfg.twentyq.flat_target_rate = args.flat_target_rate
    if args.checkpoints_dir is not None:
        cfg.paths.checkpoints = args.checkpoints_dir
    if cfg.twentyq.freeze_creator and cfg.twentyq.freeze_solver:
        print("WARNING: both roles frozen — this run plays and measures games "
              "but trains nothing.")
    elif ((cfg.twentyq.freeze_creator or cfg.twentyq.freeze_solver)
            and cfg.roles.swap_interval):
        # Freezing is by ROLE: under rotation the frozen role changes hands at
        # every swap, so BOTH adapters still train (each while it plays the
        # unfrozen role) — almost never what a freeze is meant to express.
        frozen = "creator" if cfg.twentyq.freeze_creator else "solver"
        print(f"WARNING: freeze_{frozen}=true with roles.swap_interval="
              f"{cfg.roles.swap_interval}: freezing is BY ROLE, so both "
              "adapters will still be updated (each while it plays the "
              "unfrozen role). Use --swap-interval 0 to freeze one adapter.")
    run_name = args.run_name or ("q-" + time.strftime("%Y%m%d-%H%M%S"))
    log_path = Path(cfg.paths.runs) / f"{run_name}.jsonl"
    resume_log = Path(args.resume_log) if args.resume_log else log_path
    if (args.resume_step is not None and cfg.twentyq.recent_secret_window > 0
            and not resume_log.exists()):
        ap.error(
            f"recent-secret resume log not found: {resume_log} "
            "(reuse --run-name or pass --resume-log)"
        )
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

    logger = JsonlLogger(log_path, meta={"config": args.config, "run": run_name,
                                         "mode": "twentyq",
                                         "credit": cfg.twentyq.credit,
                                         "validation_every": cfg.twentyq.validation_every,
                                         "validation_secret_set": cfg.twentyq.validation_secret_set,
                                         "recent_secret_window": cfg.twentyq.recent_secret_window,
                                         "repeat_handling": cfg.twentyq.repeat_handling,
                                         "generation_batch_size": cfg.twentyq.generation_batch_size,
                                         "ensemble_batch_size": cfg.twentyq.ensemble_batch_size,
                                         "swap_interval": cfg.roles.swap_interval,
                                         "difficulty_mode": cfg.twentyq.difficulty_mode,
                                         "flat_target_rate": cfg.twentyq.flat_target_rate,
                                         "freeze_creator": cfg.twentyq.freeze_creator,
                                         "freeze_solver": cfg.twentyq.freeze_solver,
                                         "resume_step": start_iter})
    reward_logger = None
    reward_log_path = None
    if cfg.twentyq.reward_log:
        reward_log_path = Path(cfg.paths.runs) / f"{run_name}.rewards.jsonl"
        reward_logger = JsonlLogger(reward_log_path, meta={
            "config": args.config, "run": run_name, "mode": "twentyq_rewards",
            "credit": cfg.twentyq.credit,
            "w_ensemble": cfg.twentyq.w_ensemble,
            "generation_batch_size": cfg.twentyq.generation_batch_size,
            "ensemble_batch_size": cfg.twentyq.ensemble_batch_size,
            "swap_interval": cfg.roles.swap_interval,
            "resume_step": start_iter,
        })
        print(f"Reward log: {reward_log_path}")
    transcript = None
    tree = None
    if not args.no_transcript:
        transcript_path = Path(cfg.paths.runs) / f"{run_name}.transcript.txt"
        transcript = TranscriptLogger(
            transcript_path,
            meta={"config": args.config, "run": run_name,
                  "generation_batch_size": cfg.twentyq.generation_batch_size,
                  "ensemble_batch_size": cfg.twentyq.ensemble_batch_size,
                  "resume_step": start_iter})
        print(f"Transcript: {transcript_path}")
        # Kept alongside the flat file (per user): a GRPO-structured folder tree.
        tree_root = Path(cfg.paths.runs) / f"{run_name}.transcript"
        tree = TwentyQTranscriptTree(tree_root, meta={
            "run": run_name, "config": args.config, "mode": "twentyq",
            "credit": cfg.twentyq.credit, "swap_interval": cfg.roles.swap_interval,
            "model": cfg.model.path, "N_secrets": cfg.twentyq.n_secrets,
            "K_episodes": cfg.twentyq.episodes_per_secret,
            "T_max_turns": cfg.twentyq.max_turns,
            "recent_secret_window": cfg.twentyq.recent_secret_window,
            "repeat_handling": cfg.twentyq.repeat_handling,
            "generation_batch_size": cfg.twentyq.generation_batch_size,
            "ensemble_batch_size": cfg.twentyq.ensemble_batch_size,
            "validation_every": cfg.twentyq.validation_every,
            "validation_secret_set": cfg.twentyq.validation_secret_set,
            "difficulty_mode": cfg.twentyq.difficulty_mode,
            "flat_target_rate": cfg.twentyq.flat_target_rate,
            "freeze_creator": cfg.twentyq.freeze_creator,
            "freeze_solver": cfg.twentyq.freeze_solver,
            "resume_step": start_iter})
        print(f"Transcript tree: {tree_root}/")
    trainer = TwentyQTrainer(base, adapters, cfg, logger=logger,
                             transcript=transcript, backend=backend)
    trainer.transcript_tree = tree
    if args.resume_step is not None:
        restored = trainer.restore_recent_secrets(
            resume_log, before_iteration=start_iter)
        print(f"  restored {restored} recent secrets from {resume_log}")

    iters = args.iters if args.iters is not None else cfg.train.iters
    if start_iter >= iters:
        ap.error(f"--resume-step {start_iter} >= target iters {iters}: nothing to do")
    scheduled_validation = set(validation_steps(
        start_iter, iters, cfg.twentyq.validation_every))
    print(f"Running iterations {start_iter}..{iters - 1}; logging to {log_path}\n")
    print(
        "TwentyQ batching: "
        f"generation={cfg.twentyq.generation_batch_size}, "
        f"ensemble={cfg.twentyq.ensemble_batch_size}"
    )
    difficulty = cfg.twentyq.difficulty_mode
    if difficulty == "flat":
        difficulty += f" @ {cfg.twentyq.flat_target_rate:.2f} target guess rate"
    print(
        f"TwentyQ updates: "
        f"creator={'FROZEN' if cfg.twentyq.freeze_creator else 'training'}, "
        f"solver={'FROZEN' if cfg.twentyq.freeze_solver else 'training'} | "
        f"difficulty={difficulty}\n"
    )

    def log_reward_signals(record, *, validation: bool = False):
        if reward_logger is None:
            return
        signals = record["reward_signals"]
        compact = {
            "mode": record["mode"],
            "credit": record.get("credit", cfg.twentyq.credit),
            "ensemble_applied_to_training": (
                not validation and cfg.twentyq.credit == "ensemble"),
            "diagnostic_ensemble": validation,
            "terminal_reward_mean": signals["terminal_mean"],
            "terminal_reward_total": signals["terminal_total"],
            "ensemble_reward_mean": signals["dense_immediate_mean"],
            "ensemble_reward_total": signals["dense_immediate_total"],
            "dense_reward_mean": signals["dense_immediate_mean"],
            "dense_reward_total": signals["dense_immediate_total"],
            "combined_reward_mean": signals["combined_immediate_mean"],
            "combined_return_start_mean": signals["combined_return_start_mean"],
            "n_episodes": signals["n_episodes"],
            "n_steps": signals["n_steps"],
        }
        if validation:
            compact["step"] = record["step"]
            reward_logger.log_validation(compact)
        else:
            compact.update({
                "iter": record["iter"], "category": record["category"],
                "creator": record["creator"], "solver": record["solver"],
            })
            reward_logger.log(compact)

    def run_and_report_validation(step: int):
        val = trainer.run_validation(step)
        log_reward_signals(val, validation=True)
        parts = [
            f"{name}={metrics['guess_rate']:.3f}"
            for name, metrics in val["adapters"].items()
        ]
        sig = val["reward_signals"]
        print(
            f"       validation step {step}: guess_rate "
            f"{' '.join(parts)} | terminal={sig['terminal_mean']:+.3f} "
            f"dense={sig['dense_immediate_mean']:+.3f} "
            f"combined_R0={sig['combined_return_start_mean']:+.3f}"
        )
        return val

    try:
        # True pre-training baseline, followed by completed steps X, 2X, ...
        if 0 in scheduled_validation:
            run_and_report_validation(0)
        for it in range(start_iter, iters):
            rec = trainer.run_iteration(it)
            log_reward_signals(rec)
            ep = rec["episodes"]
            print(
                f"[{it:>4}] {rec['creator']}->create {rec['solver']}->guess "
                f"cat={rec['category']} | "
                f"Rc={rec['creator_reward_mean']:+.3f} Rs={rec['solver_reward_mean']:+.3f} "
                f"guess%={rec['guess_rate_mean']:.2f} rgrad={rec['r_gradient']:.2f} | "
                f"parse={rec['parse_ok_rate']:.2f} valid={rec['validity_rate']:.2f} | "
                f"eps {ep['guessed']}/{ep['total']} "
                f"fmt={ep['format_ended']} | phi={rec['phi_mean']:.2f}"
                f" | mem={rec.get('peak_mem_gb', '?')}G"
            )
            sig = rec["reward_signals"]
            print(
                f"       rewards: terminal={sig['terminal_mean']:+.3f} "
                f"dense={sig['dense_immediate_mean']:+.3f} "
                f"combined={sig['combined_immediate_mean']:+.3f} "
                f"combined_R0={sig['combined_return_start_mean']:+.3f}"
            )
            ce = cfg.train.checkpoint_every
            if ce and (it + 1) % ce == 0:
                trainer.save_checkpoints(it + 1)
                print(f"       checkpointed at step {it + 1}")
            if it + 1 in scheduled_validation:
                run_and_report_validation(it + 1)
    finally:
        logger.close()
        if reward_logger is not None:
            reward_logger.close()
        if transcript is not None:
            transcript.close()
    print(f"\nDone. Log: {log_path}")
    if reward_log_path is not None:
        print(f"Reward log: {reward_log_path}")


if __name__ == "__main__":
    main()
