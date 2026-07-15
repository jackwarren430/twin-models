#!/usr/bin/env python
"""Play ONE fully-instrumented 21-questions game (K = 1, one secret / G_c = 1)
and print EVERYTHING: every system + user prompt, every creator/solver/answerer
completion, and — at every step — the ensemble reward.

This is the end-to-end exercise of the new dense ensemble reward
(games/twentyq/ensemble_reward.py) wired ALONGSIDE the sparse terminal reward:
the frozen base-LLM ensemble scores the secret's log-prob at each Q/A history
prefix, and the per-turn shaping is

    r_t = w_ensemble · (gamma · score(history_t) - score(history_{t-1}))

with score(history_0) the empty-history baseline. The last turn additionally
carries the sparse episode reward (w_guess / efficiency / format). Reward-to-go
R_t = r_t + gamma * R_{t+1} is what a per-turn solver trajectory would train on.

    # DGX Spark (torch/CUDA) — configs/twentyq-tiny.yaml sets compute.backend: torch
    .venv/bin/python -u scripts/twentyq_single_game.py --config configs/twentyq-tiny.yaml

    # cheaper: score with a subset of the ensemble and force a known secret
    .venv/bin/python -u scripts/twentyq_single_game.py \
        --config configs/twentyq-tiny.yaml \
        --ensemble-models gemma-4-e2b qwen3-4b --secret elephant --category animal

    # save the transcript to a file AND let the players think (show the reasoning)
    .venv/bin/python -u scripts/twentyq_single_game.py --config configs/twentyq-tiny.yaml \
        --thinking --out runs/single_game.txt

Prompts are shown as the fully rendered chat template (special tokens included),
and completions are decoded with special tokens kept, so a <think>/thought block
and the turn terminators are visible.
"""

import argparse
import atexit
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin.backends import get_backend                       # noqa: E402
from twin.config import Config                               # noqa: E402
from twin.games.twentyq.ensemble_reward import (             # noqa: E402
    DEFAULT_SYSTEM as ENSEMBLE_SYSTEM,
    ENSEMBLE_MODELS,
    EnsembleReward,
    format_qa_history,
)
from twin.games.twentyq.episode import run_episode           # noqa: E402
from twin.games.twentyq.judge import judge_secret_validity   # noqa: E402
from twin.games.twentyq.prompts import (                     # noqa: E402
    ANSWERER_SYSTEM,
    CREATOR_SYSTEM,
    GUESSER_SYSTEM,
    answerer_user,
    creator_secret_user,
    guesser_user,
)
from twin.games.twentyq.rewards import (                     # noqa: E402
    ensemble_shaped_returns,
    episode_reward,
    history_states,
    per_turn_secret_trajectories,
)
from twin.games.twentyq.schema import Secret, parse_secret   # noqa: E402
from twin.games.twentyq.trainer import TwentyQTrainer        # noqa: E402

RULE = "=" * 100
SUB = "-" * 100


class _Tee:
    """Duplicate everything written to stdout into a file as well (--out)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def _banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}", flush=True)


def _show_prompt(base, tag: str, system: str, user: str, *,
                 enable_thinking: bool) -> None:
    """Print the EXACT string the model conditions on — the chat template with
    all special tokens (<bos>, <start_of_turn>/<end_of_turn>, the generation
    prompt) rendered in, not the bare system/user text."""
    rendered = base.render(user, system=system, enable_thinking=enable_thinking)
    print(f"\n[{tag}] RENDERED PROMPT (chat template — special tokens shown, "
          f"enable_thinking={enable_thinking}):\n{rendered}", flush=True)


def _show_completion(base, tag: str, gen) -> None:
    """Print the completion with special tokens kept (decoded straight from the
    sampled ids) — so any <think>/thought block and the turn terminator are
    visible, not stripped."""
    raw = base.tokenizer.decode(gen.completion_tokens, skip_special_tokens=False)
    print(f"\n[{tag}] COMPLETION (special tokens shown):\n{raw}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "twentyq-tiny.yaml"))
    ap.add_argument("--category", default="animal")
    ap.add_argument("--secret", default=None,
                    help="skip the creator rollout and force this secret string")
    ap.add_argument("--difficulty", type=float, default=0.5)
    ap.add_argument("--target-rate", type=float, default=0.5)
    ap.add_argument("--max-turns", type=int, default=None,
                    help="override twentyq.max_turns for this single game")
    ap.add_argument("--gamma", type=float, default=None,
                    help="override twentyq.gamma (per-turn discount)")
    ap.add_argument("--w-ensemble", type=float, default=None,
                    help="override twentyq.w_ensemble (shaping scale)")
    ap.add_argument("--ensemble-models", nargs="*", default=None,
                    help=f"subset of {list(ENSEMBLE_MODELS)} (default: all)")
    ap.add_argument("--ensemble-device", default="cuda")
    ap.add_argument("--no-validity", action="store_true",
                    help="skip the judge validity gate (saves one base call)")
    ap.add_argument("--out", default=None,
                    help="also write all output to this file (tee)")
    ap.add_argument("--thinking", action="store_true",
                    help="let the creator/guesser/answerer THINK before answering "
                         "and show the trace; raises the per-call token budget so "
                         "the reasoning doesn't truncate the contract line")
    ap.add_argument("--think-budget", type=int, default=1024,
                    help="per-call max_tokens when --thinking (default 1024)")
    args = ap.parse_args()

    if args.out:
        real_stdout = sys.stdout
        fh = open(args.out, "w")
        sys.stdout = _Tee(real_stdout, fh)

        def _restore_and_close():
            # Restore the real stdout BEFORE closing the file, so the interpreter's
            # final stream flush never touches the now-closed file (that ordering
            # bug exits 120). atexit runs before the std-stream flush at shutdown.
            sys.stdout = real_stdout
            fh.flush()
            fh.close()

        atexit.register(_restore_and_close)
        print(f"(teeing all output to {args.out})", flush=True)

    cfg = Config.from_yaml(args.config)
    qcfg = cfg.twentyq
    if args.max_turns is not None:
        qcfg.max_turns = args.max_turns
    if args.gamma is not None:
        qcfg.gamma = args.gamma
    if args.w_ensemble is not None:
        qcfg.w_ensemble = args.w_ensemble
    if args.thinking:
        # Turn thinking ON for the three player roles and give the reasoning room
        # (the q-shakeout lesson: a tight budget truncates mid-<think> into an
        # empty contract line). The ensemble scorer is unaffected (it always
        # scores with thinking OFF).
        b = args.think_budget
        qcfg.creator_thinking = qcfg.guesser_thinking = qcfg.answerer_thinking = True
        qcfg.secret_max_tokens = max(qcfg.secret_max_tokens, b)
        qcfg.question_max_tokens = max(qcfg.question_max_tokens, b)
        qcfg.answer_max_tokens = max(qcfg.answer_max_tokens, b)
    gamma, w_ens = qcfg.gamma, qcfg.w_ensemble
    category = args.category

    # --- policy model + trainer (for its render / generate / judge closures) ---
    backend = get_backend(cfg.compute.backend)
    print(f"Backend: {backend.name} | loading base: {cfg.model.path}", flush=True)
    base = backend.load_base(cfg.model, cfg.compute)
    adapters = backend.build_adapters(base, cfg.lora)
    trainer = TwentyQTrainer(base, adapters, cfg, logger=None,
                             transcript=None, backend=backend)
    assign = trainer.roles.assignment(0)     # creator / solver adapter names
    creator, solver = assign.creator, assign.solver
    print(f"adapters: creator={creator} guesser={solver} | "
          f"category={category} | max_turns={qcfg.max_turns} | "
          f"gamma={gamma} w_ensemble={w_ens}", flush=True)

    # --- creator: one secret (rank 0 of 1) -------------------------------------
    if args.secret:
        secret = Secret(secret=args.secret, category=category,
                        difficulty=args.difficulty, notes="(forced via --secret)")
        _banner("CREATOR — secret forced via --secret (rollout skipped)")
        print(f"secret = {secret.secret!r} (category={secret.category})", flush=True)
    else:
        _banner("CREATOR — pick one secret")
        cuser = creator_secret_user(category, 0, 1, args.difficulty,
                                    args.target_rate, previous=None)
        _show_prompt(base, "creator", CREATOR_SYSTEM, cuser,
                     enable_thinking=qcfg.creator_thinking)
        gen = trainer._generate(
            creator, CREATOR_SYSTEM, cuser,
            max_tokens=qcfg.secret_max_tokens, temp=cfg.gen.creator_temp,
            enable_thinking=qcfg.creator_thinking,
        )
        _show_completion(base, "creator", gen)
        secret = parse_secret(gen.text, default_category=category)
        secret.difficulty = float(args.difficulty)
        print(f"\n[creator] PARSED secret = {secret.secret!r} "
              f"(category={secret.category}, difficulty={secret.difficulty})", flush=True)

    if not args.no_validity:
        _banner("JUDGE — secret validity gate")
        valid = judge_secret_validity(secret, trainer._grade).correct
        print(f"\nvalid = {valid}", flush=True)

    # --- instrumented players --------------------------------------------------
    def guesser_fn(qa_pairs, turn_index):
        _banner(f"SOLVER (guesser) — turn {turn_index}")
        user = guesser_user(category, qa_pairs, turn_index, qcfg.max_turns)
        _show_prompt(base, "guesser", GUESSER_SYSTEM, user,
                     enable_thinking=qcfg.guesser_thinking)
        gen = trainer._generate(
            solver, GUESSER_SYSTEM, user,
            max_tokens=qcfg.question_max_tokens, temp=cfg.gen.solver_temp,
            enable_thinking=qcfg.guesser_thinking,
        )
        _show_completion(base, "guesser", gen)
        return gen

    def answerer_fn(question, qa_pairs):
        _banner("CREATOR (answerer) — reply to the question")
        user = answerer_user(secret.secret, secret.category, question)
        _show_prompt(base, "answerer", ANSWERER_SYSTEM, user,
                     enable_thinking=qcfg.answerer_thinking)
        gen = trainer._generate(
            creator, ANSWERER_SYSTEM, user,
            max_tokens=qcfg.answer_max_tokens, temp=cfg.gen.oracle_temp,
            enable_thinking=qcfg.answerer_thinking,
        )
        _show_completion(base, "answerer", gen)
        return gen.text

    # --- play the single episode ----------------------------------------------
    _banner(f"EPISODE — playing 1 game (K=1), secret={secret.secret!r}")
    ep = run_episode(guesser_fn, answerer_fn, secret, max_turns=qcfg.max_turns)
    _banner(f"EPISODE OVER — ended={ep.ended!r} guessed={ep.guessed} "
            f"turns_used={ep.turns_used}")
    for t in ep.turns:
        print(f"  turn {t.index}: [{t.kind}] {t.content!r} -> {t.answer}", flush=True)

    # --- ensemble reward at each step -----------------------------------------
    _banner("ENSEMBLE REWARD — score(history_t) at every prefix")
    ens = EnsembleReward.load(args.ensemble_models or None,
                              device=args.ensemble_device, dtype=qcfg.ensemble_dtype)
    states = history_states(ep)              # [Φ_0 (empty), ... Φ_T]
    scores = [ens.score_history(s, secret.secret, system=ENSEMBLE_SYSTEM)
              for s in states]
    potentials = [s.score for s in scores]

    for i, (state, sc) in enumerate(zip(states, scores)):
        label = "history_0 (empty baseline)" if i == 0 else f"history_{i} (after turn {i - 1})"
        print(f"\n{SUB}\nΦ_{i} — {label}\n{SUB}", flush=True)
        print(f"history:\n{format_qa_history(state)}", flush=True)
        print(f"\nscore(history_{i}) = {sc.score:+.4f}   "
              f"per-model: " + ", ".join(f"{k}={v:+.3f}" for k, v in sc.per_model.items()),
              flush=True)

    # sparse terminal reward (phi_final=None: the ensemble supplies the dense signal)
    term = episode_reward(
        qcfg, guessed=ep.guessed, turns_used=ep.turns_used,
        max_turns=qcfg.max_turns, phi_final=None,
        format_fail=(ep.ended == "format"),
    )
    returns = ensemble_shaped_returns(potentials, term.total,
                                      gamma=gamma, scale=w_ens)

    _banner("PER-TURN REWARD")
    print(f"sparse terminal reward (added to the LAST turn) = {term.total:+.4f}  "
          f"(guessed={term.guessed} r_eff={term.r_efficiency:.3f} "
          f"format_fail={term.format_fail})\n", flush=True)
    print(f"{'turn':>4}  {'Φ_t':>9}  {'Φ_t+1':>9}  {'shaping r_t':>14}  "
          f"{'+terminal':>11}  {'return R_t':>11}", flush=True)
    print(SUB, flush=True)
    n_turns = len(potentials) - 1
    for t in range(n_turns):
        shaping = w_ens * (gamma * potentials[t + 1] - potentials[t])
        term_add = term.total if t == n_turns - 1 else 0.0
        print(f"{t:>4}  {potentials[t]:>+9.4f}  {potentials[t + 1]:>+9.4f}  "
              f"{shaping:>+14.4f}  {term_add:>+11.4f}  {returns[t]:>+11.4f}", flush=True)

    # trajectory view (advantages are 0 with a single episode — nothing to
    # baseline against; shown to confirm the wiring the trainer would use).
    trajs = per_turn_secret_trajectories([ep], [returns],
                                         adv_mode=cfg.train.adv_mode)
    _banner("SOLVER TRAJECTORIES (what the trainer would push to GRPO)")
    print("(advantage = 0 with K=1: a per-turn group of one mean-centers to zero)\n",
          flush=True)
    for tr in trajs:
        print(f"  turn {tr.meta['turn']} [{tr.meta['kind']}]  "
              f"reward(R_t)={tr.reward:+.4f}  advantage={tr.advantage:+.4f}", flush=True)
    if not trajs:
        print("  (no trained turns — scripted/answer-only or empty episode)", flush=True)


if __name__ == "__main__":
    main()
