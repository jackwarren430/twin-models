"""21-questions reward + advantage wiring (twentyq/DESIGN.md §2.5, Sprint Q4).

Pure arithmetic, RewardEngine-style: the trainer measures (episode outcomes,
judge Φ, void flags) and this module turns them into scalars and grouped
:class:`Trajectory` lists. Nothing here runs a model.

The creator side deliberately has almost no code: secrets are adapted into a
:class:`ProblemSuite` (statement = the secret, difficulty = the dictated rank)
so ``RewardEngine.creator_reward`` / ``creator_problem_rewards`` — the whole
calibration-fit apparatus, scored-fraction scaling, expected_n semantics —
apply verbatim with per-secret guess rates standing in for solve rates.
"""

from dataclasses import dataclass

from twin.config import TwentyQConfig
from twin.games.twentyq.episode import Episode
from twin.games.twentyq.schema import Secret
from twin.problems.schema import Problem, ProblemSuite
from twin.rl import Trajectory
from twin.rl.core import group_advantages


@dataclass
class QEpisodeReward:
    total: float
    guessed: bool
    r_efficiency: float          # raw 0..1, paid only on guessed episodes
    phi_final: float | None      # judge closeness of the final state (None = no signal)
    format_fail: bool


def episode_reward(
    cfg: TwentyQConfig,
    *,
    guessed: bool,
    turns_used: int,
    max_turns: int,
    phi_final: float | None,
    format_fail: bool = False,
) -> QEpisodeReward:
    """One episode's scalar (DESIGN §2.5)::

        R = w_guess·guessed
          + w_efficiency·(1 − turns_used/T_max)·guessed
          + w_close·Φ(final)·(1 − guessed)
          − w_format·format_fail

    Efficiency is gated on success (a fast wrong episode earns nothing from
    speed) and Φ partial credit is paid only on failures (a win is already a
    win — Φ would just add noise on top of w_guess). With the default weights
    (w_close < w_guess) the worst win, w_guess at full budget, strictly beats
    the best failure, w_close·1. Φ None (judge gave no signal) degrades to the
    bare binary outcome."""
    r_efficiency = 0.0
    total = 0.0
    if guessed:
        total += cfg.w_guess
        if max_turns > 0:
            r_efficiency = min(1.0, max(0.0, 1.0 - turns_used / max_turns))
        total += cfg.w_efficiency * r_efficiency
    elif phi_final is not None:
        total += cfg.w_close * phi_final
    if format_fail:
        total -= cfg.w_format
    return QEpisodeReward(
        total=total, guessed=bool(guessed), r_efficiency=r_efficiency,
        phi_final=phi_final, format_fail=bool(format_fail),
    )


def secret_turn_trajectories(
    episodes: list[Episode],
    rewards: list[QEpisodeReward],
    *,
    adv_mode: str = "mean",
) -> list[Trajectory]:
    """Solver trajectories for ONE secret's K episodes, v1 broadcast credit
    (DESIGN §2.3): the episode-level group advantage A_k = R_k − mean(R) is
    carried by every turn trajectory of episode k. Turns without recorded
    token ids (scripted players, answerer-side turns) are skipped.

    Voided episodes must be filtered out by the caller BEFORE this — a lied-to
    guesser's data is corrupted and trains nothing."""
    if len(episodes) != len(rewards):
        raise ValueError(f"{len(episodes)} episodes vs {len(rewards)} rewards")
    advantages = group_advantages([r.total for r in rewards], mode=adv_mode)
    trajs: list[Trajectory] = []
    for ep, rew, adv in zip(episodes, rewards, advantages):
        for turn in ep.turns:
            if turn.prompt_tokens is None or turn.completion_tokens is None:
                continue
            trajs.append(Trajectory(
                prompt_ids=turn.prompt_tokens,
                completion_ids=turn.completion_tokens,
                reward=rew.total,
                advantage=adv,
                meta={
                    "secret_id": ep.secret.secret_id,
                    "turn": turn.index,
                    "kind": turn.kind,
                    "guessed": rew.guessed,
                    "ended": ep.ended,
                },
            ))
    return trajs


def guess_rate(episode_guessed: list[bool]) -> float:
    """Realized guess rate over one secret's (non-void) episodes — the 21Q
    stand-in for a problem's solve rate. No episodes (all void) => 0.0; the
    caller marks such secrets unscored so the engine's scored-fraction scaling
    handles them (same posture as inconsistent problems)."""
    if not episode_guessed:
        return 0.0
    return sum(1 for g in episode_guessed if g) / len(episode_guessed)


def secrets_as_suite(secrets: list[Secret], *, category: str = "") -> ProblemSuite:
    """Adapt parsed secrets into a ProblemSuite so RewardEngine.creator_reward
    applies verbatim. statement/answer = the secret, difficulty = the dictated
    rank value (trainer-owned, exactly like per-problem mode overwriting
    claimed difficulty), so `is_valid`'s spread check and the rank sort both
    behave. Domain is tagged "twentyq" for logging."""
    return ProblemSuite(
        problems=[
            Problem(
                statement=s.secret,
                difficulty=s.difficulty,
                answer=s.secret,
                solution=s.notes,
                domain="twentyq",
                problem_id=s.secret_id,
            )
            for s in secrets
        ],
        theme=category,
        domain="twentyq",
    )
