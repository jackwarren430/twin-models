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

from collections import defaultdict
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


@dataclass(frozen=True)
class StepReward:
    """The reward signals attached to one solver turn.

    ``dense`` is the immediate potential-shaping term, ``terminal`` is the
    sparse episode scalar (non-zero only on the final turn), ``total`` is their
    immediate sum, and ``return_`` is the value actually assigned to that
    turn's GRPO trajectory.  Potentials are included so logs can be audited
    without reconstructing the shaping calculation.
    """

    dense: float
    terminal: float
    total: float
    return_: float
    potential_before: float | None = None
    potential_after: float | None = None
    local: float = 0.0


def _step_trace(
    dense: list[float],
    terminal: float,
    *,
    gamma: float,
    potentials: list[float] | None = None,
    broadcast_return: bool = False,
    local: list[float] | None = None,
) -> list[StepReward]:
    """Build an auditable per-step trace from immediate dense rewards.

    ``local[t]`` is a NON-PROPAGATING per-turn term: it lands on turn ``t``'s
    return and nowhere else, unlike ``dense``/``terminal``, which flow backwards
    through the reward-to-go accumulation. That asymmetry is deliberate and is
    the whole point of the field. Winning is a shared property of an episode —
    every turn helped get there, so outcome credit is rightly spread. Asking a
    question the transcript already answered is not shared: it is a defect of
    exactly one turn, and propagating it backwards would debit the informative
    turns that preceded it for a mistake they did not make. Under the per-turn-
    index GRPO baseline that backwards smear is pure variance, since siblings
    that repeat at DIFFERENT turns would penalize each other's good turns.
    """
    if not dense:
        return []
    local = list(local) if local is not None else [0.0] * len(dense)
    if len(local) != len(dense):
        raise ValueError(f"local has {len(local)} entries for {len(dense)} turns")
    terminal_by_step = [0.0] * len(dense)
    terminal_by_step[-1] = float(terminal)
    immediate = [d + term for d, term in zip(dense, terminal_by_step)]
    if broadcast_return:
        returns = [float(terminal)] * len(dense)
    else:
        returns = [0.0] * len(dense)
        acc = 0.0
        for t in reversed(range(len(dense))):
            acc = immediate[t] + gamma * acc
            returns[t] = acc
    return [
        StepReward(
            dense=float(dense[t]),
            terminal=terminal_by_step[t],
            total=immediate[t] + local[t],
            return_=returns[t] + local[t],
            potential_before=(potentials[t] if potentials is not None else None),
            potential_after=(potentials[t + 1] if potentials is not None else None),
            local=float(local[t]),
        )
        for t in range(len(dense))
    ]


def broadcast_reward_trace(n_turns: int, terminal: float) -> list[StepReward]:
    """Trace for legacy broadcast credit.

    The sparse scalar occurs at episode termination, while the legacy GRPO
    assignment broadcasts that same scalar to every turn (hence
    ``return_ == terminal`` for every row).
    """
    return _step_trace(
        [0.0] * n_turns, terminal, gamma=1.0, broadcast_return=True)


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
    for ki, (ep, rew, adv) in enumerate(zip(episodes, rewards, advantages)):
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
                    "ep_in_secret": ki,
                    "turn": turn.index,
                    "kind": turn.kind,
                    "guessed": rew.guessed,
                    "ended": ep.ended,
                },
            ))
    return trajs


def shaped_returns(
    intermediate_phis: list[float | None],
    terminal: float,
    *,
    gamma: float,
    w_close: float,
) -> list[float]:
    """Per-turn reward-to-go for ONE episode under potential-based shaping
    (Sprint Q7, DESIGN §2.3 v2).

    ``intermediate_phis[t]`` is the judge's closeness Φ of the state after
    turn ``t`` (``t = 0..T-2`` — the non-terminal states). Terminal-state
    potential is pinned to 0 (the Ng et al. condition for policy invariance),
    and the start-state potential is 0 by definition ("nothing established").
    A ``None`` (judge gave no signal) carries the previous potential forward
    (that turn's shaping delta is 0).

        r_t     = w_close · (γ·Φ_{t+1} − Φ_t)          t = 0..T-1
        r_{T-1} += terminal                             (the v1 episode scalar)
        R_t     = r_t + γ·R_{t+1}

    Because the shaping telescopes to zero at γ=1, ``R_0 == terminal``
    exactly: per-turn credit REDISTRIBUTES the episode total across turns
    without changing it — the broadcast-vs-per-turn ablation compares credit
    schemes, not reward scales."""
    return [s.return_ for s in shaped_reward_trace(
        intermediate_phis, terminal, gamma=gamma, w_close=w_close)]


def shaped_reward_trace(
    intermediate_phis: list[float | None],
    terminal: float,
    *,
    gamma: float,
    w_close: float,
) -> list[StepReward]:
    """Component-level counterpart of :func:`shaped_returns`."""
    pots = [0.0]
    prev = 0.0
    for p in intermediate_phis:
        if p is not None:
            prev = float(p)
        pots.append(prev)
    pots.append(0.0)                        # terminal state, both outcomes
    n_turns = len(intermediate_phis) + 1
    dense = [w_close * (gamma * pots[t + 1] - pots[t])
             for t in range(n_turns)]
    return _step_trace(dense, terminal, gamma=gamma, potentials=pots)


def history_states(ep: Episode) -> list[list[tuple[str, str]]]:
    """The Q/A history visible AFTER each turn, as a list of length
    ``turns_used + 1``: ``states[0]`` is empty (the score(history_0) baseline,
    before turn 0) and ``states[i + 1]`` is the history after turn ``i``.

    A turn that produced no answer (a correct GUESS or a format fail ends the
    episode; both have ``answer is None``) carries the previous history forward
    unchanged — the ensemble sees no new evidence, so that turn's shaping delta
    is ~0 and the credit for it flows from the sparse terminal reward instead.
    Wrong guesses appear as ``('Is it <x>?', 'NO')``, matching :attr:`qa_pairs`."""
    states = [[]]
    pairs: list[tuple[str, str]] = []
    for t in ep.turns:
        if t.answer is not None:
            q = f"Is it {t.content}?" if t.kind == "guess" else t.content
            pairs = pairs + [(q, t.answer)]
        states.append(list(pairs))
    return states


def ensemble_shaped_returns(
    potentials: list[float],
    terminal: float,
    *,
    gamma: float,
    scale: float = 1.0,
) -> list[float]:
    """Per-turn reward-to-go for ONE episode under ENSEMBLE-potential shaping
    (the new dense reward). ``potentials[i]`` is the ensemble score Φ of the
    secret given history state ``i`` (``potentials = [Φ_0, Φ_1, ..., Φ_T]`` from
    :func:`history_states`, length ``turns_used + 1``). Turn ``t`` earns the
    potential-based shaping reward

        r_t = scale · (γ·Φ_{t+1} − Φ_t)            t = 0..T-1

    exactly the ``r_t = γ·score(history_t) − score(history_{t-1})`` form of the
    reward spec, and the LAST turn additionally carries the sparse episode
    scalar (``terminal`` — the w_guess/efficiency/format reward), so the dense
    ensemble signal is paid ALONGSIDE the terminal outcome. Reward-to-go is the
    usual γ-discounted accumulation ``R_t = r_t + γ·R_{t+1}``.

    Unlike :func:`shaped_returns` (judge-Φ, start/terminal potentials pinned to
    0 for policy invariance) the ensemble potentials are used raw — score(history_0)
    is a real baseline, per the reward spec — so the shaping does NOT telescope to
    ``terminal`` at γ=1; it adds the honest ensemble-belief gain across the game."""
    return [s.return_ for s in ensemble_reward_trace(
        potentials, terminal, gamma=gamma, scale=scale)]


def ensemble_reward_trace(
    potentials: list[float],
    terminal: float,
    *,
    gamma: float,
    scale: float = 1.0,
    local: list[float] | None = None,
) -> list[StepReward]:
    """Component-level counterpart of :func:`ensemble_shaped_returns`.

    ``local`` (see :func:`_step_trace`) carries non-propagating per-turn terms —
    the repeat penalty from :func:`repeat_penalties`."""
    n_turns = len(potentials) - 1
    if n_turns <= 0:
        return []
    dense = [scale * (gamma * potentials[t + 1] - potentials[t])
             for t in range(n_turns)]
    return _step_trace(dense, terminal, gamma=gamma, potentials=potentials,
                       local=local)


def repeat_penalties(ep: Episode, w_repeat: float) -> list[float]:
    """Per-turn local penalty for turns flagged ``Turn.repeat`` (DESIGN §9).

    Under terminal credit with gamma=1 every turn of a winning episode receives
    the SAME return, so GRPO raises the log-probability of the repeated question
    exactly as much as the informative ones. The v7 penguin transcript is the
    worst case: ten consecutive "Is the animal a parrot?" turns in an episode
    that won, i.e. ten reinforced repetitions of the lock that caused the near
    loss. This term is what breaks the tie between the turns of a won episode.

    Note discounting is NOT a substitute. The lock occupies the LATE turns and
    the informative questions the early ones, so gamma < 1 would move credit
    towards the repeats rather than away from them."""
    if w_repeat <= 0:
        return [0.0] * len(ep.turns)
    return [(-w_repeat if t.repeat else 0.0) for t in ep.turns]


def per_turn_secret_trajectories(
    episodes: list[Episode],
    returns_by_episode: list[list[float]],
    *,
    adv_mode: str = "mean",
) -> list[Trajectory]:
    """Per-turn credit for ONE secret's K episodes: turn t of episode k
    carries its own reward-to-go ``R_{k,t}``, baselined against the same turn
    index of the sibling episodes (``group_advantages`` per index). A turn
    index only one episode reached has no counterfactual — its advantage is 0
    (a singleton group mean-centers to zero anyway). Voided episodes must be
    filtered by the caller, same as :func:`secret_turn_trajectories`."""
    if len(episodes) != len(returns_by_episode):
        raise ValueError(f"{len(episodes)} episodes vs {len(returns_by_episode)} returns")
    for ep, rets in zip(episodes, returns_by_episode):
        if len(rets) != ep.turns_used:
            raise ValueError(
                f"episode has {ep.turns_used} turns but {len(rets)} returns")
    by_index: dict[int, list[float]] = defaultdict(list)
    for rets in returns_by_episode:
        for t, r in enumerate(rets):
            by_index[t].append(r)
    adv_by_index = {t: group_advantages(v, mode=adv_mode)
                    for t, v in by_index.items()}
    cursor = {t: 0 for t in by_index}
    trajs: list[Trajectory] = []
    for ki, (ep, rets) in enumerate(zip(episodes, returns_by_episode)):
        for turn, ret in zip(ep.turns, rets):
            adv = adv_by_index[turn.index][cursor[turn.index]]
            cursor[turn.index] += 1
            if turn.prompt_tokens is None or turn.completion_tokens is None:
                continue
            trajs.append(Trajectory(
                prompt_ids=turn.prompt_tokens,
                completion_ids=turn.completion_tokens,
                reward=ret,
                advantage=adv,
                meta={
                    "secret_id": ep.secret.secret_id,
                    "ep_in_secret": ki,
                    "turn": turn.index,
                    "kind": turn.kind,
                    "ended": ep.ended,
                    "credit": "per_turn",
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
