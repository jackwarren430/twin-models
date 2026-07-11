"""Sprint Q4 — episode reward math + broadcast advantage wiring
(twentyq/DESIGN.md §2.3/§2.5). Pure arithmetic, no model."""

import itertools

import pytest

from twin.config import TwentyQConfig
from twin.games.twentyq.episode import Episode, Turn
from twin.games.twentyq.rewards import (
    QEpisodeReward,
    episode_reward,
    guess_rate,
    secret_turn_trajectories,
    secrets_as_suite,
)
from twin.games.twentyq.schema import Secret
from twin.rewards import RewardEngine
from twin.rl import all_zero_advantages

CFG = TwentyQConfig()


def _reward(**kw):
    defaults = dict(guessed=False, turns_used=5, max_turns=10, phi_final=None)
    defaults.update(kw)
    return episode_reward(CFG, **defaults)


# ----- episode reward invariants --------------------------------------------

def test_any_win_beats_any_clean_failure():
    """Worst win (full budget) > best failure (Φ=1), across the whole grid."""
    worst_win = _reward(guessed=True, turns_used=10).total
    for turns, phi in itertools.product(range(11), [0.0, 0.3, 0.7, 1.0]):
        fail = _reward(guessed=False, turns_used=turns, phi_final=phi).total
        assert worst_win > fail


def test_fewer_turns_pay_more_when_guessed():
    totals = [_reward(guessed=True, turns_used=t).total for t in range(1, 11)]
    assert totals == sorted(totals, reverse=True)
    assert totals[0] > totals[-1]


def test_efficiency_not_paid_on_failure():
    fast_fail = _reward(guessed=False, turns_used=1)
    slow_fail = _reward(guessed=False, turns_used=10)
    assert fast_fail.total == slow_fail.total == 0.0
    assert fast_fail.r_efficiency == 0.0


def test_phi_orders_failures_and_is_ignored_on_wins():
    close = _reward(phi_final=0.9).total
    far = _reward(phi_final=0.2).total
    assert close > far
    win_hi = _reward(guessed=True, phi_final=0.9).total
    win_lo = _reward(guessed=True, phi_final=0.1).total
    assert win_hi == win_lo                     # Φ only shapes failures


def test_phi_none_degrades_to_binary():
    assert _reward(phi_final=None).total == 0.0


def test_format_fail_strictly_worse_than_clean_failure():
    clean = _reward(phi_final=0.4)
    broken = _reward(phi_final=0.4, format_fail=True)
    assert broken.total == pytest.approx(clean.total - CFG.w_format)


# ----- trajectory assembly ----------------------------------------------------

def _episode(secret, outcomes, ended="budget"):
    """Episode with one tokenized turn per outcome entry."""
    ep = Episode(secret=secret, ended=ended)
    for i in range(outcomes):
        ep.turns.append(Turn(
            index=i, kind="question", content=f"q{i}", answer="YES",
            raw_text="", prompt_tokens=[1, i], completion_tokens=[2, i],
        ))
    return ep


def test_broadcast_advantages_group_relative_over_episodes():
    secret = Secret(secret="octopus", category="animal", difficulty=0.5)
    eps = [_episode(secret, 3), _episode(secret, 2)]
    rewards = [
        episode_reward(CFG, guessed=True, turns_used=3, max_turns=10, phi_final=None),
        episode_reward(CFG, guessed=False, turns_used=2, max_turns=10, phi_final=0.4),
    ]
    trajs = secret_turn_trajectories(eps, rewards)
    assert len(trajs) == 5                      # 3 + 2 turns, all tokenized
    r0, r1 = rewards[0].total, rewards[1].total
    mean = (r0 + r1) / 2
    # Every turn of episode 0 carries A_0, every turn of episode 1 carries A_1.
    assert all(t.advantage == pytest.approx(r0 - mean) for t in trajs[:3])
    assert all(t.advantage == pytest.approx(r1 - mean) for t in trajs[3:])
    assert sum(t.advantage for t in trajs[:1] + trajs[3:4]) == pytest.approx(0.0)
    assert trajs[0].meta["secret_id"] == secret.secret_id


def test_tied_episodes_yield_all_zero_advantages():
    secret = Secret(secret="cat", category="animal", difficulty=0.2)
    eps = [_episode(secret, 2), _episode(secret, 2)]
    same = [episode_reward(CFG, guessed=True, turns_used=4, max_turns=10,
                           phi_final=None)] * 2
    trajs = secret_turn_trajectories(eps, same)
    assert all_zero_advantages(trajs)           # trainer will skip the update


def test_tokenless_turns_skipped():
    secret = Secret(secret="cat", category="animal", difficulty=0.2)
    ep = _episode(secret, 2)
    ep.turns.append(Turn(index=2, kind="question", content="q2", answer="YES",
                         raw_text=""))          # no tokens (scripted)
    rewards = [episode_reward(CFG, guessed=False, turns_used=3, max_turns=10,
                              phi_final=0.1)]
    other = _episode(secret, 1)
    rewards.append(episode_reward(CFG, guessed=True, turns_used=1, max_turns=10,
                                  phi_final=None))
    trajs = secret_turn_trajectories([ep, other], rewards)
    assert len(trajs) == 3                      # 2 tokenized + 1, tokenless dropped


def test_mismatched_lengths_raise():
    secret = Secret(secret="cat", category="animal", difficulty=0.2)
    with pytest.raises(ValueError):
        secret_turn_trajectories([_episode(secret, 1)], [])


# ----- creator-side reuse -----------------------------------------------------

def test_guess_rate():
    assert guess_rate([]) == 0.0
    assert guess_rate([True, False, True, True]) == 0.75


def test_secrets_as_suite_feeds_reward_engine_verbatim():
    """The whole point of the adapter: creator calibration via the existing
    engine, guess rates standing in for solve rates."""
    secrets = [
        Secret(secret="dog", category="animal", difficulty=0.0),
        Secret(secret="octopus", category="animal", difficulty=0.5),
        Secret(secret="axolotl", category="animal", difficulty=1.0),
    ]
    suite = secrets_as_suite(secrets, category="animal")
    assert suite.is_valid()                     # dictated ranks give the spread
    engine = RewardEngine()
    # Perfectly calibrated: realized guess rates == default 1->0 target ramp.
    perfect = engine.creator_reward(suite, [1.0, 0.5, 0.0], [True] * 3)
    # Miscalibrated: everything guessed instantly (all too easy).
    flat = engine.creator_reward(suite, [1.0, 1.0, 1.0], [True] * 3)
    assert perfect.r_gradient == pytest.approx(1.0)
    assert perfect.total > flat.total
    # Voided secret (invalid / lying creator): unscored + inconsistent.
    voided = engine.creator_reward(
        suite, [1.0, 0.5, 0.0], [True, True, False],
        scored_mask=[True, True, False], expected_n=3)
    assert voided.total < perfect.total
    assert voided.r_consistency == pytest.approx(2 / 3)
