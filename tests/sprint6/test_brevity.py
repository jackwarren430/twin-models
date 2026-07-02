"""Brevity bonus for correct solver attempts (DESIGN.md §6.2). Fast: pure
arithmetic, no model. R += w_brevity * (1 - L/L_max), paid only when solved."""

import pytest

from twin.config import RewardsConfig
from twin.rewards import RewardEngine


def _engine(w_brevity=0.2):
    return RewardEngine(RewardsConfig(w_brevity=w_brevity))


def test_default_off():
    # w_brevity defaults to 0.0 — passing lengths changes nothing.
    r = RewardEngine(RewardsConfig()).solver_reward(True, n_tokens=10, max_tokens=100)
    assert r.total == pytest.approx(1.0)
    assert r.r_brevity == 0.0


def test_shorter_correct_earns_more():
    short = _engine().solver_reward(True, n_tokens=100, max_tokens=1000)
    long = _engine().solver_reward(True, n_tokens=900, max_tokens=1000)
    assert short.r_brevity == pytest.approx(0.9)
    assert long.r_brevity == pytest.approx(0.1)
    assert short.total == pytest.approx(1.0 + 0.2 * 0.9)
    assert short.total > long.total


def test_not_paid_when_wrong():
    # A wrong answer gets no brevity bonus — truncation can't be farmed.
    r = _engine().solver_reward(False, n_tokens=1, max_tokens=1000)
    assert r.r_brevity == 0.0
    assert r.total == pytest.approx(0.0)


def test_correct_always_beats_incorrect():
    # Worst correct (at budget) still out-scores best incorrect (1 token).
    worst_correct = _engine().solver_reward(True, n_tokens=1000, max_tokens=1000)
    best_wrong = _engine().solver_reward(False, n_tokens=1, max_tokens=1000)
    assert worst_correct.total > best_wrong.total


def test_disabled_without_lengths():
    # Trainer paths that don't pass lengths keep the old formula exactly.
    r = _engine().solver_reward(True)
    assert r.r_brevity == 0.0
    assert r.total == pytest.approx(1.0)


def test_clamped_to_unit_interval():
    # Over-budget completions clamp at 0 rather than going negative.
    over = _engine().solver_reward(True, n_tokens=1500, max_tokens=1000)
    assert over.r_brevity == 0.0
    assert over.total == pytest.approx(1.0)


def test_stacks_with_oracle_tax():
    r = _engine().solver_reward(True, n_oracle_calls=2, n_tokens=500, max_tokens=1000)
    assert r.total == pytest.approx(1.0 + 0.2 * 0.5 - 0.05 * 2)
