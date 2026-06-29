"""Reward engine tests (DESIGN.md §6). Pure arithmetic — no model."""

import math

import pytest

from twin.config import RewardsConfig
from twin.problems.schema import Problem, ProblemSuite
from twin.rewards import RewardEngine, solve_rate


def _suite(difficulties):
    return ProblemSuite(
        problems=[Problem(statement=f"p{i}", difficulty=d) for i, d in enumerate(difficulties)]
    )


def _engine():
    return RewardEngine(RewardsConfig())


# ----- solver ---------------------------------------------------------------
def test_solver_reward_solved():
    r = _engine().solver_reward(True, n_oracle_calls=0)
    assert r.total == pytest.approx(1.0)
    assert r.solved


def test_solver_reward_oracle_tax():
    r = _engine().solver_reward(True, n_oracle_calls=2)
    # 1.0 - 0.05*2
    assert r.total == pytest.approx(0.9)


def test_solver_reward_unsolved():
    assert _engine().solver_reward(False).total == pytest.approx(0.0)


def test_solver_parse_gate_is_zero():
    assert _engine().solver_parse_gate().total == 0.0


# ----- creator gradient -----------------------------------------------------
def test_creator_perfect_ramp_scores_high():
    suite = _suite([0.0, 0.5, 1.0])
    r = _engine().creator_reward(suite, solve_rates=[1.0, 0.5, 0.0],
                                 consistent_flags=[True, True, True])
    assert r.r_gradient == pytest.approx(1.0)
    assert r.r_consistency == pytest.approx(1.0)
    assert r.r_valid == 1.0
    # 1*1 + 0.5*1 - 0 + 0.1*1
    assert r.total == pytest.approx(1.6)


def test_creator_flat_curve_scores_low_gradient():
    suite = _suite([0.0, 0.5, 1.0])
    r = _engine().creator_reward(suite, solve_rates=[1.0, 1.0, 1.0],
                                 consistent_flags=[True, True, True])
    mse = ((1 - 1) ** 2 + (1 - 0.5) ** 2 + (1 - 0) ** 2) / 3
    assert r.r_gradient == pytest.approx(math.exp(-4.0 * mse))
    assert r.r_gradient < 0.3


def test_creator_gradient_uses_difficulty_order_not_list_order():
    # Problems given hardest-first; solve rates parallel to that order.
    suite = _suite([1.0, 0.0, 0.5])
    r = _engine().creator_reward(suite, solve_rates=[0.0, 1.0, 0.5],
                                 consistent_flags=[True, True, True])
    # Reordered easy->hard this is the perfect ramp [1.0, 0.5, 0.0].
    assert r.r_gradient == pytest.approx(1.0)
    assert r.solve_rates_by_rank == [1.0, 0.5, 0.0]


def test_creator_consistency_fraction():
    suite = _suite([0.0, 0.5, 1.0])
    r = _engine().creator_reward(suite, solve_rates=[1.0, 0.5, 0.0],
                                 consistent_flags=[True, False, True])
    assert r.r_consistency == pytest.approx(2 / 3)


def test_creator_oracle_tax_subtracts():
    suite = _suite([0.0, 0.5, 1.0])
    base = _engine().creator_reward(suite, [1.0, 0.5, 0.0], [True, True, True], n_oracle_calls=0)
    taxed = _engine().creator_reward(suite, [1.0, 0.5, 0.0], [True, True, True], n_oracle_calls=3)
    assert taxed.total == pytest.approx(base.total - 0.1 * 3)


def test_creator_invalid_suite_no_valid_bonus():
    # spread too small -> is_valid() False -> r_valid 0
    suite = _suite([0.1, 0.1, 0.1])
    r = _engine().creator_reward(suite, [1.0, 1.0, 1.0], [True, True, True])
    assert r.r_valid == 0.0


def test_creator_parse_gate_is_fixed_low():
    assert _engine().creator_parse_gate().total == pytest.approx(-1.0)


def test_length_mismatch_raises():
    suite = _suite([0.0, 0.5, 1.0])
    with pytest.raises(ValueError):
        _engine().creator_reward(suite, [1.0, 0.0], [True, True, True])


def test_clip_bounds_total():
    eng = RewardEngine(RewardsConfig(clip=0.5))
    r = eng.solver_reward(True)  # raw 1.0 -> clipped to 0.5
    assert r.total == 0.5


# ----- helper ---------------------------------------------------------------
def test_solve_rate_helper():
    assert solve_rate([True, True, False, False]) == 0.5
    assert solve_rate([]) == 0.0
