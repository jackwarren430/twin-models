"""Sprint 3 — creator reward with void-problem masking (DESIGN §6.3). Fast: no model.

A void problem (failed consistency) must not be farmable as a "hard" problem:
it's dropped from the gradient curve but still drags R_consistency down. This is
the reward-side change Sprint 3 adds so the loop can't be gamed by shipping
unsolvable problems with fake answers.
"""

from twin.problems.schema import Problem, ProblemSuite
from twin.rewards import RewardEngine


def _suite(diffs):
    return ProblemSuite(problems=[Problem(statement=f"p{i}", difficulty=d)
                                  for i, d in enumerate(diffs)])


def test_mask_excludes_void_from_gradient():
    suite = _suite([0.1, 0.5, 0.9])
    eng = RewardEngine()
    # problem 2 is void; the two scored problems form a perfect 1->0 ramp
    masked = eng.creator_reward(
        suite, solve_rates=[1.0, 0.0, 0.0],
        consistent_flags=[True, True, False],
        scored_mask=[True, True, False],
    )
    assert masked.solve_rates_by_rank == [1.0, 0.0]      # only scored, easy->hard
    # Perfect fit over the subset, but scaled by the scored fraction (2/3):
    # a mostly-void suite must not out-earn an honest fully-consistent ramp
    # (Sprint 5 — mini-01 found the unscaled version farmable).
    assert abs(masked.r_gradient - 2 / 3) < 1e-6
    assert abs(masked.r_consistency - 2 / 3) < 1e-9      # void still counted here


def test_voiding_problems_is_unprofitable():
    """The mini-01 exploit: [1.0] over one consistent problem used to earn
    r_gradient=1.0 and near-max total reward. With scored-fraction scaling an
    honest full ramp strictly dominates every partially-void suite."""
    suite = _suite([0.1, 0.5, 0.9])
    eng = RewardEngine()
    honest = eng.creator_reward(
        suite, [1.0, 0.5, 0.0], [True] * 3, scored_mask=[True] * 3)
    one_easy = eng.creator_reward(
        suite, [1.0, 0.0, 0.0], [True, False, False],
        scored_mask=[True, False, False])
    easy_hard = eng.creator_reward(
        suite, [1.0, 0.0, 0.0], [True, False, True],
        scored_mask=[True, False, True])
    assert abs(honest.r_gradient - 1.0) < 1e-6           # full suite unscaled
    assert honest.total > easy_hard.total > one_easy.total


def test_all_void_zero_gradient():
    suite = _suite([0.2, 0.8])
    eng = RewardEngine()
    r = eng.creator_reward(
        suite, solve_rates=[0.0, 0.0],
        consistent_flags=[False, False],
        scored_mask=[False, False],
    )
    assert r.r_gradient == 0.0
    assert r.solve_rates_by_rank == []
    assert r.r_consistency == 0.0


def test_mask_length_validated():
    suite = _suite([0.2, 0.8])
    eng = RewardEngine()
    try:
        eng.creator_reward(suite, [0.0, 0.0], [True, True], scored_mask=[True])
        assert False, "expected ValueError"
    except ValueError:
        pass
