"""Sprint 5 — gradient reward scaled by scored fraction. Fast: no model.

mini-01 exposed the exploit: the curve fit is measured over the consistent-only
subset against a re-stretched target ramp, so one consistent easy problem gave
a perfect [1.0]-vs-[1.0] fit (r_gradient=1.0, total 1.27) and nearly out-earned
the honest 3-problem ideal (1.6). Scaling by n_scored/n makes voiding problems
strictly unprofitable.
"""

from twin.problems.schema import Problem, ProblemSuite
from twin.rewards import RewardEngine


def _suite(diffs):
    return ProblemSuite(problems=[Problem(statement=f"p{i}", difficulty=d)
                                  for i, d in enumerate(diffs)])


def test_full_suite_unscaled():
    suite = _suite([0.1, 0.5, 0.9])
    r = RewardEngine().creator_reward(
        suite, [1.0, 0.5, 0.0], [True] * 3, scored_mask=[True] * 3)
    assert abs(r.r_gradient - 1.0) < 1e-6


def test_single_consistent_scaled_to_third():
    suite = _suite([0.1, 0.5, 0.9])
    r = RewardEngine().creator_reward(
        suite, [1.0, 0.0, 0.0], [True, False, False],
        scored_mask=[True, False, False])
    # perfect fit over the 1-problem subset, scaled by 1/3
    assert abs(r.r_gradient - 1 / 3) < 1e-6


def test_no_mask_backcompat_unscaled():
    suite = _suite([0.1, 0.9])
    r = RewardEngine().creator_reward(suite, [1.0, 0.0], [True, True])
    assert abs(r.r_gradient - 1.0) < 1e-6


def test_mini01_exploit_closed():
    """Every partially-void suite earns strictly less than the honest ramp,
    including the two shapes mini-01 actually produced (reward 1.27 and 1.43
    vs the 1.6 ideal, pre-fix)."""
    suite = _suite([0.1, 0.5, 0.9])
    eng = RewardEngine()
    honest = eng.creator_reward(
        suite, [1.0, 0.5, 0.0], [True] * 3, scored_mask=[True] * 3)
    one_easy = eng.creator_reward(
        suite, [1.0, 0.0, 0.0], [True, False, False],
        scored_mask=[True, False, False])
    easy_impossible = eng.creator_reward(
        suite, [1.0, 0.0, 0.0], [True, False, True],
        scored_mask=[True, False, True])
    assert honest.total > easy_impossible.total
    assert honest.total > one_easy.total
