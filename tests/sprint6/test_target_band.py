"""Sprint 6 — interior target curve (rewards.target_hi/target_lo).

The default 1→0 ramp asks for an always-solved problem at rank 0 and a
never-solved one at rank N−1 — ranks whose realized solver K-groups are
zero-variance (gradient-free) by design once the creator is on-target. An
interior band (0.9→0.1) keeps outcome variance at every rank. The band changes
the game's incentives, so defaults stay 1→0 and the band is an explicit
ablation knob (configs/mini2.yaml).
"""

import pytest

from twin.config import RewardsConfig
from twin.problems.schema import Problem, ProblemSuite
from twin.rewards import RewardEngine


def _suite(diffs):
    return ProblemSuite(problems=[Problem(statement=f"p{i}", difficulty=d)
                                  for i, d in enumerate(diffs)])


def test_target_curve_default_unchanged():
    assert ProblemSuite.target_curve(3) == pytest.approx([1.0, 0.5, 0.0])
    assert ProblemSuite.target_curve(1) == [1.0]
    assert ProblemSuite.target_curve(0) == []


def test_target_curve_interior_band():
    curve = ProblemSuite.target_curve(3, hi=0.9, lo=0.1)
    assert curve == pytest.approx([0.9, 0.5, 0.1])
    assert ProblemSuite.target_curve(1, hi=0.9, lo=0.1) == [0.9]
    curve5 = ProblemSuite.target_curve(5, hi=0.9, lo=0.1)
    assert curve5[0] == pytest.approx(0.9) and curve5[-1] == pytest.approx(0.1)
    assert all(a > b for a, b in zip(curve5, curve5[1:]))   # still descending


def test_engine_honors_band():
    suite = _suite([0.1, 0.5, 0.9])
    eng = RewardEngine(RewardsConfig(target_hi=0.9, target_lo=0.1))
    # on-band rates fit perfectly...
    on_band = eng.creator_reward(
        suite, [0.9, 0.5, 0.1], [True] * 3, scored_mask=[True] * 3)
    assert on_band.r_gradient == pytest.approx(1.0)
    assert on_band.target_curve == pytest.approx([0.9, 0.5, 0.1])
    # ...and the old 1→0 endpoints now miss the target
    endpoints = eng.creator_reward(
        suite, [1.0, 0.5, 0.0], [True] * 3, scored_mask=[True] * 3)
    assert endpoints.r_gradient < on_band.r_gradient


def test_engine_default_band_is_legacy_ramp():
    suite = _suite([0.1, 0.5, 0.9])
    r = RewardEngine().creator_reward(
        suite, [1.0, 0.5, 0.0], [True] * 3, scored_mask=[True] * 3)
    assert r.r_gradient == pytest.approx(1.0)
    assert r.target_curve == pytest.approx([1.0, 0.5, 0.0])
