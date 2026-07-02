"""Sprint 6 — mean-baseline group advantages + the all-zero tie detector.

mini-01 diagnosis: with G_c=2, ÷std standardization turns ANY non-tie into ±1
— a 0.7889-vs-0.7738 gap (solve-rate noise) trained exactly as hard as a
1.6-vs-−1.0 one. The Dr.GRPO mean baseline preserves magnitude, so near-tie
groups get near-zero gradients and real gaps proportionally strong ones.
"""

import pytest

from twin.rl import Trajectory, all_zero_advantages, group_advantages


def test_mean_mode_is_default_and_preserves_magnitude():
    advs = group_advantages([0.0, 1.0, 2.0, 3.0])
    assert advs == [-1.5, -0.5, 0.5, 1.5]
    assert advs == group_advantages([0.0, 1.0, 2.0, 3.0], mode="mean")


def test_mean_mode_near_tie_gives_near_zero_not_pm1():
    # the actual mini-01 noise gap: 0.7889 vs 0.7738
    near_tie = group_advantages([0.7889, 0.7738])
    real_gap = group_advantages([1.6, -1.0])
    assert max(abs(a) for a in near_tie) < 0.01
    assert max(abs(a) for a in real_gap) == pytest.approx(1.3)
    # std mode collapses both to the same ±1 signal — that's the bug
    std_tie = group_advantages([0.7889, 0.7738], mode="std")
    std_gap = group_advantages([1.6, -1.0], mode="std")
    assert std_tie == pytest.approx(std_gap, abs=1e-3)


def test_std_mode_is_legacy_standardization():
    advs = group_advantages([0.0, 1.0, 2.0, 3.0], mode="std")
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)
    n = len(advs)
    var = sum(a * a for a in advs) / n
    assert var == pytest.approx(1.0, rel=1e-3)        # unit variance


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        group_advantages([1.0, 2.0], mode="zscore")


def test_zero_variance_zero_in_both_modes():
    # float residue from centering identical values: ~1e-17 raw ("mean"),
    # amplified to ~1e-11 by the ÷(std+eps) in "std"
    for mode in ("mean", "std"):
        assert group_advantages([0.1, 0.1, 0.1], mode=mode) == pytest.approx(
            [0.0, 0.0, 0.0], abs=1e-9)
    assert group_advantages([]) == []


def _traj(adv):
    return Trajectory([0], [1], advantage=adv)


def test_all_zero_advantages_detects_ties_with_float_residue():
    # mean-centering identical floats can leave ~1e-17 residue; the detector
    # must still call it a tie
    advs = group_advantages([0.1, 0.1, 0.1])
    assert all_zero_advantages([_traj(a) for a in advs])
    assert all_zero_advantages([])                    # empty is trivially tied


def test_all_zero_advantages_false_when_any_signal():
    trajs = [_traj(0.0), _traj(0.0), _traj(0.05)]
    assert not all_zero_advantages(trajs)
