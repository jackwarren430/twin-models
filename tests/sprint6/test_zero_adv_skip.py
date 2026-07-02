"""Sprint 6 — the trainer skips the whole GRPO update on a fully tied batch.

mini-01 paid the full reference pass + policy pass + backward on iterations
whose advantages were all zero (creator pg==0 in 15/30 iters) — several
minutes per saturated 4096-token math iteration for a pure-KL gradient that is
≈0 right after the reference pass anyway.

The tests call ``SelfPlayTrainer._grpo`` with a *dummy* ``self`` carrying no
model/adapters/optimizers: if the tied batch is skipped before any scoring, the
call succeeds and returns the no-op metrics; if the gate ever moves after the
reference pass, the dummy raises AttributeError and the test fails. The
non-tied case must reach for the machinery (AttributeError == it tried to
score), proving the gate doesn't swallow real updates. Fast: no model.
"""

import pytest

from twin.rl import Trajectory, group_advantages
from twin.train.loop import SelfPlayTrainer


class _NoMachinery:
    """Stands in for a trainer; any attribute access = the update wasn't skipped."""


def _traj(adv, completion=(1, 2)):
    return Trajectory([0], list(completion), advantage=adv)


def test_tied_batch_skips_before_touching_the_model():
    trajs = [_traj(0.0), _traj(0.0), _traj(0.0)]
    m = SelfPlayTrainer._grpo(_NoMachinery(), "A", trajs)
    assert m["skipped_zero_adv"] is True
    assert m["n_traj"] == 3
    assert m["loss"] == m["pg"] == m["kl"] == m["grad_norm"] == 0.0
    assert m["n_tokens"] == 0


def test_mean_centered_tie_residue_still_skips():
    # advantages produced by mean-centering identical rewards carry ~1e-17
    # float residue; the gate must treat that as a tie
    advs = group_advantages([0.1, 0.1, 0.1])
    trajs = [_traj(a) for a in advs]
    m = SelfPlayTrainer._grpo(_NoMachinery(), "A", trajs)
    assert m.get("skipped_zero_adv") is True


def test_empty_or_tokenless_batch_still_noop():
    m = SelfPlayTrainer._grpo(_NoMachinery(), "A", [])
    assert m["n_traj"] == 0 and m["n_tokens"] == 0
    m = SelfPlayTrainer._grpo(_NoMachinery(), "A", [_traj(1.0, completion=())])
    assert m["n_traj"] == 0 and m["n_tokens"] == 0


def test_nonzero_advantage_reaches_for_the_machinery():
    trajs = [_traj(0.5), _traj(-0.5)]
    with pytest.raises(AttributeError):               # tried to run the update
        SelfPlayTrainer._grpo(_NoMachinery(), "A", trajs)
