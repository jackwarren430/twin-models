"""Component-level Twenty Questions reward traces (model-free)."""

import pytest

from twin.games.twentyq.episode import Episode, Turn
from twin.games.twentyq.rewards import (
    broadcast_reward_trace,
    ensemble_reward_trace,
    repeat_penalties,
    shaped_reward_trace,
)
from twin.games.twentyq.schema import Secret


def test_ensemble_trace_separates_dense_terminal_and_return():
    trace = ensemble_reward_trace(
        [-3.0, -2.0, -1.0], terminal=1.0, gamma=1.0, scale=0.5)
    assert [s.dense for s in trace] == pytest.approx([0.5, 0.5])
    assert [s.terminal for s in trace] == pytest.approx([0.0, 1.0])
    assert [s.total for s in trace] == pytest.approx([0.5, 1.5])
    assert [s.return_ for s in trace] == pytest.approx([2.0, 1.5])
    assert trace[0].potential_before == -3.0
    assert trace[-1].potential_after == -1.0


def test_broadcast_trace_distinguishes_event_from_assignment():
    trace = broadcast_reward_trace(3, terminal=1.2)
    assert [s.terminal for s in trace] == pytest.approx([0.0, 0.0, 1.2])
    assert [s.total for s in trace] == pytest.approx([0.0, 0.0, 1.2])
    assert [s.return_ for s in trace] == pytest.approx([1.2, 1.2, 1.2])


def test_judge_trace_preserves_existing_telescoping_identity():
    trace = shaped_reward_trace(
        [0.25, 0.75], terminal=0.4, gamma=1.0, w_close=0.5)
    assert trace[0].return_ == pytest.approx(0.4)
    assert sum(s.dense for s in trace) == pytest.approx(0.0)
    assert sum(s.total for s in trace) == pytest.approx(0.4)


# ----- local (non-propagating) repeat penalty --------------------------------

def _ep_with_repeats(flags: list[bool]) -> Episode:
    ep = Episode(secret=Secret(secret="octopus", category="animal", difficulty=0.4))
    for i, is_repeat in enumerate(flags):
        ep.turns.append(Turn(index=i, kind="question", content=f"q{i}",
                             answer="NO", raw_text="", repeat=is_repeat))
    return ep


def test_repeat_penalties_flags_only_repeat_turns():
    ep = _ep_with_repeats([False, True, False, True])
    assert repeat_penalties(ep, 0.25) == [0.0, -0.25, 0.0, -0.25]


def test_repeat_penalties_disabled_at_zero_weight():
    ep = _ep_with_repeats([True, True])
    assert repeat_penalties(ep, 0.0) == [0.0, 0.0]


def test_local_penalty_does_not_propagate_backwards():
    """The whole point of `local`: a repeat at turn 2 must not debit turns 0-1.

    Terminal credit with gamma=1 gives every turn return == terminal; adding a
    local -0.25 on turn 2 must change turn 2 and nothing else."""
    trace = ensemble_reward_trace(
        [0.0] * 5, 1.0, gamma=1.0, scale=0.0,
        local=[0.0, 0.0, -0.25, 0.0])
    assert [round(s.return_, 6) for s in trace] == [1.0, 1.0, 0.75, 1.0]
    assert [round(s.local, 6) for s in trace] == [0.0, 0.0, -0.25, 0.0]
    # terminal/dense components are untouched by the local term
    assert [round(s.terminal, 6) for s in trace] == [0.0, 0.0, 0.0, 1.0]


def test_local_penalty_contrasts_with_propagating_dense():
    """A dense term at turn 2 DOES flow backwards; that is the difference."""
    dense_flows = ensemble_reward_trace(
        [0.0, 0.0, 0.0, -0.25, -0.25], 0.0, gamma=1.0, scale=1.0)
    assert dense_flows[0].return_ != pytest.approx(0.0)
    local_stays = ensemble_reward_trace(
        [0.0] * 5, 0.0, gamma=1.0, scale=0.0, local=[0.0, 0.0, -0.25, 0.0])
    assert local_stays[0].return_ == pytest.approx(0.0)


def test_local_length_mismatch_raises():
    with pytest.raises(ValueError, match="local has"):
        ensemble_reward_trace([0.0] * 4, 1.0, gamma=1.0, scale=0.0,
                              local=[0.0, 0.0])


def test_local_defaults_to_zero_and_preserves_legacy_trace():
    with_local = ensemble_reward_trace([0.0] * 4, 1.0, gamma=1.0, scale=0.0,
                                       local=[0.0] * 3)
    without = ensemble_reward_trace([0.0] * 4, 1.0, gamma=1.0, scale=0.0)
    assert [s.return_ for s in with_local] == [s.return_ for s in without]
    assert all(s.local == 0.0 for s in without)
