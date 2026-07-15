"""Component-level Twenty Questions reward traces (model-free)."""

import pytest

from twin.games.twentyq.rewards import (
    broadcast_reward_trace,
    ensemble_reward_trace,
    shaped_reward_trace,
)


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
