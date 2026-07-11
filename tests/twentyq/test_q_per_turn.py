"""Sprint Q7 — per-turn credit: potential-based shaping, reward-to-go,
turn-index baselines, and the trainer's per_turn path (twentyq/DESIGN.md §2.3
v2). Judge and models scripted throughout."""

import pytest

from twin.config import Config, TwentyQConfig
from twin.games.twentyq.episode import Episode, Turn
from twin.games.twentyq.rewards import (
    episode_reward,
    per_turn_secret_trajectories,
    shaped_returns,
)
from twin.games.twentyq.schema import Secret

from test_q_trainer import CREATOR_OK, GUESSER_OK, _config, _make_trainer

CFG = TwentyQConfig()


# ----- shaped_returns ---------------------------------------------------------

@pytest.mark.parametrize("phis,terminal", [
    ([], 1.15),                       # single-turn win
    ([0.2, 0.5, 0.8], 0.35),          # rising closeness, failed episode
    ([0.9, 0.1], -0.5),               # judge zig-zag, format-fail terminal
    ([None, 0.4, None], 0.2),         # parse failures carried forward
])
def test_telescoping_identity_at_gamma_1(phis, terminal):
    """At γ=1 the shaping sums to zero: R_0 == the v1 episode scalar exactly."""
    returns = shaped_returns(phis, terminal, gamma=1.0, w_close=0.5)
    assert returns[0] == pytest.approx(terminal)
    assert len(returns) == len(phis) + 1


def test_reward_to_go_recursion():
    gamma, w = 0.9, 0.5
    phis = [0.3, 0.6]
    terminal = 1.0
    returns = shaped_returns(phis, terminal, gamma=gamma, w_close=w)
    pots = [0.0, 0.3, 0.6, 0.0]
    rs = [w * (gamma * pots[t + 1] - pots[t]) for t in range(3)]
    rs[-1] += terminal
    expected_last = rs[2]
    expected_mid = rs[1] + gamma * expected_last
    expected_first = rs[0] + gamma * expected_mid
    assert returns == pytest.approx([expected_first, expected_mid, expected_last])


def test_progress_turns_earn_more_than_stalls():
    """Within one failed episode, the turn that raised Φ carries a larger
    immediate shaping term than the turn that stalled."""
    w = 0.5
    # turn 0 raises Φ 0->0.8; turn 1 stalls at 0.8; turn 2 terminal.
    returns = shaped_returns([0.8, 0.8], 0.4, gamma=1.0, w_close=w)
    r0 = returns[0] - returns[1]      # immediate reward of turn 0
    r1 = returns[1] - returns[2]
    assert r0 == pytest.approx(w * 0.8)
    assert r1 == pytest.approx(0.0)
    assert r0 > r1


def test_none_phi_means_zero_delta():
    base = shaped_returns([0.5, 0.5], 0.0, gamma=1.0, w_close=0.5)
    with_none = shaped_returns([0.5, None], 0.0, gamma=1.0, w_close=0.5)
    assert with_none == pytest.approx(base)


# ----- per-turn trajectory assembly -------------------------------------------

def _ep(secret, n_turns):
    ep = Episode(secret=secret)
    for i in range(n_turns):
        ep.turns.append(Turn(index=i, kind="question", content=f"q{i}",
                             answer="YES", raw_text="",
                             prompt_tokens=[1, i], completion_tokens=[2, i]))
    return ep


def test_turn_index_baseline_across_siblings():
    secret = Secret(secret="octopus", category="animal", difficulty=0.5)
    eps = [_ep(secret, 2), _ep(secret, 2)]
    returns = [[1.0, 0.6], [0.2, 0.4]]
    trajs = per_turn_secret_trajectories(eps, returns)
    assert len(trajs) == 4
    by = {(t.meta["turn"], t.reward): t.advantage for t in trajs}
    assert by[(0, 1.0)] == pytest.approx(+0.4)   # 1.0 - mean(1.0, 0.2)
    assert by[(0, 0.2)] == pytest.approx(-0.4)
    assert by[(1, 0.6)] == pytest.approx(+0.1)   # 0.6 - mean(0.6, 0.4)
    assert by[(1, 0.4)] == pytest.approx(-0.1)


def test_ragged_lengths_singleton_turn_gets_zero_advantage():
    secret = Secret(secret="octopus", category="animal", difficulty=0.5)
    eps = [_ep(secret, 3), _ep(secret, 1)]
    returns = [[1.0, 0.8, 0.6], [0.2]]
    trajs = per_turn_secret_trajectories(eps, returns)
    solo = [t for t in trajs if t.meta["turn"] >= 1]
    assert all(t.advantage == pytest.approx(0.0) for t in solo)
    shared = [t for t in trajs if t.meta["turn"] == 0]
    assert sorted(round(t.advantage, 6) for t in shared) == [-0.4, 0.4]


def test_length_mismatches_raise():
    secret = Secret(secret="octopus", category="animal", difficulty=0.5)
    with pytest.raises(ValueError):
        per_turn_secret_trajectories([_ep(secret, 2)], [[1.0]])
    with pytest.raises(ValueError):
        per_turn_secret_trajectories([_ep(secret, 1)], [])


# ----- trainer integration ------------------------------------------------------

def _per_turn_config():
    return _config(twentyq={
        "n_secrets": 2, "episodes_per_secret": 2, "max_turns": 2,
        "categories": ["animal"], "credit": "per_turn", "gamma": 1.0,
    })


def test_trainer_per_turn_judge_budget_and_rewards():
    t = _make_trainer(_per_turn_config(), list(CREATOR_OK), list(GUESSER_OK))
    rec = t.run_iteration(0)
    assert rec["credit"] == "per_turn"
    calls = t.captured["judge"]
    # Intermediate states: one per non-terminal turn of each non-void episode
    # (A1: 0, A2: 1, B1: 1, B2: 1) = 3, plus the final-state calls for the two
    # failed episodes (B1, B2) = 5 total closeness calls.
    assert sum(1 for q in calls if "scoring how close" in q) == 5
    trajs = t.captured["grpo"]["B"]
    assert len(trajs) == 7
    assert all(tr.meta["credit"] == "per_turn" for tr in trajs)
    # γ=1 telescoping: turn-0 reward (reward-to-go from the start) equals the
    # v1 episode scalar for each episode.
    qcfg = t.cfg.twentyq
    ep_a1 = episode_reward(qcfg, guessed=True, turns_used=1, max_turns=2,
                           phi_final=None).total
    ep_a2 = episode_reward(qcfg, guessed=True, turns_used=2, max_turns=2,
                           phi_final=None).total
    turn0 = sorted(tr.reward for tr in trajs
                   if tr.meta["turn"] == 0 and tr.meta["ended"] == "guessed")
    assert turn0 == pytest.approx(sorted([ep_a1, ep_a2]))


def test_trainer_broadcast_path_untouched_by_default():
    t = _make_trainer(_config(), list(CREATOR_OK), list(GUESSER_OK))
    rec = t.run_iteration(0)
    assert rec["credit"] == "broadcast"
    # v1 budget: closeness only for the two failed episodes.
    assert sum(1 for q in t.captured["judge"] if "scoring how close" in q) == 2
    assert all("credit" not in tr.meta for tr in t.captured["grpo"]["B"])
