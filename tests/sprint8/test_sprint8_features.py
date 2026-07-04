"""Sprint 8 — batched solver generation (chunking + config), think-share
telemetry, and per-problem credit decomposition."""

import math
import random

import pytest

from twin.config import Config
from twin.models.base import GenResult, ReactResult
from twin.problems.schema import Problem, ProblemSuite
from twin.rewards import RewardEngine
from twin.roles import RoleManager
from twin.think import strip_think, think_share
from twin.tools.protocol import ToolCall, ToolResult
from twin.train.loop import SelfPlayTrainer


# --------------------------------------------------------------------------- #
# think_share
# --------------------------------------------------------------------------- #
def test_think_share_basic():
    assert think_share("") == 0.0
    assert think_share("no reasoning here") == 0.0
    text = "<think>aaaa</think>bbbb"
    # closed span (19 chars incl. tags) over 23 total
    assert think_share(text) == pytest.approx(19 / 23)


def test_think_share_unclosed_counts_to_end():
    # A truncation spiral (mini-03a) must read ~1.0, not 0.
    assert think_share("<think>never stops thinking") == 1.0
    # prose then an unclosed think
    t = "intro " + "<think>rest"
    assert think_share(t) == pytest.approx(len("<think>rest") / len(t))


def test_think_share_all_think():
    assert think_share("<think>x</think>") == 1.0


# --------------------------------------------------------------------------- #
# per-problem credit decomposition (engine)
# --------------------------------------------------------------------------- #
def _suite(difficulties):
    return ProblemSuite(problems=[
        Problem(statement=f"p{i}", difficulty=d, answer="1")
        for i, d in enumerate(difficulties)
    ])


def test_problem_rewards_split_credit():
    eng = RewardEngine()
    suite = _suite([0.0, 0.5, 1.0])
    # rank 0 nails its target, rank 1 misses badly, rank 2 is void
    rates = [0.9, 0.9, 0.0]
    flags = [True, True, False]
    rewards = eng.creator_problem_rewards(
        suite, rates, flags,
        scored_mask=[True, True, False],
        target_by_problem=[0.9, 0.5, 0.1],
        valid=True,
    )
    c = eng.cfg
    assert rewards[0] == pytest.approx(
        c.w_gradient * 1.0 + c.w_consistency + c.w_valid)
    assert rewards[1] == pytest.approx(
        c.w_gradient * math.exp(-c.mse_beta * 0.4 ** 2)
        + c.w_consistency + c.w_valid)
    # void rank: no gradient term, no consistency credit, shared validity only
    assert rewards[2] == pytest.approx(c.w_valid)
    assert rewards[0] > rewards[1] > rewards[2]


def test_problem_rewards_default_targets_from_ranks():
    eng = RewardEngine()
    suite = _suite([1.0, 0.0])           # deliberately out of order
    # targets by rank: difficulty 0.0 -> 1.0 (easy), 1.0 -> 0.0 (hard)
    rewards = eng.creator_problem_rewards(suite, [0.0, 1.0], [True, True])
    c = eng.cfg
    perfect = c.w_gradient + c.w_consistency  # validity fails (n=2 spread ok… check)
    assert rewards[0] == rewards[1]
    assert rewards[0] >= perfect            # both perfectly on-target


def test_problem_rewards_oracle_tax_is_per_problem():
    eng = RewardEngine()
    suite = _suite([0.0, 1.0])
    with_tax = eng.creator_problem_rewards(
        suite, [1.0, 0.0], [True, True], n_oracle_by_problem=[2, 0])
    no_tax = eng.creator_problem_rewards(suite, [1.0, 0.0], [True, True])
    assert with_tax[0] == pytest.approx(no_tax[0] - 2 * eng.cfg.w_oracle)
    assert with_tax[1] == pytest.approx(no_tax[1])


def test_problem_rewards_validate_lengths():
    eng = RewardEngine()
    suite = _suite([0.0, 1.0])
    with pytest.raises(ValueError):
        eng.creator_problem_rewards(suite, [1.0], [True, True])
    with pytest.raises(ValueError):
        eng.creator_problem_rewards(
            suite, [1.0, 0.0], [True, True], target_by_problem=[0.5])


# --------------------------------------------------------------------------- #
# trainer wiring: decomposed credit + batched solver group (scripted, no model)
# --------------------------------------------------------------------------- #
class _FakeAdapters:
    NAMES = ["A", "B"]

    def activate(self, name):
        pass

    def global_norm(self, name):
        return 0.0

    def drift_from(self, name, tree):
        return 0.0


class _FakeHarness:
    def __init__(self, outputs=()):
        self.calls = [ToolResult(ToolCall("solve", "q"), out, ok=True)
                      for out in outputs]


class _FakeBase:
    """Just enough of TwinBase for the batched-solver path."""

    def __init__(self):
        self.batch_calls: list[int] = []

    def render(self, user, system=None, enable_thinking=False, tools=None):
        return f"[sys]{system}[user]{user}"

    def generate_batch(self, prompts, *, max_tokens, temp, top_p,
                       completion_batch_size=32, seed=None):
        self.batch_calls.append(len(prompts))
        return [GenResult(text="<think>hm</think>\nANSWER: 42",
                          prompt_tokens=[1], completion_tokens=[2, 3])
                for _ in prompts]


def _react(text):
    return ReactResult(
        text=text, prompt_tokens=[1, 2], completion_tokens=[3, 4, 5],
        loss_mask=[1, 1, 1], n_rounds=1, n_tool_calls=1,
    )


def _problem_json(statement, answer, check):
    return (
        '{"statement": "%s", "difficulty": 0.5, '
        '"solution": "via tool", "answer": "%s", '
        '"verification": {"type": "math", "symbol": "x", "check": "%s"}}'
        % (statement, answer, check)
    )


def _make_trainer(cfg, creator_texts):
    t = SelfPlayTrainer.__new__(SelfPlayTrainer)
    t.cfg = cfg
    t.engine = RewardEngine(cfg.rewards)
    t.roles = RoleManager(swap_interval=100, warmup_iterations=0,
                          injection_rate=0.0, names=["A", "B"])
    t.rng = random.Random(0)
    t.logger = None
    t.transcript = None
    t.adapters = _FakeAdapters()
    t.base = _FakeBase()
    t._init_trees = {"A": None, "B": None}
    script = list(creator_texts)

    def fake_generate_creator(adapter, system, user):
        return _react(script.pop(0)), _FakeHarness(outputs=("42",))

    def fake_grpo(adapter, trajs):
        t.captured_grpo = getattr(t, "captured_grpo", {})
        t.captured_grpo[adapter] = list(trajs)
        return {"n_traj": len(trajs), "n_tokens": 0, "loss": 0.0, "pg": 0.0,
                "kl": 0.0, "grad_norm": 0.0}

    t._generate_creator = fake_generate_creator
    t._grpo = fake_grpo
    t._judge = lambda q: "VERDICT: INCORRECT"
    return t


def _config(**game_overrides):
    game = {"n_problems": 3, "creator_group": 1, "solver_attempts": 8,
            "domains": ["math"], "creator_mode": "per_problem",
            "condition_on_previous": False, "personas": False,
            "require_tool_use": False}
    game.update(game_overrides)
    return Config.from_dict({
        "game": game,
        "gen": {"solver_batch": 3},
        "rewards": {"target_hi": 0.9, "target_lo": 0.1},
        "roles": {"swap_interval": 100, "warmup_iterations": 0},
    })


_SCRIPT = [
    _problem_json("What is 40 + 2?", "42", "x = 40 + 2"),
    _problem_json("Solve 2y = 84 for y.", "42", "2*x = 84"),
    _problem_json("Solve 3z = 126 for z.", "42", "3*x = 126"),
]


def test_batched_solver_group_chunks_and_records():
    cfg = _config()
    trainer = _make_trainer(cfg, list(_SCRIPT))
    record = trainer.run_iteration(0)
    # K=8 at solver_batch=3 -> chunks of 3,3,2 per problem, 3 problems
    assert trainer.base.batch_calls[:3] == [3, 3, 2]
    assert len(trainer.base.batch_calls) == 9
    assert record["solver_stats"]["attempts"] == 24
    # think-share telemetry present and sane (solver text is ~half think)
    assert 0.0 < record["solver_think_share"] < 1.0
    assert record["creator_think_share"] == 0.0   # scripted creators: no think


def test_decomposed_credit_gives_ranks_their_own_rewards():
    cfg = _config(credit="per_problem")
    trainer = _make_trainer(cfg, list(_SCRIPT))
    record = trainer.run_iteration(0)
    creator = record["creator"]
    trajs = trainer.captured_grpo[creator]
    assert len(trajs) == 3
    # All three solved at rate 1.0 vs targets 0.9/0.5/0.1: the easy rank is
    # closest to target, the hard rank furthest -> strictly ordered rewards.
    by_rank = sorted(trajs, key=lambda t: t.meta["rank"])
    rs = [t.reward for t in by_rank]
    assert rs[0] > rs[1] > rs[2]
    # summary values are rounded to 4 decimals
    assert record["suites"][0]["problem_rewards"] == [
        pytest.approx(r, abs=1e-4) for r in rs]


def test_broadcast_credit_unchanged_by_default():
    cfg = _config()                        # credit defaults to "broadcast"
    assert cfg.game.credit == "broadcast"
    trainer = _make_trainer(cfg, list(_SCRIPT))
    record = trainer.run_iteration(0)
    creator = record["creator"]
    trajs = trainer.captured_grpo[creator]
    assert len({round(t.reward, 9) for t in trajs}) == 1
    assert "problem_rewards" not in record["suites"][0]


def test_solver_batch_default_is_sequential():
    cfg = Config.from_dict({})
    assert cfg.gen.solver_batch == 1


def test_creator_enable_thinking_config():
    # None (default) = follow model.enable_thinking; explicit override wins.
    assert Config.from_dict({}).model.creator_enable_thinking is None
    cfg = Config.from_dict({"model": {"enable_thinking": True,
                                      "creator_enable_thinking": False}})
    assert cfg.model.enable_thinking is True
    assert cfg.model.creator_enable_thinking is False
