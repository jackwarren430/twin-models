"""Sprint 8 — the trainer's per-problem path must grade partial parses
against the PROMPTED targets, scaled by the parsed fraction of the intended N
(audit 2026-07-03 bug 3), through the REAL ``run_iteration`` with scripted
generation (same pattern as tests/sprint7/test_loop_per_problem.py)."""

import math
import random

import pytest

from twin.config import Config
from twin.models.base import GenResult, ReactResult
from twin.rewards import RewardEngine
from twin.roles import RoleManager
from twin.tools.protocol import ToolCall, ToolResult
from twin.train.loop import SelfPlayTrainer


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


def _make_trainer(cfg, creator_texts, solver_answer):
    t = SelfPlayTrainer.__new__(SelfPlayTrainer)
    t.cfg = cfg
    t.engine = RewardEngine(cfg.rewards)
    t.roles = RoleManager(swap_interval=100, warmup_iterations=0,
                          injection_rate=0.0, names=["A", "B"])
    t.rng = random.Random(0)
    t.logger = None
    t.transcript = None
    t.adapters = _FakeAdapters()
    t._init_trees = {"A": None, "B": None}
    script = list(creator_texts)

    def fake_generate_creator(adapter, system, user):
        return _react(script.pop(0)), _FakeHarness(outputs=("42",))

    def fake_generate(adapter, system, user, *, max_tokens, temp):
        return GenResult(text=f"ANSWER: {solver_answer}",
                         prompt_tokens=[1], completion_tokens=[2, 3])

    def fake_grpo(adapter, trajs):
        return {"n_traj": len(trajs), "n_tokens": 0, "loss": 0.0, "pg": 0.0,
                "kl": 0.0, "grad_norm": 0.0}

    t._generate_creator = fake_generate_creator
    t._generate = fake_generate
    t._grpo = fake_grpo
    t._judge = lambda q: "VERDICT: INCORRECT"
    return t


def test_partial_parse_uses_prompted_targets_and_intended_n():
    cfg = Config.from_dict({
        "game": {"n_problems": 3, "creator_group": 1, "solver_attempts": 2,
                 "domains": ["math"], "creator_mode": "per_problem",
                 "condition_on_previous": True, "personas": False,
                 "require_tool_use": False},
        "rewards": {"target_hi": 0.9, "target_lo": 0.1},
        "roles": {"swap_interval": 100, "warmup_iterations": 0},
    })
    # Ranks 0 and 1 parse and are consistent; rank 2 (the HARD one, prompted
    # target 0.1) is dropped. Solver answers everything right -> rates [1, 1].
    trainer = _make_trainer(cfg, [
        _problem_json("What is 40 + 2?", "42", "x = 40 + 2"),
        _problem_json("Solve 2y = 84 for y.", "42", "2*x = 84"),
        "I ran out of budget mid-think",
    ], solver_answer="42")
    record = trainer.run_iteration(0)
    sm = record["suites"][0]
    assert sm["n_problems"] == 2 and sm["n_consistent"] == 2

    # Prompted targets for ranks 0,1 over the 0.9->0.1 band with N=3:
    # [0.9, 0.5] — NOT the re-stretched [0.9, 0.1]. Scaled by 2/3.
    mse = ((1.0 - 0.9) ** 2 + (1.0 - 0.5) ** 2) / 2
    expect = math.exp(-cfg.rewards.mse_beta * mse) * 2 / 3
    assert sm["r_gradient"] == pytest.approx(expect, abs=1e-4)
    # The buggy value (re-stretched ramp, no intended-N scaling) is far off.
    buggy_mse = ((1.0 - 0.9) ** 2 + (1.0 - 0.1) ** 2) / 2
    buggy = math.exp(-cfg.rewards.mse_beta * buggy_mse)
    assert abs(sm["r_gradient"] - buggy) > 0.1
