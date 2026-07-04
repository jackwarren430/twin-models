"""Sprint 9 (landed with the sprint8 batch) — diversity telemetry + the
config-gated cross-suite repetition penalty (the R-Zero-style mechanism the
lit review flagged as twin's biggest missing piece)."""

import random

import pytest

from twin.analysis.diversity import (
    cross_suite_penalties,
    jaccard,
    mean_pairwise_similarity,
)
from twin.config import Config
from twin.models.base import GenResult, ReactResult
from twin.rewards import RewardEngine
from twin.roles import RoleManager
from twin.tools.protocol import ToolCall, ToolResult
from twin.train.loop import SelfPlayTrainer


# --------------------------------------------------------------------------- #
# pure functions
# --------------------------------------------------------------------------- #
def test_jaccard_extremes():
    assert jaccard("solve for x in 2x + 3 = 11", "solve for x in 2x + 3 = 11") == 1.0
    assert jaccard("integrate x squared", "the train leaves at noon") == 0.0
    assert jaccard("", "anything at all") == 0.0


def test_jaccard_template_edit_stays_high():
    # Swapping one constant in a template leaves most bigrams intact — the
    # collapse signature the metric must catch.
    a = "A store sells 3 apples and 2 pears for 22 dollars"
    b = "A store sells 3 apples and 2 pears for 21 dollars"
    assert jaccard(a, b) > 0.6


def test_quadratic_coefficient_swap_reads_as_duplicate():
    # mini-04b's live collapse signature (EXPERIMENTS checkpoint-10 review):
    # the raw-token metric read pairs like these at ~0.3-0.7 because "5x" and
    # "2x" tokenize as different words; digit-normalized they are the same
    # template and must read as duplicates.
    a = "Solve the quadratic equation $x^2 - 5x + 6 = 0$ and select the maximum root."
    b = "Solve the quadratic equation $x^2 - 2x + 1 = 0$ and select the maximum root."
    assert jaccard(a, b) == 1.0
    c = "Find the smallest positive integer divisible by 12, 15, and 20."
    assert jaccard(a, c) < 0.3


def test_mean_pairwise_similarity():
    assert mean_pairwise_similarity([]) == 0.0
    assert mean_pairwise_similarity(["one problem only"]) == 0.0
    sims = mean_pairwise_similarity(["a b c d", "a b c d", "x y z w"])
    assert 0.0 < sims < 1.0


def test_cross_suite_penalties_flag_the_copycat():
    original = ["solve the system x + y = 10 and x - y = 2",
                "a chemist mixes two salt solutions of 20 and 50 percent"]
    copycat = ["solve the system x + y = 12 and x - y = 4",
               "a chemist mixes two salt solutions of 30 and 60 percent"]
    fresh = ["how many primes are less than 100",
             "a rectangle has perimeter 36 and area 80, find its sides"]
    p_orig, p_copy, p_fresh = cross_suite_penalties([original, copycat, fresh])
    # the copier and the copied both read high vs each other; fresh reads low
    assert p_copy > p_fresh and p_orig > p_fresh
    assert p_fresh < 0.2


def test_cross_suite_penalties_edge_cases():
    assert cross_suite_penalties([]) == []
    assert cross_suite_penalties([["only suite"]]) == [0.0]
    assert cross_suite_penalties([[], ["a b c"]]) == [0.0, 0.0]


# --------------------------------------------------------------------------- #
# trainer wiring (scripted, no model)
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
    t._init_trees = {"A": None, "B": None}
    script = list(creator_texts)

    def fake_generate_creator(adapter, system, user):
        return _react(script.pop(0)), _FakeHarness(outputs=("42",))

    def fake_generate(adapter, system, user, *, max_tokens, temp):
        return GenResult(text="ANSWER: 42", prompt_tokens=[1],
                         completion_tokens=[2, 3])

    def fake_grpo(adapter, trajs):
        t.captured_grpo = getattr(t, "captured_grpo", {})
        t.captured_grpo[adapter] = list(trajs)
        return {"n_traj": len(trajs), "n_tokens": 0, "loss": 0.0, "pg": 0.0,
                "kl": 0.0, "grad_norm": 0.0}

    t._generate_creator = fake_generate_creator
    t._generate = fake_generate
    t._grpo = fake_grpo
    t._judge = lambda q: "VERDICT: INCORRECT"
    return t


def _config(w_diversity=0.0):
    return Config.from_dict({
        "game": {"n_problems": 2, "creator_group": 2, "solver_attempts": 1,
                 "domains": ["math"], "creator_mode": "per_problem",
                 "condition_on_previous": False, "personas": False,
                 "require_tool_use": False},
        "rewards": {"target_hi": 0.9, "target_lo": 0.1,
                    "w_diversity": w_diversity},
        "roles": {"swap_interval": 100, "warmup_iterations": 0},
    })


# suite 0 and suite 1 rank-0 problems are near-duplicates; rank-1s differ
_SCRIPT = [
    _problem_json("What is 40 plus 2 in total?", "42", "x = 40 + 2"),
    _problem_json("A rectangle of area 84 has width 2, find its length.",
                  "42", "2*x = 84"),
    _problem_json("What is 40 plus 2 in total, friend?", "42", "x = 40 + 2"),
    _problem_json("Three crates hold 126 melons equally, how many each?",
                  "42", "3*x = 126"),
]


def test_repetition_telemetry_always_logged():
    trainer = _make_trainer(_config(), list(_SCRIPT))
    record = trainer.run_iteration(0)
    assert record["problem_similarity"] > 0.0
    reps = [sm["repetition"] for sm in record["suites"]]
    assert len(reps) == 2 and all(r > 0.0 for r in reps)
    # rewards untouched when the penalty weight is 0
    creator = record["creator"]
    baseline = {round(t.reward, 6) for t in trainer.captured_grpo[creator]}
    assert len(baseline) == 1


def test_diversity_penalty_lowers_repetitive_suite_rewards():
    base_trainer = _make_trainer(_config(0.0), list(_SCRIPT))
    base_rec = base_trainer.run_iteration(0)
    pen_trainer = _make_trainer(_config(0.5), list(_SCRIPT))
    pen_rec = pen_trainer.run_iteration(0)
    creator = pen_rec["creator"]
    base_rewards = sorted(t.reward for t in base_trainer.captured_grpo[creator])
    pen_rewards = sorted(t.reward for t in pen_trainer.captured_grpo[creator])
    assert all(p < b for p, b in zip(pen_rewards, base_rewards))
    # penalty magnitude matches the logged per-suite repetition
    reps = [sm["repetition"] for sm in pen_rec["suites"]]
    assert max(base_rewards) - max(pen_rewards) == pytest.approx(
        0.5 * min(reps), abs=1e-3)
