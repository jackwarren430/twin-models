"""Sprint 7 — per-problem creator mode through the REAL ``run_iteration``:
N separate rollouts per suite, conditioning on previous problem JSONs,
rank-dictated difficulty, broadcast credit (suite reward shared by parsed
ranks, parse-gate for garbage ranks), the strict tool gate, and the new
tool-adoption telemetry. Every model call is stubbed (no weights, fast) —
the machinery around them is the real code."""

import random

from twin.config import Config
from twin.models.base import GenResult, ReactResult
from twin.rewards import RewardEngine
from twin.roles import RoleManager
from twin.tools.protocol import ToolCall, ToolResult
from twin.train.loop import SelfPlayTrainer


# --------------------------------------------------------------------------- #
# Scripted stand-ins
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
    """Mimics ToolHarness after a rollout: .calls with ok/output."""

    def __init__(self, outputs=(), ok=True):
        self.calls = [
            ToolResult(ToolCall("solve", "q"), out, ok=ok) for out in outputs
        ]


def _react(text, n_tool_calls=1):
    return ReactResult(
        text=text, prompt_tokens=[1, 2], completion_tokens=[3, 4, 5],
        loss_mask=[1, 1, 1], n_rounds=1, n_tool_calls=n_tool_calls,
    )


def _problem_json(statement, difficulty, answer, check):
    return (
        '{"statement": "%s", "difficulty": %s, '
        '"solution": "derived with the tool", "answer": "%s", '
        '"verification": {"type": "math", "symbol": "x", "check": "%s"}}'
        % (statement, difficulty, answer, check)
    )


def _config(**game_overrides):
    game = {
        "n_problems": 3, "creator_group": 2, "solver_attempts": 2,
        "domains": ["math"], "creator_mode": "per_problem",
        "condition_on_previous": True, "personas": True,
        "require_tool_use": False,
    }
    game.update(game_overrides)
    return Config.from_dict({
        "game": game,
        "rewards": {"target_hi": 0.9, "target_lo": 0.1},
        "tools": {"protocol": "native"},
        "roles": {"swap_interval": 100, "warmup_iterations": 0},
    })


def _make_trainer(cfg, creator_script, solver_answers):
    """A SelfPlayTrainer with real reward/role/record machinery and scripted
    generation. ``creator_script``: list of (text, harness) per creator
    rollout, consumed in order. ``solver_answers``: cycled answers."""
    t = SelfPlayTrainer.__new__(SelfPlayTrainer)
    t.cfg = cfg
    t.engine = RewardEngine(cfg.rewards)
    t.roles = RoleManager(
        swap_interval=cfg.roles.swap_interval,
        warmup_iterations=cfg.roles.warmup_iterations,
        injection_rate=0.0, names=["A", "B"],
    )
    t.rng = random.Random(0)
    t.logger = None
    t.transcript = None
    t.adapters = _FakeAdapters()
    t._init_trees = {"A": None, "B": None}

    t.captured = {"creator_prompts": [], "creator_systems": [], "grpo": {}}
    script = list(creator_script)

    def fake_generate_creator(adapter, system, user):
        t.captured["creator_systems"].append(system)
        t.captured["creator_prompts"].append(user)
        text, harness = script.pop(0)
        return _react(text), harness

    answers = list(solver_answers)

    def fake_generate(adapter, system, user, *, max_tokens, temp):
        ans = answers.pop(0) if answers else "0"
        return GenResult(
            text=f"working...\nANSWER: {ans}",
            prompt_tokens=[1], completion_tokens=[2, 3],
        )

    def fake_grpo(adapter, trajs):
        t.captured["grpo"][adapter] = list(trajs)
        return {"n_traj": len(trajs), "n_tokens": 0, "loss": 0.0, "pg": 0.0,
                "kl": 0.0, "grad_norm": 0.0}

    def fake_judge(question):
        return "VERDICT: INCORRECT"  # cert-less problems would fail closed

    t._generate_creator = fake_generate_creator
    t._generate = fake_generate
    t._grpo = fake_grpo
    t._judge = fake_judge
    return t


# Suite 0: three parseable problems (answers 4, 5, 6). Suite 1: two parseable,
# middle rank is garbage (no JSON).
def _script():
    p = [
        _problem_json("What is 1 + 3?", 0.0, 4, "x = 1 + 3"),
        _problem_json("Solve 2y = 10 for y.", 0.5, 5, "2*x = 10"),
        _problem_json("Solve 3z = 18 for z.", 1.0, 6, "3*x = 18"),
        _problem_json("What is 2 + 2?", 0.0, 4, "x = 2 + 2"),
        "I ran out of ideas mid-thought",
        _problem_json("Solve 4w = 28 for w.", 1.0, 7, "4*x = 28"),
    ]
    return [(text, _FakeHarness(outputs=("4", "5", "6", "4", "", "7")[i:i + 1]))
            for i, text in enumerate(p)]


def test_per_problem_generation_and_broadcast_credit():
    cfg = _config()
    # 5 scored problems reach the solver, K=2 -> 10 attempts; make them all
    # correct-by-echo? No: answer them right for suite 0, wrong for suite 1.
    trainer = _make_trainer(cfg, _script(),
                            solver_answers=["4", "4", "5", "5", "6", "6",
                                            "4", "4", "7", "7"])
    record = trainer.run_iteration(0)

    creator = record["creator"]
    creator_trajs = trainer.captured["grpo"][creator]
    # 2 suites x 3 ranks = 6 creator trajectories
    assert len(creator_trajs) == 6

    # broadcast: suite 0's three parsed ranks share one reward
    s0 = [t for t in creator_trajs if t.meta.get("suite_id") ==
          creator_trajs[0].meta["suite_id"]]
    assert len(s0) == 3
    assert len({round(t.reward, 9) for t in s0}) == 1
    assert all(t.meta["parsed"] for t in s0)

    # suite 1: the garbage middle rank got the parse gate, the others share
    s1 = [t for t in creator_trajs if t not in s0]
    failed = [t for t in s1 if not t.meta["parsed"]]
    parsed = [t for t in s1 if t.meta["parsed"]]
    assert len(failed) == 1 and failed[0].reward == -1.0
    assert len({round(t.reward, 9) for t in parsed}) == 1
    assert parsed[0].reward != -1.0

    # summaries carry the per-suite bookkeeping
    sm0, sm1 = record["suites"]
    assert sm0["n_rollouts"] == 3 and sm0["n_parse_failed"] == 0
    assert sm1["n_rollouts"] == 3 and sm1["n_parse_failed"] == 1
    assert sm1["n_problems"] == 2          # suite shrank, ranks kept order

    # rank-dictated difficulties (0, 0.5, 1 for N=3), not the model's claims
    ranks = sorted(t.meta["rank"] for t in s0)
    assert ranks == [0, 1, 2]

    # answer-in-obs diagnostic: every parsed answer was scripted into an obs
    assert record["creator_answer_in_obs"] == 5
    assert record["creator_tool_ok"] == 6   # one ok call per rollout scripted


def test_conditioning_injects_previous_jsons_and_targets():
    cfg = _config()
    trainer = _make_trainer(cfg, _script(),
                            solver_answers=["4"] * 10)
    trainer.run_iteration(0)
    prompts = trainer.captured["creator_prompts"]
    assert len(prompts) == 6
    # rank 0 of suite 0: no conditioning block, easy target from the band
    assert "already wrote" not in prompts[0]
    assert "90%" in prompts[0]
    # rank 1 sees problem 0's JSON; rank 2 sees both
    assert "What is 1 + 3?" in prompts[1]
    assert "What is 1 + 3?" in prompts[2] and "2y = 10" in prompts[2]
    # target ramp for N=3 over 0.9->0.1: 90 / 50 / 10 %
    assert "50%" in prompts[1] and "10%" in prompts[2]
    # suite 1 starts fresh — no carryover across suites
    assert "What is 1 + 3?" not in prompts[3]
    # the garbage rank parsed nothing, so rank 2 of suite 1 sees only rank 0
    assert "2 + 2" in prompts[5] and "ran out of ideas" not in prompts[5]
    # personas: creator system names alpha/omega
    sys0 = trainer.captured["creator_systems"][0]
    assert '"alpha"' in sys0 or '"omega"' in sys0


def test_conditioning_off_keeps_prompts_independent():
    cfg = _config(condition_on_previous=False)
    trainer = _make_trainer(cfg, _script(), solver_answers=["4"] * 10)
    trainer.run_iteration(0)
    for u in trainer.captured["creator_prompts"]:
        assert "already wrote" not in u


def test_strict_tool_gate_voids_toolless_problems():
    cfg = _config(require_tool_use=True)
    script = _script()
    # strip the tool calls from suite 0's middle rank -> gated
    script[1] = (script[1][0], _FakeHarness(outputs=()))
    trainer = _make_trainer(cfg, script, solver_answers=["4"] * 10)
    record = trainer.run_iteration(0)
    sm0 = record["suites"][0]
    assert sm0["n_tool_gated"] == 1
    assert record["creator_tool_gated"] == 1
    # the gated problem is void: only 2 of suite 0's problems were scored
    assert sm0["n_consistent"] <= 2


def test_suite_mode_regression_single_rollout_semantics():
    """creator_mode=suite still produces one trajectory per suite with the
    same reward semantics as v1 (broadcast over a single rollout is v1)."""
    cfg = _config(creator_mode="suite")
    suite_json = (
        '{"theme": "t", "domain": "math", "problems": ['
        + _problem_json("What is 1 + 3?", 0.0, 4, "x = 1 + 3") + ","
        + _problem_json("Solve 2y = 10 for y.", 0.5, 5, "2*x = 10") + ","
        + _problem_json("Solve 3z = 18 for z.", 1.0, 6, "3*x = 18")
        + "]}"
    )
    script = [(suite_json, _FakeHarness(outputs=("4",))),
              ("no json here", _FakeHarness(outputs=()))]
    trainer = _make_trainer(cfg, script, solver_answers=["4", "4", "5", "5", "6", "6"])
    record = trainer.run_iteration(0)
    creator = record["creator"]
    trajs = trainer.captured["grpo"][creator]
    assert len(trajs) == 2                       # one per suite, as in v1
    assert trajs[1].reward == -1.0               # parse gate
    assert record["parse_ok_rate"] == 0.5
