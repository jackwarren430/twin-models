"""Sprint 8 — coding pipeline. mini-02's coding domain scored 0/141
consistent because the creator prompt NEVER asked for ``verification.tests``
(the field/rules blocks were math-only empty strings) while ``verify_code``
fails closed with "no tests supplied". These tests lock the whole repaired
path: contract in the prompts, run_python creator tool, code-aware solver
prompt + extraction, and a scripted end-to-end iteration where consistency
and solver grading run through the REAL sandbox."""

import json
import random

from twin.config import Config
from twin.models.base import GenResult, ReactResult
from twin.problems.schema import Problem
from twin.prompts import (
    SOLVER_CODE_SYSTEM,
    creator_problem_user,
    creator_user,
    is_code_problem,
    solver_system,
    solver_user,
)
from twin.rewards import RewardEngine
from twin.roles import RoleManager
from twin.tools import tool_schemas
from twin.tools.native import parse_native_tool_calls
from twin.tools.protocol import ToolCall, ToolResult
from twin.tools.sandbox import SandboxResult, format_sandbox_result
from twin.train.extract import extract_code_block
from twin.train.loop import SelfPlayTrainer


# --------------------------------------------------------------------------- #
# creator contract
# --------------------------------------------------------------------------- #
def test_coding_creator_contract_demands_tests():
    u = creator_user("coding", "string manipulation", 3)
    for needle in ("solution_code", "tests", "entry_point", "run_python",
                   '"type": "code"', "assert"):
        assert needle in u, needle
    # and no math certificate leakage
    assert '"symbol"' not in u and '"check"' not in u


def test_coding_per_problem_contract_demands_tests():
    u = creator_problem_user(
        "coding", "recursion basics", rank=2, n_problems=3,
        difficulty=1.0, target_rate=0.1)
    for needle in ("solution_code", "tests", "entry_point", "run_python"):
        assert needle in u, needle


def test_math_contract_unchanged():
    u = creator_user("math", "linear equations", 3)
    assert '"check"' in u and '"symbol"' in u and "solve" in u
    assert "solution_code" not in u and "run_python" not in u


# --------------------------------------------------------------------------- #
# run_python tool plumbing
# --------------------------------------------------------------------------- #
def test_run_python_schema_exposed():
    schemas = tool_schemas(["solve", "calc", "run_python"])
    names = [s["function"]["name"] for s in schemas]
    assert names == ["solve", "calc", "run_python"]


def test_native_parser_maps_code_argument():
    calls = parse_native_tool_calls(
        '<tool_call>{"name": "run_python", "arguments": {"code": '
        '"print(6*7)"}}</tool_call>')
    assert len(calls) == 1
    assert calls[0].name == "run_python" and calls[0].arg == "print(6*7)"


def test_format_sandbox_result():
    ok = SandboxResult(ok=True, returncode=0, stdout="42\n", stderr="", timed_out=False)
    assert format_sandbox_result(ok) == "42"
    silent = SandboxResult(ok=True, returncode=0, stdout="", stderr="", timed_out=False)
    assert "print()" in format_sandbox_result(silent)
    err = SandboxResult(ok=False, returncode=1, stdout="",
                        stderr="Traceback...\nAssertionError", timed_out=False)
    out = format_sandbox_result(err)
    assert out.startswith("ERROR (exit 1)") and "AssertionError" in out
    to = SandboxResult(ok=False, returncode=None, stdout="", stderr="", timed_out=True)
    assert format_sandbox_result(to).startswith("TIMEOUT")


def test_tool_runner_executes_real_sandboxed_python():
    t = SelfPlayTrainer.__new__(SelfPlayTrainer)
    t.cfg = Config.from_dict({})
    runner, harness = t._build_tool_runner(
        ["solve", "run_python"], max_tool_calls=4, native=True)
    obs = runner('<tool_call>{"name": "run_python", "arguments": '
                 '{"code": "print(6*7)"}}</tool_call>')
    assert obs is not None and "42" in obs
    assert harness.calls[0].ok


# --------------------------------------------------------------------------- #
# solver code path
# --------------------------------------------------------------------------- #
def _code_problem(entry="add_two", solution="def add_two(x):\n    return x + 2",
                  tests="assert add_two(1) == 3\nassert add_two(-2) == 0"):
    return Problem(
        statement=f"Write a function {entry}(x) returning x plus two.",
        difficulty=0.0,
        answer=entry,
        solution="Add 2 to the input.",
        domain="coding",
        verification={"type": "code", "entry_point": entry,
                      "solution_code": solution, "tests": tests},
    )


def test_is_code_problem_and_prompts():
    p = _code_problem()
    assert is_code_problem(p)
    assert not is_code_problem(Problem(statement="2+2?", difficulty=0.0))
    u = solver_user(p)
    assert "```python" in u and "`add_two`" in u and "ANSWER:" not in u
    assert "```python" in solver_system(code=True)
    assert solver_system(code=True).endswith(SOLVER_CODE_SYSTEM)


def test_extract_code_block_think_aware():
    text = ("<think>draft:\n```python\ndef add_two(x): return x\n```\n"
            "no wait</think>\n```python\ndef add_two(x):\n    return x + 2\n```")
    assert "x + 2" in extract_code_block(text)
    assert "return x\n" not in extract_code_block(text)
    # unclosed think: fall back to the raw text's block
    assert "x + 2" in extract_code_block(
        "<think>```python\ndef add_two(x):\n    return x + 2\n```")
    assert extract_code_block("no fence at all") == "no fence at all"


# --------------------------------------------------------------------------- #
# end-to-end scripted iteration (REAL sandbox verification, no model)
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
        self.calls = [ToolResult(ToolCall("run_python", "q"), out, ok=True)
                      for out in outputs]


def _react(text):
    return ReactResult(
        text=text, prompt_tokens=[1, 2], completion_tokens=[3, 4, 5],
        loss_mask=[1, 1, 1], n_rounds=1, n_tool_calls=1,
    )


def _problem_payload(entry, solution_code, tests, statement):
    return json.dumps({
        "statement": statement,
        "difficulty": 0.5,
        "solution": "implement directly",
        "answer": entry,
        "verification": {"type": "code", "entry_point": entry,
                         "solution_code": solution_code, "tests": tests},
    })


def test_coding_iteration_end_to_end():
    cfg = Config.from_dict({
        "game": {"n_problems": 2, "creator_group": 1, "solver_attempts": 1,
                 "domains": ["coding"], "creator_mode": "per_problem",
                 "condition_on_previous": False, "personas": False,
                 "require_tool_use": False},
        "roles": {"swap_interval": 100, "warmup_iterations": 0},
    })
    good = _problem_payload(
        "add_two", "def add_two(x):\n    return x + 2",
        "assert add_two(1) == 3\nassert add_two(-2) == 0",
        "Write a function add_two(x) returning x plus 2.")
    broken = _problem_payload(
        "triple", "def triple(x):\n    return x * 2",     # WRONG solution
        "assert triple(3) == 9",
        "Write a function triple(x) returning 3*x.")
    script = [(good, _FakeHarness(outputs=("ok",))),
              (broken, _FakeHarness(outputs=("ok",)))]

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
    captured = {"solver_systems": [], "solver_users": []}

    def fake_generate_creator(adapter, system, user):
        text, harness = script.pop(0)
        return _react(text), harness

    def fake_generate(adapter, system, user, *, max_tokens, temp):
        captured["solver_systems"].append(system)
        captured["solver_users"].append(user)
        return GenResult(
            text=("<think>easy</think>\n```python\ndef add_two(x):\n"
                  "    return x + 2\n```"),
            prompt_tokens=[1], completion_tokens=[2, 3])

    t._generate_creator = fake_generate_creator
    t._generate = fake_generate
    t._grpo = lambda adapter, trajs: {
        "n_traj": len(trajs), "n_tokens": 0, "loss": 0.0, "pg": 0.0,
        "kl": 0.0, "grad_norm": 0.0}
    t._judge = lambda q: "VERDICT: INCORRECT"

    record = t.run_iteration(0)
    sm = record["suites"][0]
    # the good problem passed its own tests (REAL sandbox), the broken one
    # was voided by its own failing tests — the exact mini-02 bug class
    assert sm["n_consistent"] == 1
    # the solver saw the CODE contract, not the math ANSWER-line one
    assert captured["solver_systems"][0].endswith(SOLVER_CODE_SYSTEM)
    assert "```python" in captured["solver_users"][0]
    # and its extracted code block passed the creator's tests
    assert record["solver_stats"]["problems_solved"] == 1
