"""Domain expansion (2026-07-04) — Knights & Knaves logic domain.

The verifiable-logic staple of the RLVR literature (Logic-RL arXiv:2502.14768;
Reasoning Gym arXiv:2505.24760), added because mini-04b measured consistency
as the creator's binding constraint: K&K authoring with the ``logic_solve``
tool gives by-construction answers (AZR's property) while solving stays
genuinely hard for a text model as characters/nesting grow.

Hand-verified fixtures (truth tables worked in the comments, not trusted to
memory):

* ``A: ~B; B: ~(A ^ C); C: ~A`` — unique: a=knight, b=knave, c=knave.
* ``A: B & C; B: ~A`` — C silent but pinned; unique: a=knave, b=knight,
  c=knave (A's lie forces c false once b is true).
* ``A: ~B; B: ~A`` — TWO solutions (either one knight) -> ambiguous.
* ``A: B; B: ~A`` — zero solutions -> contradictory.
"""

import random

import pytest

from twin.config import Config
from twin.models.base import GenResult, ReactResult
from twin.problems.schema import Problem
from twin.prompts import (
    SOLVER_LOGIC_SYSTEM,
    creator_user,
    is_logic_problem,
    solver_system,
    solver_user,
)
from twin.rewards import RewardEngine
from twin.roles import RoleManager
from twin.tools.logic import logic_solve
from twin.tools.native import tool_schemas
from twin.tools.protocol import ToolCall, ToolResult
from twin.train.loop import SelfPlayTrainer
from twin.verifiers.dispatch import check_consistency, verify_answer
from twin.verifiers.logic_verifier import (
    ClaimParseError,
    check_logic_consistency,
    format_assignment,
    parse_assignment,
    solve_claims,
    verify_logic,
)

_UNIQUE = "A: ~B; B: ~(A ^ C); C: ~A"
_UNIQUE_ANSWER = "A=knight, B=knave, C=knave"
_SILENT = "A: B & C; B: ~A"
_SILENT_ANSWER = "A=knave, B=knight, C=knave"
_AMBIGUOUS = "A: ~B; B: ~A"
_CONTRADICTORY = "A: B; B: ~A"


# --------------------------------------------------------------------------- #
# parser + enumeration
# --------------------------------------------------------------------------- #
def test_unique_puzzle_enumerates_one_solution():
    people, sols = solve_claims(_UNIQUE)
    assert people == ["a", "b", "c"]
    assert sols == [{"a": True, "b": False, "c": False}]


def test_silent_character_is_inferred_and_pinned():
    people, sols = solve_claims(_SILENT)
    assert people == ["a", "b", "c"]          # c never speaks, only mentioned
    assert sols == [{"a": False, "b": True, "c": False}]


def test_ambiguous_and_contradictory_counts():
    assert len(solve_claims(_AMBIGUOUS)[1]) == 2
    assert solve_claims(_CONTRADICTORY)[1] == []


def test_operator_words_and_literals():
    # words normalize to symbols; bare name = knight; literals allowed
    people, sols = solve_claims("A: not B; B: not (A xor C); C: not A")
    assert sols == [{"a": True, "b": False, "c": False}]
    # 'A: false' forces A to be a knave (a knight cannot say a falsehood)
    _, sols = solve_claims("A: false; B: ~A")
    assert sols == [{"a": False, "b": True}]


def test_implication_is_right_associative():
    # A -> B -> C must parse as A -> (B -> C): under all-false it is
    # F -> (F -> F) = True, where left association would give
    # (F -> F) -> F = False.
    from twin.verifiers.logic_verifier import _eval, _Parser, _tokenize
    ast = _Parser(_tokenize("A -> B -> C")).parse()
    assert _eval(ast, {"a": False, "b": False, "c": False}) is True


def test_newline_separated_claims_parse():
    _, sols = solve_claims("A: ~B\nB: ~(A ^ C)\nC: ~A")
    assert len(sols) == 1


def test_bad_dsl_fails_closed():
    for bad in ["", "no colon here", "A: B ++ C", "A: (B", "A: ", "true: B"]:
        with pytest.raises(ClaimParseError):
            solve_claims(bad)


def test_people_cap():
    claims = "; ".join(f"P{i}: true" for i in range(9))
    with pytest.raises(ClaimParseError):
        solve_claims(claims)


# --------------------------------------------------------------------------- #
# answer parsing
# --------------------------------------------------------------------------- #
def test_parse_assignment_formats_and_synonyms():
    people = ["a", "b"]
    want = {"a": True, "b": False}
    for text in [
        "A=knight, B=knave",
        "a: knight; b: knave",
        "A=truth-teller, B=liar",
        "A = true, B = false",
    ]:
        assert parse_assignment(text, people) == want


def test_parse_assignment_fails_closed():
    people = ["a", "b"]
    for bad in ["A=knight", "A=knight, B=knight, C=knave",
                "A=knight, A=knave, B=knight", "A=wizard, B=knave", ""]:
        with pytest.raises(ClaimParseError):
            parse_assignment(bad, people)


def test_format_assignment_roundtrip():
    assign = {"a": True, "b": False}
    assert format_assignment(assign) == "a=knight, b=knave"
    assert parse_assignment(format_assignment(assign), ["a", "b"]) == assign


# --------------------------------------------------------------------------- #
# verifier entry points
# --------------------------------------------------------------------------- #
def test_consistency_ok_and_answer_mismatch():
    assert check_logic_consistency(_UNIQUE, _UNIQUE_ANSWER).correct
    r = check_logic_consistency(_UNIQUE, "A=knave, B=knight, C=knight")
    assert not r.correct and "differs" in r.detail


def test_consistency_rejects_ill_posed():
    assert not check_logic_consistency(_AMBIGUOUS, "A=knight, B=knave").correct
    assert not check_logic_consistency(_CONTRADICTORY, "A=knight, B=knave").correct
    assert not check_logic_consistency("", "A=knight").correct
    # single person is below MIN_PEOPLE even if unique
    assert not check_logic_consistency("A: true", "A=knight").correct


def test_verify_logic_grades_against_enumeration():
    assert verify_logic(_UNIQUE_ANSWER, _UNIQUE).correct
    assert verify_logic("a: liar, b: truth-teller, c: liar", _SILENT).correct
    wrong = verify_logic("A=knight, B=knight, C=knave", _UNIQUE)
    assert not wrong.correct and "b" in wrong.detail
    assert not verify_logic("gibberish", _UNIQUE).correct
    assert not verify_logic(_UNIQUE_ANSWER, _AMBIGUOUS).correct


def _logic_problem(claims=_UNIQUE, answer=_UNIQUE_ANSWER, **kw):
    return Problem(
        statement="Who is a knight and who is a knave?",
        difficulty=0.5, answer=answer,
        verification={"type": "logic", "claims": claims}, **kw)


def test_dispatch_selects_logic():
    p = _logic_problem()
    assert check_consistency(p).method == "logic"
    assert verify_answer(p, _UNIQUE_ANSWER).correct
    assert not verify_answer(p, "A=knave, B=knave, C=knave").correct
    # domain fallback without explicit vtype
    p2 = Problem(statement="s", difficulty=0.1, answer=_UNIQUE_ANSWER,
                 domain="logic", verification={"claims": _UNIQUE})
    assert check_consistency(p2).method == "logic"


# --------------------------------------------------------------------------- #
# the creator's authoring tool
# --------------------------------------------------------------------------- #
def test_logic_solve_tool_verdicts():
    assert logic_solve(_UNIQUE) == "UNIQUE solution: a=knight, b=knave, c=knave"
    assert logic_solve(_AMBIGUOUS).startswith("AMBIGUOUS: 2 ")
    assert logic_solve(_CONTRADICTORY).startswith("CONTRADICTORY")
    assert logic_solve("???").startswith("PARSE ERROR")


def test_logic_solve_native_schema_registered():
    (schema,) = tool_schemas(["logic_solve"])
    assert schema["function"]["name"] == "logic_solve"
    assert "claims" in schema["function"]["parameters"]["properties"]


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
def test_creator_contract_carries_logic_certificate():
    text = creator_user("logic", "direct accusations", 3)
    assert '"type": "logic"' in text
    assert '"claims"' in text
    assert "logic_solve" in text
    assert "exactly one solution" in text.lower() or "EXACTLY ONE" in text


def test_solver_prompts_use_assignment_contract():
    p = _logic_problem()
    assert is_logic_problem(p)
    assert not is_logic_problem("plain text")
    assert "=knight" in solver_user(p)
    assert solver_system(logic=True) == SOLVER_LOGIC_SYSTEM
    assert "ANSWER" in SOLVER_LOGIC_SYSTEM


# --------------------------------------------------------------------------- #
# scripted end-to-end iteration (no model) — mirrors test_diversity scaffolding
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
        self.calls = [ToolResult(ToolCall("logic_solve", "q"), out, ok=True)
                      for out in outputs]


def _react(text):
    return ReactResult(
        text=text, prompt_tokens=[1, 2], completion_tokens=[3, 4, 5],
        loss_mask=[1, 1, 1], n_rounds=1, n_tool_calls=1,
    )


def _problem_json(statement, answer, claims):
    return (
        '{"statement": "%s", "difficulty": 0.5, "domain": "logic", '
        '"solution": "via logic_solve", "answer": "%s", '
        '"verification": {"type": "logic", "claims": "%s"}}'
        % (statement, answer, claims)
    )


_SCRIPT = [
    _problem_json("Puzzle one: who is what?", _UNIQUE_ANSWER, _UNIQUE),
    _problem_json("Puzzle two (ill-posed on purpose).", "A=knight, B=knave",
                  _AMBIGUOUS),
]


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
        return _react(script.pop(0)), _FakeHarness(outputs=("UNIQUE solution: ...",))

    def fake_generate(adapter, system, user, *, max_tokens, temp):
        # the loop must have selected the logic solver contract
        assert "knight" in system
        return GenResult(text=f"cases...\nANSWER: {solver_answer}",
                         prompt_tokens=[1], completion_tokens=[2, 3])

    def fake_grpo(adapter, trajs):
        return {"n_traj": len(trajs), "n_tokens": 0, "loss": 0.0, "pg": 0.0,
                "kl": 0.0, "grad_norm": 0.0}

    t._generate_creator = fake_generate_creator
    t._generate = fake_generate
    t._grpo = fake_grpo
    t._judge = lambda q: "VERDICT: INCORRECT"
    return t


def _config():
    return Config.from_dict({
        "game": {"n_problems": 2, "creator_group": 1, "solver_attempts": 2,
                 "domains": ["logic"], "creator_mode": "per_problem",
                 "condition_on_previous": False, "personas": False,
                 "require_tool_use": False},
        "rewards": {"target_hi": 0.9, "target_lo": 0.1},
        "roles": {"swap_interval": 100, "warmup_iterations": 0},
    })


def test_logic_iteration_end_to_end():
    trainer = _make_trainer(_config(), list(_SCRIPT), _UNIQUE_ANSWER)
    record = trainer.run_iteration(0)
    assert record["domain"] == "logic"
    (suite,) = record["suites"]
    # puzzle 1 consistent (unique + answer matches); puzzle 2 discarded (ambiguous)
    assert suite["n_consistent"] == 1
    assert suite["n_problems"] == 2
    # the correct scripted solver answer graded correct on the consistent rank
    assert suite["solve_rates_by_rank"] == [1.0]


def test_logic_iteration_wrong_solver_grades_zero():
    trainer = _make_trainer(_config(), list(_SCRIPT),
                            "A=knave, B=knight, C=knight")
    record = trainer.run_iteration(0)
    (suite,) = record["suites"]
    assert suite["n_consistent"] == 1
    assert suite["solve_rates_by_rank"] == [0.0]
