"""Verifier tests: math (sympy), code (exec), judge (stub oracle), dispatch.

Synthetic problems only; no model. The judge oracle is a deterministic stub."""

from twin.problems.schema import Problem
from twin.verifiers import (
    check_consistency,
    check_predicate,
    judge_answer,
    verify_answer,
    verify_code,
    verify_math,
)


# ----- math -----------------------------------------------------------------
def test_math_symbolic_equality():
    assert verify_math("x + 1", "1 + x").correct
    assert verify_math("6/2", "3").correct
    assert verify_math("(x+1)**2", "x**2 + 2*x + 1").correct


def test_math_numeric_fallback_within_tolerance():
    assert verify_math("1/2", "0.5").correct
    assert verify_math("2.0000001", "2", tolerance=1e-6).correct
    assert not verify_math("2.1", "2", tolerance=1e-6).correct


def test_math_handles_answer_prefix():
    # "x = 3" should compare its RHS.
    assert verify_math("x = 3", "3").correct


def test_math_wrong_answer():
    r = verify_math("4", "3")
    assert not r.correct
    assert r.method == "sympy"


# ----- predicate (consistency certificate) ----------------------------------
def test_predicate_root_check():
    assert check_predicate("x**2 - 5*x + 6", "x", "3").correct
    assert not check_predicate("x**2 - 5*x + 6", "x", "4").correct


def test_predicate_relational():
    assert check_predicate("Eq(x**2, 9)", "x", "3").correct
    assert not check_predicate("Eq(x**2, 9)", "x", "4").correct


# ----- code -----------------------------------------------------------------
def test_code_passes_tests():
    r = verify_code("def f(x):\n    return x * 2", "assert f(3) == 6")
    assert r.correct
    assert r.method == "exec"


def test_code_fails_tests():
    r = verify_code("def f(x):\n    return x + 1", "assert f(3) == 6")
    assert not r.correct


def test_code_requires_tests():
    assert not verify_code("def f(): pass", "").correct


# ----- judge ----------------------------------------------------------------
def test_judge_parses_verdict():
    yes = lambda q: "Looks right.\nVERDICT: CORRECT"
    no = lambda q: "Reasoning...\nVERDICT: INCORRECT"
    assert judge_answer("Q", "a", "a", yes).correct
    assert not judge_answer("Q", "a", "b", no).correct


def test_judge_ambiguous_is_not_correct():
    vague = lambda q: "I am not sure about this one."
    assert not judge_answer("Q", "a", "a", vague).correct


# ----- dispatch: solver scoring ---------------------------------------------
def test_verify_answer_math():
    p = Problem(statement="2+2?", difficulty=0.1, answer="4", domain="math",
                verification={"type": "math_numeric"})
    assert verify_answer(p, "4").correct
    assert not verify_answer(p, "5").correct


def test_verify_answer_code():
    p = Problem(statement="double", difficulty=0.3, domain="coding",
                verification={"type": "code", "tests": "assert f(3) == 6"})
    assert verify_answer(p, "def f(x):\n    return x * 2").correct
    assert not verify_answer(p, "def f(x):\n    return x").correct


def test_verify_answer_judge_for_logic_domain():
    p = Problem(statement="riddle", difficulty=0.5, answer="blue", domain="logic")
    correct_oracle = lambda q: "VERDICT: CORRECT"
    assert verify_answer(p, "blue", oracle=correct_oracle).correct


# ----- dispatch: creator consistency ----------------------------------------
def test_consistency_code_runs_reference_solution():
    p = Problem(statement="double", difficulty=0.2, domain="coding",
                solution="def f(x):\n    return x * 2",
                verification={"type": "code", "tests": "assert f(3) == 6"})
    assert check_consistency(p).correct


def test_consistency_math_uses_predicate():
    p = Problem(statement="root of x^2-5x+6, larger", difficulty=0.4, answer="3",
                domain="math", verification={"type": "math", "check": "x**2 - 5*x + 6",
                                             "symbol": "x"})
    assert check_consistency(p).correct


def test_consistency_math_falls_back_to_judge():
    p = Problem(statement="hard proof", difficulty=0.9, answer="42",
                solution="some prose", domain="math", verification={"type": "math"})
    yes = lambda q: "VERDICT: CORRECT"
    assert check_consistency(p, oracle=yes).correct
