"""Sprint 5 — robust math verification certificates. Fast: no model.

`check_predicate` is what lets creator consistency be verified mechanically
(CAS) instead of by the LLM judge, so it must accept the equation forms a model
naturally writes, reject self-certifying checks, and never let model text reach
builtins through sympy's eval-based parser.
"""

from twin.tools import calc
from twin.verifiers import check_predicate, verify_math


# ----- accepted certificate forms -------------------------------------------
def test_equation_form():
    assert check_predicate("2*x + 3 = 11", "x", "4").correct
    assert not check_predicate("2*x + 3 = 11", "x", "5").correct


def test_python_style_double_equals():
    assert check_predicate("4*x == 20", "x", "5").correct
    assert not check_predicate("4*x == 20", "x", "4").correct


def test_eq_call_form():
    assert check_predicate("Eq(x**2, 9)", "x", "3").correct
    assert not check_predicate("Eq(x**2, 9)", "x", "4").correct


def test_bare_expression_vanishes():
    assert check_predicate("x**2 - 5*x + 6", "x", "3").correct
    assert not check_predicate("x**2 - 5*x + 6", "x", "4").correct


def test_recomputation_cert_is_accepted():
    # "What is 15% of 80?" -> the check recomputes the answer; NOT trivial.
    assert check_predicate("x = 0.15*80", "x", "12").correct
    assert not check_predicate("x = 0.15*80", "x", "13").correct


def test_eq_respects_tolerance():
    assert check_predicate("3*x = 1", "x", "0.333333", tolerance=1e-3).correct
    assert not check_predicate("3*x = 1", "x", "0.34", tolerance=1e-3).correct


# ----- systems / tuple answers ----------------------------------------------
def test_multi_symbol_system():
    ok = check_predicate("x + y = 10, x - y = 2", "x, y", "(6, 4)")
    assert ok.correct
    assert not check_predicate("x + y = 10, x - y = 2", "x, y", "(5, 5)").correct


def test_multi_symbol_arity_mismatch_fails_cleanly():
    r = check_predicate("x + y = 10", "x, y", "6")
    assert not r.correct


# ----- trivial (self-certifying) checks are rejected -------------------------
def test_trivial_literal_cert_rejected():
    for check in ("x = 12", "12 = x", "x == 12", "Eq(x, 12)", "x = 7/2", "x = -3.5"):
        r = check_predicate(check, "x", "12")
        assert not r.correct, check
        assert "trivial" in r.detail


def test_trivial_part_in_system_rejected():
    r = check_predicate("x + y = 10, x = 6", "x, y", "(6, 4)")
    assert not r.correct
    assert "trivial" in r.detail


# ----- locked parse namespace ------------------------------------------------
def test_check_predicate_locked_namespace(tmp_path):
    probe = tmp_path / "pwned"
    evil = f"__import__('pathlib').Path('{probe}').write_text('x')"
    r = check_predicate(evil, "x", "1")
    assert not r.correct
    assert not probe.exists()


def test_verify_math_locked_namespace(tmp_path):
    probe = tmp_path / "pwned"
    evil = f"__import__('pathlib').Path('{probe}').write_text('x')"
    r = verify_math(evil, "3")
    assert not r.correct
    assert not probe.exists()


def test_calc_locked_namespace(tmp_path):
    probe = tmp_path / "pwned"
    out = calc(f"__import__('pathlib').Path('{probe}').write_text('x')")
    assert not probe.exists()
    assert isinstance(out, str)


# ----- empty / malformed ------------------------------------------------------
def test_empty_inputs_fail_cleanly():
    assert not check_predicate("", "x", "3").correct
    assert not check_predicate("x = 3", "", "3").correct
    assert not check_predicate("x = 3", "not a symbol!!", "3").correct
