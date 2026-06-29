"""Tests for the calc tool. Fast — no model, no subprocess."""

from twin.tools.calc import calc


def test_integer_arithmetic():
    assert calc("2 + 2") == "4"
    assert calc("7 * 6") == "42"


def test_exact_rationals_stay_exact():
    assert calc("1/2 + 1/3") == "5/6"


def test_simplifies_radicals():
    assert calc("sqrt(16)") == "4"


def test_symbolic_expression_returned_symbolically():
    # implicit multiplication: "2x" -> 2*x
    assert calc("x + x") == "2*x"


def test_irrational_gives_decimal():
    out = calc("sqrt(2)")
    assert out.startswith("1.41421")


def test_empty_is_error():
    assert calc("   ").startswith("error:")


def test_bad_input_is_error_not_exception():
    assert calc("2 +").startswith("error:")
    # must never raise into the rollout
    assert calc("import os").startswith("error:")
