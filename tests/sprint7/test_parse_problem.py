"""Sprint 7 — parse_problem: single-problem JSON extraction for per-problem
creator mode. Fast: no model."""

import pytest

from twin.problems.schema import SuiteParseError, parse_problem


GOOD = '{"statement": "Solve 2x = 8.", "difficulty": 0.3, "solution": "x = 8/2", "answer": "4"}'


def test_parses_bare_json():
    p = parse_problem(GOOD)
    assert p.statement == "Solve 2x = 8."
    assert p.answer == "4"
    assert p.difficulty == 0.3


def test_parses_json_with_think_prefix_and_fence():
    text = f"<think>brief plan</think>\nHere it is:\n```json\n{GOOD}\n```"
    p = parse_problem(text)
    assert p.answer == "4"


def test_unwraps_accidental_suite_wrapper():
    text = '{"theme": "x", "problems": [' + GOOD + "]}"
    p = parse_problem(text)
    assert p.statement == "Solve 2x = 8."


def test_default_domain_applied():
    p = parse_problem(GOOD, default_domain="arithmetic")
    assert p.domain == "arithmetic"


def test_missing_statement_raises():
    with pytest.raises(SuiteParseError):
        parse_problem('{"difficulty": 0.5, "answer": "1"}')


def test_no_json_raises():
    with pytest.raises(SuiteParseError):
        parse_problem("I could not come up with a problem.")


def test_empty_wrapper_raises():
    with pytest.raises(SuiteParseError):
        parse_problem('{"problems": []}')
