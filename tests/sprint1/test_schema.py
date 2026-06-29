"""Sprint 1 — problem schema & parsing. Fast: no model."""

import json

import pytest

from twin.problems.schema import (
    Problem,
    ProblemSuite,
    SuiteParseError,
    parse_suite,
)


def _suite_dict(n=3, spread=True):
    diffs = [0.0, 0.5, 1.0] if (spread and n == 3) else [0.1] * n
    return {
        "theme": "quadratics",
        "domain": "math",
        "problems": [
            {
                "statement": f"problem {i}",
                "difficulty": diffs[i] if i < len(diffs) else 0.5,
                "answer": str(i),
                "solution": f"because {i}",
                "verification": {"type": "math_numeric"},
            }
            for i in range(n)
        ],
    }


def test_target_curve_is_linear_descending():
    assert ProblemSuite.target_curve(1) == [1.0]
    curve = ProblemSuite.target_curve(5)
    assert curve[0] == 1.0 and curve[-1] == 0.0
    # strictly descending, evenly spaced
    diffs = [round(curve[i] - curve[i + 1], 6) for i in range(len(curve) - 1)]
    assert all(d == diffs[0] for d in diffs)
    assert ProblemSuite.target_curve(0) == []


def test_roundtrip_dict():
    suite = ProblemSuite.from_dict(_suite_dict())
    again = ProblemSuite.from_dict(suite.to_dict())
    assert len(again) == 3
    assert again.theme == "quadratics"
    assert again.problems[2].answer == "2"
    # JSON serialisation works
    json.loads(suite.to_json())


def test_sorted_by_difficulty_stable():
    d = _suite_dict()
    d["problems"][0]["difficulty"] = 0.9
    d["problems"][2]["difficulty"] = 0.1
    suite = ProblemSuite.from_dict(d)
    order = [p.difficulty for p in suite.sorted_by_difficulty()]
    assert order == sorted(order)


def test_validate_flags_small_spread():
    suite = ProblemSuite.from_dict(_suite_dict(n=3, spread=False))
    issues = suite.validate()
    assert any("spread" in s for s in issues)
    assert not suite.is_valid()


def test_validate_flags_difficulty_out_of_range():
    d = _suite_dict()
    d["problems"][0]["difficulty"] = 1.5
    suite = ProblemSuite.from_dict(d)
    assert any("out of [0,1]" in s for s in suite.validate())


def test_valid_suite_passes():
    suite = ProblemSuite.from_dict(_suite_dict())
    assert suite.is_valid()
    assert suite.validate() == []


def test_from_dict_requires_problems():
    with pytest.raises(SuiteParseError):
        ProblemSuite.from_dict({"theme": "x"})
    with pytest.raises(SuiteParseError):
        ProblemSuite.from_dict({"problems": []})


def test_problem_bad_difficulty_raises():
    with pytest.raises(SuiteParseError):
        Problem.from_dict({"statement": "s", "difficulty": "hard"})


def test_problem_requires_statement():
    with pytest.raises(SuiteParseError):
        Problem.from_dict({"difficulty": 0.5})


# ----- parsing free-form model text ----------------------------------------
def test_parse_plain_json():
    text = json.dumps(_suite_dict())
    suite = parse_suite(text)
    assert len(suite) == 3


def test_parse_fenced_json_with_prose():
    text = (
        "Sure! Here is the suite you asked for:\n\n"
        "```json\n" + json.dumps(_suite_dict()) + "\n```\n"
        "Let me know if you want harder ones."
    )
    suite = parse_suite(text)
    assert suite.domain == "math"
    assert len(suite) == 3


def test_parse_embedded_object_no_fence():
    text = "blah blah " + json.dumps(_suite_dict()) + " trailing commentary {not json}"
    suite = parse_suite(text)
    assert len(suite) == 3


def test_parse_handles_braces_inside_strings():
    d = _suite_dict()
    d["problems"][0]["statement"] = "compute f(x) = {x : x>0} cardinality"
    text = "here:\n```\n" + json.dumps(d) + "\n```"
    suite = parse_suite(text)
    assert "cardinality" in suite.problems[0].statement


def test_parse_failure_raises():
    with pytest.raises(SuiteParseError):
        parse_suite("no json here, sorry")
