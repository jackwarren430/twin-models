"""Sprint 3 — creator/solver prompt builders + theme picking. Fast: no model."""

import random

from twin.problems.schema import Problem
from twin.prompts import (
    CREATOR_SYSTEM,
    JUDGE_SYSTEM,
    THEMES,
    creator_user,
    pick_theme,
    solver_user,
)


def test_creator_user_mentions_contract_and_args():
    u = creator_user("math", "quadratic equations", 4)
    assert "quadratic equations" in u
    assert "math" in u
    assert "4" in u
    # asks for the exact JSON schema parse_suite consumes
    for key in ('"problems"', '"difficulty"', '"answer"', '"solution"'):
        assert key in u
    # nudges the model to use the CAS tool
    assert "solve tool" in u


def test_creator_system_documents_react_tool_protocol():
    s = CREATOR_SYSTEM
    assert "<tool>" in s and "</tool>" in s     # ReAct call syntax
    assert "<obs>" in s                          # observation feedback
    assert "solve(" in s                         # the CAS tool
    assert "JSON" in s                           # still asked for the JSON contract


def test_judge_system_documents_cas_tool_and_verdict():
    s = JUDGE_SYSTEM
    # The judge gets the same ReAct CAS protocol as the creator so it recomputes
    # rather than eyeballing arithmetic.
    assert "<tool>" in s and "</tool>" in s
    assert "<obs>" in s
    assert "solve(" in s
    # and must still terminate with the parseable verdict line.
    assert "VERDICT:" in s
    assert "CORRECT" in s and "INCORRECT" in s


def test_solver_user_includes_statement_and_answer_marker():
    p = Problem(statement="Solve x^2 - 5x + 6 = 0; give the larger root.", difficulty=0.4)
    u = solver_user(p)
    assert "Solve x^2 - 5x + 6 = 0" in u
    assert "ANSWER:" in u
    # also accepts a bare string
    assert "just this" in solver_user("just this")


def test_pick_theme_in_pool_and_deterministic():
    rng = random.Random(0)
    t = pick_theme("math", rng)
    assert t in THEMES["math"]
    # same seed -> same sequence
    assert pick_theme("math", random.Random(7)) == pick_theme("math", random.Random(7))


def test_pick_theme_unknown_domain_falls_back_to_math():
    t = pick_theme("astrology", random.Random(3))
    assert t in THEMES["math"]
