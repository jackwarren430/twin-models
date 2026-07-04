"""Sprint 5 — creator contract carries the math verification certificate. Fast."""

from twin.prompts import creator_user


def test_math_prompt_asks_for_certificate():
    u = creator_user("math", "linear equations", 3)
    assert '"verification"' in u
    assert '"check"' in u
    assert '"symbol"' in u
    # the anti-collusion rule is stated
    assert "trivial" in u
    # and the multi-unknown convention
    assert "x, y" in u


def test_coding_prompt_carries_code_contract():
    # The Sprint-8 coding pipeline landed: coding prompts now demand the
    # executable contract (tests/sprint8/test_coding_pipeline.py has the
    # full coverage); the base suite contract is intact either way.
    u = creator_user("coding", "recursion basics", 3)
    assert '"verification"' in u and "solution_code" in u and "tests" in u
    for key in ('"problems"', '"difficulty"', '"answer"', '"solution"'):
        assert key in u
