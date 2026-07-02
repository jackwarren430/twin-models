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


def test_coding_prompt_unchanged_until_coding_fix():
    u = creator_user("coding", "recursion basics", 3)
    assert '"verification"' not in u
    # the base contract is intact
    for key in ('"problems"', '"difficulty"', '"answer"', '"solution"'):
        assert key in u
