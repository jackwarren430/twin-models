"""Grading of model responses per verification type. Fast: no model.

Code grading runs the real sandbox (same path the trainer uses), so those few
cases spawn a subprocess; everything else is pure string work.
"""

import pytest

from twin.bench.dataset import BenchItem
from twin.bench.grade import (
    extract_code_block,
    extract_mcq_letter,
    grade,
    strip_think,
)


def _item(vtype, **kw):
    kw.setdefault("answer", "")
    return BenchItem(id="t", category="c", prompt="q",
                     verification={"type": vtype, **kw.pop("verification", {})}, **kw)


# ----- think stripping + extraction helpers --------------------------------
def test_strip_think_removes_closed_block():
    out = strip_think("<think>secret 999</think>\nANSWER: 42")
    assert "999" not in out and "42" in out


def test_strip_think_keeps_unclosed_block():
    text = "<think>never finished"
    assert strip_think(text) == text


def test_extract_code_block_prefers_last_fence():
    text = "```python\ndef a(): pass\n```\nthen\n```python\ndef b(): return 1\n```"
    assert "def b()" in extract_code_block(text)
    assert "def a()" not in extract_code_block(text)


def test_extract_code_block_falls_back_to_raw():
    assert extract_code_block("def f(): return 1").startswith("def f()")


# ----- math_numeric --------------------------------------------------------
def test_math_correct_and_wrong():
    it = _item("math_numeric", answer="5/6")
    assert grade(it, "work...\nANSWER: 5/6").correct
    assert not grade(it, "ANSWER: 1/2").correct


def test_math_ignores_think_block_answer():
    it = _item("math_numeric", answer="42")
    # the decoy in the think block must not be picked up
    assert grade(it, "<think>ANSWER: 999</think>\nANSWER: 42").correct


# ----- code ----------------------------------------------------------------
@pytest.mark.parametrize("body,ok", [
    ("def add(a, b):\n    return a + b", True),
    ("def add(a, b):\n    return a - b", False),
])
def test_code_grading(body, ok):
    it = _item("code", verification={"tests": "assert add(2, 3) == 5\nassert add(0, 0) == 0"})
    resp = f"Here:\n```python\n{body}\n```"
    assert grade(it, resp).correct is ok


# ----- mcq -----------------------------------------------------------------
def _mcq(answer="C"):
    return BenchItem(id="m", category="knowledge", prompt="capital?",
                     choices=["Sydney", "Melbourne", "Canberra", "Perth"],
                     answer=answer, verification={"type": "mcq"})


def test_mcq_letter_forms():
    it = _mcq("C")
    assert grade(it, "reasoning\nANSWER: C").correct
    assert grade(it, "ANSWER: (C)").correct
    assert grade(it, "ANSWER: C) Canberra").correct
    assert not grade(it, "ANSWER: A").correct


def test_mcq_echoed_choice_text_resolves_to_letter():
    it = _mcq("C")
    assert grade(it, "ANSWER: Canberra").correct  # echoes option text, not letter


def test_mcq_choice_starting_with_letter_not_mistaken_for_label():
    # "Canberra" starts with 'C' but must resolve by text match (-> C), not by
    # grabbing the leading letter of the word.
    assert extract_mcq_letter("ANSWER: Canberra",
                              ["Sydney", "Melbourne", "Canberra", "Perth"]) == "C"


def test_mcq_unparseable_is_wrong():
    it = _mcq("C")
    assert not grade(it, "I am not sure about this one.").correct


# ----- exact ---------------------------------------------------------------
def test_exact_match_and_alias_and_containment():
    it = BenchItem(id="e", category="knowledge", prompt="formula for water?",
                   answer="H2O", verification={"type": "exact", "aliases": ["h2o"]})
    assert grade(it, "ANSWER: H2O").correct
    assert grade(it, "ANSWER: h2o").correct
    assert grade(it, "ANSWER: The formula is H2O").correct  # whole-word containment


def test_exact_negative_does_not_partial_match():
    it = BenchItem(id="e", category="reasoning", prompt="yes or no?",
                   answer="no", verification={"type": "exact"})
    assert grade(it, "ANSWER: no").correct
    assert not grade(it, "ANSWER: winner").correct  # 'no' must not match inside 'winner'


def test_exact_wrong_answer():
    it = BenchItem(id="e", category="knowledge", prompt="capital of japan?",
                   answer="Tokyo", verification={"type": "exact"})
    assert not grade(it, "ANSWER: Kyoto").correct
