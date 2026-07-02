"""Sprint 4 — output parsers survive Qwen3 thinking-mode prefixes. Fast: no model.

With ``enable_thinking: true`` (base.yaml) the model emits a ``<think>…</think>``
block before its real output. The creator's JSON and the solver's ``ANSWER:``
line must still be recovered. These tests feed think-prefixed text through the
existing parsers so turning thinking on doesn't silently break extraction.
"""

import json

from twin.problems.schema import parse_suite
from twin.train.extract import count_oracle_calls, extract_final_answer

_SUITE = {
    "theme": "linear equations",
    "domain": "math",
    "problems": [
        {"statement": "Solve 2x = 4.", "difficulty": 0.1, "answer": "2",
         "solution": "x = 2."},
        {"statement": "Solve 3x + 1 = 10.", "difficulty": 0.6, "answer": "3",
         "solution": "x = 3."},
    ],
}


def test_extract_answer_after_think_block():
    text = (
        "<think>\nHmm, 3x+1=10 so 3x=9, maybe x=3. Let me double check {scratch}.\n"
        "</think>\n"
        "The solution gives x = 3.\nANSWER: 3"
    )
    assert extract_final_answer(text) == "3"


def test_extract_answer_ignores_numbers_inside_think():
    # The think block mentions other numbers; only the ANSWER line should win.
    text = (
        "<think>could be 7 or 11 or 42</think>\n"
        "Reasoning omitted.\nANSWER: 11"
    )
    assert extract_final_answer(text) == "11"


def test_parse_suite_after_think_block():
    text = "<think>\nI'll make two problems, easy then harder.\n</think>\n" + json.dumps(_SUITE)
    suite = parse_suite(text)
    assert len(suite.problems) == 2
    assert suite.problems[0].answer == "2"


def test_parse_suite_skips_non_json_braces_in_think():
    # Stray braces in the think block are NOT valid JSON, so they're skipped and
    # the real suite (later) is parsed.
    text = (
        "<think> set s = {a, b, c}; consider f(x) = {x: 4} ... </think>\n"
        "```json\n" + json.dumps(_SUITE) + "\n```"
    )
    suite = parse_suite(text)
    assert len(suite.problems) == 2


def test_oracle_count_unaffected_by_think_block():
    text = "<think>should I ask the oracle?</think>\n<tool>oracle(what is pi)</tool>"
    assert count_oracle_calls(text) == 1
