"""Sprint 3 — solver-output answer extraction + oracle-call counting. Fast: no model."""

from twin.train.extract import count_oracle_calls, extract_final_answer


def test_extract_answer_line():
    assert extract_final_answer("Work...\nANSWER: 42") == "42"
    assert extract_final_answer("a\nb\nFinal Answer = -3/2") == "-3/2"


def test_extract_boxed_preferred():
    txt = "Reasoning. \\boxed{7}\nANSWER: 99"
    assert extract_final_answer(txt) == "7"


def test_extract_last_nonempty_fallback():
    assert extract_final_answer("some reasoning\n13\n") == "13"
    assert extract_final_answer("") == ""


def test_extract_takes_last_answer_when_repeated():
    assert extract_final_answer("ANSWER: 1\nmore\nANSWER: 2") == "2"


def test_count_oracle_calls():
    txt = (
        "Let me check. <tool>oracle(capital of France)</tool> "
        "and <tool>calc(2+2)</tool> then <tool>oracle(population)</tool>"
    )
    assert count_oracle_calls(txt) == 2
    assert count_oracle_calls("no tools here") == 0
