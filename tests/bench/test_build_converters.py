"""Unit tests for the hard-tier source converters (scripts/build_hard_bench.py).

Pure row->item conversion logic, exercised with synthetic rows — no network, no
model. Guards the gradeability filters (numeric-only MATH answers, MBPP entry
extraction, MMLU-Pro letter range, BBH letter-vs-exact routing) that decide
whether an adapted item can actually be graded.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "build_hard_bench", ROOT / "scripts" / "build_hard_bench.py"
)
bhb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bhb)


# ----- MATH-500 ------------------------------------------------------------
def test_conv_math_keeps_numeric_hard():
    row = {"problem": "Find n.", "answer": "144", "level": 4,
           "subject": "Counting", "unique_id": "test/counting/525.json"}
    it = bhb.conv_math(row)
    assert it and it["verification"]["type"] == "math_numeric"
    assert it["answer"] == "144" and it["id"].startswith("math500-")


def test_conv_math_drops_easy_levels():
    assert bhb.conv_math({"problem": "p", "answer": "3", "level": 2}) is None


def test_conv_math_keeps_fraction_drops_irrational_and_text():
    assert bhb.conv_math({"problem": "p", "answer": "\\frac{3}{2}", "level": 3})
    assert bhb.conv_math({"problem": "p", "answer": "\\sqrt{2}", "level": 5}) is None
    assert bhb.conv_math({"problem": "p", "answer": "\\text{blue}", "level": 5}) is None


# ----- MBPP ----------------------------------------------------------------
def test_conv_mbpp_extracts_entry_and_tests():
    row = {"task_id": 11, "text": "remove a char",
           "test_list": ["assert remove_Occ('hello','l') == 'heo'",
                         "assert remove_Occ('abcda','a') == 'bcd'"],
           "test_setup_code": ""}
    it = bhb.conv_mbpp(row)
    assert it["verification"]["type"] == "code"
    assert it["entry_point"] == "remove_Occ"
    assert "remove_Occ" in it["prompt"]  # name surfaced for the model
    assert it["verification"]["tests"].count("assert") == 2


def test_conv_mbpp_no_tests_skipped():
    assert bhb.conv_mbpp({"task_id": 1, "text": "x", "test_list": []}) is None


# ----- MMLU-Pro ------------------------------------------------------------
def test_conv_mmlu_pro_mcq():
    row = {"question_id": 70, "question": "Q?", "category": "law",
           "options": ["a", "b", "c", "d", "e"], "answer": "D"}
    it = bhb.conv_mmlu_pro(row)
    assert it["verification"]["type"] == "mcq"
    assert len(it["choices"]) == 5 and it["answer"] == "D"


def test_conv_mmlu_pro_letter_out_of_range_skipped():
    row = {"question_id": 1, "question": "Q", "options": ["a", "b"], "answer": "D"}
    assert bhb.conv_mmlu_pro(row) is None


# ----- BBH -----------------------------------------------------------------
def test_conv_bbh_letter_task_becomes_mcq():
    inp = ("Puzzle.\nOptions:\n(A) apples\n(B) peaches\n(C) mangoes")
    it = bhb.conv_bbh({"input": inp, "target": "(B)"}, "logical_deduction", 0)
    assert it["verification"]["type"] == "mcq"
    assert it["choices"] == ["apples", "peaches", "mangoes"]
    assert it["answer"] == "B"
    assert "Options:" not in it["prompt"]  # options stripped from the stem


def test_conv_bbh_boolean_becomes_exact():
    it = bhb.conv_bbh({"input": "( True and False ) is", "target": "False"}, "boolean", 1)
    assert it["verification"]["type"] == "exact" and it["answer"] == "False"


def test_conv_bbh_long_freeform_skipped():
    long_target = "x" * 60
    assert bhb.conv_bbh({"input": "q", "target": long_target}, "t", 0) is None


@pytest.mark.parametrize("target", ["(A)", "(C)"])
def test_conv_bbh_letter_without_options_skipped(target):
    # a "(X)" target but no parseable Options block can't be graded -> skip
    assert bhb.conv_bbh({"input": "no options here", "target": target}, "t", 0) is None
