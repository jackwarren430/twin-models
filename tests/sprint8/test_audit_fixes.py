"""Sprint 8 — regressions for the four confirmed audit-2026-07-03 bugs.

1. Draft JSON inside ``<think>`` must not beat the final post-think JSON
   (``_extract_json_object`` searched by start index).
2. ``verify_math`` named-value asymmetry: creator answer ``"x = 2, y = 1"``
   passed its own certificate but a solver's correct ``"(2, 1)"`` graded
   WRONG — a live reward-hack channel (farming fake hardness).
3. Per-problem parse-fail re-stretch exploit: a rank that fails to parse is
   absent from the suite, so the Sprint-5 n_scored/n scaling never fired and
   the target ramp re-stretched over the survivors — dropping hard ranks
   out-earned writing them.
4. A speculative ``\\boxed{}`` inside ``<think>`` beat the solver's final
   ``ANSWER:`` line in ``extract_final_answer``.
"""

import math

import pytest

from twin.config import RewardsConfig
from twin.problems.schema import Problem, ProblemSuite, parse_problem, parse_suite
from twin.rewards import RewardEngine
from twin.think import strip_think
from twin.train.extract import extract_final_answer
from twin.verifiers.math_verifier import check_predicate, verify_math


# --------------------------------------------------------------------------- #
# 1. think-scoped JSON extraction
# --------------------------------------------------------------------------- #
_DRAFT = '{"statement": "DRAFT do not use", "difficulty": 0.5, "answer": "999"}'
_FINAL = '{"statement": "Solve 2x = 10.", "difficulty": 0.5, "answer": "5"}'


def test_draft_json_in_think_loses_to_final_json():
    text = f"<think>Let me draft: {_DRAFT} ... no, better:</think>\n{_FINAL}"
    p = parse_problem(text)
    assert p.statement == "Solve 2x = 10."
    assert p.answer == "5"


def test_draft_suite_in_think_loses_to_final_suite():
    draft = '{"theme": "draft", "problems": [%s]}' % _DRAFT
    final = '{"theme": "final", "problems": [%s]}' % _FINAL
    suite = parse_suite(f"<think>{draft}</think>\nHere it is:\n{final}")
    assert suite.theme == "final"
    assert suite.problems[0].answer == "5"


def test_unclosed_think_json_still_parses():
    # Truncated rollout: the only JSON lives in an unclosed think block.
    # Pre-fix behaviour parsed it; the fallback must keep that.
    p = parse_problem(f"<think>I will write {_FINAL} and then check")
    assert p.answer == "5"


def test_json_only_in_think_with_prose_after_falls_back():
    # Post-think text exists but has no JSON -> fall back to the full text.
    p = parse_problem(f"<think>{_FINAL}</think>\nDone! See above.")
    assert p.answer == "5"


def test_strip_think_leaves_unclosed_blocks():
    assert strip_think("<think>a</think>b") == "b"
    assert strip_think("<think>never closed") == "<think>never closed"


# --------------------------------------------------------------------------- #
# 2. verify_math named-value symmetry
# --------------------------------------------------------------------------- #
def test_solver_tuple_matches_creator_named_answer():
    # THE reward-hack channel: cert passes AND the correct solver answer must
    # grade correct against the same creator answer string.
    creator_answer = "x = 2, y = 1"
    cert = check_predicate("x + y = 3, x - y = 1", "x, y", creator_answer)
    assert cert.correct
    assert verify_math("(2, 1)", creator_answer).correct
    assert verify_math("[2, 1]", creator_answer).correct


def test_named_candidate_matches_tuple_expected():
    assert verify_math("x = 2, y = 1", "(2, 1)").correct


def test_named_both_sides_aligned_by_name_not_order():
    assert verify_math("y = 1, x = 2", "x = 2, y = 1").correct
    assert not verify_math("y = 2, x = 1", "x = 2, y = 1").correct


def test_wrong_values_still_fail():
    assert not verify_math("(1, 2)", "x = 2, y = 1").correct
    assert not verify_math("(2, 1, 0)", "x = 2, y = 1").correct


def test_named_values_with_expressions():
    assert verify_math("(1/2, 3)", "x = 0.5, y = 3").correct
    assert verify_math("a = 2*3, b = 10 - 3", "(6, 7)").correct


def test_scalar_and_string_paths_untouched():
    assert verify_math("4", "4").correct
    assert verify_math("x = 4", "4").correct          # single '=' RHS rule
    assert verify_math("hexagon", "hexagon").correct  # exact-match short-circuit
    assert not verify_math("pentagon", "hexagon").correct


# --------------------------------------------------------------------------- #
# 3. per-problem parse-fail re-stretch exploit
# --------------------------------------------------------------------------- #
def _suite(difficulties):
    return ProblemSuite(problems=[
        Problem(statement=f"p{i}", difficulty=d, answer="1")
        for i, d in enumerate(difficulties)
    ])


def test_expected_n_scales_gradient_by_parsed_fraction():
    eng = RewardEngine(RewardsConfig())
    suite = _suite([0.0, 0.25, 0.5])            # ranks 0..2 of an intended 5
    rates = [1.0, 0.5, 0.0]
    flags = [True, True, True]
    base = eng.creator_reward(suite, rates, flags)
    scaled = eng.creator_reward(suite, rates, flags, expected_n=5)
    assert scaled.r_gradient == pytest.approx(base.r_gradient * 3 / 5)


def test_target_by_problem_pins_prompted_targets():
    eng = RewardEngine(RewardsConfig(target_hi=0.9, target_lo=0.1))
    # Ranks 0 and 1 of an intended 3 parsed; prompted targets 0.9 and 0.5.
    suite = _suite([0.0, 0.5])
    rates = [0.9, 0.5]                           # exactly on prompt targets
    flags = [True, True]
    pinned = eng.creator_reward(
        suite, rates, flags, expected_n=3, target_by_problem=[0.9, 0.5])
    # Perfect fit against the PROMPTED targets, scaled by 2/3.
    assert pinned.r_gradient == pytest.approx(2 / 3)
    assert pinned.target_curve == [0.9, 0.5]
    # Without the pin the ramp re-stretches to [0.9, 0.1] -> imperfect fit.
    stretched = eng.creator_reward(suite, rates, flags)
    assert stretched.target_curve == pytest.approx([0.9, 0.1])
    assert stretched.r_gradient < 1.0


def test_parse_dropping_is_strictly_unprofitable():
    """The audit repro, in principle: a 3/5-parsed suite that perfectly fits
    the re-stretched ramp must now earn strictly less than an honest 5/5."""
    eng = RewardEngine(RewardsConfig())          # hi=1.0, lo=0.0
    honest_suite = _suite([0.0, 0.25, 0.5, 0.75, 1.0])
    honest = eng.creator_reward(
        honest_suite,
        [1.0, 0.8, 0.5, 0.3, 0.0],               # near-target, not perfect
        [True] * 5,
        expected_n=5,
        target_by_problem=[1.0, 0.75, 0.5, 0.25, 0.0],
    )
    cheat_suite = _suite([0.0, 0.25, 0.5])       # dropped the two hard ranks
    cheat = eng.creator_reward(
        cheat_suite,
        [1.0, 0.5, 0.0],                          # PERFECT re-stretched fit
        [True] * 3,
        expected_n=5,
        target_by_problem=[1.0, 0.75, 0.5],       # what those ranks were told
    )
    assert cheat.total < honest.total
    # and the old engine call (no expected_n / targets) shows the exploit
    old_cheat = eng.creator_reward(cheat_suite, [1.0, 0.5, 0.0], [True] * 3)
    assert old_cheat.r_gradient == pytest.approx(1.0)   # the bug, preserved
    assert cheat.r_gradient < old_cheat.r_gradient


def test_new_params_validate():
    eng = RewardEngine(RewardsConfig())
    suite = _suite([0.0, 1.0])
    with pytest.raises(ValueError):
        eng.creator_reward(suite, [1, 0], [True, True], target_by_problem=[1.0])
    with pytest.raises(ValueError):
        eng.creator_reward(suite, [1, 0], [True, True], expected_n=1)


def test_defaults_keep_pre_audit_behaviour():
    eng = RewardEngine(RewardsConfig())
    suite = _suite([0.0, 0.5, 1.0])
    rates = [1.0, 0.5, 0.0]
    r = eng.creator_reward(suite, rates, [True] * 3)
    assert r.r_gradient == pytest.approx(1.0)
    assert r.target_curve == pytest.approx([1.0, 0.5, 0.0])
    masked = eng.creator_reward(
        suite, rates, [True, False, True], scored_mask=[True, False, True])
    # Sprint-5 scaling: n_scored/n with the re-stretched 2-rank ramp
    stretched_mse = ((1.0 - 1.0) ** 2 + (0.0 - 0.0) ** 2) / 2
    assert masked.r_gradient == pytest.approx(
        math.exp(-eng.cfg.mse_beta * stretched_mse) * 2 / 3)


# --------------------------------------------------------------------------- #
# 4. boxed-in-think vs final ANSWER line
# --------------------------------------------------------------------------- #
def test_boxed_in_think_loses_to_answer_line():
    text = "<think>could it be \\boxed{5}? no...</think>\nANSWER: 7"
    assert extract_final_answer(text) == "7"


def test_boxed_after_think_still_preferred():
    text = "<think>hmm</think>\nSo the result is \\boxed{9}.\nANSWER: 9"
    assert extract_final_answer(text) == "9"


def test_unclosed_think_falls_back_to_full_text():
    assert extract_final_answer("<think>surely \\boxed{5} is right") == "5"


def test_plain_text_unchanged():
    assert extract_final_answer("work...\nANSWER: 42") == "42"
    assert extract_final_answer("\\boxed{3}") == "3"
    assert extract_final_answer("just 11") == "just 11"
