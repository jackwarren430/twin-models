"""Sprint 7 — the per-problem creator prompt: rank/difficulty/target-rate
injection, conditioning on previous problem JSONs (and only JSONs), and the
solution-before-answer field order (CoT: derive first, state after). Fast."""

import json

from twin.prompts import creator_problem_user, creator_user


def test_contains_rank_difficulty_and_target_percentage():
    u = creator_problem_user(
        "math", "basic probability",
        rank=2, n_problems=5, difficulty=0.5, target_rate=0.5,
    )
    assert "problem 3 of 5" in u
    assert "0.50" in u                       # dictated difficulty value
    assert "50%" in u                        # target solve rate as a percentage
    assert "basic probability" in u


def test_hard_rank_asks_for_hardest_and_structural():
    u = creator_problem_user(
        "math", "systems of two equations",
        rank=4, n_problems=5, difficulty=1.0, target_rate=0.1,
        opponent="omega",
    )
    assert "10%" in u
    assert "omega" in u
    assert "hardest problem you can" in u
    assert "structurally hard" in u          # not merely bigger numbers
    # the decisive-design instruction (mini-03a's think-spiral fix) survives
    assert "Design decisively" in u


def test_easy_rank_asks_for_warmup():
    u = creator_problem_user(
        "math", "percentages and ratios",
        rank=0, n_problems=5, difficulty=0.0, target_rate=0.9,
        opponent="omega",
    )
    assert "90%" in u
    assert "never miss" in u


def test_conditioning_includes_previous_jsons_only_when_given():
    prev = [json.dumps({"statement": "What is 2+2?", "difficulty": 0.0,
                        "answer": "4"})]
    with_prev = creator_problem_user(
        "math", "divisibility", rank=1, n_problems=3,
        difficulty=0.5, target_rate=0.5, previous=prev,
    )
    without = creator_problem_user(
        "math", "divisibility", rank=1, n_problems=3,
        difficulty=0.5, target_rate=0.5, previous=None,
    )
    assert "What is 2+2?" in with_prev
    assert "strictly harder" in with_prev and "genuinely distinct" in with_prev
    assert "What is 2+2?" not in without
    assert "already wrote" not in without


def test_single_problem_contract_solution_before_answer():
    u = creator_problem_user(
        "math", "linear equations", rank=0, n_problems=3,
        difficulty=0.0, target_rate=0.9,
    )
    assert '"statement"' in u and '"solution"' in u and '"answer"' in u
    assert u.index('"solution"') < u.index('"answer"')
    # single problem: no suite wrapper
    assert '"problems"' not in u
    # math contract still carries the certificate spec
    assert '"verification"' in u and '"check"' in u and '"symbol"' in u


def test_suite_contract_also_reordered_solution_first():
    u = creator_user("math", "quadratic equations", 3)
    assert u.index('"solution"') < u.index('"answer"')
