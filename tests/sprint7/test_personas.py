"""Sprint 7 — persona injection: names glued to adapters, fair-competition
framing, and (critically) neutrality when personas are off — the pre-Sprint-7
prompts must come out byte-identical so mini-01..03 remain reproducible and
the benchmark/judge stay unbiased. Fast: no model."""

from twin.prompts import (
    CREATOR_SYSTEM,
    JUDGE_SYSTEM,
    PERSONAS,
    SOLVER_SYSTEM,
    creator_persona,
    creator_system,
    opponent_of,
    persona_of,
    solver_persona,
    solver_system,
)


def test_personas_glued_to_adapters():
    assert PERSONAS == {"A": "alpha", "B": "omega"}
    assert persona_of("A") == "alpha" and persona_of("B") == "omega"
    assert opponent_of("A") == "omega" and opponent_of("B") == "alpha"


def test_creator_persona_names_and_fair_competition():
    p = creator_persona("A")
    assert '"alpha"' in p and "omega" in p
    assert "fair competition" in p
    # calibration framing, not raw stumping: the win condition is prediction
    assert "predicting exactly what omega can and cannot solve" in p


def test_solver_persona_names_and_framing():
    p = solver_persona("B")
    assert '"omega"' in p and "alpha" in p
    assert "fair competition" in p


def test_neutral_defaults_are_byte_identical_to_v1():
    # personas off + legacy protocol == the exact pre-Sprint-7 prompts
    assert creator_system() == CREATOR_SYSTEM
    assert solver_system() == SOLVER_SYSTEM


def test_persona_prepended_not_replacing():
    s = creator_system(persona=creator_persona("A"))
    assert s.startswith('You are "alpha"')
    assert s.endswith(CREATOR_SYSTEM)  # the standard instructions survive intact
    s2 = solver_system(persona=solver_persona("B"))
    assert s2.startswith('You are "omega"')
    assert s2.endswith(SOLVER_SYSTEM)


def test_native_creator_system_drops_legacy_markup_keeps_tool_mandate():
    s = creator_system(native_tools=True)
    assert "<tool>" not in s and "<obs>" not in s  # template declares the tools
    assert "solve" in s
    # the anti-"simulate the obs" instruction (mini-04a rewrite: two-phase
    # VERIFY/DELIVER + tools cannot run inside private reasoning)
    assert "can NOT run inside your private reasoning" in s
    assert "VERIFY" in s and "DELIVER" in s
    assert "JSON" in s


def test_judge_system_has_no_persona_vocabulary():
    # the judge must stay a neutral grader
    assert "alpha" not in JUDGE_SYSTEM and "omega" not in JUDGE_SYSTEM
    assert "competition" not in JUDGE_SYSTEM
