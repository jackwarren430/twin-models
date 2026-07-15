"""Sprint Q2 — the three judge contracts parse strictly and degrade gracefully
(twentyq/DESIGN.md §2.4). Oracles are stubs; no model anywhere."""

import pytest

from twin.games.twentyq.judge import (
    judge_answer_audit,
    judge_closeness,
    judge_secret_validity,
)
from twin.games.twentyq.schema import Secret

SECRET = Secret(secret="octopus", category="animal", difficulty=0.4)
QA = [("Is it a mammal?", "NO"), ("Does it live in water?", "YES")]


def oracle_returning(text):
    return lambda prompt: text


def raising_oracle(prompt):
    raise RuntimeError("judge exploded")


# ----- validity --------------------------------------------------------------

def test_validity_valid_and_invalid():
    assert judge_secret_validity(SECRET, oracle_returning("blah\nVERDICT: VALID")).correct
    assert not judge_secret_validity(SECRET, oracle_returning("VERDICT: INVALID")).correct


def test_validity_last_verdict_wins_and_case_insensitive():
    reply = "VERDICT: INVALID ... wait, reconsidering ... verdict: valid"
    assert judge_secret_validity(SECRET, oracle_returning(reply)).correct


@pytest.mark.parametrize("reply", ["", "it seems fine to me", "VERDICT: MAYBE"])
def test_validity_default_fails_open_without_verdict(reply):
    # New default (fail_open): no parseable verdict counts as VALID so gemma's
    # format flakiness stops discarding valid secrets (q-fullv2-rot fix).
    assert judge_secret_validity(SECRET, oracle_returning(reply)).correct
    # Opt back into the pre-fix posture explicitly.
    assert not judge_secret_validity(
        SECRET, oracle_returning(reply), mode="fail_closed").correct


def test_validity_clear_invalid_voids_in_every_mode():
    # A clear INVALID is always honoured; mode only decides the fallback.
    for mode in ("fail_open", "fail_closed"):
        assert not judge_secret_validity(
            SECRET, oracle_returning("VERDICT: INVALID"), mode=mode).correct


def test_validity_off_skips_judge_and_passes():
    # "off" never calls the oracle and always returns VALID.
    assert judge_secret_validity(SECRET, raising_oracle, mode="off").correct


def test_validity_judge_exception_respects_mode():
    assert judge_secret_validity(SECRET, raising_oracle).correct  # fail_open default
    assert not judge_secret_validity(
        SECRET, raising_oracle, mode="fail_closed").correct


def test_validity_prompt_carries_secret_and_category():
    seen = {}

    def oracle(prompt):
        seen["prompt"] = prompt
        return "VERDICT: VALID"

    judge_secret_validity(SECRET, oracle)
    assert "octopus" in seen["prompt"] and "animal" in seen["prompt"]


# ----- answer audit ----------------------------------------------------------

def test_audit_parses_tf_row():
    assert judge_answer_audit(SECRET, QA, oracle_returning("thinking...\nAUDIT: T F")) == [True, False]


def test_audit_tolerates_words_commas_case():
    assert judge_answer_audit(SECRET, QA, oracle_returning("AUDIT: true, FALSE")) == [True, False]
    assert judge_answer_audit(SECRET, QA, oracle_returning("audit: Y N")) == [True, False]


def test_audit_empty_pairs_is_trivially_empty():
    assert judge_answer_audit(SECRET, [], oracle_returning("AUDIT:")) == []


@pytest.mark.parametrize("reply", [
    "",                       # nothing
    "all answers look fine",  # no AUDIT line
    "AUDIT: T",               # count mismatch (2 pairs)
    "AUDIT: T F F",           # count mismatch
    "AUDIT: T MAYBE",         # unknown token
])
def test_audit_unparseable_returns_none(reply):
    assert judge_answer_audit(SECRET, QA, oracle_returning(reply)) is None


def test_audit_judge_exception_returns_none():
    assert judge_answer_audit(SECRET, QA, raising_oracle) is None


def test_audit_prompt_numbers_pairs():
    seen = {}

    def oracle(prompt):
        seen["prompt"] = prompt
        return "AUDIT: T T"

    judge_answer_audit(SECRET, QA, oracle)
    assert "1. Q: Is it a mammal?" in seen["prompt"]
    assert "2. Q: Does it live in water?" in seen["prompt"]


# ----- closeness -------------------------------------------------------------

@pytest.mark.parametrize("reply,phi", [
    ("CLOSENESS: 0", 0.0),
    ("CLOSENESS: 5", 0.5),
    ("the guesser is nearly there\nCLOSENESS: 10", 1.0),
    ("CLOSENESS: 14", 1.0),    # clamps high
    ("CLOSENESS: -2", 0.0),    # clamps low
    ("closeness = 7", 0.7),    # separator/case tolerance
])
def test_closeness_parse_and_clamp(reply, phi):
    assert judge_closeness(SECRET, QA, oracle_returning(reply)) == pytest.approx(phi)


def test_closeness_last_match_wins():
    reply = "CLOSENESS: 3 is where they started; now...\nCLOSENESS: 8"
    assert judge_closeness(SECRET, QA, oracle_returning(reply)) == pytest.approx(0.8)


@pytest.mark.parametrize("reply", ["", "pretty close!", "CLOSENESS: high"])
def test_closeness_unparseable_returns_none(reply):
    assert judge_closeness(SECRET, QA, oracle_returning(reply)) is None


def test_closeness_judge_exception_returns_none():
    assert judge_closeness(SECRET, QA, raising_oracle) is None


def test_closeness_monotone_scale():
    phis = [judge_closeness(SECRET, QA, oracle_returning(f"CLOSENESS: {n}"))
            for n in range(11)]
    assert phis == sorted(phis) and phis[0] == 0.0 and phis[-1] == 1.0


def test_closeness_final_guess_included_when_given():
    seen = {}

    def oracle(prompt):
        seen["prompt"] = prompt
        return "CLOSENESS: 9"

    judge_closeness(SECRET, QA, oracle, final_guess="squid")
    assert "Final guess: squid" in seen["prompt"]
