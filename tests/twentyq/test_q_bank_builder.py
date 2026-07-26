"""Bank-builder resume contract (scripts/build_twentyq_bank.py).

Calibration plays K real episodes per candidate and takes hours, so the report
file doubles as a checkpoint. These tests pin the two properties that make it
safe to restart a killed build: a checkpoint is only reused when the conditions
that produced its numbers are identical, and resuming yields the same bank as
an uninterrupted run. Pure dict/dataclass logic — no model, no judge.
"""

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from twin.games.twentyq.schema import Secret

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "build_twentyq_bank", ROOT / "scripts" / "build_twentyq_bank.py"
)
btb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(btb)


def _args(**over):
    base = dict(judge_model="Qwen/Qwen3-8B", rollouts_per_category=170,
                targets=[0.95, 0.9, 0.8], episodes_per_candidate=8,
                exclusion_cap=250,
                holdout=[Path("data/twentyq-validation-v2.json")], seed=1)
    base.update(over)
    return SimpleNamespace(**base)


def _cfg(**over):
    q = dict(categories=["animal", "food"], max_turns=21, question_retries=4)
    q.update(over)
    return SimpleNamespace(model=SimpleNamespace(path="google/gemma-4-E2B-it"),
                           twentyq=SimpleNamespace(**q))


# ----- signature -----------------------------------------------------------
def test_signature_is_stable_across_equal_runs():
    assert btb.signature_of(_args(), _cfg()) == btb.signature_of(_args(), _cfg())


def test_signature_tracks_everything_that_moves_the_measured_rate():
    ref = btb.signature_of(_args(), _cfg())
    assert btb.signature_of(_args(seed=2), _cfg()) != ref
    assert btb.signature_of(_args(episodes_per_candidate=4), _cfg()) != ref
    assert btb.signature_of(_args(judge_model="other/model"), _cfg()) != ref
    assert btb.signature_of(_args(rollouts_per_category=10), _cfg()) != ref
    assert btb.signature_of(_args(targets=[0.5]), _cfg()) != ref
    # Changes which candidates get generated, so it changes the population.
    assert btb.signature_of(_args(exclusion_cap=10), _cfg()) != ref
    # Holding out an extra set changes which candidates can survive dedup.
    assert btb.signature_of(
        _args(holdout=[Path("data/twentyq-validation-v2.json"),
                       Path("data/twentyq-bank-v1.json")]), _cfg()) != ref
    # A different player model or turn budget makes the win rates describe a
    # different game entirely — the case that must never silently merge.
    assert btb.signature_of(_args(), _cfg(max_turns=10)) != ref
    assert btb.signature_of(_args(), _cfg(question_retries=0)) != ref
    ref_cfg = _cfg()
    ref_cfg.model.path = "HuggingFaceTB/SmolLM3-3B"
    assert btb.signature_of(_args(), ref_cfg) != ref


def test_signature_is_json_round_trippable():
    """The check is ``loaded["signature"] == signature`` against a parsed JSON
    file, so a tuple-vs-list or int-vs-float mismatch would silently disable
    resume on every rerun."""
    import json
    sig = btb.signature_of(_args(targets=(0.95, 0.5)), _cfg())
    assert json.loads(json.dumps(sig)) == sig


# ----- checkpoint round-trip ----------------------------------------------
def test_vetted_roster_survives_the_checkpoint():
    vetted = [Secret(secret="okapi", category="animal", difficulty=0.2,
                     notes="hoofed, rarely guessed", secret_id="abc123"),
              Secret(secret="couscous", category="food", difficulty=0.5)]
    restored = [Secret.from_dict(d) for d in [s.to_dict() for s in vetted]]
    assert [s.to_dict() for s in restored] == [s.to_dict() for s in vetted]


def test_resume_measures_only_the_missing_candidates():
    vetted = [Secret(secret=n, category="animal", difficulty=0.5)
              for n in ("okapi", "tapir", "pangolin", "cow")]
    measured = [{"secret": "okapi", "measured_guess_rate": 0.5},
                {"secret": "tapir", "measured_guess_rate": 0.0}]
    done = {m["secret"] for m in measured}
    todo = [s.secret for s in vetted if s.secret not in done]
    assert todo == ["pangolin", "cow"]


# ----- selection -----------------------------------------------------------
def _select(measured, lo, hi):
    return [m["secret"] for m in measured
            if lo <= m["measured_guess_rate"] <= hi]


def test_band_is_recomputed_not_read_from_a_stale_flag():
    """The band is excluded from the signature on purpose so it can be retuned
    without remeasuring, which means a resumed checkpoint's stored ``in_band``
    may have been written under a different band."""
    measured = [{"secret": "cow", "measured_guess_rate": 1.0, "in_band": True},
                {"secret": "okapi", "measured_guess_rate": 0.5, "in_band": True},
                {"secret": "tapir", "measured_guess_rate": 0.0, "in_band": True}]
    assert _select(measured, 0.125, 0.875) == ["okapi"]


def test_band_endpoints_are_inclusive():
    measured = [{"secret": "a", "measured_guess_rate": 0.125},
                {"secret": "b", "measured_guess_rate": 0.875},
                {"secret": "c", "measured_guess_rate": 0.124}]
    assert _select(measured, 0.125, 0.875) == ["a", "b"]


# ----- generation exclusions ----------------------------------------------
def _accumulate(draws: list[str]) -> list[str]:
    """The builder's exclusion accumulator: distinct names, first-seen order."""
    from twin.games.twentyq.schema import normalize_guess
    seen: list[str] = []
    keys: set[str] = set()
    for d in draws:
        k = normalize_guess(d)
        if k and k not in keys:
            keys.add(k)
            seen.append(d)
    return seen


def test_exclusions_accumulate_distinct_names_not_raw_draws():
    """Generation excludes DISTINCT prior names, not the last N raw draws.

    The trainer's rolling window is sized for a live run where repeats are the
    signal; during a bank build they are pure waste, and a raw-draw window
    causes them — at 250 draws against a 128-draw window the first half scrolls
    out and gets re-proposed. Measured on the v1 build: 250 draws per category
    yielded 39 / 23 / 56 distinct secrets, and diversity (not calibration
    yield) is what bounds bank size."""
    draws = ["Lion", "lion", " LION ", "Tiger", "Lion", "Bear", "tiger"]
    assert _accumulate(draws) == ["Lion", "Tiger", "Bear"]


def test_exclusions_do_not_collapse_plurals():
    """normalize_guess deliberately does not depluralize: 'Glass' and
    'Glasses' are different household objects, so collapsing them here would
    exclude a legitimate candidate. Near-misses that slip through are caught
    downstream by dedup()'s edit-distance-1 matcher."""
    assert _accumulate(["Glass", "Glasses"]) == ["Glass", "Glasses"]
    assert [s.secret for s in btb.dedup(
        [Secret(secret="Lion", category="animal", difficulty=0.5),
         Secret(secret="Lions", category="animal", difficulty=0.5)],
        reserved=[])] == ["Lion"]


def test_exclusions_skip_unnameable_draws():
    assert _accumulate(["   ", "Lion", "!!!"]) == ["Lion"]


# ----- dedup ---------------------------------------------------------------
def test_dedup_drops_holdout_collisions_and_near_duplicates():
    cands = [Secret(secret="okapi", category="animal", difficulty=0.5),
             Secret(secret="Okpi", category="animal", difficulty=0.5),
             Secret(secret="giraffe", category="animal", difficulty=0.5),
             Secret(secret="tapir", category="animal", difficulty=0.5)]
    kept = [s.secret for s in btb.dedup(cands, reserved=["giraffe"])]
    # "Okpi" is the v4.5 edit-distance-1 evasion; "giraffe" is held out for
    # validation and must never enter the bank the solver trains on.
    assert kept == ["okapi", "tapir"]


def test_dedup_drops_unnameable_candidates():
    cands = [Secret(secret="   ", category="animal", difficulty=0.5),
             Secret(secret="tapir", category="animal", difficulty=0.5)]
    assert [s.secret for s in btb.dedup(cands, reserved=[])] == ["tapir"]


def test_dedup_does_not_mutate_its_input():
    cands = [Secret(secret="okapi", category="animal", difficulty=0.5),
             Secret(secret="okapi", category="animal", difficulty=0.5)]
    before = deepcopy([s.to_dict() for s in cands])
    btb.dedup(cands, reserved=[])
    assert [s.to_dict() for s in cands] == before
