"""Power arithmetic for reading a validation null (scripts/validation_power.py).

This tool exists to stop a specific mistake: treating "validation was flat" as
evidence of no effect when the instrument could not have resolved the effect in
question. The arithmetic has to be right, because its output is used to decide
whether a run's headline result stands.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "validation_power", ROOT / "scripts" / "validation_power.py"
)
vp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vp)


def _record(step, per_adapter):
    """per_adapter: {name: {secret: [bool, ...] one entry per EPISODE}}."""
    return {
        "type": "validation", "step": step,
        "adapters": {
            name: {"secrets": [{"secret": s, "guessed": g}
                               for s, eps in by_secret.items() for g in eps]}
            for name, by_secret in per_adapter.items()
        },
    }


# ----- episode-vs-secret grouping ------------------------------------------
def test_rates_average_over_episodes_not_last_one_wins():
    """The record stores one row per EPISODE. A dict keyed by secret name
    collapses K rows to the last one and reports 0.0 or 1.0 for everything —
    the exact bug that once turned 8 episodes into a bogus per-secret count."""
    rec = _record(0, {"B": {"cow": [True, True, False, False]}})
    assert vp.per_secret_rates(rec)["B"] == {"cow": 0.5}


def test_rates_are_reported_per_adapter():
    rec = _record(0, {"A": {"cow": [False, False]}, "B": {"cow": [True, True]}})
    rates = vp.per_secret_rates(rec)
    assert rates["A"]["cow"] == 0.0 and rates["B"]["cow"] == 1.0


# ----- live vs dead --------------------------------------------------------
def test_a_secret_won_by_nobody_at_any_checkpoint_is_dead():
    recs = [_record(s, {"A": {"tapir": [False, False], "cow": [True, False]},
                        "B": {"tapir": [False, False], "cow": [False, True]}})
            for s in (0, 5)]
    assert vp.live_secrets(recs) == {"cow"}


def test_one_win_at_one_checkpoint_by_one_adapter_makes_a_secret_live():
    """Liveness is about whether the item can register a change at all, so a
    single win anywhere is enough. Being strict here would overstate how dead
    the instrument is and understate its measured resolution."""
    recs = [_record(0, {"A": {"giraffe": [False, False]},
                        "B": {"giraffe": [False, False]}}),
            _record(5, {"A": {"giraffe": [False, False]},
                        "B": {"giraffe": [True, False]}})]
    assert vp.live_secrets(recs) == {"giraffe"}


# ----- paired spread -------------------------------------------------------
def test_paired_sd_ignores_between_secret_spread():
    """The whole reason for pairing. Two secrets with wildly different rates
    but an identical B-A gap have zero paired spread; an unpaired calculation
    would charge that between-secret difference as noise."""
    rec = _record(0, {"A": {"easy": [True, True, True, False],
                            "hard": [False, False, False, False]},
                      "B": {"easy": [True, True, True, True],
                            "hard": [True, False, False, False]}})
    assert vp.paired_sd([rec], "A", "B") == 0.0


def test_paired_sd_can_be_restricted_to_live_secrets():
    """Dead secrets contribute an exact zero difference, which shrinks the sd
    while adding no information — so the two views must differ."""
    rec = _record(0, {"A": {"dead": [False, False], "live": [False, False]},
                      "B": {"dead": [False, False], "live": [True, True]}})
    assert vp.paired_sd([rec], "A", "B", restrict={"live"}) != vp.paired_sd(
        [rec], "A", "B")


# ----- power arithmetic ----------------------------------------------------
def test_mde_shrinks_as_the_square_root_of_n():
    """Quadrupling the set halves the smallest detectable effect — the reason
    'add more secrets' has sharply diminishing returns past a point."""
    assert abs(vp.mde(0.2, 100) - vp.mde(0.2, 25) / 2) < 1e-12


def test_mde_and_n_required_are_inverses():
    """They are quoted side by side in the report; if they disagreed, one of
    the two numbers used to size a new set would be silently wrong."""
    sd, effect = 0.1867, 0.10
    n = vp.n_required(sd, effect)
    assert abs(vp.mde(sd, round(n)) - effect) < 0.005


def test_smaller_effects_need_quadratically_more_secrets():
    sd = 0.2
    assert abs(vp.n_required(sd, 0.05) / vp.n_required(sd, 0.10) - 4.0) < 1e-9
