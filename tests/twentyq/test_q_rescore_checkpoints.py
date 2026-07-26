"""Checkpoint re-scoring bookkeeping (scripts/rescore_checkpoints.py).

The measurement itself is ``TwentyQTrainer.run_validation`` and is covered by
the trainer's own tests. What is only in this script is the bookkeeping that
decides WHICH checkpoints get scored — and getting that wrong is expensive and
quiet: a half-written step pairs a trained B against a stale A, and a broken
resume silently re-spends hours or, worse, skips a step and leaves a gap in the
series that looks like a missing data point rather than a bug.
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "rescore_checkpoints", ROOT / "scripts" / "rescore_checkpoints.py"
)
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)


def _ckpts(tmp_path: Path, names: list[str]) -> Path:
    for n in names:
        (tmp_path / n).write_bytes(b"")
    return tmp_path


# ----- step discovery ------------------------------------------------------
def test_step_zero_is_always_available_without_a_file(tmp_path):
    """Adapters are zero-init, so a freshly built adapter IS the base model.
    Step 0 is the pre-training policy and needs no checkpoint on disk — but the
    run never saves one, so a discovery that only globbed files would drop the
    single most important point in the series: the baseline everything else is
    measured against."""
    assert rc.discover_steps(tmp_path) == [0]


def test_discovers_steps_present_for_every_adapter(tmp_path):
    d = _ckpts(tmp_path, ["adapter_A_step5.safetensors",
                          "adapter_B_step5.safetensors",
                          "adapter_A_step10.safetensors",
                          "adapter_B_step10.safetensors"])
    assert rc.discover_steps(d) == [0, 5, 10]


def test_partial_steps_are_excluded(tmp_path):
    """A step killed between the two saves leaves only adapter A. Scoring it
    would pair a step-15 A against whatever B was last loaded — a silent
    off-by-one-checkpoint comparison, which is exactly the kind of error the
    frozen-A control exists to catch and would instead be corrupted by."""
    d = _ckpts(tmp_path, ["adapter_A_step5.safetensors",
                          "adapter_B_step5.safetensors",
                          "adapter_A_step15.safetensors"])
    assert rc.discover_steps(d) == [0, 5]


def test_unparseable_checkpoint_names_are_ignored(tmp_path):
    d = _ckpts(tmp_path, ["adapter_A_stepFINAL.safetensors",
                          "adapter_A_step5.safetensors",
                          "adapter_B_step5.safetensors",
                          "notes.txt"])
    assert rc.discover_steps(d) == [0, 5]


def test_steps_sort_numerically_not_lexically(tmp_path):
    d = _ckpts(tmp_path, [f"adapter_{n}_step{s}.safetensors"
                          for s in (5, 10, 60) for n in ("A", "B")])
    assert rc.discover_steps(d) == [0, 5, 10, 60]


# ----- resume --------------------------------------------------------------
def test_already_done_reads_only_validation_records(tmp_path):
    out = tmp_path / "rescore.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in [
        {"type": "meta", "mode": "twentyq_rescore", "step": 999},
        {"type": "validation", "step": 0},
        {"type": "validation", "step": 20},
    ]) + "\n")
    # The meta row carries no measurement; counting it would skip a real step.
    assert rc.already_done(out) == {0, 20}


def test_already_done_survives_a_truncated_final_line(tmp_path):
    """The realistic failure: the process is killed mid-write. Every complete
    record before the tear must still count, or a resume re-spends hours."""
    out = tmp_path / "rescore.jsonl"
    out.write_text(json.dumps({"type": "validation", "step": 0}) + "\n"
                   + '{"type": "validation", "step": 2')
    assert rc.already_done(out) == {0}


def test_already_done_on_a_fresh_run(tmp_path):
    assert rc.already_done(tmp_path / "missing.jsonl") == set()
