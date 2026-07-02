"""Sprint 4 — raw-text transcript logger (e2e logging). Fast: no model."""

from twin.log import TranscriptLogger


def test_writes_sections_entries_and_notes(tmp_path):
    path = tmp_path / "t.log"
    with TranscriptLogger(path, meta={"test": "demo"}) as tr:
        tr.section("iter 0 | creator=A solver=B")
        tr.entry("creator[0] adapter=A", '{"problems": []}', n_tool_calls=1)
        tr.entry("problem[0.0]", "Solve x+1=3.", consistent=True)
        tr.entry("creator[0] reward", reward=0.5)  # metadata-only => bullet line

    text = path.read_text()
    # header carries the run meta
    assert "TRANSCRIPT" in text and "test='demo'" in text
    # section divider
    assert "===== iter 0 | creator=A solver=B =====" in text
    # raw text is fenced verbatim, with its label + meta
    assert ">>> creator[0] adapter=A | n_tool_calls=1" in text
    assert '{"problems": []}' in text
    assert "Solve x+1=3." in text
    assert "consistent=True" in text
    assert "<<<" in text
    # metadata-only entry is a single bullet, not a fenced block
    assert "* creator[0] reward | reward=0.5" in text


def test_append_mode_accumulates(tmp_path):
    path = tmp_path / "t.log"
    with TranscriptLogger(path) as tr:
        tr.entry("first", "alpha")
    with TranscriptLogger(path) as tr:
        tr.entry("second", "beta")
    text = path.read_text()
    assert "alpha" in text and "beta" in text
    assert text.count("########## TRANSCRIPT") == 2  # one header per open


def test_block_writes_verbatim(tmp_path):
    path = tmp_path / "t.log"
    with TranscriptLogger(path) as tr:
        tr.section("RUN SUMMARY")
        tr.block("line one\nline two")
    text = path.read_text()
    assert "===== RUN SUMMARY =====" in text
    assert "line one\nline two" in text


def test_run_summary_counts_parses_and_solver_accuracy():
    """The end-of-run digest aggregates parse failures, creator consistency, and
    solver accuracy from synthetic iteration records (no model)."""
    from twin.analysis import format_run_summary

    records = [
        {  # iter 0: both creator suites fail to parse, no solver work
            "iter": 0,
            "suites": [
                {"parsed": False, "error": "suite missing 'problems' list"},
                {"parsed": False, "error": "suite missing 'problems' list"},
            ],
            "solver_stats": {"problems": 0, "attempts": 0,
                             "attempts_solved": 0, "problems_solved": 0},
        },
        {  # iter 1: two suites parse; 3 of 4 authored problems are consistent
            "iter": 1,
            "suites": [
                {"parsed": True, "n_problems": 2, "n_consistent": 2},
                {"parsed": True, "n_problems": 2, "n_consistent": 1},
            ],
            "solver_stats": {"problems": 3, "attempts": 3,
                             "attempts_solved": 2, "problems_solved": 2},
        },
    ]
    text = format_run_summary(records)
    assert "iterations: 2" in text
    assert "suites generated" in text and "2/4" in text       # parsed OK 2 of 4
    assert "2x  suite missing 'problems' list" in text         # grouped w/ count
    assert "3/4" in text                                       # consistent 3 of 4
    assert "2/3" in text                                       # solver solved 2 of 3
    assert "iter 0:" in text and "iter 1:" in text             # per-iteration rows


def test_run_summary_empty():
    from twin.analysis import format_run_summary

    assert format_run_summary([]) == "(no iterations)"
