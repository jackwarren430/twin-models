#!/usr/bin/env python
"""Central test runner for twin-models.

One entry point for the whole suite. Tests are organized one folder per sprint
(``tests/sprint1`` … ``tests/sprint4``); the heavy end-to-end tests that load the
real base model are marked ``model`` and exposed here as the ``e2e`` suite. This
is a thin wrapper over pytest: it builds the marker expression, verbosity flags,
and (for e2e) raw-text logging for you, then hands the rest straight to pytest.

Selecting suites
----------------
    # all fast tests — the default (every sprint, model/e2e excluded)
    python scripts/run_tests.py

    # one or more quick sprints
    python scripts/run_tests.py --suite 1
    python scripts/run_tests.py --suite 1 2 3

    # the heavy end-to-end suite (loads the real base, ~6 GB, generates)
    python scripts/run_tests.py --suite e2e

    # a quick sprint AND e2e together
    python scripts/run_tests.py --suite 4 e2e

    # everything quick (alias for the default)
    python scripts/run_tests.py --suite all

Verbosity
---------
    python scripts/run_tests.py --suite 2 -v      # show test names
    python scripts/run_tests.py --suite 2 -vv     # very verbose
    python scripts/run_tests.py --suite 2 -q      # quiet

Raw-text logging (e2e only)
---------------------------
    # write everything the models emit (creator suites, solver attempts, judge
    # Q&A, per-iteration update summaries) to a readable transcript. Implies the
    # e2e suite. With no path, defaults to runs/e2e-<timestamp>.log.
    python scripts/run_tests.py --log
    python scripts/run_tests.py --log runs/my-e2e.log
    python scripts/run_tests.py --suite 3 e2e --log     # run sprint3 + e2e, log e2e

Other
-----
    python scripts/run_tests.py --suite 2 -k consistency   # filter by name
    python scripts/run_tests.py --suite 2 --list           # collect-only, don't run
    python scripts/run_tests.py --failed                   # rerun last-failed
    python scripts/run_tests.py tests/sprint2/test_rewards.py   # any pytest target

Anything after a literal ``--`` (or any unrecognized flag) is forwarded to
pytest unchanged, e.g. ``run_tests.py -- --durations=10``.

Run it inside the project env:  ``conda run -n twin-models python scripts/run_tests.py``
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
RUNS_DIR = REPO_ROOT / "runs"

ALL_SPRINTS = (1, 2, 3, 4, 5, 6)
SUITE_TOKENS = "1 2 3 4 5 6 e2e all"  # for help/error text


def _available_sprints() -> list[int]:
    """Sprint numbers whose folder currently contains at least one test file."""
    found = []
    for n in ALL_SPRINTS:
        d = TESTS_DIR / f"sprint{n}"
        if d.is_dir() and any(d.glob("test_*.py")):
            found.append(n)
    return found


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_tests.py",
        description="Run the twin-models test suite by sprint or e2e, with "
                    "verbosity, selection, and raw-text logging options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("targets", nargs="*",
                   help="specific pytest targets (files / node ids); default: the whole selection")
    p.add_argument("--suite", nargs="+", metavar="SUITE",
                   help=f"which suite(s) to run; tokens: {SUITE_TOKENS} "
                        "(default: all quick sprints, e2e excluded)")
    p.add_argument("-k", dest="keyword", metavar="EXPR",
                   help="only tests matching this pytest -k expression")
    p.add_argument("--log", nargs="?", const="", default=None, metavar="PATH",
                   help="e2e only: write a raw-text transcript of all model output; "
                        "implies the e2e suite. Default path: runs/e2e-<timestamp>.log")

    verb = p.add_mutually_exclusive_group()
    verb.add_argument("-q", "--quiet", action="store_true", help="quiet output")
    verb.add_argument("-v", "--verbose", action="count", default=0,
                      help="verbose (-v) or very verbose (-vv)")

    p.add_argument("--list", action="store_true",
                   help="list the selected tests (collect-only) and exit")
    p.add_argument("--failed", "--lf", dest="failed", action="store_true",
                   help="rerun only the tests that failed last time")
    p.add_argument("--durations", type=int, metavar="N",
                   help="report the N slowest tests")
    p.add_argument("--dry-run", action="store_true",
                   help="print the pytest args that would run, then exit")

    # Back-compat aliases for the pre-suite interface.
    compat = p.add_argument_group("deprecated aliases (prefer --suite)")
    compat.add_argument("--sprint", "-s", nargs="+", type=int, choices=ALL_SPRINTS,
                        metavar="N", help="alias for --suite N ...")
    compat.add_argument("--model", action="store_true",
                        help="alias for adding e2e to the default quick suites")
    compat.add_argument("--only-model", action="store_true",
                        help="alias for --suite e2e")
    return p


class Selection:
    """Resolved suite selection: which quick sprints, whether e2e is included,
    and the e2e transcript path (if logging)."""

    def __init__(self):
        self.quick_sprints: set[int] = set()  # empty + quick => all sprints
        self.quick = False                    # any quick tests wanted
        self.e2e = False                      # the model/e2e suite wanted
        self.log_path: Path | None = None
        self.notes: list[str] = []

    def marker_expr(self) -> str:
        quick_part = None
        if self.quick:
            if self.quick_sprints and set(self.quick_sprints) != set(ALL_SPRINTS):
                names = " or ".join(f"sprint{n}" for n in sorted(self.quick_sprints))
                quick_part = f"({names}) and not model"
            else:
                quick_part = "not model"
        e2e_part = "model" if self.e2e else None
        if quick_part and e2e_part:
            return f"({quick_part}) or {e2e_part}"
        return quick_part or e2e_part or ""

    def describe(self) -> str:
        bits = []
        if self.quick:
            bits.append("sprints " + (",".join(map(str, sorted(self.quick_sprints)))
                                      if self.quick_sprints else "all"))
        if self.e2e:
            bits.append("e2e")
        return " + ".join(bits) if bits else "none"


def resolve_selection(args) -> Selection:
    sel = Selection()

    if args.suite:
        for tok in args.suite:
            t = tok.lower()
            if t in ("e2e", "model"):
                sel.e2e = True
            elif t == "all":
                sel.quick = True
                sel.quick_sprints |= set(ALL_SPRINTS)
            elif t.isdigit() and int(t) in ALL_SPRINTS:
                sel.quick = True
                sel.quick_sprints.add(int(t))
            else:
                raise SystemExit(
                    f"run_tests.py: unknown --suite token {tok!r}; use one of: {SUITE_TOKENS}")

    # Deprecated aliases fold into the same selection.
    if args.sprint:
        sel.quick = True
        sel.quick_sprints |= set(args.sprint)
    if args.only_model:
        sel.e2e = True
    elif args.model:
        sel.e2e = True
        sel.quick = True  # historically "also run model tests" on top of all quick

    # --log is an e2e feature; asking for it selects e2e.
    if args.log is not None:
        if not sel.e2e:
            sel.notes.append("--log selects the e2e suite (raw-text logging is e2e-only)")
        sel.e2e = True
        if args.log == "":
            sel.log_path = RUNS_DIR / f"e2e-{time.strftime('%Y%m%d-%H%M%S')}.log"
        else:
            sel.log_path = Path(args.log)

    # Default: nothing chosen => all quick sprints, e2e excluded.
    if not sel.quick and not sel.e2e:
        sel.quick = True

    return sel


def _verbosity_flags(args) -> list[str]:
    if args.quiet:
        return ["-q"]
    if args.verbose >= 2:
        return ["-vv"]
    if args.verbose == 1:
        return ["-v"]
    return []


def build_pytest_args(args, sel: Selection, passthrough: list[str]) -> list[str]:
    argv: list[str] = []
    expr = sel.marker_expr()
    if expr:
        argv += ["-m", expr]
    argv += _verbosity_flags(args)
    if args.keyword:
        argv += ["-k", args.keyword]
    if args.failed:
        argv += ["--lf"]
    if args.list:
        argv += ["--collect-only"]
    if args.durations is not None:
        argv += [f"--durations={args.durations}"]
    argv += passthrough
    argv += args.targets  # empty => pytest uses testpaths (tests/)
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)

    if not _available_sprints():
        print("No test files found under tests/sprintN/.", file=sys.stderr)
        return 1

    sel = resolve_selection(args)

    # The e2e/model tests are gated on this env var (see tests/sprint4/test_e2e_model.py).
    if sel.e2e:
        os.environ["TWIN_RUN_MODEL_TESTS"] = "1"
    # Raw-text transcript path the e2e suite reads (tests/sprint4 transcript fixture).
    if sel.log_path is not None:
        sel.log_path.parent.mkdir(parents=True, exist_ok=True)
        os.environ["TWIN_E2E_LOG"] = str(sel.log_path.resolve())

    pytest_args = build_pytest_args(args, sel, passthrough)

    os.chdir(REPO_ROOT)  # so testpaths + relative config paths resolve
    print(f"available sprints: {_available_sprints()}")
    print(f"selection: {sel.describe()}")
    for note in sel.notes:
        print(f"note: {note}")
    if sel.log_path is not None:
        print(f"raw-text transcript: {sel.log_path}")
    print(f"pytest {' '.join(pytest_args)}\n")
    if args.dry_run:
        return 0

    import pytest  # imported here so --help works without pytest installed
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
