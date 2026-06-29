#!/usr/bin/env python
"""Central test runner for twin-models.

One entry point for the whole suite, which is organized one folder per sprint
(``tests/sprint1`` … ``tests/sprint4``). Thin wrapper over pytest: it builds the
marker expression and verbosity flags for you, then hands everything else
straight through to pytest.

Examples
--------
    # all fast tests (model tests excluded) — the default
    python scripts/run_tests.py

    # just Sprint 2, verbose
    python scripts/run_tests.py --sprint 2 -v

    # Sprints 1 and 2
    python scripts/run_tests.py --sprint 1 2

    # filter by name within the selection
    python scripts/run_tests.py --sprint 2 -k consistency

    # include the heavy model tests (loads the real base)
    python scripts/run_tests.py --model
    python scripts/run_tests.py --only-model            # ONLY those

    # a specific file or node id (any pytest target works)
    python scripts/run_tests.py tests/sprint2/test_rewards.py
    python scripts/run_tests.py tests/sprint2/test_calc.py::test_integer_arithmetic

    # list what would run, without running it
    python scripts/run_tests.py --sprint 2 --list

    # rerun only last-failed
    python scripts/run_tests.py --failed

Anything after a literal ``--`` (or any unrecognized flag) is forwarded to
pytest unchanged, e.g. ``run_tests.py -- --durations=10``.

Run it inside the project env:  ``conda run -n twin-models python scripts/run_tests.py``
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"

ALL_SPRINTS = (1, 2, 3, 4)


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
        description="Run the twin-models test suite by sprint, with verbosity and selection options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("targets", nargs="*",
                   help="specific pytest targets (files / node ids); default: the whole selection")
    p.add_argument("--sprint", "-s", nargs="+", type=int, choices=ALL_SPRINTS, metavar="N",
                   help="restrict to these sprint(s); default: all sprints that have tests")
    p.add_argument("-k", dest="keyword", metavar="EXPR",
                   help="only tests matching this pytest -k expression")

    model = p.add_mutually_exclusive_group()
    model.add_argument("--model", action="store_true",
                       help="also run heavy model tests (sets TWIN_RUN_MODEL_TESTS=1)")
    model.add_argument("--only-model", action="store_true",
                       help="run ONLY the heavy model tests")

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
    return p


def _marker_expr(args) -> str:
    pieces: list[str] = []
    if args.sprint:
        names = " or ".join(f"sprint{n}" for n in sorted(set(args.sprint)))
        pieces.append(f"({names})")
    if args.only_model:
        pieces.append("model")
    elif not args.model:
        pieces.append("not model")  # default: skip the slow ones
    return " and ".join(pieces)


def _verbosity_flags(args) -> list[str]:
    if args.quiet:
        return ["-q"]
    if args.verbose >= 2:
        return ["-vv"]
    if args.verbose == 1:
        return ["-v"]
    return []


def build_pytest_args(args, passthrough: list[str]) -> list[str]:
    argv: list[str] = []
    expr = _marker_expr(args)
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

    # Model tests are gated by this env var (see tests/sprint1/test_adapters.py).
    if args.model or args.only_model:
        os.environ["TWIN_RUN_MODEL_TESTS"] = "1"

    pytest_args = build_pytest_args(args, passthrough)

    os.chdir(REPO_ROOT)  # so testpaths + relative config paths resolve
    print(f"available sprints: {_available_sprints()}")
    print(f"pytest {' '.join(pytest_args)}\n")
    if args.dry_run:
        return 0

    import pytest  # imported here so --help works without pytest installed
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
