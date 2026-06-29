"""Shared pytest setup for the twin-models test suite.

Tests are organized one directory per sprint (``tests/sprint1`` … ``tests/sprint4``).
Rather than repeat a ``pytestmark`` in every file, we derive the ``sprintN``
marker from the file's directory here, so the only thing a test file must do to
join a sprint is live in that sprint's folder. The ``model`` marker (heavy tests
that load the real base) is still applied explicitly where it's needed.

Run tests via ``scripts/run_tests.py`` (the central endpoint) or plain pytest;
both see these markers. Select a sprint with ``-m sprint2``, exclude slow model
tests with ``-m "not model"``, etc.
"""

import re
from pathlib import Path

import pytest

_SPRINT_DIR = re.compile(r"^sprint(\d+)$")


def pytest_collection_modifyitems(config, items):
    """Auto-apply a ``sprintN`` marker to every test under ``tests/sprintN/``."""
    for item in items:
        for part in Path(str(item.fspath)).parts:
            m = _SPRINT_DIR.match(part)
            if m:
                item.add_marker(getattr(pytest.mark, f"sprint{m.group(1)}"))
                break
