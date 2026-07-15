"""Local JSONL run logging (DESIGN.md §5 step 6, §11).

One JSON object per line. The first line of a fresh run is a ``meta`` record
(config snapshot, start time); each iteration appends one ``iteration`` record.
Deliberately dependency-free (no W&B for v1) and flushed eagerly so a run can be
tailed live and survives a crash.
"""

import json
import time
from pathlib import Path
from typing import Any, Iterator


class JsonlLogger:
    def __init__(self, path: str | Path, *, meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")
        if meta is not None:
            self._write({"type": "meta", **meta})

    def _write(self, obj: dict[str, Any]) -> None:
        obj.setdefault("time", time.time())
        # default=str so stray non-serializable values never crash a run.
        self._f.write(json.dumps(obj, default=str) + "\n")
        self._f.flush()

    def log(self, record: dict[str, Any]) -> None:
        self._write({"type": "iteration", **record})

    def log_validation(self, record: dict[str, Any]) -> None:
        """Append a stationary-evaluation record alongside iteration rows."""
        self._write({"type": "validation", **record})

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file back into a list of dicts (skips blank lines)."""
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
