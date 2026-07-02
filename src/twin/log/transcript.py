"""Human-readable raw-text transcript logging for e2e runs.

The JSONL run log (:class:`~twin.log.jsonl.JsonlLogger`) keeps only numeric /
metadata summaries — small, structured, and meant to be plotted. This is its
counterpart: a plain-text log of *everything the models actually wrote*, in the
order they wrote it, so a run can be read back like a transcript.

``scripts/train.py`` wires it in by default (``--no-transcript`` opts out), as
does the e2e suite / ``scripts/run_tests.py --log``; it is gated inside the
trainer, so a run without a sink carries no overhead. Like the JSONL logger it
is dependency-free, append-mode, and flushed eagerly so the file can be tailed
live and survives a crash.

Format — easy to read and to grep/extract:

    ########## TRANSCRIPT 2026-06-29 01:23:45 | test=sprint4_e2e ##########

    ===== iter 0 | domain=math theme=quadratics | creator=A solver=B =====

    >>> creator[0] adapter=A | n_tool_calls=1 n_oracle=0
    <the creator's full raw output>
    <<<

    * problem[0.0] | difficulty=0.0 creator_answer='3' consistent=True
      Solve x^2 - 5x + 6 = 0; give the larger root.
    ...
"""

import time
from pathlib import Path
from typing import Any


def _fmt_meta(meta: dict[str, Any]) -> str:
    return " ".join(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}"
                    for k, v in meta.items())


class TranscriptLogger:
    def __init__(self, path: str | Path, *, meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")
        header = f"########## TRANSCRIPT {time.strftime('%Y-%m-%d %H:%M:%S')}"
        if meta:
            header += " | " + _fmt_meta(meta)
        self._f.write(f"\n{header} ##########\n")
        self._f.flush()

    def section(self, title: str) -> None:
        """A run/iteration-level divider."""
        self._f.write(f"\n===== {title} =====\n")
        self._f.flush()

    def entry(self, label: str, text: str = "", **meta: Any) -> None:
        """One labelled event. With ``text`` it is fenced as a raw block; without
        it (metadata only) it is a single bullet line."""
        suffix = f" | {_fmt_meta(meta)}" if meta else ""
        body = text.rstrip("\n")
        if body:
            self._f.write(f"\n>>> {label}{suffix}\n{body}\n<<<\n")
        else:
            self._f.write(f"* {label}{suffix}\n")
        self._f.flush()

    def block(self, text: str) -> None:
        """Write a verbatim multi-line block (e.g. an end-of-run summary),
        unfenced, between blank lines."""
        self._f.write(f"\n{text.rstrip(chr(10))}\n")
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.flush()
            self._f.close()

    def __enter__(self) -> "TranscriptLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
