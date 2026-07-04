"""Shared ``<think>`` handling (audit 2026-07-03).

With thinking mode ON, a rollout's reasoning trace is *scratch space*: drafts
in it must never beat the final post-think output. Two live bugs came from
ignoring this — a draft problem JSON inside ``<think>`` out-ranking the final
JSON (parse order is by start index), and a speculative ``\\boxed{}`` inside
``<think>`` out-ranking the solver's final ``ANSWER:`` line. Every extractor
that consumes model text should look at :func:`strip_think` output first and
fall back to the raw text only when the visible portion yields nothing.

An *unclosed* think block (truncated reasoning that never produced a final
answer) is deliberately left in place: extraction then sees the raw text, and
whether that should count is the caller's fallback policy, not ours.
"""

import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Drop closed ``<think>...</think>`` spans; leave unclosed blocks alone."""
    return _THINK_RE.sub("", text or "")
