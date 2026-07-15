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

# Known reasoning-trace delimiters as (open, close) special-token pairs whose
# enclosed span is scratch reasoning to be dropped before extraction:
#   - Qwen3 / generic:  <think> ... </think>
#   - gemma-4 channels: <|channel>thought ... <channel|>   (open id 100, close 101)
# Matching is by PRESENCE in the text — no model identity is threaded through the
# pure-text parsers — so one extractor path handles either family; add a pair
# here (regex-escaped) to teach it a new model's thinking tokens. `|` in a token
# must be escaped (\|) or the alternation swallows it.
_THINK_PAIRS = [
    (r"<think>", r"</think>"),
    (r"<\|channel>", r"<channel\|>"),
]
_THINK_RE = re.compile(
    "|".join(rf"{o}.*?{c}" for o, c in _THINK_PAIRS), re.DOTALL | re.IGNORECASE
)
_THINK_OPEN_RE = re.compile(
    "|".join(o for o, _ in _THINK_PAIRS), re.IGNORECASE
)


def strip_think(text: str) -> str:
    """Drop closed reasoning spans (``<think>...</think>`` or gemma-4's
    ``<|channel>...<channel|>``) so extraction only sees post-think output;
    leave an unclosed block alone (the caller's fallback policy decides whether
    truncated reasoning counts)."""
    return _THINK_RE.sub("", text or "")


def think_share(text: str) -> float:
    """Fraction of ``text`` (by characters) inside think spans, in [0, 1].

    Telemetry for SPIRAL-style thinking collapse (RELATED_WORK.md §2): a
    brevity-pressured policy can learn to stop reasoning entirely — watch this
    trend toward 0 — or to never leave the trace (truncation spirals,
    mini-03a) — watch it pin at 1. An *unclosed* trailing ``<think>`` counts
    as thinking to the end, so a truncation spiral reads ~1.0, not 0."""
    text = text or ""
    if not text:
        return 0.0
    closed = sum(len(m.group(0)) for m in _THINK_RE.finditer(text))
    stripped = _THINK_RE.sub("", text)
    m = _THINK_OPEN_RE.search(stripped)
    unclosed = len(stripped) - m.start() if m else 0
    return min(1.0, (closed + unclosed) / len(text))
