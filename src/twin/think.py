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
# here to teach it a new model's thinking tokens.
#
# These are LITERAL strings, escaped for regex use below (they used to be
# hand-escaped, which made `|` a standing footgun) — because they are also
# consumed verbatim: THINK_OPEN_MARKERS is what the backend bans at the logits
# level when thinking is disabled, so this list is the single definition of
# "thinking" for both extraction and suppression (DESIGN §8).
THINK_SPANS = [
    ("<think>", "</think>"),
    ("<|channel>", "<channel|>"),
]
THINK_OPEN_MARKERS = [open_marker for open_marker, _ in THINK_SPANS]
_THINK_RE = re.compile(
    "|".join(rf"{re.escape(o)}.*?{re.escape(c)}" for o, c in THINK_SPANS),
    re.DOTALL | re.IGNORECASE,
)
_THINK_OPEN_RE = re.compile(
    "|".join(re.escape(o) for o in THINK_OPEN_MARKERS), re.IGNORECASE
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


# gemma-4 opens its reasoning span as ``<|channel>thought\n...`` — the word is
# the CHANNEL NAME, not the first line of reasoning, and rendering it verbatim
# puts a stray "thought" at the top of every transcript block. Dropped only
# when the first line is exactly a known label, so a real reasoning line that
# merely begins with the word survives.
_CHANNEL_LABELS = {"thought", "thinking", "analysis"}


def _drop_channel_label(body: str) -> str:
    head, sep, rest = body.lstrip("\n").partition("\n")
    if sep and head.strip().lower() in _CHANNEL_LABELS:
        return rest
    return body


def think_text(text: str) -> str:
    """The reasoning CONTENT of ``text``, markers stripped, spans joined by a
    blank line — the complement of :func:`strip_think`, for transcripts that
    show a model's thoughts separately from its answer (DESIGN §8).

    An unclosed trailing span (truncated reasoning) is included to its end,
    matching :func:`think_share`'s accounting: a truncation spiral must be
    visible in the transcript, not silently dropped for lacking a close tag.
    Returns "" when the model did not think — callers use that to decide
    whether to render a thoughts block at all."""
    text = text or ""
    if not text:
        return ""
    spans: list[str] = []
    for match in _THINK_RE.finditer(text):
        body = match.group(0)
        for open_marker, close_marker in THINK_SPANS:
            if body[:len(open_marker)].lower() == open_marker.lower():
                body = body[len(open_marker):]
                if body[-len(close_marker):].lower() == close_marker.lower():
                    body = body[:-len(close_marker)]
                break
        spans.append(_drop_channel_label(body).strip())
    # Unclosed remainder: whatever follows the last surviving open marker.
    stripped = _THINK_RE.sub("", text)
    open_match = _THINK_OPEN_RE.search(stripped)
    if open_match:
        spans.append(stripped[open_match.end():].strip())
    return "\n\n".join(s for s in spans if s)
