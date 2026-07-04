"""Parsing helpers for solver output and tool usage (DESIGN.md §5, §6).

Kept model-free and tiny so they're trivially testable.
"""

import re

from twin.think import strip_think
from twin.tools.protocol import parse_tool_calls

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_LINE = re.compile(r"(?im)^[ \t>*-]*(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$")
_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str:
    """The solver's code answer: the last fenced code block in the post-think
    text (drafts inside ``<think>`` must not win — same rule as
    :func:`extract_final_answer`), falling back to the raw text's last block,
    then to the whole (post-think) text for models that skip the fence.
    Sprint 8 coding pipeline; the bench grader reuses this."""
    visible = strip_think(text).strip()
    for t in ([visible, text] if visible else [text]):
        blocks = _FENCE.findall(t or "")
        if blocks:
            return blocks[-1].strip()
    return visible if visible else (text or "").strip()


def extract_final_answer(text: str) -> str:
    """Pull the solver's final answer out of free-form text.

    Extraction runs on the post-``<think>`` text when there is any — a
    speculative ``\\boxed{}`` inside the reasoning trace must not beat the
    final ``ANSWER:`` line (audit 2026-07-03). The raw text is the fallback
    for rollouts whose only content is an unclosed think block.

    Preference order within the searched text: a ``\\boxed{...}`` span, then
    the last ``ANSWER:`` line (the form the solver prompt asks for), then the
    last non-empty line."""
    visible = strip_think(text).strip()
    return _extract_from(visible) if visible else _extract_from(text)


def _extract_from(text: str) -> str:
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    lines = _ANSWER_LINE.findall(text)
    if lines:
        return lines[-1].strip()
    nonempty = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return nonempty[-1] if nonempty else ""


def count_oracle_calls(text: str) -> int:
    """Number of ``<tool>oracle(...)</tool>`` calls the model emitted.

    v1 rollouts are single-turn (we don't execute inline tools mid-generation —
    that needs <obs> masking, deferred), but the oracle **tax** must still bite,
    so we count the calls the policy *wrote*. This keeps the reward pathway live;
    full ReAct execution is a Sprint-3.5/4 extension."""
    return sum(1 for c in parse_tool_calls(text) if c.name == "oracle")
