"""Parsing helpers for solver output and tool usage (DESIGN.md §5, §6).

Kept model-free and tiny so they're trivially testable.
"""

import re

from twin.tools.protocol import parse_tool_calls

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_LINE = re.compile(r"(?im)^[ \t>*-]*(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$")


def extract_final_answer(text: str) -> str:
    """Pull the solver's final answer out of free-form text.

    Preference order: a ``\\boxed{...}`` span, then the last ``ANSWER:`` line
    (the form the solver prompt asks for), then the last non-empty line."""
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
