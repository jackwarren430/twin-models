"""ReAct-style text tool protocol (DESIGN.md §9).

The policy emits tool calls as ``<tool>name(arg)</tool>``; the harness executes
the named tool and returns its result as ``<obs>...</obs>``. Text-based (not
Qwen native function-calling) so it's robust across chat templates and trivial
to parse and sandbox. We can switch to native tool-calls later without changing
the tools themselves.

The ``arg`` is passed to the tool as a single string — tools (`calc`, `python`,
`oracle`) take free-form text, so we deliberately do **not** try to parse a
Python-style argument list. We match the outermost balanced parentheses so that
args containing ``()`` (code, expressions) survive intact.
"""

import re
from dataclasses import dataclass
from typing import Callable

_TOOL_OPEN = re.compile(r"<tool>\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", re.DOTALL)

# Sentinel tool name a parser may emit for a syntactically broken call (e.g.
# invalid JSON in a native <tool_call> block). The harness turns it into a
# format-error observation so the model gets corrective feedback instead of
# silence.
PARSE_ERROR_NAME = "__parse_error__"


@dataclass
class ToolCall:
    name: str
    arg: str


@dataclass
class ToolResult:
    call: ToolCall
    output: str
    ok: bool = True


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract ``<tool>name(arg)</tool>`` calls from model text, in order.

    Tolerant of whitespace and of ``arg`` containing parentheses; a call whose
    parens/closing tag are unbalanced is skipped rather than raising."""
    calls: list[ToolCall] = []
    for m in _TOOL_OPEN.finditer(text):
        name = m.group(1)
        arg_start = m.end()  # just past the opening '('
        arg, close_idx = _balanced_paren_arg(text, arg_start)
        if close_idx is None:
            continue
        # Require the closing </tool> to follow (allow whitespace between).
        tail = text[close_idx + 1 :]
        if re.match(r"\s*</tool>", tail):
            calls.append(ToolCall(name=name, arg=arg.strip()))
    return calls


def _balanced_paren_arg(text: str, start: int):
    """Return (arg_text, index_of_matching_close_paren) from ``start`` (which is
    just past the opening paren). Respects quotes so parens inside strings don't
    miscount. Returns (``""``, None) if unbalanced."""
    depth = 1
    in_str = None  # quote char or None
    esc = False
    for j in range(start, len(text)):
        c = text[j]
        if in_str is not None:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == in_str:
                in_str = None
            continue
        if c in ("'", '"'):
            in_str = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start:j], j
    return "", None


class ToolHarness:
    """Registry + executor for one rollout/turn.

    Holds the per-turn call budget and tracks oracle usage (the taxed quantity
    the reward engine needs). One harness instance per rollout; call
    :meth:`reset` to reuse."""

    def __init__(
        self,
        tools: dict[str, Callable[[str], str]],
        *,
        max_tool_calls: int = 4,
        parser: Callable[[str], list[ToolCall]] | None = None,
    ):
        self.tools = tools
        self.max_tool_calls = max_tool_calls
        # How calls are extracted from model text: the legacy <tool>name(arg)
        # </tool> parser by default, or twin.tools.native.parse_native_tool_calls
        # for Qwen3 function calling (Sprint 7).
        self.parser = parser or parse_tool_calls
        self.calls: list[ToolResult] = []
        self.reset()

    def reset(self) -> None:
        self.calls = []
        oracle = self.tools.get("oracle")
        if hasattr(oracle, "reset"):
            oracle.reset()

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def n_oracle_calls(self) -> int:
        oracle = self.tools.get("oracle")
        if hasattr(oracle, "n_calls"):
            return oracle.n_calls
        return sum(1 for r in self.calls if r.call.name == "oracle")

    def run(self, text: str) -> list[ToolResult]:
        """Execute every tool call found in ``text`` (up to the remaining
        budget) and return their results in order."""
        results: list[ToolResult] = []
        for call in self.parser(text):
            if self.n_calls >= self.max_tool_calls:
                results.append(ToolResult(call, "error: tool call budget exhausted", ok=False))
                self.calls.append(results[-1])
                break
            if call.name == PARSE_ERROR_NAME:
                res = ToolResult(
                    call,
                    'error: malformed tool call — the body must be one JSON '
                    'object like {"name": "solve", "arguments": '
                    '{"expression": "..."}}',
                    ok=False,
                )
                results.append(res)
                self.calls.append(res)
                continue
            fn = self.tools.get(call.name)
            if fn is None:
                res = ToolResult(call, f"error: unknown tool '{call.name}'", ok=False)
            else:
                try:
                    out = fn(call.arg)
                    res = ToolResult(call, str(out), ok=not str(out).startswith("error:"))
                except Exception as e:  # noqa: BLE001 - tool boundary
                    res = ToolResult(call, f"error: {type(e).__name__}: {e}", ok=False)
            results.append(res)
            self.calls.append(res)
        return results

    @staticmethod
    def format_observations(results: list[ToolResult]) -> str:
        """Render results as ``<obs>...</obs>`` blocks to feed back to the model."""
        return "\n".join(f"<obs>{r.output}</obs>" for r in results)
