"""Qwen3 native function-calling protocol (Sprint 7, DESIGN.md §9).

The legacy ReAct text protocol (``<tool>name(arg)</tool>``) is not something
Qwen3 was trained to emit: with thinking ON the model plans tool calls inside
``<think>`` and hallucinates the results — mini-03b's transcript contains ZERO
real calls but plenty of "The tool would return x=3, y=2". Qwen3's own chat
template declares tools in the system block and the model emits

    <tool_call>
    {"name": "solve", "arguments": {"expression": "2*x + 3 = 11"}}
    </tool_call>

which it *was* post-trained on. This module provides the pieces to run that
protocol through the existing segmented-generation machinery
(``TwinBase.generate_react``):

  * :data:`NATIVE_STOP` — the stop string (``</tool_call>``).
  * :func:`parse_native_tool_calls` — extract calls from a generated segment
    (drop-in ``parser`` for :class:`~twin.tools.protocol.ToolHarness`).
  * :func:`format_tool_responses` — the masked glue spliced back into the
    stream: closes the assistant turn, renders a ``<tool_response>`` user turn
    exactly as the chat template would, and opens a fresh assistant turn.
  * :func:`tool_schemas` — JSON function signatures for ``render(tools=...)``.

The glue strings mirror the Qwen3 chat template's ``tool``-role rendering; a
model-gated test (tests/sprint7) verifies them against
``apply_chat_template`` ground truth so template drift is caught.
"""

import json
import re

from twin.tools.cas import SOLVE_TOOL_DOC
from twin.tools.protocol import PARSE_ERROR_NAME, ToolCall, ToolResult

NATIVE_STOP = "</tool_call>"

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

_IM_END = "<|im_end|>"
_IM_START = "<|im_start|>"


def parse_native_tool_calls(text: str) -> list[ToolCall]:
    """Extract ``<tool_call>{json}</tool_call>`` calls from model text, in
    order. A block whose body is not valid JSON (or lacks a name) yields a
    :data:`PARSE_ERROR_NAME` call so the harness feeds a format error back to
    the model instead of silently ignoring the attempt."""
    calls: list[ToolCall] = []
    for m in _TOOL_CALL_RE.finditer(text):
        body = m.group(1).strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            calls.append(ToolCall(name=PARSE_ERROR_NAME, arg=body))
            continue
        if not isinstance(obj, dict) or not obj.get("name"):
            calls.append(ToolCall(name=PARSE_ERROR_NAME, arg=body))
            continue
        calls.append(ToolCall(name=str(obj["name"]), arg=_extract_arg(obj)))
    return calls


def _extract_arg(obj: dict) -> str:
    """Flatten the ``arguments`` object to the single string our tools take.
    Accepts ``{"expression": "..."}`` (the declared schema), a bare string, or
    any single-value object; anything else is passed through as JSON so the
    tool's own error text tells the model what went wrong."""
    args = obj.get("arguments", "")
    if isinstance(args, str):
        # Some models double-encode: arguments = "{\"expression\": \"...\"}"
        s = args.strip()
        if s.startswith("{"):
            try:
                args = json.loads(s)
            except json.JSONDecodeError:
                return args
        else:
            return args
    if isinstance(args, dict):
        if "expression" in args:
            return str(args["expression"])
        if len(args) == 1:
            return str(next(iter(args.values())))
        return json.dumps(args)
    return str(args)


def format_tool_responses(results: list[ToolResult]) -> str:
    """Render tool results as the exact text the Qwen3 chat template emits for
    ``tool``-role messages, bracketed so it splices into an in-progress
    completion: close the assistant turn (the model stopped at
    ``</tool_call>``, before emitting its own ``<|im_end|>``), one user turn
    holding every ``<tool_response>``, then a fresh assistant turn. All of it
    is injected — the caller records it with loss mask 0."""
    responses = "".join(
        f"\n<tool_response>\n{r.output}\n</tool_response>" for r in results
    )
    return (
        f"{_IM_END}\n{_IM_START}user{responses}{_IM_END}\n{_IM_START}assistant\n"
    )


# --------------------------------------------------------------------------- #
# Function signatures for the chat template (render(tools=...))
# --------------------------------------------------------------------------- #
SOLVE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "solve",
        "description": (
            "Computer-algebra tool (SymPy). Solve an equation, a system, or an "
            "expression for a variable; evaluate arithmetic; or simplify "
            "(integrate/diff/factor run automatically). Examples: "
            "expression='x**2 - 5*x + 6, x' -> [2, 3]; "
            "expression='x**2 - 5*x + 6, x, max' -> 3; "
            "expression='2*x + 3 = 11' -> [4]; "
            "expression='x + y = 10, x - y = 2, x, y' -> {x: 6, y: 4}; "
            "expression='7*8 + 3' -> 59. " + SOLVE_TOOL_DOC
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The expression or equation(s), optionally followed by "
                        "the variable(s) to solve for and a selector "
                        "(max|min|real|first), comma-separated."
                    ),
                }
            },
            "required": ["expression"],
        },
    },
}

CALC_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calc",
        "description": (
            "Evaluate/simplify a single mathematical expression exactly "
            "(SymPy). Example: expression='(3/8)*(2/7)' -> 3/28."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to evaluate.",
                }
            },
            "required": ["expression"],
        },
    },
}

_SCHEMAS = {"solve": SOLVE_TOOL_SCHEMA, "calc": CALC_TOOL_SCHEMA}


def tool_schemas(names: list[str]) -> list[dict]:
    """Schemas for the named tools (unknown names are skipped — the config
    lists what the run enables; only CAS tools exist natively so far)."""
    return [_SCHEMAS[n] for n in names if n in _SCHEMAS]
