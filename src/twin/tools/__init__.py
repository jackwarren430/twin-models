"""Tools exposed to the policy (DESIGN.md §9).

  * ``calc``   — SymPy evaluator (cheap, untaxed).
  * ``solve``  — SymPy computer-algebra solver (creator-only generation aid).
  * ``run_python`` / ``SandboxResult`` — sandboxed subprocess exec (also powers
    the code verifier).
  * ``OracleTool`` — base-only query, counted and taxed.
  * ``ToolHarness`` / ``parse_tool_calls`` — the legacy ReAct text protocol
    (``<tool>name(arg)</tool>``) + per-turn budget.
  * ``twin.tools.native`` — Qwen3 native function calling (``<tool_call>``
    JSON blocks + ``<tool_response>`` splice; Sprint 7): the protocol the
    model was actually trained on.
"""

from twin.tools.calc import calc, evaluate
from twin.tools.cas import SOLVE_TOOL_DOC, solve, split_top_level_commas
from twin.tools.native import (
    NATIVE_STOP,
    format_tool_responses,
    parse_native_tool_calls,
    tool_schemas,
)
from twin.tools.oracle import OracleTool
from twin.tools.protocol import (
    PARSE_ERROR_NAME,
    ToolCall,
    ToolHarness,
    ToolResult,
    parse_tool_calls,
)
from twin.tools.sandbox import SandboxResult, run_python

__all__ = [
    "calc",
    "evaluate",
    "solve",
    "split_top_level_commas",
    "SOLVE_TOOL_DOC",
    "run_python",
    "SandboxResult",
    "OracleTool",
    "ToolHarness",
    "ToolCall",
    "ToolResult",
    "parse_tool_calls",
    "PARSE_ERROR_NAME",
    "NATIVE_STOP",
    "parse_native_tool_calls",
    "format_tool_responses",
    "tool_schemas",
]
