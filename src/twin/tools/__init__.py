"""Tools exposed to the policy via a ReAct text protocol (DESIGN.md §9).

  * ``calc``   — SymPy evaluator (cheap, untaxed).
  * ``run_python`` / ``SandboxResult`` — sandboxed subprocess exec (also powers
    the code verifier).
  * ``OracleTool`` — base-only query, counted and taxed.
  * ``ToolHarness`` / ``parse_tool_calls`` — the text protocol + per-turn budget.
"""

from twin.tools.calc import calc, evaluate
from twin.tools.oracle import OracleTool
from twin.tools.protocol import (
    ToolCall,
    ToolHarness,
    ToolResult,
    parse_tool_calls,
)
from twin.tools.sandbox import SandboxResult, run_python

__all__ = [
    "calc",
    "evaluate",
    "run_python",
    "SandboxResult",
    "OracleTool",
    "ToolHarness",
    "ToolCall",
    "ToolResult",
    "parse_tool_calls",
]
