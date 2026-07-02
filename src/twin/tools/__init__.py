"""Tools exposed to the policy via a ReAct text protocol (DESIGN.md §9).

  * ``calc``   — SymPy evaluator (cheap, untaxed).
  * ``solve``  — SymPy computer-algebra solver (creator-only generation aid).
  * ``run_python`` / ``SandboxResult`` — sandboxed subprocess exec (also powers
    the code verifier).
  * ``OracleTool`` — base-only query, counted and taxed.
  * ``ToolHarness`` / ``parse_tool_calls`` — the text protocol + per-turn budget.
"""

from twin.tools.calc import calc, evaluate
from twin.tools.cas import SOLVE_TOOL_DOC, solve, split_top_level_commas
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
]
