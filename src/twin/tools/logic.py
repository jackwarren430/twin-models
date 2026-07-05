"""``logic_solve`` — the creator's authoring tool for Knights & Knaves.

The logic-domain analog of ``solve`` (math) and ``run_python`` (coding): the
creator drafts a claims string, the tool enumerates every satisfying
knight/knave assignment, and the response says whether the puzzle is
well-posed (exactly one solution) and what that solution is. The creator's
answer is therefore correct BY CONSTRUCTION — it reads the assignment off
the tool response instead of solving its own puzzle — which removes the
self-consistency risk mini-04b measured as the binding constraint.
"""

from twin.verifiers.logic_verifier import (
    ClaimParseError,
    format_assignment,
    solve_claims,
)

_MAX_SHOWN = 4

LOGIC_SOLVE_TOOL_DOC = (
    "Input: the full claims string, e.g. 'A: B & ~C; B: ~A; C: A | B' "
    "(one 'Speaker: statement' per claim, ';'-separated; a bare name means "
    "'that person is a knight'; operators ~ & | ^ -> and parentheses)."
)


def logic_solve(claims: str) -> str:
    """Enumerate knight/knave assignments for a claims string. Returns a
    one-line verdict the creator can act on directly: the unique solution,
    or why the puzzle is not yet well-posed."""
    try:
        people, solutions = solve_claims(claims)
    except ClaimParseError as e:
        return f"PARSE ERROR: {e}"
    if not solutions:
        return (
            f"CONTRADICTORY: no knight/knave assignment over "
            f"{', '.join(people)} satisfies these claims — relax one."
        )
    if len(solutions) == 1:
        return f"UNIQUE solution: {format_assignment(solutions[0])}"
    shown = "; ".join(format_assignment(s) for s in solutions[:_MAX_SHOWN])
    more = "" if len(solutions) <= _MAX_SHOWN else f" (+{len(solutions) - _MAX_SHOWN} more)"
    return (
        f"AMBIGUOUS: {len(solutions)} satisfying assignments — add or "
        f"sharpen claims until exactly one remains. Found: {shown}{more}"
    )
