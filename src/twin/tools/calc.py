"""``calc`` tool: a cheap, untaxed SymPy evaluator.

Exists so the policy has a deterministic way to do arithmetic/algebra without
reaching for the (taxed) oracle (DESIGN.md §9). Parsing uses
``sympy.parse_expr`` rather than ``sympify``/``eval`` so a model can't smuggle
arbitrary Python through it.
"""

from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def evaluate(expr: str):
    """Parse and evaluate ``expr``, returning a SymPy object. Raises on bad input."""
    return parse_expr(expr, transformations=_TRANSFORMS, evaluate=True)


def calc(expr: str) -> str:
    """Evaluate a mathematical expression and return a string result.

    Returns a numeric value when the expression is closed-form, otherwise the
    simplified symbolic form. Errors are returned as ``"error: ..."`` text (the
    tool contract is text-in/text-out; it must not throw into the rollout)."""
    expr = (expr or "").strip()
    if not expr:
        return "error: empty expression"
    try:
        result = evaluate(expr)
    except Exception as e:  # noqa: BLE001 - tool boundary, surface as text
        return f"error: {type(e).__name__}: {e}"
    # Symbolic result (has variables): return the simplified form.
    if getattr(result, "free_symbols", set()):
        return str(result)
    # Exact rational/integer: keep it exact; otherwise give a decimal value.
    if getattr(result, "is_Integer", False) or getattr(result, "is_Rational", False):
        return str(result)
    try:
        return str(result.evalf())
    except Exception:  # noqa: BLE001
        return str(result)
