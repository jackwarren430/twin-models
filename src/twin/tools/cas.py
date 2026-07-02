"""``solve`` tool: a SymPy-backed computer-algebra ("Wolfram-Alpha-like") solver.

Creator-only, untaxed generation aid (DESIGN.md §9). It lets the creator compute
exact answers *while building a suite*, so it can one-shot correct
problem+answer pairs instead of hallucinating answers (which fail the
consistency check and become void problems). This is distinct from the
verifier/judge, which only *checks* a proposed answer.

Like ``calc`` it parses with ``parse_expr`` (never ``eval``/``sympify`` of raw
input) and returns text — it must never throw into the rollout. Dispatch:

  * **Solve** when a variable is named or an ``=`` is present::

        solve(x**2 - 5*x + 6, x)   -> [2, 3]
        solve(x**2 - 5*x + 6, x, max) -> 3
        solve(2*x + 3 = 11)        -> [4]
        solve(x + y = 10, x - y = 2, x, y) -> {x: 6, y: 4}

    A trailing selector ``max``/``min``/``real``/``all``/``first`` narrows a
    scalar solution list.
  * **Evaluate / simplify** a single expression otherwise::

        solve(2*3 + 4)             -> 10
        solve(integrate(x**2, x))  -> x**3/3   (unevaluated Integral -> .doit())
"""

import re
import threading

import sympy as _sympy
from sympy import Eq, Float
from sympy import solve as sympy_solve
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_SELECTORS = {"max", "min", "real", "all", "first"}
_SYMBOL_RE = re.compile(r"^[A-Za-z_]\w*$")

# Locked-down eval namespace for parse_expr. SymPy's parser ultimately ``eval``s
# the transformed source, so a bare ``parse_expr`` will happily run e.g.
# ``__import__("os").system(...)`` (auto_symbol skips dunder names, and Python
# injects ``__builtins__`` when globals lack it). We pass an explicit global dict
# that exposes only the SymPy namespace (sin, integrate, factor, Symbol, ...)
# with ``__builtins__`` emptied, so undefined names become Symbols and builtins
# are unreachable.
_GLOBAL_DICT = {k: getattr(_sympy, k) for k in dir(_sympy) if not k.startswith("_")}
_GLOBAL_DICT["__builtins__"] = {}

# Public alias: the one locked namespace every parse of model text should use.
# `calc` and `twin.verifiers.math_verifier` import this so no parser in the
# project evals model text with builtins reachable.
SAFE_GLOBAL_DICT = _GLOBAL_DICT


# --------------------------------------------------------------------------- #
# Argument parsing (quote/paren-aware, mirrors tools/protocol._balanced_paren_arg)
# --------------------------------------------------------------------------- #
def split_top_level_commas(s: str) -> list[str]:
    """Split ``s`` on commas that are not nested in parens/brackets/braces or a
    string literal. ``"f(x, y), z"`` -> ``["f(x, y)", "z"]``."""
    parts: list[str] = []
    depth = 0
    in_str = None  # quote char or None
    esc = False
    start = 0
    for j, c in enumerate(s):
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
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        elif c == "," and depth == 0:
            parts.append(s[start:j])
            start = j + 1
    parts.append(s[start:])
    return [p.strip() for p in parts if p.strip()]


def split_top_level_eq(p: str):
    """If ``p`` has a top-level single ``=`` (not ``==``/``<=``/``>=``/``!=``),
    return ``(lhs, rhs)``; otherwise ``None``."""
    depth = 0
    in_str = None
    esc = False
    for j, c in enumerate(p):
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
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        elif c == "=" and depth == 0:
            # skip relational operators (==, <=, >=, !=)
            prev = p[j - 1] if j > 0 else ""
            nxt = p[j + 1] if j + 1 < len(p) else ""
            if nxt == "=" or prev in "<>=!":
                continue
            return p[:j], p[j + 1 :]
    return None


def _is_symbol(p: str) -> bool:
    return bool(_SYMBOL_RE.match(p))


def normalize_eq_ops(s: str) -> str:
    """Collapse Python-style ``==`` equality into the single ``=`` the solver
    treats as an equation, outside of string literals. Models reach for ``==``
    (``4*x == 20``); the bare parser reads that as a boolean and silently solves
    nothing (``[]``), which once made the judge reject a correct problem. Leaves
    ``<=`` / ``>=`` / ``!=`` (single ``=``) untouched."""
    out: list[str] = []
    in_str = None
    esc = False
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if in_str is not None:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == in_str:
                in_str = None
            i += 1
            continue
        if c in ("'", '"'):
            in_str = c
            out.append(c)
            i += 1
            continue
        if c == "=" and i + 1 < n and s[i + 1] == "=" and (i == 0 or s[i - 1] not in "<>!="):
            out.append("=")
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _parse(s: str):
    return parse_expr(
        s, transformations=_TRANSFORMS, global_dict=_GLOBAL_DICT, evaluate=True
    )


def _to_relation_or_expr(part: str):
    """Parse a part into an ``Eq`` (if it has a top-level ``=``), an already-built
    ``Relational``, or a bare expression (meaning ``expr = 0`` for solving)."""
    eq = split_top_level_eq(part)
    if eq is not None:
        return Eq(_parse(eq[0]), _parse(eq[1]))
    return _parse(part)


# --------------------------------------------------------------------------- #
# Result selection / formatting
# --------------------------------------------------------------------------- #
def _select(sols, selector: str | None):
    """Apply a selector to a flat list of scalar solutions. Selectors are no-ops
    for structured solutions (systems -> dict/tuples)."""
    if not isinstance(sols, (list, tuple)):
        return sols
    if selector in (None, "all"):
        return list(sols)
    if selector == "first":
        return sols[0] if sols else sols
    if selector == "real":
        return [s for s in sols if getattr(s, "is_real", None) is not False]
    if selector in ("max", "min"):
        reals = [s for s in sols if getattr(s, "is_real", None) is not False]
        pool = reals or list(sols)
        try:
            return max(pool) if selector == "max" else min(pool)
        except TypeError:
            return list(sols)  # not orderable (complex/symbolic) -> return all
    return list(sols)


def _fmt_scalar(x) -> str:
    """Render a solution/value. Symbolic forms are returned as-is (so ``factor``/
    ``integrate`` output survives); exact constants stay exact; explicit decimals
    are evaluated."""
    if getattr(x, "free_symbols", set()):
        return str(x)
    if getattr(x, "is_Integer", False) or getattr(x, "is_Rational", False):
        return str(x)
    try:
        if x.has(Float):
            return str(x.evalf())
    except Exception:  # noqa: BLE001 - non-sympy result; fall through to str()
        pass
    return str(x)


def _format(result, selector: str | None) -> str:
    result = _select(result, selector)
    if isinstance(result, dict):
        return "{" + ", ".join(f"{k}: {_fmt_scalar(v)}" for k, v in result.items()) + "}"
    if isinstance(result, (list, tuple)):
        return "[" + ", ".join(_fmt_scalar(v) for v in result) + "]"
    return _fmt_scalar(result)


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def _solve_impl(arg: str) -> str:
    parts = split_top_level_commas(normalize_eq_ops(arg))
    if not parts:
        return "error: empty expression"

    selector: str | None = None
    syms = []
    expr_parts: list[str] = []
    for i, p in enumerate(parts):
        low = p.lower()
        if low in _SELECTORS:
            selector = low
        elif i > 0 and _is_symbol(p):
            syms.append(_parse(p))
        else:
            expr_parts.append(p)

    if not expr_parts:
        return "error: nothing to solve"

    has_eq = any(split_top_level_eq(p) is not None for p in expr_parts)
    solve_intent = bool(syms) or has_eq

    if solve_intent:
        relations = [_to_relation_or_expr(p) for p in expr_parts]
        # Unwrap singletons: solve(expr, x) yields a flat [2, 3], whereas
        # solve([expr], [x]) yields [(2,), (3,)] — we want the flat form.
        rel_arg = relations[0] if len(relations) == 1 else relations
        if syms:
            tgt = syms[0] if len(syms) == 1 else syms
            sols = sympy_solve(rel_arg, tgt)
        else:
            sols = sympy_solve(rel_arg)  # infer free symbols
        return _format(sols, selector)

    # Evaluate / simplify a single expression (calculus via .doit()).
    expr = _parse(expr_parts[0])
    doit = getattr(expr, "doit", None)
    if callable(doit):
        expr = expr.doit()
    return _fmt_scalar(expr)


def solve(arg: str, *, timeout_s: float | None = 3.0) -> str:
    """Solve / evaluate ``arg`` and return a string result. Errors (including a
    best-effort ``timeout_s`` guard) are returned as ``"error: ..."`` text — the
    tool contract is text-in/text-out and must not throw into a rollout.

    The timeout runs the work on a daemon thread and gives up waiting after
    ``timeout_s``; SymPy is CPU-bound so this bounds *waiting*, not the worker
    itself (the simple math themes don't hit pathological solves)."""
    arg = (arg or "").strip()
    if not arg:
        return "error: empty expression"

    box: dict[str, str] = {}

    def _run():
        try:
            box["out"] = _solve_impl(arg)
        except Exception as e:  # noqa: BLE001 - tool boundary, surface as text
            box["out"] = f"error: {type(e).__name__}: {e}"

    if not timeout_s or timeout_s <= 0:
        _run()
        return box["out"]

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return f"error: timed out after {timeout_s}s"
    return box.get("out", "error: no result")


# Documentation reused by the creator prompt and tests (single source of truth).
SOLVE_TOOL_DOC = (
    "solve(expr_or_equation[, variable[, max|min|real]]) — a computer-algebra "
    "tool. Solve an equation or expression for a variable, evaluate arithmetic, "
    "or simplify. Examples: solve(x**2 - 5*x + 6, x) -> [2, 3]; "
    "solve(x**2 - 5*x + 6, x, max) -> 3; solve(2*x + 3 = 11) -> [4]; "
    "solve(7*8 + 3) -> 59."
)
