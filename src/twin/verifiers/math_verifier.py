"""Math verifier: SymPy exact/symbolic equality with a numeric fallback.

Used two ways (DESIGN.md §7):
  * **solver scoring** — compare the solver's answer to the creator's reference
    answer (``verify_math``);
  * **creator consistency** — check the creator's own answer satisfies a
    machine-checkable predicate the creator supplied (``check_predicate``).

Both parse model text with ``parse_expr`` (not ``eval``/``sympify``) so no
arbitrary Python executes here.
"""

from sympy import Eq, simplify
from sympy.core.relational import Relational
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from twin.verifiers.result import VerificationResult

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_METHOD = "sympy"


def _normalize(s: str) -> str:
    s = (s or "").strip()
    # Take the right-hand side if the model wrote "answer = <value>".
    if s.count("=") == 1 and not any(op in s for op in ("==", "<=", ">=", "!=")):
        s = s.split("=", 1)[1].strip()
    for tok in ("$", "\\(", "\\)", "\\[", "\\]"):
        s = s.replace(tok, "")
    return s.strip()


def _parse(s: str):
    return parse_expr(_normalize(s), transformations=_TRANSFORMS, evaluate=True)


def verify_math(candidate: str, expected: str, *, tolerance: float = 1e-6) -> VerificationResult:
    """True iff ``candidate`` equals ``expected`` symbolically or numerically.

    Falls back to a normalized string compare when neither side parses (e.g. the
    answer is a word, not an expression)."""
    cand_raw, exp_raw = (candidate or "").strip(), (expected or "").strip()
    try:
        c, e = _parse(cand_raw), _parse(exp_raw)
    except Exception:  # noqa: BLE001 - not parseable as math; degrade to text
        same = _normalize(cand_raw).lower() == _normalize(exp_raw).lower()
        return _binary(same, f"string compare: {cand_raw!r} vs {exp_raw!r}")

    # Symbolic equality.
    try:
        if simplify(c - e) == 0:
            return VerificationResult.ok(_METHOD, f"symbolic match: {c} == {e}")
    except (TypeError, ValueError):
        pass
    # Numeric fallback (closed-form only).
    try:
        if not (c.free_symbols or e.free_symbols):
            cn, en = float(c.evalf()), float(e.evalf())
            if abs(cn - en) <= tolerance:
                return VerificationResult.ok(_METHOD, f"numeric match: {cn} ~= {en}")
            return VerificationResult.fail(_METHOD, f"numeric mismatch: {cn} != {en}")
    except (TypeError, ValueError):
        pass
    return VerificationResult.fail(_METHOD, f"no match: {c} != {e}")


def check_predicate(check: str, symbol: str, value: str, *, tolerance: float = 1e-6) -> VerificationResult:
    """Substitute ``value`` for ``symbol`` in ``check`` and test it holds.

    ``check`` is a creator-supplied certificate: either a relational
    (``Eq(x**2-5*x+6, 0)``) or a bare expression that should evaluate to 0 when
    the answer is correct. Lets math problems be verified *without* an
    independent solver or an LLM judge."""
    try:
        sym = parse_expr(symbol, transformations=_TRANSFORMS)
        val = _parse(value)
        expr = parse_expr(check, transformations=_TRANSFORMS, evaluate=True)
    except Exception as e:  # noqa: BLE001
        return VerificationResult.fail(_METHOD, f"predicate parse error: {e}")

    substituted = expr.subs(sym, val)
    try:
        if isinstance(substituted, (Relational, Eq)) or isinstance(expr, (Relational, Eq)):
            holds = bool(simplify(substituted) == True)  # noqa: E712 - sympy truth
            return _binary(holds, f"predicate {expr} with {sym}={val} -> {holds}")
        # Bare expression: correct when it simplifies to ~0.
        s = simplify(substituted)
        if s == 0:
            return VerificationResult.ok(_METHOD, f"predicate {expr}=0 with {sym}={val}")
        if not s.free_symbols:
            close = abs(float(s.evalf())) <= tolerance
            return _binary(close, f"predicate {expr}={float(s.evalf())} with {sym}={val}")
        return VerificationResult.fail(_METHOD, f"predicate did not vanish: {s}")
    except (TypeError, ValueError) as e:
        return VerificationResult.fail(_METHOD, f"predicate eval error: {e}")


def _binary(passed: bool, detail: str) -> VerificationResult:
    return VerificationResult.ok(_METHOD, detail) if passed else VerificationResult.fail(_METHOD, detail)
