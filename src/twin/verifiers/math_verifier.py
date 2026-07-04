"""Math verifier: SymPy exact/symbolic equality with a numeric fallback.

Used two ways (DESIGN.md §7):
  * **solver scoring** — compare the solver's answer to the creator's reference
    answer (``verify_math``);
  * **creator consistency** — check the creator's own answer satisfies a
    machine-checkable **certificate** the creator supplied (``check_predicate``:
    ``verification.check`` + ``verification.symbol``, Sprint 5). Accepted check
    forms: ``"2*x + 3 = 11"``, ``"4*x == 20"``, ``"Eq(x**2, 9)"``, a bare
    expression that must vanish, or several comma-separated relations for
    multi-unknown answers (``symbol: "x, y"``, ``check: "x + y = 10, x - y = 2"``).
    A check that merely restates the answer (``"x = 12"``) is rejected as
    trivial — it would certify anything.

All parsing goes through ``parse_expr`` with the locked ``SAFE_GLOBAL_DICT``
namespace (see ``twin.tools.cas``): SymPy's parser ultimately ``eval``s the
transformed source, so this is what keeps model-written text from reaching
builtins.
"""

import re

from sympy import Eq, Matrix, simplify
from sympy.core.containers import Tuple
from sympy.core.relational import Relational
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from twin.tools.cas import (
    SAFE_GLOBAL_DICT,
    normalize_eq_ops,
    split_top_level_commas,
    split_top_level_eq,
)
from twin.verifiers.result import VerificationResult

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_METHOD = "sympy"

# LaTeX the solver sometimes wraps answers in (e.g. "$\left(\frac{28}{11}, ...\right)$").
# We strip it to a plain expression so ordered pairs / fractions parse — see
# DESIGN.md §7 and the e2e finding that LaTeX-wrapped tuples never verified.
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")


def _strip_latex(s: str) -> str:
    s = _BOXED_RE.sub(r"(\1)", s)
    prev = None
    while prev != s:  # collapse possibly-repeated \frac{a}{b} -> ((a)/(b))
        prev = s
        s = _FRAC_RE.sub(r"((\1)/(\2))", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    for tok in ("\\,", "\\;", "\\:", "\\!", "\\quad", "\\qquad"):
        s = s.replace(tok, " ")
    s = s.replace("\\\\", " ").replace("\\ ", " ")
    return s.replace("{", "(").replace("}", ")")


def _normalize(s: str) -> str:
    s = _strip_latex((s or "").strip())
    # Take the right-hand side if the model wrote "answer = <value>".
    if s.count("=") == 1 and not any(op in s for op in ("==", "<=", ">=", "!=")):
        s = s.split("=", 1)[1].strip()
    for tok in ("$", "\\(", "\\)", "\\[", "\\]"):
        s = s.replace(tok, "")
    return s.strip()


def _parse(s: str):
    return parse_expr(
        _normalize(s), transformations=_TRANSFORMS,
        global_dict=SAFE_GLOBAL_DICT, evaluate=True,
    )


def _parse_raw(s: str):
    """Parse without ``_normalize``'s take-the-RHS-of-'=' step (which would eat
    an equation-form certificate). LaTeX still stripped; namespace locked."""
    return parse_expr(
        _strip_latex((s or "").strip()), transformations=_TRANSFORMS,
        global_dict=SAFE_GLOBAL_DICT, evaluate=True,
    )


def _is_seq(obj) -> bool:
    """Ordered-pair / vector answer: a list/tuple/Matrix, not a scalar."""
    return isinstance(obj, (tuple, list, Tuple, Matrix))


def _as_list(obj) -> list:
    return list(obj)


def _scalar_equal(c, e, tolerance: float) -> bool:
    """Symbolic equality, then a closed-form numeric compare within ``tolerance``."""
    try:
        if simplify(c - e) == 0:
            return True
    except (TypeError, ValueError):
        pass
    try:
        if not (c.free_symbols or e.free_symbols):
            return abs(float(c.evalf()) - float(e.evalf())) <= tolerance
    except (TypeError, ValueError, AttributeError):
        pass
    return False


_NAMED_PART = re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.+?)\s*$")


def _named_parts(s: str) -> dict[str, str] | None:
    """``"x = 2, y = 1"`` -> ``{"x": "2", "y": "1"}`` (insertion-ordered).

    ``None`` unless EVERY top-level comma part is ``name = expr`` with distinct
    names and there are at least two parts (a single ``name = value`` is
    already handled by ``_normalize``'s take-the-RHS step)."""
    parts = split_top_level_commas(s or "")
    if len(parts) < 2:
        return None
    named: dict[str, str] = {}
    for p in parts:
        m = _NAMED_PART.match(p)
        if not m or m.group(1) in named or "=" in m.group(2):
            return None
        named[m.group(1)] = m.group(2)
    return named


def _unify_named_values(cand: str, exp: str) -> tuple[str, str]:
    """Rewrite named-assignment multi-value answers into bare tuples so both
    sides parse and compare element-wise (audit 2026-07-03).

    ``_normalize_named_value`` landed only in ``check_predicate`` (mini-03),
    which made the two verification paths ASYMMETRIC: a creator answer
    ``"x = 2, y = 1"`` passed its own certificate, but a solver's correct
    ``"(2, 1)"`` graded WRONG against it — a live channel for farming fake
    hardness. When both sides are named, the candidate is re-ordered to the
    expected side's names (so ``"y = 1, x = 2"`` still matches); when only one
    side is named, it is rewritten in its own appearance order and compared to
    the other side's bare tuple. Non-named inputs pass through untouched."""
    cn, en = _named_parts(_strip_latex(cand)), _named_parts(_strip_latex(exp))
    if en and cn and sorted(cn) == sorted(en):
        cand = "(" + ", ".join(cn[k] for k in en) + ")"
        exp = "(" + ", ".join(en.values()) + ")"
    elif en and not cn:
        exp = "(" + ", ".join(en.values()) + ")"
    elif cn and not en:
        cand = "(" + ", ".join(cn.values()) + ")"
    return cand, exp


def verify_math(candidate: str, expected: str, *, tolerance: float = 1e-6) -> VerificationResult:
    """True iff ``candidate`` equals ``expected`` symbolically or numerically.

    Handles scalars, ordered pairs / vectors (compared element-wise, so
    ``(3, 2)``, ``[3, 2]`` and ``$\\left(\\tfrac31,\\tfrac21\\right)$`` all unify),
    named-assignment forms (``"x = 2, y = 1"`` on EITHER side — see
    ``_unify_named_values``), and degrades to a normalized string compare when
    neither side parses (e.g. the answer is a word, not an expression)."""
    cand_raw, exp_raw = _unify_named_values(
        (candidate or "").strip(), (expected or "").strip())

    # Cheap exact match on the normalized text — also the safety net for
    # sequence answers whose elements are byte-identical (e.g. "(3, 2)").
    if _normalize(cand_raw) and _normalize(cand_raw) == _normalize(exp_raw):
        return VerificationResult.ok(_METHOD, f"exact match: {cand_raw!r}")

    try:
        c, e = _parse(cand_raw), _parse(exp_raw)
    except Exception:  # noqa: BLE001 - not parseable as math; degrade to text
        same = _normalize(cand_raw).lower() == _normalize(exp_raw).lower()
        return _binary(same, f"string compare: {cand_raw!r} vs {exp_raw!r}")

    # Ordered pairs / vectors: same length and element-wise equal.
    if _is_seq(c) or _is_seq(e):
        if not (_is_seq(c) and _is_seq(e)):
            return VerificationResult.fail(_METHOD, f"shape mismatch: {c} vs {e}")
        cl, el = _as_list(c), _as_list(e)
        if len(cl) != len(el):
            return VerificationResult.fail(_METHOD, f"length {len(cl)} != {len(el)}")
        if all(_scalar_equal(ci, ei, tolerance) for ci, ei in zip(cl, el)):
            return VerificationResult.ok(_METHOD, f"elementwise match: {c} == {e}")
        return VerificationResult.fail(_METHOD, f"elementwise mismatch: {c} != {e}")

    # Scalar.
    if _scalar_equal(c, e, tolerance):
        return VerificationResult.ok(_METHOD, f"match: {c} == {e}")
    return VerificationResult.fail(_METHOD, f"no match: {c} != {e}")


# A bare numeric literal (int / decimal / simple fraction, optional sign). Used
# to reject trivial certificates like "x = 12" or "x = 7/2" — a check that just
# restates the answer certifies anything, so the creator must recompute the
# answer from the problem's quantities ("4*x = 20", "x = 0.15*80", ...).
_NUM_LITERAL = re.compile(
    r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?(\s*/\s*\d+(\.\d*)?)?$"
)
_EQ_CALL = re.compile(r"^\s*Eq\s*\((.*)\)\s*$", re.DOTALL)


def _check_sides(part: str) -> tuple[str, str] | None:
    """The (lhs, rhs) of one certificate part, for ``a = b`` / ``a == b`` /
    ``Eq(a, b)`` forms; ``None`` for bare expressions / inequalities."""
    eq = split_top_level_eq(normalize_eq_ops(part))
    if eq is not None:
        return eq[0].strip(), eq[1].strip()
    m = _EQ_CALL.match(part)
    if m:
        args = split_top_level_commas(m.group(1))
        if len(args) == 2:
            return args[0].strip(), args[1].strip()
    return None


def _is_trivial_part(part: str, symbol_names: set[str]) -> bool:
    """True for ``<symbol> = <numeric literal>`` (either order) — a
    self-certifying check that must be rejected."""
    sides = _check_sides(part)
    if sides is None:
        return False
    for a, b in (sides, sides[::-1]):
        if a in symbol_names and _NUM_LITERAL.match(b):
            return True
    return False


def _parse_relation(part: str):
    """Parse one certificate part into an ``Eq``/``Relational``/bare expression."""
    sides = _check_sides(part)
    if sides is not None:
        return Eq(_parse_raw(sides[0]), _parse_raw(sides[1]), evaluate=False)
    return _parse_raw(part)


def _vanishes(expr, tolerance: float) -> bool:
    """``expr`` is 0, exactly or numerically within ``tolerance``."""
    s = simplify(expr)
    if s == 0:
        return True
    if not s.free_symbols:
        try:
            return abs(complex(s.evalf())) <= tolerance
        except (TypeError, ValueError):
            return False
    return False


_NAMED_VALUE = re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.+?)\s*$")


def _normalize_named_value(value: str, sym_parts: list[str]) -> str:
    """Rewrite a ``"x = 2, y = 1"``-style answer into the declared symbols'
    order (``"(2, 1)"``). Creators frequently state multi-unknown answers as
    named assignments instead of the spec'd bare tuple, which voided *correct*
    problems in mini-03 (the '=' broke the predicate parse). Only fires when
    every comma-separated part is ``name = expr`` and the names are exactly
    the declared symbols; anything else is returned untouched."""
    parts = split_top_level_commas(value or "")
    named: dict[str, str] = {}
    for p in parts:
        m = _NAMED_VALUE.match(p)
        if not m:
            return value
        named[m.group(1)] = m.group(2)
    names = [s.strip() for s in sym_parts]
    if sorted(named) != sorted(names):
        return value
    ordered = [named[n] for n in names]
    return ordered[0] if len(ordered) == 1 else "(" + ", ".join(ordered) + ")"


def check_predicate(check: str, symbol: str, value: str, *, tolerance: float = 1e-6) -> VerificationResult:
    """Substitute ``value`` for ``symbol`` in ``check`` and test it holds.

    ``check`` is the creator's certificate (see module docstring for the
    accepted forms); ``symbol`` may name several unknowns (``"x, y"``) with
    ``value`` a matching tuple. Every comma-separated relation in ``check``
    must hold. Lets math problems be verified *without* an independent solver
    or an LLM judge."""
    check = (check or "").strip()
    symbol = (symbol or "").strip()
    if not check or not symbol:
        return VerificationResult.fail(_METHOD, "empty predicate/symbol")

    sym_parts = split_top_level_commas(symbol)
    try:
        syms = [_parse_raw(s) for s in sym_parts]
        if not all(getattr(s, "is_Symbol", False) for s in syms):
            return VerificationResult.fail(_METHOD, f"bad symbol spec: {symbol!r}")
        val = _parse(_normalize_named_value(value, sym_parts))
        vals = list(val) if (len(syms) > 1 and _is_seq(val)) else [val]
        if len(vals) != len(syms):
            return VerificationResult.fail(
                _METHOD, f"{len(syms)} symbols but {len(vals)} values")
        parts = split_top_level_commas(normalize_eq_ops(check))
        if any(_is_trivial_part(p, set(sym_parts)) for p in parts):
            return VerificationResult.fail(
                _METHOD,
                "trivial certificate (restates the answer); the check must "
                "recompute the answer from the problem's quantities",
            )
        relations = [_parse_relation(p) for p in parts]
    except Exception as e:  # noqa: BLE001
        return VerificationResult.fail(_METHOD, f"predicate parse error: {e}")

    pairs = list(zip(syms, vals))
    try:
        for rel in relations:
            if isinstance(rel, Eq):
                # Compare via lhs-rhs so `tolerance` applies (Eq.subs would
                # auto-evaluate two numbers to a hard boolean).
                holds = _vanishes((rel.lhs - rel.rhs).subs(pairs), tolerance)
            elif isinstance(rel, Relational):
                holds = bool(simplify(rel.subs(pairs)) == True)  # noqa: E712 - sympy truth
            else:  # bare expression: must vanish at the answer
                holds = _vanishes(rel.subs(pairs), tolerance)
            if not holds:
                return VerificationResult.fail(
                    _METHOD, f"predicate {rel} fails at {dict(pairs)}")
        return VerificationResult.ok(
            _METHOD, f"predicate holds at {dict(pairs)}: {check}")
    except (TypeError, ValueError, AttributeError) as e:
        return VerificationResult.fail(_METHOD, f"predicate eval error: {e}")


def _binary(passed: bool, detail: str) -> VerificationResult:
    return VerificationResult.ok(_METHOD, detail) if passed else VerificationResult.fail(_METHOD, detail)
