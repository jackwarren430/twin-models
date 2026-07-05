"""Verifier dispatch: pick a method per problem and run it (DESIGN.md §7).

Two entry points:
  * :func:`verify_answer` — score a **solver** attempt against the creator's
    reference answer / hidden tests.
  * :func:`check_consistency` — score whether the **creator's own** problem is
    self-consistent (its reference solution holds up).

Method selection: an explicit ``problem.verification['type']`` wins; otherwise we
default by domain (math -> sympy, coding -> exec, anything else -> judge).
Verifier-first, judge only as the documented fallback.
"""

from twin.problems.schema import Problem
from twin.verifiers.code_verifier import verify_code
from twin.verifiers.judge import judge_answer, judge_consistency
from twin.verifiers.logic_verifier import check_logic_consistency, verify_logic
from twin.verifiers.math_verifier import check_predicate, verify_math
from twin.verifiers.result import VerificationResult

_MATH_DOMAINS = {"math", "arithmetic", "algebra"}
_CODE_DOMAINS = {"coding", "code", "python"}
_LOGIC_DOMAINS = {"logic", "puzzle", "puzzles", "knights-and-knaves"}


def _method_for(problem: Problem) -> str:
    vtype = str(problem.verification.get("type", "")).strip().lower()
    if vtype:
        if vtype.startswith("math"):
            return "math"
        if vtype.startswith(("code", "exec", "python")):
            return "code"
        if vtype.startswith("logic"):
            return "logic"
        if vtype.startswith("judge"):
            return "judge"
    if problem.domain.lower() in _CODE_DOMAINS:
        return "code"
    if problem.domain.lower() in _MATH_DOMAINS:
        return "math"
    if problem.domain.lower() in _LOGIC_DOMAINS:
        return "logic"
    return "judge"


def _sandbox_opts(problem: Problem) -> dict:
    v = problem.verification
    return {
        k: v[k]
        for k in ("timeout_s", "cpu_s", "mem_mb")
        if k in v
    }


def verify_answer(problem: Problem, candidate: str, *, oracle=None) -> VerificationResult:
    """Grade a solver's ``candidate`` answer against the problem's ground truth."""
    method = _method_for(problem)
    if method == "code":
        tests = problem.verification.get("tests", "")
        setup = problem.verification.get("setup", "")
        return verify_code(candidate, tests, setup=setup, **_sandbox_opts(problem))
    if method == "math":
        tol = float(problem.verification.get("tolerance", 1e-6))
        return verify_math(candidate, problem.answer, tolerance=tol)
    if method == "logic":
        return verify_logic(candidate, str(problem.verification.get("claims", "")))
    return judge_answer(problem.statement, candidate, problem.answer, oracle)


def check_consistency(problem: Problem, *, oracle=None) -> VerificationResult:
    """Does the creator's own solution/answer hold up under the verifier?"""
    method = _method_for(problem)
    if method == "code":
        code = (
            problem.verification.get("solution_code")
            or problem.solution
            or problem.answer
        )
        tests = problem.verification.get("tests", "")
        setup = problem.verification.get("setup", "")
        return verify_code(code, tests, setup=setup, **_sandbox_opts(problem))

    if method == "math":
        # Deterministic certificate first (no model needed)...
        check = problem.verification.get("check")
        symbol = problem.verification.get("symbol")
        if check and symbol:
            tol = float(problem.verification.get("tolerance", 1e-6))
            return check_predicate(check, symbol, problem.answer, tolerance=tol)
        # ...else fall back to the LLM judge on the worked solution.
        return judge_consistency(problem.statement, problem.answer, problem.solution, oracle)

    if method == "logic":
        return check_logic_consistency(
            str(problem.verification.get("claims", "")), problem.answer)

    return judge_consistency(problem.statement, problem.answer, problem.solution, oracle)
