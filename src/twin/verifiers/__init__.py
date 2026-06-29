"""Verifiers + consistency (DESIGN.md §7).

  * :class:`VerificationResult` — the common result type.
  * deterministic: :func:`verify_math` / :func:`check_predicate` (sympy),
    :func:`verify_code` (sandboxed exec).
  * fallback: :func:`judge_answer` / :func:`judge_consistency` (oracle/base).
  * dispatch: :func:`verify_answer` (solver scoring) and
    :func:`check_consistency` (creator scoring).
"""

from twin.verifiers.code_verifier import verify_code
from twin.verifiers.dispatch import check_consistency, verify_answer
from twin.verifiers.judge import judge_answer, judge_consistency
from twin.verifiers.math_verifier import check_predicate, verify_math
from twin.verifiers.result import VerificationResult

__all__ = [
    "VerificationResult",
    "verify_math",
    "check_predicate",
    "verify_code",
    "judge_answer",
    "judge_consistency",
    "verify_answer",
    "check_consistency",
]
