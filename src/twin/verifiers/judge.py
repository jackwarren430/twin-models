"""LLM-judge fallback for non-executable problems (DESIGN.md §7).

The judge is the frozen **base/oracle** (zeroed adapter) — no extra weights and
no conflict of interest with the trained policy (§12 Q3). It is used only when a
deterministic verifier can't apply (free-form logic/reasoning answers).

``oracle`` here is any ``Callable[[str], str]`` — in the loop it's an
:class:`~twin.tools.oracle.OracleTool` bound to the base; in tests it's a stub.
The grading instruction lives in the prompt, and we parse a ``VERDICT:`` line so
the signal degrades gracefully if the base rambles.
"""

import re
from typing import Callable

from twin.verifiers.result import VerificationResult

_METHOD = "judge"
_VERDICT = re.compile(r"VERDICT\s*[:=]\s*(CORRECT|INCORRECT)", re.IGNORECASE)

_GRADE_PROMPT = """You are grading a candidate answer to a problem. Decide whether \
the candidate answer is correct.

Problem:
{statement}

Reference answer (ground truth):
{reference}

Candidate answer:
{candidate}

Think briefly, then end your reply with exactly one line:
VERDICT: CORRECT   or   VERDICT: INCORRECT"""

_CONSISTENCY_PROMPT = """You are checking whether a problem's stated solution is \
internally correct — does the worked solution actually solve the problem and \
yield the stated answer?

Problem:
{statement}

Stated answer:
{answer}

Worked solution:
{solution}

Think briefly, then end your reply with exactly one line:
VERDICT: CORRECT   or   VERDICT: INCORRECT"""


def _parse_verdict(text: str) -> bool | None:
    matches = _VERDICT.findall(text or "")
    if matches:
        return matches[-1].upper() == "CORRECT"
    # Loose fallback: trailing yes/no.
    tail = (text or "").strip().lower()[-40:]
    if "incorrect" in tail or re.search(r"\bno\b", tail):
        return False
    if "correct" in tail or re.search(r"\byes\b", tail):
        return True
    return None


def judge_answer(statement: str, candidate: str, reference: str,
                 oracle: Callable[[str], str]) -> VerificationResult:
    """Grade a solver's free-form answer against the reference via the oracle."""
    prompt = _GRADE_PROMPT.format(statement=statement, reference=reference, candidate=candidate)
    return _run(prompt, oracle, what="answer")


def judge_consistency(statement: str, answer: str, solution: str,
                      oracle: Callable[[str], str]) -> VerificationResult:
    """Grade whether the creator's own solution/answer is self-consistent."""
    prompt = _CONSISTENCY_PROMPT.format(statement=statement, answer=answer, solution=solution)
    return _run(prompt, oracle, what="consistency")


def _run(prompt: str, oracle: Callable[[str], str], *, what: str) -> VerificationResult:
    if oracle is None:
        return VerificationResult(False, 0.0, f"no judge available for {what}", "unverifiable")
    try:
        reply = oracle(prompt)
    except Exception as e:  # noqa: BLE001
        return VerificationResult(False, 0.0, f"judge error: {e}", _METHOD)
    verdict = _parse_verdict(reply)
    if verdict is None:
        return VerificationResult(False, 0.0, f"judge gave no clear verdict: {reply[:120]!r}", _METHOD)
    return (VerificationResult.ok if verdict else VerificationResult.fail)(_METHOD, f"judge: {what}")
