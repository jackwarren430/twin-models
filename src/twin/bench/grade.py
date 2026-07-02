"""Deterministic grading of a model response to a benchmark item.

One entry point, :func:`grade`, dispatches on ``item.vtype`` and returns the
project's standard :class:`~twin.verifiers.result.VerificationResult`. The
deterministic verifiers are reused verbatim so the benchmark grades answers the
exact same way training does:

* ``math_numeric`` -> :func:`twin.verifiers.verify_math` (SymPy)
* ``code``         -> :func:`twin.verifiers.verify_code` (sandboxed exec)

and two benchmark-local graders cover knowledge/reasoning:

* ``mcq``   -> extract the chosen letter, compare to the reference letter
* ``exact`` -> normalized whole-word match against ``answer`` + ``aliases``

Everything here is model-free string work (plus the existing verifiers), so it
is fully unit-testable. Reasoning model output is handled first by stripping any
``<think>...</think>`` block so extraction sees only the final answer.
"""

from __future__ import annotations

import re

from twin.bench.dataset import BenchItem
from twin.train.extract import extract_final_answer
from twin.verifiers.result import VerificationResult
from twin.verifiers import verify_code, verify_math

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
_MCQ_LABEL_RE = re.compile(r"[\(\[]?\s*([A-Za-z])\s*[\)\].:]")


def strip_think(text: str) -> str:
    """Drop ``<think>...</think>`` spans so answer extraction sees the response.

    An *unclosed* think block (a truncated reasoning trace that never produced a
    final answer) is left in place — extraction will then fail and the item is
    scored wrong, which is the correct outcome for a model that didn't finish."""
    return _THINK_RE.sub("", text or "")


def extract_code_block(text: str) -> str:
    """Return the last fenced code block, or the whole text if none is fenced."""
    blocks = _FENCE_RE.findall(text or "")
    if blocks:
        return blocks[-1].strip()
    return (text or "").strip()


def _normalize(s: str) -> str:
    """Lowercase and reduce to alphanumeric tokens separated by single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _accepts(candidate: str, accepted: list[str]) -> bool:
    """True if a normalized accepted answer equals, or appears as whole words in,
    the normalized candidate (so 'Tokyo' matches 'The capital is Tokyo' but 'no'
    does not match 'winner')."""
    nc = _normalize(candidate)
    if not nc:
        return False
    padded = f" {nc} "
    for a in accepted:
        na = _normalize(a)
        if na and (nc == na or f" {na} " in padded):
            return True
    return False


def extract_mcq_letter(text: str, choices: list[str]) -> str:
    """Pull the chosen option letter (e.g. 'B') from a multiple-choice response.

    Tries, in order: a labelled letter on the answer line ('ANSWER: B', '(B)',
    'B)'), then matching the answer text back to exactly one choice. Returns ''
    when nothing resolves (scored as wrong)."""
    n = len(choices)
    valid = {chr(ord("A") + i) for i in range(n)}
    ans = extract_final_answer(text)

    # 1) a labelled / bare letter on the answer line.
    m = _MCQ_LABEL_RE.search(ans)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    bare = ans.strip()
    if len(bare) == 1 and bare.upper() in valid:
        return bare.upper()
    lead = re.match(r"\s*([A-Za-z])\b", ans)
    if lead and lead.group(1).upper() in valid:
        return lead.group(1).upper()

    # 2) fall back to matching the answer text against the option texts.
    na = _normalize(ans)
    if na:
        hits = [i for i, c in enumerate(choices) if _normalize(c) and _normalize(c) == na]
        if len(hits) != 1:
            padded = f" {na} "
            hits = [i for i, c in enumerate(choices)
                    if _normalize(c) and f" {_normalize(c)} " in padded]
        if len(hits) == 1:
            return chr(ord("A") + hits[0])
    return ""


def grade(item: BenchItem, response: str) -> VerificationResult:
    """Grade ``response`` against ``item``, returning a ``VerificationResult``."""
    text = strip_think(response)
    vtype = item.vtype

    if vtype == "math_numeric":
        tol = float(item.verification.get("tolerance", 1e-6))
        return verify_math(extract_final_answer(text), item.answer, tolerance=tol)

    if vtype == "code":
        v = item.verification
        candidate = extract_code_block(text)
        opts = {k: v[k] for k in ("timeout_s", "cpu_s", "mem_mb") if k in v}
        return verify_code(candidate, v.get("tests", ""), setup=v.get("setup", ""), **opts)

    if vtype == "mcq":
        letter = extract_mcq_letter(text, item.choices)
        ref = item.answer.strip().upper()
        if not letter:
            return VerificationResult.fail("mcq", "no option letter found in response")
        detail = f"chose {letter}, reference {ref}"
        return (VerificationResult.ok if letter == ref else VerificationResult.fail)("mcq", detail)

    if vtype == "exact":
        accepted = [item.answer, *item.verification.get("aliases", [])]
        candidate = extract_final_answer(text)
        ok = _accepts(candidate, accepted)
        detail = f"{candidate!r} vs accepted {accepted}"
        return (VerificationResult.ok if ok else VerificationResult.fail)("exact", detail)

    return VerificationResult.fail("unverifiable", f"unknown verification type: {vtype!r}")
