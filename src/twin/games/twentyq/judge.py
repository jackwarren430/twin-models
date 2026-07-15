"""The three frozen-base judge contracts for 21 questions (twentyq/DESIGN.md
§2.4, Sprint Q2). Same shape as :mod:`twin.verifiers.judge`: pure functions
over an ``oracle: Callable[[str], str]`` (the trainer binds ``_judge``; tests
bind stubs), strict last-match regex parsing, graceful degradation — a judge
that rambles or fails to answer NEVER raises.

Degradation policy per contract:

- validity: degradation posture is configurable via ``mode`` (default
  "fail_open"). fail-open => no clear verdict counts as VALID; only a clear
  ``VERDICT: INVALID`` voids. This is the default because gemma4-E2B routinely
  role-plays the guesser and ends its turn before emitting any verdict, so
  fail-CLOSED voided valid, guessable secrets (Axolotl/Salmon/Octopus in
  q-fullv2-rot) and discarded all their episodes — the same "throws away good
  data" pathology that retired the answer audit. "fail_closed" keeps the old
  posture (unjudgeable => INVALID); "off" skips the judge call entirely.
- answer audit: unparseable / wrong count => ``None`` (unauditable). The
  trainer treats None as "not proven lying" (episode counts) but logs it —
  fail-open, because voiding on judge failure would punish the creator for
  the judge's mistakes, inviting noise-farming of R_consistency.
- closeness: unparseable => ``None`` (no shaping term, reward falls back to
  the binary outcome).
"""

import re
from typing import Callable, Optional

from twin.games.twentyq.schema import Secret
from twin.verifiers.result import VerificationResult

_VALIDITY = re.compile(r"VERDICT\s*[:=]\s*(VALID|INVALID)", re.IGNORECASE)
_CLOSENESS = re.compile(r"CLOSENESS\s*[:=]\s*(-?\d+)", re.IGNORECASE)
_AUDIT_LINE = re.compile(r"AUDIT\s*[:=]\s*([A-Za-z01](?:[\sA-Za-z01,]*[A-Za-z01])?)", re.IGNORECASE)
_TRUE_TOKENS = {"t", "true", "y", "yes", "1"}
_FALSE_TOKENS = {"f", "false", "n", "no", "0"}

CLOSENESS_MAX = 10


_VALIDITY_PROMPT = """You are vetting a secret for a game of 20 questions. A valid \
secret is a real, commonly known, unambiguous instance of its declared category — \
something a well-read person could identify through yes/no questions. Invalid: \
fictional-only mashups, vague abstractions, category mismatches, multi-entity \
answers, or things too obscure to be guessable.

Category: {category}
Secret: {secret}

Think briefly, then end with exactly one line:
VERDICT: VALID   or   VERDICT: INVALID"""


_AUDIT_PROMPT = """You are auditing the answers given in a 20-questions game. The \
answerer knew the secret and answered each question. For EACH numbered question, \
mark T if the answer is truthful for the secret, or F ONLY if it is clearly, \
definitely false. A defensible, borderline, or reasonable-judgment-call answer is \
T — mark F only for an unambiguous lie. Answers of SOMETIMES and UNKNOWN are T \
unless plainly contradicted. Judge only truthfulness, not helpfulness.

Secret: {secret}  (category: {category})

Question/answer pairs:
{pairs}

End with exactly one line containing one T or F per question, in order, \
space-separated, e.g. for three questions: AUDIT: T T F
AUDIT: {slots}"""


_CLOSENESS_PROMPT = """You are scoring how close a guesser is to identifying a \
secret in 20 questions, given the questions asked and the truthful answers so far.

Secret: {secret}  (category: {category})

Transcript:
{pairs}
{guess_line}
Score on this scale:
- 0: nothing established; questions have not constrained even the broad category.
- 3: broad category established, but little else.
- 5: category plus several key attributes pinned down; the field is narrowed.
- 8: the answers so far are consistent with only a handful of candidates.
- 10: the secret is uniquely determined by the information gathered (or the \
final guess is essentially it).

Think briefly, then end with exactly one line:
CLOSENESS: <integer 0-10>"""


def _format_pairs(qa_pairs: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"{i + 1}. Q: {q}\n   A: {a}" for i, (q, a) in enumerate(qa_pairs)
    ) or "(no questions asked)"


def judge_secret_validity(
    secret: Secret, oracle: Callable[[str], str], mode: str = "fail_open"
) -> VerificationResult:
    """VALID/INVALID vetting of one secret before any episode.

    ``mode`` sets what happens when the judge emits no parseable ``VERDICT:``
    line (gemma4-E2B frequently role-plays the guesser and ends its turn first):
      - "fail_open" (default): unparseable / judge error => VALID; only a clear
        ``VERDICT: INVALID`` voids. Stops discarding valid secrets on format flakiness.
      - "fail_closed": unparseable => INVALID (original posture).
      - "off": skip the judge call entirely — every parsed secret is played.
    A clear verdict is always honoured; ``mode`` only decides the fallback.
    """
    if mode == "off":
        return VerificationResult.ok("q_validity", "validity gate off")
    # Fallback when the judge gives us nothing usable.
    unresolved = VerificationResult.ok if mode == "fail_open" else VerificationResult.fail
    prompt = _VALIDITY_PROMPT.format(category=secret.category, secret=secret.secret)
    try:
        reply = oracle(prompt)
    except Exception as e:  # noqa: BLE001 - judge boundary, never crash a run
        return unresolved("q_validity", f"judge error ({mode}): {e}")
    matches = _VALIDITY.findall(reply or "")
    if not matches:
        return unresolved(
            "q_validity", f"no clear verdict ({mode}): {(reply or '')[:120]!r}")
    ok = matches[-1].upper() == "VALID"
    return (VerificationResult.ok if ok else VerificationResult.fail)(
        "q_validity", "judge: secret validity")


def judge_answer_audit(
    secret: Secret,
    qa_pairs: list[tuple[str, str]],
    oracle: Callable[[str], str],
) -> Optional[list[bool]]:
    """One batched truthfulness audit per episode: per-pair True (truthful) /
    False (lie). Returns ``None`` when the episode is unauditable (judge error,
    no AUDIT line, or a verdict count that doesn't match the pair count)."""
    if not qa_pairs:
        return []
    prompt = _AUDIT_PROMPT.format(
        secret=secret.secret, category=secret.category,
        pairs=_format_pairs(qa_pairs),
        slots=" ".join("_" for _ in qa_pairs),
    )
    try:
        reply = oracle(prompt)
    except Exception:  # noqa: BLE001
        return None
    matches = _AUDIT_LINE.findall(reply or "")
    if not matches:
        return None
    # A parroted "AUDIT: _ _ _" slots row has no letter/digit token, so it
    # never reaches here; unknown tokens (or a count mismatch) => unauditable.
    tokens = re.split(r"[\s,]+", matches[-1].strip())
    if len(tokens) != len(qa_pairs):
        return None
    verdicts: list[bool] = []
    for tok in tokens:
        low = tok.lower()
        if low in _TRUE_TOKENS:
            verdicts.append(True)
        elif low in _FALSE_TOKENS:
            verdicts.append(False)
        else:
            return None
    return verdicts


def judge_closeness(
    secret: Secret,
    qa_pairs: list[tuple[str, str]],
    oracle: Callable[[str], str],
    *,
    final_guess: str | None = None,
) -> Optional[float]:
    """Φ ∈ [0, 1]: anchored 0-10 integer scale mapped down. v1 calls this once
    per episode (final state); v2 per-turn credit calls it after every turn.
    Out-of-range integers are clamped; no parseable integer => ``None``."""
    guess_line = f"Final guess: {final_guess}\n" if final_guess else ""
    prompt = _CLOSENESS_PROMPT.format(
        secret=secret.secret, category=secret.category,
        pairs=_format_pairs(qa_pairs), guess_line=guess_line,
    )
    try:
        reply = oracle(prompt)
    except Exception:  # noqa: BLE001
        return None
    matches = _CLOSENESS.findall(reply or "")
    if not matches:
        return None
    n = max(0, min(CLOSENESS_MAX, int(matches[-1])))
    return n / CLOSENESS_MAX
