"""Problem / ProblemSuite data structures and robust parsing of creator output.

The creator emits a *suite* of problems spanning an easy->hard difficulty ramp.
Each problem carries the creator's own ``solution`` and final ``answer`` (used
for the consistency check) plus an optional ``verification`` spec telling the
verifier how to check an answer (Sprint 2). The expected JSON contract:

    {
      "theme": "quadratics",
      "domain": "math",
      "problems": [
        {
          "statement": "Solve x^2 - 5x + 6 = 0; give the larger root.",
          "difficulty": 0.0,
          "answer": "3",
          "solution": "Factor (x-2)(x-3)=0 ...",
          "verification": {"type": "math_numeric", "tolerance": 1e-6}
        },
        ...
      ]
    }

Parsing is deliberately forgiving: model output is often wrapped in prose or a
```json fenced block, so we extract the first balanced JSON object.
"""

import json
import math
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from twin.think import strip_think


class SuiteParseError(ValueError):
    """Raised when creator output cannot be parsed into a ProblemSuite."""


@dataclass
class Problem:
    statement: str
    difficulty: float                     # creator's claimed difficulty, 0..1
    answer: str = ""                      # creator's final answer (for consistency)
    solution: str = ""                    # creator's worked solution / explanation
    domain: str = "math"
    verification: dict[str, Any] = field(default_factory=dict)
    problem_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, default_domain: str = "math") -> "Problem":
        if "statement" not in d or not str(d.get("statement", "")).strip():
            raise SuiteParseError("problem missing non-empty 'statement'")
        try:
            difficulty = float(d.get("difficulty"))
        except (TypeError, ValueError):
            raise SuiteParseError(f"problem has non-numeric 'difficulty': {d.get('difficulty')!r}")
        return cls(
            statement=str(d["statement"]).strip(),
            difficulty=difficulty,
            answer=str(d.get("answer", "")).strip(),
            solution=str(d.get("solution", "")).strip(),
            domain=str(d.get("domain", default_domain)),
            verification=dict(d.get("verification", {}) or {}),
            problem_id=str(d.get("problem_id", uuid.uuid4().hex[:12])),
        )


@dataclass
class ProblemSuite:
    problems: list[Problem]
    theme: str = ""
    domain: str = "math"
    metadata: dict[str, Any] = field(default_factory=dict)
    suite_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # ----- views -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.problems)

    def sorted_by_difficulty(self) -> list[Problem]:
        """Problems ordered easy->hard by claimed difficulty (stable)."""
        return sorted(self.problems, key=lambda p: p.difficulty)

    @staticmethod
    def target_curve(n: int, hi: float = 1.0, lo: float = 0.0) -> list[float]:
        """Target realised solve-rate at each rank, easy->hard: linear hi -> lo.

        For n==1 returns [hi]. This is what the creator's gradient reward is
        measured against (DESIGN.md §6.1).

        The default 1 -> 0 ramp asks for a certainly-solved problem at rank 0
        and a never-solved one at rank n-1 — ranks where the realized solver
        K-group has zero outcome variance and therefore zero gradient, *by
        design*, once the creator is on-target. An interior band (e.g.
        0.9 -> 0.1 via ``rewards.target_hi``/``target_lo``) keeps every rank's
        outcomes stochastic so solver groups stay informative at
        creator-optimum. Changing the band changes the game's incentives, so
        treat a non-default band as a measured ablation (DESIGN.md §11
        Sprint 6)."""
        if n <= 0:
            return []
        if n == 1:
            return [hi]
        return list(np.linspace(hi, lo, n))

    # ----- validation ------------------------------------------------------
    def validate(self, *, min_n: int = 2, max_n: int = 20) -> list[str]:
        """Return a list of issues (empty == valid). Used by the parse gate and
        the creator's validity-shaping reward."""
        issues: list[str] = []
        n = len(self.problems)
        if n < min_n:
            issues.append(f"need >= {min_n} problems, got {n}")
        if n > max_n:
            issues.append(f"need <= {max_n} problems, got {n}")
        for i, p in enumerate(self.problems):
            if not (0.0 <= p.difficulty <= 1.0) or math.isnan(p.difficulty):
                issues.append(f"problem {i} difficulty out of [0,1]: {p.difficulty}")
            if not p.statement.strip():
                issues.append(f"problem {i} has empty statement")
        # Want a genuine spread, not all-easy / all-hard (anti-collapse).
        if n >= 2:
            diffs = [p.difficulty for p in self.problems]
            if max(diffs) - min(diffs) < 0.3:
                issues.append(
                    f"difficulty spread too small: {max(diffs) - min(diffs):.2f} < 0.30"
                )
        return issues

    def is_valid(self, *, min_n: int = 2, max_n: int = 20) -> bool:
        return not self.validate(min_n=min_n, max_n=max_n)

    # ----- serialization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "theme": self.theme,
            "domain": self.domain,
            "metadata": self.metadata,
            "problems": [p.to_dict() for p in self.problems],
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProblemSuite":
        if "problems" not in d or not isinstance(d["problems"], list):
            raise SuiteParseError("suite missing 'problems' list")
        if not d["problems"]:
            raise SuiteParseError("suite has no problems")
        default_domain = str(d.get("domain", "math"))
        problems = [
            Problem.from_dict(p, default_domain=default_domain) for p in d["problems"]
        ]
        return cls(
            problems=problems,
            theme=str(d.get("theme", "")),
            domain=default_domain,
            metadata=dict(d.get("metadata", {}) or {}),
            suite_id=str(d.get("suite_id", uuid.uuid4().hex[:12])),
        )


# ---------------------------------------------------------------------------
# Robust extraction of a JSON object from free-form model text.
# ---------------------------------------------------------------------------
_TOOL_MARKUP_RE = re.compile(
    r"<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>",
    re.DOTALL | re.IGNORECASE,
)


def _strip_tool_markup(text: str) -> str:
    """Drop native ``<tool_call>``/``<tool_response>`` blocks: a tool-call's
    OWN JSON (``{"name": "solve", ...}``) is a parseable dict that sits
    earlier in the rollout than the final problem JSON, so leaving it in made
    every tool-USING rollout parse-fail (probe 2026-07-04 — the model did
    everything right and the extractor grabbed the wrong object)."""
    return _TOOL_MARKUP_RE.sub("", text or "")


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the intended JSON object from creator output.

    The post-``</think>``, post-tool-markup text is searched FIRST: with
    thinking on, the trace routinely contains *draft* problem JSONs (audit
    2026-07-03), and with native tools the rollout contains the tool-call
    JSON itself (probe 2026-07-04) — either would beat the final answer by
    start index. The raw text is the fallback so an unclosed think block
    whose JSON is the only JSON (truncation) still parses, matching pre-fix
    behaviour there."""
    text = text.strip()
    visible = _strip_tool_markup(strip_think(text)).strip()
    candidates = [visible, text] if (visible and visible != text) else [text]
    for t in candidates:
        obj = _extract_json_object_raw(t)
        if obj is not None:
            return obj
    raise SuiteParseError("no parseable JSON object found in text")


def _extract_json_object_raw(text: str) -> dict[str, Any] | None:
    # 1) whole string is JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # 2) fenced ```json ... ``` (or plain ``` ... ```) block, first that parses
    fences = _iter_code_fences(text)
    for block in fences:
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    # 3) first balanced {...} span that parses as a dict
    for span in _iter_balanced_braces(text):
        try:
            obj = json.loads(span)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _iter_code_fences(text: str) -> list[str]:
    blocks: list[str] = []
    i = 0
    while True:
        start = text.find("```", i)
        if start == -1:
            break
        # skip the optional language tag up to end of line
        nl = text.find("\n", start)
        if nl == -1:
            break
        end = text.find("```", nl + 1)
        if end == -1:
            break
        blocks.append(text[nl + 1 : end].strip())
        i = end + 3
    return blocks


def _iter_balanced_braces(text: str):
    """Yield substrings that are balanced-brace spans starting at each '{',
    respecting strings/escapes. Yields outermost-first by start index."""
    n = len(text)
    for start in range(n):
        if text[start] != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for j in range(start, n):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : j + 1]
                    break


def parse_suite(text: str) -> ProblemSuite:
    """Parse creator output text into a ProblemSuite, raising SuiteParseError
    on failure. The training loop catches this and applies the parse-gate
    penalty (DESIGN.md §6.3)."""
    obj = _extract_json_object(text)
    return ProblemSuite.from_dict(obj)


def parse_problem(text: str, *, default_domain: str = "math") -> Problem:
    """Parse creator output text into a single Problem (Sprint 7 per-problem
    creator mode), raising SuiteParseError on failure. Tolerates the model
    wrapping the one problem in a suite-style ``{"problems": [...]}`` object —
    the first entry is taken."""
    obj = _extract_json_object(text)
    if "problems" in obj and isinstance(obj["problems"], list):
        if not obj["problems"]:
            raise SuiteParseError("wrapper object has an empty 'problems' list")
        inner = obj["problems"][0]
        if not isinstance(inner, dict):
            raise SuiteParseError("wrapper 'problems' entry is not an object")
        obj = inner
    return Problem.from_dict(obj, default_domain=default_domain)
