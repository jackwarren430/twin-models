"""Secret data structure + parsing of the 21-questions creator output
(twentyq/DESIGN.md §2.4, Sprint Q2).

The creator emits ONE secret per rollout (per-secret creator mode, mirroring
``creator_mode: per_problem``). Expected JSON contract::

    {
      "secret": "octopus",
      "category": "animal",
      "difficulty": 0.4,
      "notes": "invertebrate; guessers anchor on mammals first"
    }

Parsing reuses the suite parser's balanced-JSON extraction, so prose- and
fence-wrapped objects work identically to problem parsing. ``difficulty`` is
the creator's claim; the trainer overwrites it with the dictated rank value,
exactly as the per-problem loop does.
"""

import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from twin.problems.schema import SuiteParseError, _extract_json_object


class SecretParseError(SuiteParseError):
    """Raised when creator output cannot be parsed into a Secret."""


@dataclass
class Secret:
    secret: str                    # the entity to be guessed, e.g. "octopus"
    category: str                  # broad guessable class, e.g. "animal"
    difficulty: float              # dictated rank value 0..1 (trainer-owned)
    notes: str = ""                # creator's own rationale (never shown to solver)
    secret_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, default_category: str = "thing") -> "Secret":
        if "secret" not in d or not str(d.get("secret", "")).strip():
            raise SecretParseError("secret missing non-empty 'secret'")
        try:
            difficulty = float(d.get("difficulty"))
        except (TypeError, ValueError):
            raise SecretParseError(
                f"secret has non-numeric 'difficulty': {d.get('difficulty')!r}")
        return cls(
            secret=str(d["secret"]).strip(),
            category=str(d.get("category", default_category)).strip() or default_category,
            difficulty=difficulty,
            notes=str(d.get("notes", "")).strip(),
            secret_id=str(d.get("secret_id", uuid.uuid4().hex[:12])),
        )


def parse_secret(text: str, *, default_category: str = "thing") -> Secret:
    """Parse creator output text into a Secret, raising SecretParseError on
    failure. Tolerates a ``{"secrets": [...]}`` wrapper (first entry taken),
    the same forgiveness ``parse_problem`` extends to suite wrappers."""
    try:
        obj = _extract_json_object(text)
    except SuiteParseError as e:
        raise SecretParseError(str(e)) from e
    if "secrets" in obj and isinstance(obj["secrets"], list):
        if not obj["secrets"]:
            raise SecretParseError("wrapper object has an empty 'secrets' list")
        inner = obj["secrets"][0]
        if not isinstance(inner, dict):
            raise SecretParseError("wrapper 'secrets' entry is not an object")
        obj = inner
    return Secret.from_dict(obj, default_category=default_category)


# ----- guess matching --------------------------------------------------------

_ARTICLES = ("a ", "an ", "the ")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize_guess(text: str) -> str:
    """Canonical form for guess/secret comparison: lowercase, leading article
    dropped, punctuation stripped, whitespace collapsed. Deliberately does NOT
    attempt synonym matching — 'automobile' != 'car' in v1; a judge-arbitrated
    match is a deferred ablation (twentyq/DESIGN.md §5)."""
    s = (text or "").strip().lower()
    for art in _ARTICLES:
        if s.startswith(art):
            s = s[len(art):]
            break
    s = _PUNCT.sub("", s)
    return _WS.sub(" ", s).strip()


def guess_matches(guess: str, secret: str) -> bool:
    """Normalized exact match, with bare plural tolerance (exactly one
    trailing 's' on either side): 'the Octopus.' matches 'octopus', 'cats'
    matches 'cat'. Irregular plurals ('glasses'/'glass') don't match — the
    creator's contract asks for singular secrets, so this stays simple."""
    g, s = normalize_guess(guess), normalize_guess(secret)
    if not g or not s:
        return False
    return g == s or g == s + "s" or s == g + "s"
