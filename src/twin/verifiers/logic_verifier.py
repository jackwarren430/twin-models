"""Knights & Knaves logic verifier (domain expansion, 2026-07-04).

The domain the literature calls the canonical verifiable-logic RLVR testbed
(Logic-RL, arXiv:2502.14768: procedurally generated K&K, difficulty dialed by
character count and statement complexity; Reasoning Gym, arXiv:2505.24760,
ships it among its 100+ generator/verifier pairs). Every character is a
knight (always tells the truth) or a knave (always lies); the solver must
identify everyone from their statements.

Why this domain fits twin better than more math (EXPERIMENTS.md mini-04b):
consistency was the measured binding constraint — in math, a problem the
solver finds hard is usually one the *creator* also risks self-answering
wrong. K&K is asymmetric: solving is search (genuinely hard for a text
model as characters/nesting grow) while authoring + verification is a
2^n enumeration (n <= 8 -> trivial, exact, fail-closed). With the
``logic_solve`` tool the creator's answer is correct *by construction*
(AZR's executor-derived-truth property, imported to logic).

The certificate is ONE string field, ``verification.claims``, in a tiny
formal DSL::

    A: B & ~C; B: ~A; C: A | B

read as "A says 'B is a knight and C is a knave'; B says ...". A bare name
means "that person is a knight". Semantics: for each claim ``(speaker, s)``
the constraint is ``speaker <-> s`` (a knight's statement is true, a
knave's is false). Operators: ``~`` not, ``&`` and, ``|`` or, ``^`` xor,
``->`` implies (right-assoc), parentheses; the word forms NOT/AND/OR/XOR/
IMPLIES are normalized to symbols. Literals ``true``/``false`` are allowed
("A: false" forces A to be a knave). People are inferred from the claims
(speakers + names mentioned); an unmentioned non-speaker is unconstrainable,
which the uniqueness requirement rejects on its own.

Fail-closed everywhere: a cert that is missing, unparseable, contradictory
(0 solutions) or ambiguous (>1 solution) DISCARDS the problem; a candidate
answer that is unparseable or incomplete grades wrong with a reason.
"""

import itertools
import re

from twin.verifiers.result import VerificationResult

MAX_PEOPLE = 8
MIN_PEOPLE = 2

_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_WORD_OPS = {"not": "~", "and": "&", "or": "|", "xor": "^", "implies": "->"}
_TRUE_WORDS = {"knight", "knights", "truth-teller", "truthteller", "true", "honest"}
_FALSE_WORDS = {"knave", "knaves", "liar", "false"}
_LITERALS = {"true": True, "false": False}


class ClaimParseError(ValueError):
    """Raised when the claims DSL or an assignment answer cannot be parsed."""


# --------------------------------------------------------------------------- #
# Boolean-expression parser (tiny recursive descent — no eval, no sympy)
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"->|[()&|^~]|[A-Za-z][A-Za-z0-9_]*")


def _tokenize(expr: str) -> list[str]:
    tokens = []
    pos = 0
    for m in _TOKEN_RE.finditer(expr):
        if expr[pos:m.start()].strip():
            raise ClaimParseError(f"unexpected characters {expr[pos:m.start()].strip()!r}")
        tok = m.group()
        low = tok.lower()
        tokens.append(_WORD_OPS.get(low, tok))
        pos = m.end()
    if expr[pos:].strip():
        raise ClaimParseError(f"unexpected characters {expr[pos:].strip()!r}")
    if not tokens:
        raise ClaimParseError("empty claim expression")
    return tokens


class _Parser:
    """expr := or_ ('->' expr)?   (implication, right-associative)
    or_  := xor ('|' xor)*
    xor  := and_ ('^' and_)*
    and_ := unary ('&' unary)*
    unary := '~' unary | name | 'true' | 'false' | '(' expr ')'
    Returns an AST of nested tuples evaluated by :func:`_eval`."""

    def __init__(self, tokens: list[str]):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def parse(self):
        node = self.expr()
        if self.peek() is not None:
            raise ClaimParseError(f"unexpected token {self.peek()!r}")
        return node

    def expr(self):
        left = self.or_()
        if self.peek() == "->":
            self.take()
            return ("->", left, self.expr())
        return left

    def or_(self):
        node = self.xor()
        while self.peek() == "|":
            self.take()
            node = ("|", node, self.xor())
        return node

    def xor(self):
        node = self.and_()
        while self.peek() == "^":
            self.take()
            node = ("^", node, self.and_())
        return node

    def and_(self):
        node = self.unary()
        while self.peek() == "&":
            self.take()
            node = ("&", node, self.unary())
        return node

    def unary(self):
        tok = self.peek()
        if tok == "~":
            self.take()
            return ("~", self.unary())
        if tok == "(":
            self.take()
            node = self.expr()
            if self.take() != ")":
                raise ClaimParseError("unbalanced parentheses")
            return node
        if tok is None or tok in {")", "&", "|", "^", "->"}:
            raise ClaimParseError(f"expected a name, got {tok!r}")
        self.take()
        low = tok.lower()
        if low in _LITERALS:
            return ("lit", _LITERALS[low])
        return ("var", low)


def _eval(node, assign: dict[str, bool]) -> bool:
    op = node[0]
    if op == "var":
        return assign[node[1]]
    if op == "lit":
        return node[1]
    if op == "~":
        return not _eval(node[1], assign)
    a = _eval(node[1], assign)
    b = _eval(node[2], assign)
    if op == "&":
        return a and b
    if op == "|":
        return a or b
    if op == "^":
        return a != b
    if op == "->":
        return (not a) or b
    raise ClaimParseError(f"unknown operator {op!r}")


def _variables(node, out: set[str]):
    if node[0] == "var":
        out.add(node[1])
    elif node[0] == "~":
        _variables(node[1], out)
    elif node[0] in {"&", "|", "^", "->"}:
        _variables(node[1], out)
        _variables(node[2], out)


# --------------------------------------------------------------------------- #
# Claims DSL + enumeration
# --------------------------------------------------------------------------- #
def parse_claims(claims: str) -> tuple[list[str], list[tuple[str, object]]]:
    """Parse ``"A: B & ~C; B: ~A"`` into (people, [(speaker, ast), ...]).

    People are the union of speakers and names mentioned in claims, sorted;
    all names are case-normalized to lower. Raises :class:`ClaimParseError`.
    """
    text = (claims or "").strip()
    if not text:
        raise ClaimParseError("no claims supplied")
    parts = [p.strip() for p in re.split(r"[;\n]+", text) if p.strip()]
    parsed: list[tuple[str, object]] = []
    people: set[str] = set()
    for part in parts:
        if ":" not in part:
            raise ClaimParseError(
                f"claim {part!r} is not of the form 'Speaker: statement'")
        speaker_raw, expr = part.split(":", 1)
        speaker = speaker_raw.strip()
        if not _NAME_RE.fullmatch(speaker):
            raise ClaimParseError(f"bad speaker name {speaker_raw.strip()!r}")
        if speaker.lower() in _LITERALS:
            raise ClaimParseError(f"speaker cannot be named {speaker!r}")
        ast = _Parser(_tokenize(expr)).parse()
        parsed.append((speaker.lower(), ast))
        people.add(speaker.lower())
        _variables(ast, people)
    return sorted(people), parsed


def solve_claims(claims: str) -> tuple[list[str], list[dict[str, bool]]]:
    """Enumerate every knight/knave assignment satisfying the claims.

    Returns ``(people, solutions)``. Raises :class:`ClaimParseError` on bad
    DSL or a people count outside [1, MAX_PEOPLE] (enumeration is 2^n)."""
    people, parsed = parse_claims(claims)
    if len(people) > MAX_PEOPLE:
        raise ClaimParseError(
            f"{len(people)} people > max {MAX_PEOPLE} (keep puzzles enumerable)")
    solutions = []
    for values in itertools.product([True, False], repeat=len(people)):
        assign = dict(zip(people, values))
        # knight <-> statement true, for every claim
        if all(assign[s] == _eval(ast, assign) for s, ast in parsed):
            solutions.append(assign)
    return people, solutions


def format_assignment(assign: dict[str, bool]) -> str:
    """Canonical answer string: ``"a=knight, b=knave"`` (sorted by name)."""
    return ", ".join(
        f"{name}={'knight' if val else 'knave'}"
        for name, val in sorted(assign.items())
    )


def parse_assignment(text: str, people: list[str]) -> dict[str, bool]:
    """Parse a solver/creator answer like ``"A=knight, B=knave"`` (also
    ``A: knight``; synonyms truth-teller/liar/true/false) into a complete
    assignment over ``people``. Fail-closed: unknown names, missing people,
    duplicates and unknown roles all raise :class:`ClaimParseError`."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ClaimParseError("empty answer")
    assign: dict[str, bool] = {}
    known = set(people)
    for part in re.split(r"[;,\n]+", cleaned):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(
            r"([A-Za-z][A-Za-z0-9_]*)\s*[:=]\s*([A-Za-z-]+)", part)
        if not m:
            raise ClaimParseError(
                f"cannot read {part!r} as 'name=knight' or 'name=knave'")
        name, role = m.group(1).lower(), m.group(2).lower()
        if name not in known:
            raise ClaimParseError(f"unknown person {m.group(1)!r}")
        if name in assign:
            raise ClaimParseError(f"{m.group(1)!r} assigned twice")
        if role in _TRUE_WORDS:
            assign[name] = True
        elif role in _FALSE_WORDS:
            assign[name] = False
        else:
            raise ClaimParseError(f"unknown role {role!r} (use knight/knave)")
    missing = known - set(assign)
    if missing:
        raise ClaimParseError(f"no role given for: {', '.join(sorted(missing))}")
    return assign


# --------------------------------------------------------------------------- #
# Verifier entry points (dispatch.py)
# --------------------------------------------------------------------------- #
def check_logic_consistency(claims: str, answer: str) -> VerificationResult:
    """Creator-side certificate: the claims must pin EXACTLY ONE assignment
    (well-posedness — an ambiguous puzzle can grade a correct solver wrong)
    and the creator's stated answer must be that assignment."""
    try:
        people, solutions = solve_claims(claims)
    except ClaimParseError as e:
        return VerificationResult.fail("logic", f"bad claims: {e}")
    if len(people) < MIN_PEOPLE:
        return VerificationResult.fail(
            "logic", f"need >= {MIN_PEOPLE} people, got {len(people)}")
    if not solutions:
        return VerificationResult.fail("logic", "claims are contradictory (0 solutions)")
    if len(solutions) > 1:
        return VerificationResult.fail(
            "logic",
            f"ambiguous: {len(solutions)} satisfying assignments "
            f"(e.g. {format_assignment(solutions[0])} / {format_assignment(solutions[1])})",
        )
    try:
        stated = parse_assignment(answer, people)
    except ClaimParseError as e:
        return VerificationResult.fail("logic", f"bad answer: {e}")
    if stated != solutions[0]:
        return VerificationResult.fail(
            "logic",
            f"stated answer differs from the unique solution "
            f"({format_assignment(solutions[0])})",
        )
    return VerificationResult.ok("logic", f"unique solution over {len(people)} people")


def verify_logic(candidate: str, claims: str) -> VerificationResult:
    """Solver-side grading: recompute the unique solution from the certificate
    and compare. Independent of the creator's stated answer string, so a
    format quirk there cannot void a correct solver (the mini-02 lesson)."""
    try:
        people, solutions = solve_claims(claims)
    except ClaimParseError as e:
        return VerificationResult.fail("logic", f"bad claims: {e}")
    if len(solutions) != 1:
        return VerificationResult.fail(
            "logic", f"certificate does not pin a unique solution ({len(solutions)})")
    try:
        cand = parse_assignment(candidate, people)
    except ClaimParseError as e:
        return VerificationResult.fail("logic", f"unparseable answer: {e}")
    if cand == solutions[0]:
        return VerificationResult.ok("logic", "matches the unique solution")
    wrong = [n for n in people if cand[n] != solutions[0][n]]
    return VerificationResult.fail("logic", f"wrong on: {', '.join(wrong)}")
