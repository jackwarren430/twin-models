"""The 21-questions episode engine (twentyq/DESIGN.md §4 Q3).

Model-free: the two players are injected callables, so tests script them
deterministically and the trainer (Sprint Q5) binds real generate closures.

    guesser_fn(qa_pairs, turn_index) -> gen
        ``qa_pairs`` is the full (question, answer) history so far; ``gen``
        must expose ``.text`` (raw completion) and — for trained players —
        ``.prompt_tokens`` / ``.completion_tokens`` (the exact rendered-prompt
        and completion ids, per-turn trajectory granularity, DESIGN §2.2).
    answerer_fn(question, qa_pairs) -> str
        Raw answerer completion text; parsed here to a canonical answer.

Contract lines (parsed on think-stripped text, last occurrence wins):

    guesser:  ``QUESTION: <yes/no question>``  or  ``GUESS: <entity>``
    answerer: ``ANSWER: YES | NO | SOMETIMES | UNKNOWN``

Rules: a correct GUESS ends the episode (``ended="guessed"``). A wrong GUESS
consumes the turn and the ENGINE answers "NO" (ground truth by construction —
recorded in the history but excluded from the truthfulness audit, which covers
only creator-authored answers). A guesser turn with neither contract line ends
the episode as a failure (``ended="format"``). Budget exhaustion without a
correct guess is ``ended="budget"``. An unparseable answerer reply is recorded
as UNKNOWN and counted in ``n_answer_format_fails`` (it stays in the audited
pairs: UNKNOWN about one's own secret is itself suspicious, let the judge see
the raw evasion).
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from twin.games.twentyq.schema import Secret, guess_matches
from twin.think import strip_think

_QUESTION = re.compile(r"^\s*QUESTION\s*[:=]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_GUESS = re.compile(r"^\s*(?:FINAL\s+)?GUESS\s*[:=]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_ANSWER = re.compile(r"ANSWER\s*[:=]\s*(YES|NO|SOMETIMES|UNKNOWN)\b", re.IGNORECASE)

ANSWER_TOKENS = ("YES", "NO", "SOMETIMES", "UNKNOWN")


def parse_guesser_turn(text: str) -> tuple[Optional[str], str]:
    """``(kind, content)`` from a raw guesser completion: kind is ``"guess"``
    (takes precedence — a turn that commits to a guess IS a guess), or
    ``"question"``, or ``None`` on format failure. Contract lines are searched
    on think-stripped text so a QUESTION drafted inside <think> doesn't count."""
    visible = strip_think(text or "")
    guesses = _GUESS.findall(visible)
    if guesses:
        return "guess", guesses[-1].strip()
    questions = _QUESTION.findall(visible)
    if questions:
        return "question", questions[-1].strip()
    return None, ""


def parse_answer(text: str) -> Optional[str]:
    """Canonical ``YES|NO|SOMETIMES|UNKNOWN`` from a raw answerer completion,
    or ``None`` when no contract line is present (caller records UNKNOWN)."""
    matches = _ANSWER.findall(strip_think(text or ""))
    return matches[-1].upper() if matches else None


@dataclass
class Turn:
    """One guesser turn plus the reply it received."""
    index: int
    kind: str                        # "question" | "guess" | "format_fail"
    content: str                     # parsed question text or guessed entity
    answer: Optional[str]            # canonical reply (None when episode ended here)
    raw_text: str                    # guesser's full completion (with thinking)
    prompt_tokens: Any = None        # exact rendered-prompt ids (trained player)
    completion_tokens: Any = None    # exact completion ids (trained player)
    answer_raw: str = ""             # answerer's full completion, "" for engine "NO"
    creator_answered: bool = False   # True => audited (creator-authored answer)


@dataclass
class Episode:
    secret: Secret
    turns: list[Turn] = field(default_factory=list)
    guessed: bool = False
    ended: str = "budget"            # "guessed" | "budget" | "format"
    n_answer_format_fails: int = 0

    @property
    def turns_used(self) -> int:
        return len(self.turns)

    @property
    def qa_pairs(self) -> list[tuple[str, str]]:
        """Full (question, answer) history — what the guesser saw. Wrong
        guesses appear as 'Is it <x>?' / 'NO'; the ending turn (correct guess
        or format fail) has no answer and is not a pair."""
        return [
            (f"Is it {t.content}?" if t.kind == "guess" else t.content, t.answer)
            for t in self.turns if t.answer is not None
        ]

    @property
    def audited_pairs(self) -> list[tuple[str, str]]:
        """The subset of :attr:`qa_pairs` with creator-authored answers — the
        truthfulness audit's input (engine-generated 'NO's are ground truth)."""
        return [
            (t.content, t.answer)
            for t in self.turns if t.answer is not None and t.creator_answered
        ]


def run_episode(
    guesser_fn: Callable[[list[tuple[str, str]], int], Any],
    answerer_fn: Callable[[str, list[tuple[str, str]]], str],
    secret: Secret,
    *,
    max_turns: int,
) -> Episode:
    """Play one episode to termination. See module docstring for the rules."""
    ep = Episode(secret=secret)
    for i in range(max_turns):
        gen = guesser_fn(ep.qa_pairs, i)
        kind, content = parse_guesser_turn(gen.text)
        turn = Turn(
            index=i, kind=kind or "format_fail", content=content,
            answer=None, raw_text=gen.text,
            prompt_tokens=getattr(gen, "prompt_tokens", None),
            completion_tokens=getattr(gen, "completion_tokens", None),
        )
        ep.turns.append(turn)

        if kind is None:
            ep.ended = "format"
            return ep

        if kind == "guess":
            if guess_matches(content, secret.secret):
                ep.guessed = True
                ep.ended = "guessed"
                return ep
            # Wrong guess: consumes the turn, engine answers truthfully.
            turn.answer = "NO"
            continue

        # Plain question -> the answerer (creator) replies.
        raw_answer = answerer_fn(content, ep.qa_pairs)
        answer = parse_answer(raw_answer)
        if answer is None:
            ep.n_answer_format_fails += 1
            answer = "UNKNOWN"
        turn.answer = answer
        turn.answer_raw = raw_answer
        turn.creator_answered = True

    ep.ended = "budget"
    return ep
