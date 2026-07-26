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

from twin.games.twentyq.schema import Secret, guess_matches, normalize_guess
from twin.think import strip_think

# Accept "QUESTION:" and the natural abbreviation "Q:" — weaker models (gemma4)
# drift to "Q:" by mimicking the flattened Q/A history. Semantically identical.
_QUESTION = re.compile(r"^\s*(?:QUESTION|Q)\s*[:=]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_GUESS = re.compile(r"^\s*(?:FINAL\s+)?GUESS\s*[:=]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
# Same contract words, but not line-initial: "Closure: GUESS: clear broth".
# Tried only after the anchored forms miss, so a well-formed line is never
# reinterpreted by these.
_GUESS_INLINE = re.compile(r"\bGUESS\s*[:=]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_QUESTION_INLINE = re.compile(r"\bQUESTION\s*[:=]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
# Last resort: a line that simply reads as a question (ends with '?'). Keeps an
# episode alive when a weak guesser drops the contract prefix entirely, so the
# closeness shaping signal still flows instead of the turn format-failing.
# '？' (U+FF1F) included: gemma-4-E2B code-switches to CJK punctuation.
_BARE_QUESTION = re.compile(r"^\s*(\S.*[?？])\s*$", re.MULTILINE)
_ANSWER = re.compile(r"ANSWER\s*[:=]\s*(YES|NO|SOMETIMES|UNKNOWN)\b", re.IGNORECASE)
# Punctuation -> space (not deletion), so "bigger/smaller" splits into two tokens.
_PUNCT_ONLY = re.compile(r"[^\w\s]")

ANSWER_TOKENS = ("YES", "NO", "SOMETIMES", "UNKNOWN")

# ----- question de-duplication ----------------------------------------------
#
# Weak guessers do not merely repeat occasionally — they lock. In the v7
# validation transcripts gemma-4-E2B asked "Is the animal a parrot?" on ten
# consecutive turns after learning the secret was a non-tropical modern bird,
# then won on the forced last-turn guess (penguin). Those ten turns carry zero
# information AND, under terminal credit, receive the winning episode's full
# return — so GRPO reinforces the lock. Detecting the repeat lets the engine
# resample before the turn is spent.
#
# Matching is on CONTENT WORDS, not surface form: "Is the animal a parrot?" and
# "Is it a parrot?" are the same question and must collide. Function words, the
# category noun, and hedges are dropped; the remainder is depluralized and
# compared as a set.
_QUESTION_STOPWORDS = frozenset({
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "being", "by",
    "can", "commonly", "considered", "could", "did", "do", "does", "for",
    "from", "generally", "had", "has", "have", "if", "in", "is", "it", "its",
    "kind", "mainly", "mostly", "normally", "of", "often", "on", "one", "or",
    "primarily", "sort", "that", "the", "there", "these", "they", "this",
    "those", "to", "type", "typically", "usually", "was", "were", "with",
    "would", "you", "your",
    # generic stand-ins for the referent
    "anything", "entity", "item", "object", "secret", "something", "thing",
})


def _depluralize(token: str) -> str:
    """Bare trailing-'s' strip for tokens long enough that it is a plural and
    not the word itself ('parrots' -> 'parrot', but 'gas' and 'its' survive)."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_question(text: str, *, category: str = "") -> str:
    """Canonical content-word form used to detect a repeated question.

    Lowercased, punctuation-stripped, function words and the category noun
    removed, remaining tokens depluralized and sorted. Two questions with the
    same content-word set normalize equal, which is the property the repeat
    check needs. Word ORDER is deliberately discarded: it makes the comparison
    robust to the phrasings weak guessers cycle through, at the cost of merging
    genuinely order-sensitive questions ("is A bigger than B"), which do not
    occur in observed transcripts.
    """
    s = _PUNCT_ONLY.sub(" ", (text or "").lower())
    drop = set(_QUESTION_STOPWORDS)
    drop.update(_depluralize(w) for w in category.lower().split())
    tokens = {
        _depluralize(w) for w in s.split()
        if w and _depluralize(w) not in drop and w not in drop
    }
    return " ".join(sorted(tokens))


# Presentation markup a chat model wraps its answer in. Stripped BEFORE the
# contract regexes run, because those are line-anchored and any prefix defeats
# them: "<h3>QUESTION: Is the food animal-based?</h3>" is a perfectly good turn
# that v7 scored as a format failure and ended the episode over.
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_LATEX_WRAP = re.compile(r"\\(?:text|mathrm|textbf|mathbf)\s*\{([^{}]*)\}")
_LATEX_DELIM = re.compile(r"\\[()\[\]]|\$+")
# Leading/trailing markdown emphasis, heading marks, list bullets, quotes.
_EDGE_MARKUP = re.compile(
    r"^[\s>*_~`#\-+{\[\u2018\u201c\"']+|[\s*_~`}\]\u2019\u201d\"']+$")


def _strip_decoration(text: str) -> str:
    """Remove presentation markup while preserving line structure.

    The contract regexes are anchored with ``^``/``$`` per line, so a model that
    obeys the contract but decorates it — HTML, markdown emphasis, LaTeX math,
    surrounding quotes — was previously indistinguishable from one that ignored
    the contract entirely. In the v7 run that misread 200 of 632 format failures
    outright, and another 159 quoted questions ('"Is it a nut?"') that the bare-
    question fallback missed only because the line ends in a quote, not a '?'.
    """
    out = []
    for line in (text or "").splitlines():
        line = _HTML_TAG.sub(" ", line)
        line = _LATEX_WRAP.sub(r"\1", line)
        line = _LATEX_DELIM.sub("", line)
        line = _EDGE_MARKUP.sub("", line)
        out.append(line.strip())
    return "\n".join(out)


# A bare entity name on its own line, with no contract prefix: "Key lime pie",
# "Taco", "Truffle". 238 of v7's 632 format failures look like this, and in
# context they are unambiguously guesses — the prompt asked for a QUESTION or a
# GUESS and this is not a question. Reading it as a guess is also the SAFE
# error: a wrong guess costs one turn and play continues, whereas a format
# failure ends the episode outright. The guards below keep meta-commentary
# ("Let me think", "Okay, so...") from being fed to the matcher as a guess.
_BARE_ENTITY = re.compile(r"^[\w][\w '\-\.]{0,48}$")
_META_OPENERS = (
    "let", "i ", "i'", "im ", "ok", "okay", "sure", "hmm", "well", "so ",
    "now", "alright", "right", "yes", "no", "maybe", "based", "given",
    "the answer", "my ", "we ", "this ", "that ", "it ", "here",
)


def parse_guesser_turn(text: str) -> tuple[Optional[str], str]:
    """``(kind, content)`` from a raw guesser completion: kind is ``"guess"``
    (takes precedence — a turn that commits to a guess IS a guess), or
    ``"question"``, or ``None`` on format failure. Contract lines are searched
    on think-stripped, decoration-stripped text, so a QUESTION drafted inside
    <think> doesn't count but one wrapped in <h3> or ** does.

    The fallback ladder is ordered by how much it infers: explicit contract
    line, then a line that simply reads as a question, then a bare entity name
    read as a guess. Each rung only fires when every rung above it missed.
    """
    visible = _strip_decoration(strip_think(text or ""))
    guesses = _GUESS.findall(visible)
    if guesses:
        return "guess", guesses[-1].strip()
    questions = _QUESTION.findall(visible)
    if questions:
        return "question", questions[-1].strip()
    bare = _BARE_QUESTION.findall(visible)
    if bare:
        return "question", bare[-1].strip()
    inline_guesses = _GUESS_INLINE.findall(visible)
    if inline_guesses:
        return "guess", inline_guesses[-1].strip()
    inline_questions = _QUESTION_INLINE.findall(visible)
    if inline_questions:
        return "question", inline_questions[-1].strip()
    lines = [ln for ln in visible.splitlines() if ln.strip()]
    if len(lines) == 1:
        candidate = lines[0].strip()
        low = candidate.lower()
        if (_BARE_ENTITY.match(candidate)
                and len(candidate.split()) <= 5
                and not low.startswith(_META_OPENERS)):
            return "guess", candidate.rstrip(".")
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
    repeat: bool = False             # content-word duplicate of an earlier turn


@dataclass
class Episode:
    secret: Secret
    turns: list[Turn] = field(default_factory=list)
    guessed: bool = False
    ended: str = "budget"            # "guessed" | "budget" | "format"
    n_answer_format_fails: int = 0
    n_question_retries: int = 0      # turns resampled because they repeated
    n_repeat_turns: int = 0          # turns that repeated even after retries

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

    def asked_keys(self) -> set[str]:
        """Normalized content-word keys of every question already ASKED, plus
        every entity already GUESSED (a wrong guess is answered NO and is just
        as wasteful to repeat). Used by the dedup retry."""
        keys = set()
        for t in self.turns:
            if t.kind == "question":
                key = normalize_question(t.content, category=self.secret.category)
                if key:
                    keys.add(key)
            elif t.kind == "guess":
                key = normalize_guess(t.content)
                if key:
                    keys.add(f"\0guess\0{key}")
        return keys


def _repeat_key(kind: str, content: str, category: str) -> str:
    """The :meth:`Episode.asked_keys` key a fresh turn would occupy ("" when
    the turn carries no comparable content)."""
    if kind == "guess":
        key = normalize_guess(content)
        return f"\0guess\0{key}" if key else ""
    return normalize_question(content, category=category)


def run_episode(
    guesser_fn: Callable[[list[tuple[str, str]], int], Any],
    answerer_fn: Callable[[str, list[tuple[str, str]]], str],
    secret: Secret,
    *,
    max_turns: int,
    question_retries: int = 0,
) -> Episode:
    """Play one episode to termination. See module docstring for the rules.

    ``question_retries`` > 0 resamples a turn that repeats a question already
    asked (or an entity already guessed) in this episode, up to that many extra
    draws; the last draw is accepted whatever it is, so the turn is never lost.
    """
    ep = Episode(secret=secret)
    for i in range(max_turns):
        gen = guesser_fn(ep.qa_pairs, i)
        kind, content = parse_guesser_turn(gen.text)
        asked = ep.asked_keys()
        for _ in range(question_retries):
            key = _repeat_key(kind or "", content, secret.category)
            if kind is None or not key or key not in asked:
                break
            ep.n_question_retries += 1
            gen = guesser_fn(ep.qa_pairs, i)
            kind, content = parse_guesser_turn(gen.text)
        key = _repeat_key(kind or "", content, secret.category)
        is_repeat = bool(kind is not None and key and key in asked)
        if is_repeat:
            ep.n_repeat_turns += 1
        turn = Turn(
            index=i, kind=kind or "format_fail", content=content,
            answer=None, raw_text=gen.text,
            prompt_tokens=getattr(gen, "prompt_tokens", None),
            completion_tokens=getattr(gen, "completion_tokens", None),
            repeat=is_repeat,
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


def run_episodes_batched(
    guesser_batch_fn: Callable[[list[tuple[list[tuple[str, str]], int]]], list[Any]],
    answerer_batch_fn: Callable[[list[tuple[str, list[tuple[str, str]]]]], list[str]],
    secret: Secret,
    *,
    n_episodes: int,
    max_turns: int,
    question_retries: int = 0,
) -> list[Episode]:
    """Play sibling episodes in lockstep while preserving :func:`run_episode`.

    At turn ``t`` every still-active episode contributes one guesser prompt to
    a batch. Questions from that batch then form a separate answerer batch;
    correct guesses, wrong engine-answered guesses, format failures, and ragged
    termination retain the scalar engine's exact semantics.

    ``question_retries`` matches :func:`run_episode`. Retries are batched too:
    only the episodes that actually repeated are re-requested, so a resample
    round costs one small extra generation call rather than a full batch.
    """
    episodes = [Episode(secret=secret) for _ in range(n_episodes)]
    active = list(range(n_episodes))
    for turn_index in range(max_turns):
        if not active:
            break
        asked = {i: episodes[i].asked_keys() for i in active}
        resolved: dict[int, tuple] = {}
        pending = list(active)
        for attempt in range(question_retries + 1):
            if not pending:
                break
            requests = [(list(episodes[i].qa_pairs), turn_index) for i in pending]
            generations = list(guesser_batch_fn(requests))
            if len(generations) != len(pending):
                raise ValueError(
                    f"guesser batch returned {len(generations)} results "
                    f"for {len(pending)} active episodes")
            repeats: list[int] = []
            for episode_index, gen in zip(pending, generations):
                kind, content = parse_guesser_turn(gen.text)
                key = _repeat_key(kind or "", content, secret.category)
                is_repeat = bool(
                    kind is not None and key and key in asked[episode_index])
                # Latest draw always wins, so exhausting retries still spends
                # the turn on a real generation rather than dropping it.
                resolved[episode_index] = (gen, kind, content, is_repeat)
                if is_repeat:
                    if attempt < question_retries:
                        episodes[episode_index].n_question_retries += 1
                        repeats.append(episode_index)
                    else:
                        episodes[episode_index].n_repeat_turns += 1
            pending = repeats

        question_indices: list[int] = []
        answer_requests: list[tuple[str, list[tuple[str, str]]]] = []
        next_active: list[int] = []
        for episode_index in active:
            ep = episodes[episode_index]
            gen, kind, content, is_repeat = resolved[episode_index]
            turn = Turn(
                index=turn_index,
                kind=kind or "format_fail",
                content=content,
                answer=None,
                raw_text=gen.text,
                prompt_tokens=getattr(gen, "prompt_tokens", None),
                completion_tokens=getattr(gen, "completion_tokens", None),
                repeat=is_repeat,
            )
            ep.turns.append(turn)

            if kind is None:
                ep.ended = "format"
                continue
            if kind == "guess":
                if guess_matches(content, secret.secret):
                    ep.guessed = True
                    ep.ended = "guessed"
                    continue
                turn.answer = "NO"
                next_active.append(episode_index)
                continue

            question_indices.append(episode_index)
            answer_requests.append((content, list(ep.qa_pairs)))
            next_active.append(episode_index)

        if answer_requests:
            raw_answers = list(answerer_batch_fn(answer_requests))
            if len(raw_answers) != len(answer_requests):
                raise ValueError(
                    f"answerer batch returned {len(raw_answers)} results "
                    f"for {len(answer_requests)} questions")
            for episode_index, raw_answer in zip(question_indices, raw_answers):
                ep = episodes[episode_index]
                turn = ep.turns[-1]
                answer = parse_answer(raw_answer)
                if answer is None:
                    ep.n_answer_format_fails += 1
                    answer = "UNKNOWN"
                turn.answer = answer
                turn.answer_raw = raw_answer
                turn.creator_answered = True

        active = next_active

    return episodes
