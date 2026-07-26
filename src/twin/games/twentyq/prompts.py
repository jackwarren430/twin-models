"""Prompt templates for 21 questions (Sprint Q5).

The guesser prompt is a SINGLE flattened user message (rules + Q/A transcript
+ instruction), not a multi-turn chat render. That keeps the per-turn
trajectory honest by construction — the prompt IS exactly what the model
conditioned on, with no chat-template think-stripping between "what was
generated" and "what later turns saw" (twentyq/DESIGN.md §2.2) — and it
reuses the plain single-turn ``BaseTrainer._generate`` path unchanged.
"""

import json

CREATOR_SYSTEM = """You are the secret-setter in a game of 20 questions. You pick \
a secret; an opponent of about your own ability then tries to identify it with \
yes/no questions. Your goal is CALIBRATION, not maximum difficulty: each secret \
comes with a target guess rate, and you score highest when the opponent's actual \
success rate hits that target. Secrets must be real, commonly known, unambiguous, \
singular instances of the requested category, and must clearly belong to that \
category — a borderline edge case (e.g. water as a "food"), an unguessable, or \
an invalid secret scores zero. Decide directly on ONE secret; do not deliberate \
over many candidates. Reply with exactly one JSON object and nothing else."""

ANSWERER_SYSTEM = """You are the answerer in a game of 20 questions. You know the \
secret. Answer each question about it TRUTHFULLY and accurately — a correct game \
depends on it. End your reply with exactly one line:
ANSWER: YES   or   ANSWER: NO   or   ANSWER: SOMETIMES   or   ANSWER: UNKNOWN"""

JUDGE_SYSTEM = """You are a careful, neutral judge for a game of 20 questions. \
Read the task and follow it exactly. Do not use any tools. Base your ruling only \
on the information given. End your reply with exactly the single labelled line the \
task asks for (VERDICT, AUDIT, or CLOSENESS) and nothing after it."""

GUESSER_SYSTEM = """You are the guesser in a game of 20 questions. A secret has \
been chosen; you probe it with yes/no questions and win by naming it within the \
turn budget — the fewer turns, the better. Ask about broad splits first, then \
narrow. Never repeat a question the transcript already answers. A wrong GUESS \
does not end the game: it costs one turn, you are told NO, and play continues — \
so name your best candidate whenever the answers point to one.

Your ENTIRE reply must be exactly one line, either:
QUESTION: <one yes/no question>
or:
GUESS: <the secret>
Write nothing before or after that line."""


def creator_secret_user(
    category: str,
    rank: int,
    n_secrets: int,
    difficulty: float,
    target_rate: float,
    previous: list[str] | None = None,
    recent: list[str] | None = None,
    *,
    difficulty_mode: str = "gradient",
) -> str:
    """``difficulty_mode`` selects the dictation block (DESIGN §7.4).

    "gradient" states this rank's own point on the 0-1 difficulty scale — the
    prompt that builds an easy->hard ladder across the round. "flat" states
    that every secret this round shares one target rate, so the rank number
    orders the round (and anchors the exclusion list) without implying a ramp.
    Everything else — the round framing, the exclusion block, the JSON
    contract line — is identical in both modes.
    """
    exclusions = list(recent or []) + list(previous or [])
    exclusion_block = ""
    if exclusions:
        # Names only (not prior JSON objects) keep this bounded block compact.
        # The caller has already category-filtered the rolling-history entries.
        exclusions = list(dict.fromkeys(exclusions))
        exclusion_block = (
            "\nExcluded secrets (recently used in this category or already "
            "picked this round):\n"
            + "\n".join(f"- {p}" for p in exclusions)
            + "\nChoose neither an exact repeat nor an obvious variant of any "
            "item above (such as a plural, alternate spelling, or qualified "
            "version).\n"
        )
    if difficulty_mode == "flat":
        # The steer must TRACK the target. Until 2026-07-26 this block ended
        # with a fixed "not so obvious ... not so obscure" clause, i.e. an
        # unconditional aim-for-the-middle instruction that flatly contradicts
        # its own sentence at extreme targets. The v7 run asked for a 90% guess
        # rate and got creator notes rejecting winnable secrets BY NAME ("known
        # but not universally recognized, like a tiger or elephant") — the
        # prompt told it to, and the run's bank came out harder than the 50%
        # one it replaced.
        if target_rate >= 0.75:
            steer = ("Err toward the FAMILIAR: a common, instantly "
                     "recognizable member of the category, the kind of thing "
                     "most people would name in the first handful of guesses.")
        elif target_rate <= 0.25:
            steer = ("Err toward the OBSCURE: something a well-read person "
                     "would recognize but rarely think of early.")
        else:
            steer = ("Pitch it in the middle — not so obvious it is named in "
                     "a few questions, not so obscure it is rarely identified "
                     "within 20.")
        dictation = (
            f"Every secret this round has the SAME target: the guesser should "
            f"succeed on about {int(round(target_rate * 100))}% of games. "
            f"{steer}"
        )
    elif difficulty_mode == "gradient":
        dictation = (
            f"Difficulty for this secret: {difficulty:.2f} on a 0-1 scale "
            f"(0 = something anyone names in a few questions; 1 = something "
            f"rarely identified within 20). Target: the guesser should succeed "
            f"on about {int(round(target_rate * 100))}% of games at this secret."
        )
    else:
        raise ValueError(f"unknown difficulty_mode: {difficulty_mode!r} "
                         "(expected gradient | flat)")
    return f"""Pick secret {rank + 1} of {n_secrets} for this round. Category: {category}.

{dictation}
{exclusion_block}
Reply with exactly one JSON object:
{{"secret": "<the entity>", "category": "{category}", "difficulty": {difficulty:.2f}, "notes": "<one line: why this fits the target>"}}"""


def answerer_user(secret: str, category: str, question: str) -> str:
    return f"""The secret is: {secret}  (category: {category})

Question: {question}

Answer truthfully for the secret."""


def guesser_user(
    category: str,
    qa_pairs: list[tuple[str, str]],
    turn_index: int,
    max_turns: int,
) -> str:
    if qa_pairs:
        # Deliberately avoid a "Q:" prefix here: weaker guessers mimic the
        # transcript's notation and emit "Q: ..." instead of the "QUESTION:"
        # contract line (observed on gemma4-E2B). Quote-and-arrow carries no
        # prefix token to copy.
        transcript = "\n".join(
            f'{i + 1}. "{q}" -> {a}' for i, (q, a) in enumerate(qa_pairs)
        )
    else:
        transcript = "(none yet — this is your first question)"
    remaining = max_turns - turn_index
    if remaining <= 1:
        instruction = ("This is your LAST turn: reply with exactly one GUESS: "
                       "line naming the secret most consistent with ALL the "
                       "answers above.")
    else:
        instruction = (f"You have {remaining} turns left (this one included). "
                       "Reply with exactly one QUESTION: line, or one GUESS: "
                       "line if you have a strong candidate.")
    # "Category: X" (not "The secret is a X."): article-free, so it stays
    # grammatical for any category string. "The secret is a animal." measurably
    # destabilized gemma4-E2B — 83% of animal episodes opened with a degenerate
    # parsed question vs 2% for the grammatical "a household object" (v4/v4.5/v5
    # transcript audit, DESIGN §6.8).
    return f"""The secret is in the category: {category}.

Questions so far:
{transcript}

{instruction}"""


def secret_json_for_conditioning(secret: str, category: str, difficulty: float) -> str:
    """The compact JSON shown to later ranks (never the creator's notes —
    same "JSONs, never thinking" rule as per-problem conditioning)."""
    return json.dumps({"secret": secret, "category": category,
                       "difficulty": difficulty})
