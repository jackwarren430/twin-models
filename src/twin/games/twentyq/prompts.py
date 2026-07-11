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
secret. Answer each question about it TRUTHFULLY. A neutral judge audits every \
answer against the secret afterwards; a false or misleading answer voids the whole \
game and costs you. End your reply with exactly one line:
ANSWER: YES   or   ANSWER: NO   or   ANSWER: SOMETIMES   or   ANSWER: UNKNOWN"""

GUESSER_SYSTEM = """You are the guesser in a game of 20 questions. A secret has \
been chosen; you probe it with yes/no questions and win by naming it within the \
turn budget — the fewer turns, the better. Ask about broad splits first, then \
narrow. Never repeat a question the transcript already answers. Each turn, end \
your reply with exactly ONE line, either:
QUESTION: <one yes/no question>
or, when you are ready to name the secret:
GUESS: <the secret>"""


def creator_secret_user(
    category: str,
    rank: int,
    n_secrets: int,
    difficulty: float,
    target_rate: float,
    previous: list[str] | None = None,
) -> str:
    prev_block = ""
    if previous:
        prev_block = (
            "\nSecrets you already picked this round (yours must be clearly "
            "different from all of them):\n"
            + "\n".join(f"- {p}" for p in previous) + "\n"
        )
    return f"""Pick secret {rank + 1} of {n_secrets} for this round. Category: {category}.

Difficulty for this secret: {difficulty:.2f} on a 0-1 scale (0 = something anyone \
names in a few questions; 1 = something rarely identified within 20). Target: the \
guesser should succeed on about {int(round(target_rate * 100))}% of games at this \
secret.
{prev_block}
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
        transcript = "\n".join(
            f"{i + 1}. Q: {q} -> {a}" for i, (q, a) in enumerate(qa_pairs)
        )
    else:
        transcript = "(none yet — this is your first question)"
    remaining = max_turns - turn_index
    if remaining <= 1:
        instruction = ("This is your LAST turn: you MUST end with a GUESS: line "
                       "naming the secret.")
    else:
        instruction = (f"You have {remaining} turns left (this one included). "
                       "End with one QUESTION: line, or a GUESS: line if you are "
                       "confident.")
    return f"""The secret is a {category}.

Questions so far:
{transcript}

{instruction}"""


def secret_json_for_conditioning(secret: str, category: str, difficulty: float) -> str:
    """The compact JSON shown to later ranks (never the creator's notes —
    same "JSONs, never thinking" rule as per-problem conditioning)."""
    return json.dumps({"secret": secret, "category": category,
                       "difficulty": difficulty})
