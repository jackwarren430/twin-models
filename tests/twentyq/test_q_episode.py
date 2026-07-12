"""Sprint Q3 — episode engine with scripted players (twentyq/DESIGN.md §4 Q3).
No model anywhere: players are deterministic callables."""

from dataclasses import dataclass, field

from twin.games.twentyq.episode import (
    parse_answer,
    parse_guesser_turn,
    run_episode,
)
from twin.games.twentyq.schema import Secret

SECRET = Secret(secret="octopus", category="animal", difficulty=0.4)


@dataclass
class FakeGen:
    text: str
    prompt_tokens: list = field(default_factory=lambda: [1, 2])
    completion_tokens: list = field(default_factory=lambda: [3, 4])


def scripted_guesser(lines):
    """Guesser that plays the given raw completions in order and records the
    qa_pairs it was shown each turn."""
    seen = []

    def fn(qa_pairs, turn_index):
        seen.append(list(qa_pairs))
        return FakeGen(lines[turn_index])

    fn.seen = seen
    return fn


def yes_answerer(question, qa_pairs):
    return "ANSWER: YES"


# ----- contract parsing ------------------------------------------------------

def test_parse_guesser_question_and_guess():
    assert parse_guesser_turn("QUESTION: Is it alive?") == ("question", "Is it alive?")
    assert parse_guesser_turn("GUESS: the octopus") == ("guess", "the octopus")
    assert parse_guesser_turn("FINAL GUESS: octopus") == ("guess", "octopus")


def test_parse_guesser_guess_takes_precedence():
    text = "QUESTION: Is it a cephalopod?\nActually I'm sure now.\nGUESS: octopus"
    assert parse_guesser_turn(text) == ("guess", "octopus")


def test_parse_guesser_last_line_wins():
    text = "QUESTION: Is it big?\nhmm no wait\nQUESTION: Is it small?"
    assert parse_guesser_turn(text) == ("question", "Is it small?")


def test_parse_guesser_lenient_q_prefix_and_bare_question():
    # Weaker models (gemma4-E2B) drop the "QUESTION:" contract, emitting "Q:"
    # (mimicking the history) or a bare question line. Both parse as a question
    # so the episode stays alive; genuine non-question text still fails.
    assert parse_guesser_turn("Q: Is it a fruit?") == ("question", "Is it a fruit?")
    assert parse_guesser_turn("Is the food sweet?") == ("question", "Is the food sweet?")
    assert parse_guesser_turn("I really can't tell what it is.") == (None, "")
    # a real GUESS still wins over a trailing bare question
    assert parse_guesser_turn("Could it be a squid?\nGUESS: octopus") == ("guess", "octopus")


def test_parse_guesser_ignores_thinking():
    only_think = "<think>QUESTION: Is it alive?</think>I wonder..."
    assert parse_guesser_turn(only_think) == (None, "")
    after_think = "<think>drafting: GUESS: squid</think>QUESTION: Is it alive?"
    assert parse_guesser_turn(after_think) == ("question", "Is it alive?")


def test_parse_answer_variants():
    assert parse_answer("ANSWER: YES") == "YES"
    assert parse_answer("Well...\nanswer: sometimes") == "SOMETIMES"
    assert parse_answer("<think>it is</think>ANSWER: no") == "NO"
    assert parse_answer("it depends") is None
    assert parse_answer("") is None


# ----- termination -----------------------------------------------------------

def test_correct_guess_terminates():
    g = scripted_guesser([
        "QUESTION: Is it an animal?",
        "GUESS: The Octopus.",          # normalization handles article + punct
    ])
    ep = run_episode(g, yes_answerer, SECRET, max_turns=6)
    assert ep.guessed and ep.ended == "guessed"
    assert ep.turns_used == 2
    assert ep.turns[-1].kind == "guess" and ep.turns[-1].answer is None


def test_budget_exhaustion():
    g = scripted_guesser(["QUESTION: Is it alive?"] * 3)
    ep = run_episode(g, yes_answerer, SECRET, max_turns=3)
    assert not ep.guessed and ep.ended == "budget"
    assert ep.turns_used == 3


def test_format_failure_ends_episode_as_failure():
    # A genuine non-question statement (no "?" and no contract line) fails; a
    # bare question would now be accepted (see the lenient-parse test above).
    g = scripted_guesser(["QUESTION: Is it alive?", "I give up, I have no idea."])
    ep = run_episode(g, yes_answerer, SECRET, max_turns=6)
    assert not ep.guessed and ep.ended == "format"
    assert ep.turns_used == 2
    assert ep.turns[-1].kind == "format_fail"


def test_wrong_guess_consumes_turn_and_continues():
    g = scripted_guesser([
        "GUESS: squid",
        "QUESTION: Does it have eight arms?",
        "GUESS: octopus",
    ])
    ep = run_episode(g, yes_answerer, SECRET, max_turns=6)
    assert ep.guessed and ep.turns_used == 3
    wrong = ep.turns[0]
    assert wrong.kind == "guess" and wrong.answer == "NO"
    assert not wrong.creator_answered            # engine-authored ground truth


# ----- history bookkeeping ---------------------------------------------------

def test_guesser_sees_accumulated_history():
    g = scripted_guesser([
        "QUESTION: Is it alive?",
        "GUESS: squid",
        "QUESTION: Is it a mollusc?",
    ])
    ep = run_episode(g, yes_answerer, SECRET, max_turns=3)
    assert g.seen[0] == []
    assert g.seen[1] == [("Is it alive?", "YES")]
    assert g.seen[2] == [("Is it alive?", "YES"), ("Is it squid?", "NO")]
    assert ep.qa_pairs == [
        ("Is it alive?", "YES"),
        ("Is it squid?", "NO"),
        ("Is it a mollusc?", "YES"),
    ]


def test_audited_pairs_exclude_engine_answers():
    g = scripted_guesser([
        "QUESTION: Is it alive?",
        "GUESS: squid",
        "QUESTION: Is it a mollusc?",
    ])
    ep = run_episode(g, yes_answerer, SECRET, max_turns=3)
    assert ep.audited_pairs == [
        ("Is it alive?", "YES"),
        ("Is it a mollusc?", "YES"),
    ]


def test_unparseable_answer_recorded_unknown_and_still_audited():
    def vague_answerer(question, qa_pairs):
        return "hmm, that's hard to say"

    g = scripted_guesser(["QUESTION: Is it alive?", "QUESTION: Is it heavy?"])
    ep = run_episode(g, vague_answerer, SECRET, max_turns=2)
    assert ep.n_answer_format_fails == 2
    assert ep.qa_pairs == [("Is it alive?", "UNKNOWN"), ("Is it heavy?", "UNKNOWN")]
    assert ep.audited_pairs == ep.qa_pairs      # creator-authored, judge sees them


def test_turn_carries_tokens_and_raw_text():
    g = scripted_guesser(["<think>hmm</think>QUESTION: Is it alive?"])
    ep = run_episode(g, yes_answerer, SECRET, max_turns=1)
    t = ep.turns[0]
    assert t.prompt_tokens == [1, 2] and t.completion_tokens == [3, 4]
    assert "<think>" in t.raw_text               # raw completion, thinking intact


def test_answerer_receives_question_text():
    seen = {}

    def answerer(question, qa_pairs):
        seen["q"] = question
        return "ANSWER: YES"

    g = scripted_guesser(["QUESTION: Is it alive?"])
    run_episode(g, answerer, SECRET, max_turns=1)
    assert seen["q"] == "Is it alive?"
