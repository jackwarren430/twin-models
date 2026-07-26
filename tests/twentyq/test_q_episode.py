"""Sprint Q3 — episode engine with scripted players (twentyq/DESIGN.md §4 Q3).
No model anywhere: players are deterministic callables."""

from dataclasses import asdict
from dataclasses import dataclass, field

import pytest

from twin.games.twentyq.episode import (
    normalize_question,
    parse_answer,
    parse_guesser_turn,
    run_episode,
    run_episodes_batched,
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


# ----- lockstep sibling batching ---------------------------------------------

def test_batched_episode_engine_matches_scalar_with_ragged_termination():
    guesser_batches = iter([
        [FakeGen("QUESTION: Is it alive?"), FakeGen("GUESS: squid"),
         FakeGen("QUESTION: Is it alive?")],
        [FakeGen("GUESS: octopus"), FakeGen("Unable to determine anything from these answers"),
         FakeGen("QUESTION: Is it aquatic?")],
        [FakeGen("QUESTION: Does it have arms?")],
    ])
    answer_batches = iter([
        ["ANSWER: YES", "vague"],
        ["ANSWER: NO"],
        ["ANSWER: YES"],
    ])
    seen_batch_sizes = []

    def guesser_batch(requests):
        seen_batch_sizes.append(("g", len(requests)))
        return next(guesser_batches)

    def answerer_batch(requests):
        seen_batch_sizes.append(("a", len(requests)))
        return next(answer_batches)

    batched = run_episodes_batched(
        guesser_batch, answerer_batch, SECRET, n_episodes=3, max_turns=3)

    scalar_scripts = [
        (["QUESTION: Is it alive?", "GUESS: octopus"],
         ["ANSWER: YES"]),
        (["GUESS: squid", "Unable to determine anything from these answers"], []),
        (["QUESTION: Is it alive?", "QUESTION: Is it aquatic?",
          "QUESTION: Does it have arms?"],
         ["vague", "ANSWER: NO", "ANSWER: YES"]),
    ]
    scalar = []
    for guesses, answers in scalar_scripts:
        guess_iter, answer_iter = iter(guesses), iter(answers)
        scalar.append(run_episode(
            lambda qa, i, it=guess_iter: FakeGen(next(it)),
            lambda q, qa, it=answer_iter: next(it),
            SECRET, max_turns=3))

    assert [asdict(ep) for ep in batched] == [asdict(ep) for ep in scalar]
    assert seen_batch_sizes == [
        ("g", 3), ("a", 2), ("g", 3), ("a", 1), ("g", 1), ("a", 1)]


def test_batched_episode_engine_rejects_wrong_result_count():
    with pytest.raises(ValueError, match="guesser batch returned"):
        run_episodes_batched(
            lambda requests: [], lambda requests: [], SECRET,
            n_episodes=2, max_turns=1)


# ----- question de-duplication (repeat retry) --------------------------------

def test_normalize_question_collapses_surface_variants():
    key = lambda t: normalize_question(t, category="animal")
    # The v7 parrot lock: same content word, three phrasings, one key.
    assert key("Is the animal a parrot?") == key("Is it a parrot?")
    assert key("Is it a parrot?") == key("Are they parrots?")
    # Hedges and the category noun are not content.
    assert key("Is the animal usually a parrot?") == key("Is it a parrot?")
    # Genuinely different questions stay apart.
    assert key("Is it a parrot?") != key("Is it a penguin?")
    assert key("Is it bigger than a cat?") != key("Is it smaller than a cat?")
    # Multi-word categories drop wholly.
    assert normalize_question("Is the household object made of metal?",
                              category="household object") == "made metal"


def test_normalize_question_ignores_pure_punctuation_and_case():
    assert (normalize_question("IS IT A PARROT???", category="animal")
            == normalize_question("is it, a parrot", category="animal"))


def test_repeat_retry_resamples_until_question_is_new():
    draws = iter([
        "QUESTION: Is it a parrot?",     # turn 1: fresh
        "QUESTION: Is the animal a parrot?",  # turn 2: repeat -> retry
        "QUESTION: Are they parrots?",        # still a repeat -> retry
        "QUESTION: Is it aquatic?",           # fresh, accepted
    ])
    ep = run_episode(
        lambda qa, i: FakeGen(next(draws)),
        yes_answerer, SECRET, max_turns=2, question_retries=3)
    assert [t.content for t in ep.turns] == ["Is it a parrot?", "Is it aquatic?"]
    assert ep.n_question_retries == 2
    assert ep.n_repeat_turns == 0


def test_repeat_retry_accepts_last_draw_when_retries_exhausted():
    draws = iter(["QUESTION: Is it a parrot?"] * 4)
    ep = run_episode(
        lambda qa, i: FakeGen(next(draws)),
        yes_answerer, SECRET, max_turns=2, question_retries=2)
    # Turn is spent, never dropped: the exhausted draw still plays.
    assert [t.content for t in ep.turns] == ["Is it a parrot?", "Is it a parrot?"]
    assert ep.n_question_retries == 2
    assert ep.n_repeat_turns == 1


def test_repeat_retry_covers_repeated_guesses():
    draws = iter([
        "GUESS: squid",     # wrong, engine answers NO
        "GUESS: squid",     # repeat of a dead guess -> retry
        "GUESS: octopus",   # fresh and correct
    ])
    ep = run_episode(
        lambda qa, i: FakeGen(next(draws)),
        yes_answerer, SECRET, max_turns=3, question_retries=1)
    assert ep.guessed and ep.ended == "guessed"
    assert ep.n_question_retries == 1


def test_repeat_retry_does_not_fire_on_format_failure():
    draws = iter(["QUESTION: Is it a parrot?", "no contract line here"])
    ep = run_episode(
        lambda qa, i: FakeGen(next(draws)),
        yes_answerer, SECRET, max_turns=3, question_retries=3)
    assert ep.ended == "format"
    assert ep.n_question_retries == 0


def test_batched_repeat_retry_reissues_only_the_repeating_episodes():
    batches = iter([
        # turn 0: both fresh
        [FakeGen("QUESTION: Is it a parrot?"), FakeGen("QUESTION: Is it aquatic?")],
        # turn 1: ep0 repeats, ep1 fresh
        [FakeGen("QUESTION: Is the animal a parrot?"), FakeGen("QUESTION: Does it swim?")],
        # retry round: ONLY ep0 is re-requested
        [FakeGen("QUESTION: Is it a penguin?")],
    ])
    sizes = []

    def guesser_batch(requests):
        sizes.append(len(requests))
        return next(batches)

    eps = run_episodes_batched(
        guesser_batch, lambda reqs: ["ANSWER: YES"] * len(reqs),
        SECRET, n_episodes=2, max_turns=2, question_retries=1)

    assert sizes == [2, 2, 1]
    assert [t.content for t in eps[0].turns] == ["Is it a parrot?", "Is it a penguin?"]
    assert [t.content for t in eps[1].turns] == ["Is it aquatic?", "Does it swim?"]
    assert eps[0].n_question_retries == 1
    assert eps[1].n_question_retries == 0


def test_batched_repeat_retry_matches_scalar_engine():
    lines = ["QUESTION: Is it a parrot?", "QUESTION: Is it a parrot?",
             "QUESTION: Is it aquatic?"]

    batch_iter = iter(lines)
    batched = run_episodes_batched(
        lambda reqs: [FakeGen(next(batch_iter))],
        lambda reqs: ["ANSWER: YES"] * len(reqs),
        SECRET, n_episodes=1, max_turns=2, question_retries=1)

    scalar_iter = iter(lines)
    scalar = run_episode(
        lambda qa, i: FakeGen(next(scalar_iter)),
        yes_answerer, SECRET, max_turns=2, question_retries=1)

    assert asdict(batched[0]) == asdict(scalar)


def test_question_retries_default_is_off():
    draws = iter(["QUESTION: Is it a parrot?"] * 3)
    ep = run_episode(
        lambda qa, i: FakeGen(next(draws)), yes_answerer, SECRET, max_turns=3)
    assert ep.n_question_retries == 0
    assert ep.n_repeat_turns == 2
    assert [t.content for t in ep.turns] == ["Is it a parrot?"] * 3


# ----- decoration-tolerant contract parsing -----------------------------------
# Every raw string below is a VERBATIM guesser completion from the v7 run that
# the old parser scored as a format failure, ending the episode. 558 of the
# run's 632 format failures are of these shapes.

@pytest.mark.parametrize("raw,expected", [
    ("<h3>QUESTION: Is the food animal-based?</h3>", "Is the food animal-based?"),
    ("<strong>QUESTION: Is it a type of cured meat?</strong>",
     "Is it a type of cured meat?"),
    ("**QUESTION: Is it a processed food?**", "Is it a processed food?"),
    ("$\\text{QUESTION: Is the food a cooked item?}$", "Is the food a cooked item?"),
    ("{QUESTION: Is it a liquid?}", "Is it a liquid?"),
    ('"Is it a nut?"', "Is it a nut?"),
    ("*Is the food item a fruit?*", "Is the food item a fruit?"),
])
def test_parse_recovers_decorated_questions(raw, expected):
    assert parse_guesser_turn(raw) == ("question", expected)


@pytest.mark.parametrize("raw,expected", [
    ("Key lime pie", "Key lime pie"),
    ("Taco", "Taco"),
    ("Truffle", "Truffle"),
    ("Closure: GUESS: clear broth", "clear broth"),
])
def test_parse_recovers_bare_and_inline_guesses(raw, expected):
    assert parse_guesser_turn(raw) == ("guess", expected)


@pytest.mark.parametrize("raw", [
    "",
    "Key: The previous questions have established that the food is edible,",
    "Not a specific food item yet.",
    "Let's see if it's a fruit.",
    "I give up, I have no idea.",
    "Unable to determine anything from these answers",
])
def test_parse_still_rejects_genuine_non_answers(raw):
    assert parse_guesser_turn(raw) == (None, "")


def test_bare_entity_guess_is_a_last_resort_only():
    """The ladder must not let the bare-entity rung reinterpret a good line."""
    # A contract line anywhere wins over the bare-entity reading.
    assert parse_guesser_turn("GUESS: octopus") == ("guess", "octopus")
    # Multi-line output is never read as a bare entity.
    assert parse_guesser_turn("Squid\nOctopus") == (None, "")
    # Nor is anything long enough to be prose.
    assert parse_guesser_turn("a b c d e f") == (None, "")


def test_decoration_stripping_preserves_inner_punctuation():
    # Apostrophes and hyphens inside the entity must survive edge-stripping.
    assert parse_guesser_turn("**GUESS: shepherd's pie**") == ("guess", "shepherd's pie")
    assert parse_guesser_turn("Ice-cream") == ("guess", "Ice-cream")


def test_full_width_question_mark_is_a_question():
    assert parse_guesser_turn("それは果物ですか？") == ("question", "それは果物ですか？")
