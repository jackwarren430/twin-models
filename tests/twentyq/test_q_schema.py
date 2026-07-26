"""Sprint Q2 — Secret parsing + guess matching (twentyq/DESIGN.md §4 Q2)."""

import pytest

from twin.games.twentyq.schema import (
    Secret,
    SecretParseError,
    guess_matches,
    normalize_guess,
    parse_secret,
)


# ----- parse_secret ----------------------------------------------------------

def test_parse_plain_json():
    s = parse_secret('{"secret": "octopus", "category": "animal", "difficulty": 0.4}')
    assert s.secret == "octopus"
    assert s.category == "animal"
    assert s.difficulty == 0.4


def test_parse_fenced_and_prose_wrapped():
    text = (
        "Sure! Here is my secret for this game.\n"
        "```json\n"
        '{"secret": "Eiffel Tower", "category": "place", "difficulty": 0.8,\n'
        ' "notes": "landmark, but non-US guessers get it fast"}\n'
        "```\nGood luck."
    )
    s = parse_secret(text)
    assert s.secret == "Eiffel Tower"
    assert s.notes.startswith("landmark")


def test_parse_secrets_wrapper_takes_first():
    text = '{"secrets": [{"secret": "cat", "category": "animal", "difficulty": 0.1}, {"secret": "dog", "category": "animal", "difficulty": 0.2}]}'
    assert parse_secret(text).secret == "cat"


def test_parse_default_category():
    s = parse_secret('{"secret": "violin", "difficulty": 0.5}')
    assert s.category == "thing"
    s2 = parse_secret('{"secret": "violin", "category": "  ", "difficulty": 0.5}',
                      default_category="object")
    assert s2.category == "object"


@pytest.mark.parametrize("text", [
    "no json here at all",
    '{"category": "animal", "difficulty": 0.4}',            # missing secret
    '{"secret": "   ", "difficulty": 0.4}',                 # blank secret
    '{"secret": "cat", "difficulty": "hardish"}',           # non-numeric difficulty
    '{"secret": "cat"}',                                    # missing difficulty
    '{"secrets": []}',                                      # empty wrapper
])
def test_parse_failures_raise(text):
    with pytest.raises(SecretParseError):
        parse_secret(text)


def test_round_trip_dict():
    s = Secret(secret="octopus", category="animal", difficulty=0.4, notes="n")
    assert Secret.from_dict(s.to_dict()) == s


# ----- guess matching --------------------------------------------------------

def test_normalize_strips_articles_case_punct():
    assert normalize_guess("  The Octopus!  ") == "octopus"
    assert normalize_guess("an   apple") == "apple"
    assert normalize_guess("GUESS: it's a cat") != ""  # no crash on noise


@pytest.mark.parametrize("guess,secret,expect", [
    ("octopus", "octopus", True),
    ("The Octopus.", "octopus", True),
    ("cats", "cat", True),                # plural tolerance
    ("cat", "cats", True),
    ("Eiffel  Tower", "eiffel tower", True),
    ("automobile", "car", False),         # no synonyms in v1
    ("dog", "octopus", False),
    ("", "octopus", False),
    ("octopus", "", False),
    ("glasses", "glass", False),          # only exact single-'s' plurals match
])
def test_guess_matches(guess, secret, expect):
    assert guess_matches(guess, secret) is expect


@pytest.mark.parametrize("candidate,secret,expect", [
    # Everything guess_matches catches is a repeat too.
    ("The Okapis", "okapi", True),
    ("cats", "cat", True),
    # Ban-evasion misspellings observed live in q-fullv45-ctrl-terminal:
    # one-letter deletions of a banned attractor.
    ("Okpi", "Okapi", True),
    ("Wasbi", "Wasabi", True),
    ("Black Cardamon", "Black Cardamom", True),   # substitution
    # Unicode decoration (iteration 13): ASCII fold before edit distance.
    ("Axolotல்", "Axolotl", True),
    ("Ökapi", "Okapi", True),
    # Accents on genuinely different entities stay distinct.
    ("Açai berry", "Jalapeño", False),
    # Length floor: distinct short words stay distinct.
    ("tea", "pea", False),
    ("cat", "bat", False),
    # Legitimately adjacent entities remain playable.
    ("Truffle oil", "Truffle", False),
    ("Truffled Chocolate", "Truffle", False),
    ("Okapi", "Ocelot", False),
    ("", "Okapi", False),
])
def test_repeat_matches_catches_misspell_evasion(candidate, secret, expect):
    from twin.games.twentyq.schema import repeat_matches
    assert repeat_matches(candidate, secret) is expect


# ----- flat-mode difficulty steer must track the target ------------------------

def test_flat_dictation_steers_familiar_at_high_targets():
    """v7 asked for a 90% guess rate while the prompt unconditionally said
    'not so obvious', and the creator duly rejected winnable secrets by name."""
    from twin.games.twentyq.prompts import creator_secret_user
    user = creator_secret_user("animal", 0, 3, 0.1, 0.9, difficulty_mode="flat")
    assert "90% of games" in user
    assert "FAMILIAR" in user
    assert "not so obvious" not in user


def test_flat_dictation_steers_obscure_at_low_targets():
    from twin.games.twentyq.prompts import creator_secret_user
    user = creator_secret_user("animal", 0, 3, 0.9, 0.1, difficulty_mode="flat")
    assert "10% of games" in user
    assert "OBSCURE" in user


def test_flat_dictation_keeps_the_middle_steer_mid_range():
    from twin.games.twentyq.prompts import creator_secret_user
    user = creator_secret_user("animal", 0, 3, 0.5, 0.5, difficulty_mode="flat")
    assert "50% of games" in user
    assert "not so obvious" in user and "not so obscure" in user
    assert "FAMILIAR" not in user and "OBSCURE" not in user


def test_gradient_mode_dictation_is_untouched():
    from twin.games.twentyq.prompts import creator_secret_user
    user = creator_secret_user("animal", 1, 3, 0.5, 0.5, difficulty_mode="gradient")
    assert "Difficulty for this secret: 0.50" in user
    assert "FAMILIAR" not in user and "OBSCURE" not in user
