"""Sprint 9 — grounded theme weighting (SPICE-style target grounding):
pick_theme's weighted draw + the bench-failure-profile builder."""

import importlib.util
import random
from pathlib import Path

from twin.config import Config
from twin.prompts import pick_theme
from twin.prompts.templates import THEMES

_ROOT = Path(__file__).resolve().parents[2]


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_grounded_themes", _ROOT / "scripts" / "build_grounded_themes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pick_theme_default_uniform_pool_unchanged():
    rng = random.Random(0)
    assert pick_theme("math", rng) in THEMES["math"]
    assert pick_theme("unknown-domain", rng) in THEMES["math"]


def test_pick_theme_weighted_draw():
    weights = {"math": [["number theory", 100.0], ["geometry", 0.0001]]}
    rng = random.Random(0)
    draws = {pick_theme("math", rng, weights=weights) for _ in range(50)}
    assert "number theory" in draws
    assert draws <= {"number theory", "geometry"}
    # overwhelmingly the heavy theme
    heavy = sum(1 for _ in range(200)
                if pick_theme("math", rng, weights=weights) == "number theory")
    assert heavy > 190


def test_pick_theme_falls_back_when_domain_missing_or_degenerate():
    rng = random.Random(1)
    assert pick_theme("math", rng, weights={"coding": [["x", 1.0]]}) in THEMES["math"]
    assert pick_theme("math", rng, weights={"math": []}) in THEMES["math"]
    assert pick_theme("math", rng, weights={"math": [["dead", 0.0]]}) in THEMES["math"]


def test_build_weights_failure_profile():
    mod = _load_builder()
    result = {"adapters": {"base": {"items": [
        {"category": "math", "subcategory": "Counting & Probability", "correct": False},
        {"category": "math", "subcategory": "Counting & Probability", "correct": False},
        {"category": "math", "subcategory": "Algebra", "correct": True},
        {"category": "math", "subcategory": "Algebra", "correct": False},
        {"category": "coding", "subcategory": "mbpp", "correct": False},
        {"category": "knowledge", "subcategory": "law", "correct": False},  # skipped
        {"category": "math", "subcategory": "", "correct": False},          # skipped
    ]}}}
    w = mod.build_weights(result, "base", floor=1.0)
    assert set(w) == {"math", "coding"}
    math_w = dict((t, x) for t, x in w["math"])
    assert math_w == {"counting and probability": 3.0, "algebra": 2.0}
    # sorted heaviest first
    assert w["math"][0][0] == "counting and probability"
    assert dict((t, x) for t, x in w["coding"]) == {"mbpp": 2.0}


def test_config_accepts_theme_weights_path():
    cfg = Config.from_dict({"game": {"theme_weights": "data/themes.json"}})
    assert cfg.game.theme_weights == "data/themes.json"
    assert Config.from_dict({}).game.theme_weights is None
