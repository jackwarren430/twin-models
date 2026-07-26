"""Thinking control (DESIGN §8): the reasoning-content extractor used by the
transcript, and the single place BaseTrainer resolves the thinking flag into
BOTH a template invitation and a decode-level ban."""

import pytest

from twin.config import Config
from twin.think import THINK_OPEN_MARKERS, strip_think, think_text
from twin.train.base import BaseTrainer


# ----- think_text -------------------------------------------------------------

def test_think_text_is_the_complement_of_strip_think():
    raw = "<think>weighing it up</think>ANSWER: YES"
    assert think_text(raw) == "weighing it up"
    assert strip_think(raw).strip() == "ANSWER: YES"


def test_think_text_handles_the_gemma_channel_family():
    raw = "<|channel>thought\nit is a mammal\n<channel|>ANSWER: YES"
    assert "it is a mammal" in think_text(raw)
    assert strip_think(raw).strip() == "ANSWER: YES"


def test_think_text_keeps_an_unclosed_truncation_spiral_visible():
    # think_share counts an unclosed span as thinking to the end; the
    # transcript must show it rather than drop it for lacking a close tag.
    assert think_text("<think>ran out of budget") == "ran out of budget"


def test_think_text_joins_multiple_spans_and_is_empty_when_absent():
    assert think_text("<think>a</think>mid<think>b</think>end") == "a\n\nb"
    assert think_text("QUESTION: Is it alive?") == ""
    assert think_text("") == "" and think_text(None) == ""


# ----- BaseTrainer resolution -------------------------------------------------

class _Base:
    """Records what the trainer asked the model to do."""

    def __init__(self):
        self.calls = []

    def render(self, user, system=None, enable_thinking=False):
        self.calls.append({"phase": "render", "enable_thinking": enable_thinking})
        return f"<prompt enable_thinking={enable_thinking}>"

    def generate(self, prompt, **kw):
        self.calls.append({"phase": "generate", **kw})
        return type("G", (), {"text": "ok", "prompt_tokens": [1],
                              "completion_tokens": [2]})()

    def generate_batch(self, prompts, **kw):
        self.calls.append({"phase": "generate_batch", **kw})
        return [type("G", (), {"text": "ok", "prompt_tokens": [1],
                               "completion_tokens": [2]})()
                for _ in prompts]


class _Adapters:
    NAMES = ["A", "B"]

    def activate(self, name):
        pass


def _trainer(enable_thinking=False):
    t = BaseTrainer.__new__(BaseTrainer)
    t.cfg = Config.from_dict({"model": {"enable_thinking": enable_thinking}})
    t.base = _Base()
    t.adapters = _Adapters()
    t.transcript = None
    return t


@pytest.mark.parametrize("think", [True, False])
def test_generate_applies_the_flag_to_template_and_decoder(think):
    t = _trainer()
    t._generate("A", "sys", "user", max_tokens=8, temp=0.7,
                enable_thinking=think)
    render, gen = t.base.calls
    assert render["enable_thinking"] is think
    # The enforcement half: suppression is exactly the negation, so the two
    # halves can never disagree about what the config asked for.
    assert gen["suppress_thinking"] is (not think)


@pytest.mark.parametrize("think", [True, False])
def test_generate_batch_applies_the_flag_to_template_and_decoder(think):
    t = _trainer()
    t._generate_batch("A", "sys", ["u1", "u2"], max_tokens=8, temp=0.7,
                      completion_batch_size=2, enable_thinking=think)
    renders = [c for c in t.base.calls if c["phase"] == "render"]
    gen = next(c for c in t.base.calls if c["phase"] == "generate_batch")
    assert all(r["enable_thinking"] is think for r in renders)
    assert gen["suppress_thinking"] is (not think)


def test_unset_role_override_follows_the_global_model_flag():
    for global_flag in (True, False):
        t = _trainer(enable_thinking=global_flag)
        t._generate("A", "sys", "user", max_tokens=8, temp=0.7)
        gen = t.base.calls[1]
        assert gen["suppress_thinking"] is (not global_flag)


def test_markers_are_the_shared_registry():
    # The backend bans exactly what the extractor recognises — one definition
    # of "thinking", so suppression and telemetry can never drift apart.
    assert "<think>" in THINK_OPEN_MARKERS
    assert "<|channel>" in THINK_OPEN_MARKERS
