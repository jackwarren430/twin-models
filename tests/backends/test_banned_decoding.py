"""Logits-level phrase banning for the twentyq masked repeat-resample
(DESIGN §6.5): variant expansion, ``bad_words_ids`` encoding, and the
``generate_batch`` wiring that hands them to ``transformers.generate``."""

import torch

from twin.backends.torch_backend import (
    TorchTwinBase,
    banned_phrase_variants,
    banned_token_sequences,
)


def test_banned_phrase_variants_cover_case_space_and_plural():
    v = banned_phrase_variants("Okapi")
    for s in ("Okapi", "okapi", "OKAPI", "Okapis", " Okapi", " okapis"):
        assert s in v
    # Bare plural runs the other way too (one trailing 's' stripped).
    assert "extension cord" in banned_phrase_variants("Extension cords")
    assert banned_phrase_variants("") == []
    assert banned_phrase_variants("   ") == []


class _CharTok:
    """One token per character; enough to make encodings inspectable."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None,
                 padding=False):
        if return_tensors == "pt":
            ids = [[ord(c) for c in t] for t in text]
            width = max(len(x) for x in ids)
            return {
                "input_ids": torch.tensor(
                    [[0] * (width - len(x)) + x for x in ids]),
                "attention_mask": torch.tensor(
                    [[0] * (width - len(x)) + [1] * len(x) for x in ids]),
            }
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)


def test_banned_token_sequences_encode_dedupe_and_skip_empty():
    seqs = banned_token_sequences(_CharTok(), ["ab", "ab", "", "  "])
    tuples = {tuple(s) for s in seqs}
    assert tuple(ord(c) for c in "ab") in tuples
    assert tuple(ord(c) for c in " ab") in tuples
    assert tuple(ord(c) for c in "AB") in tuples
    assert len(seqs) == len(tuples)          # exact-duplicate sequences merged
    assert all(seq for seq in seqs)          # never an empty ban entry


class _RecordingModel:
    """Echoes the prompt plus a fixed two-token completion ('x' then EOS '!'),
    recording every generate() kwarg."""

    def __init__(self):
        self.calls = []

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.calls.append(kwargs)
        completion = torch.tensor([[ord("x"), ord("!")]] * input_ids.shape[0])
        return torch.cat([input_ids, completion], dim=1)


def _stub_base():
    b = TorchTwinBase.__new__(TorchTwinBase)
    b.tokenizer = _CharTok()
    b.model = _RecordingModel()
    b.device = torch.device("cpu")
    b._eos = {ord("!")}
    b._pad = 0
    return b


def test_generate_batch_passes_banned_token_sequences():
    b = _stub_base()
    out = b.generate_batch(["hi"], temp=0.7, banned_strings=["No"])
    kwargs = b.model.calls[0]
    assert kwargs["bad_words_ids"] is not None
    assert [ord(c) for c in "No"] in kwargs["bad_words_ids"]
    assert [ord(c) for c in " no"] in kwargs["bad_words_ids"]
    # Masking never changes the decode bookkeeping.
    assert out[0].completion_tokens == [ord("x"), ord("!")]
    assert out[0].text == "x"


def test_generate_and_batch_default_to_no_ban():
    b = _stub_base()
    b.generate("hi", temp=0.7)
    b.generate_batch(["hi", "yo"], temp=0.7)
    assert all(c["bad_words_ids"] is None for c in b.model.calls)
