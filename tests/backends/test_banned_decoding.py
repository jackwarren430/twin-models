"""Logits-level phrase banning for the twentyq masked repeat-resample
(DESIGN §6.5b): variant expansion, the STRING-level BannedStringsProcessor
(robust to alternate BPE segmentations — the leak bad_words_ids has), and the
``generate_batch`` wiring."""

import torch

from twin.backends.torch_backend import (
    BannedStringsProcessor,
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
    """One token per printable ASCII code point; enough to make encodings and
    surfaces inspectable."""

    pad_token_id = 0
    name_or_path = "test-char-tok"

    def __len__(self):
        return 128

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

    def batch_decode(self, seqs, skip_special_tokens=False):
        return [self.decode(ids) for ids in seqs]


class _WordTok(_CharTok):
    """Multi-character vocab so one string has several token segmentations —
    the shape of the live 'Black Cardamom' leak."""

    name_or_path = "test-word-tok"
    VOCAB = ["<pad>", "card", "amom", "am", "om", "oms", "x", "cardamom", "n"]

    def __len__(self):
        return len(self.VOCAB)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.VOCAB[int(i)] for i in ids)


def test_banned_token_sequences_encode_dedupe_and_skip_empty():
    seqs = banned_token_sequences(_CharTok(), ["ab", "ab", "", "  "])
    tuples = {tuple(s) for s in seqs}
    assert tuple(ord(c) for c in "ab") in tuples
    assert tuple(ord(c) for c in " ab") in tuples
    assert tuple(ord(c) for c in "AB") in tuples
    assert len(seqs) == len(tuples)          # exact-duplicate sequences merged
    assert all(seq for seq in seqs)          # never an empty ban entry


def _scores(tok):
    return torch.zeros((1, len(tok)))


def test_string_ban_blocks_the_completing_token_case_insensitively():
    tok = _CharTok()
    proc = BannedStringsProcessor(tok, ["cardamom"], prompt_len=0)
    # Tail "CARDAMO": only 'm'/'M' complete the phrase; 'n' stays legal.
    ids = torch.tensor([[ord(c) for c in "CARDAMO"]])
    scores = proc(ids, _scores(tok))
    assert scores[0, ord("m")] == float("-inf")
    assert scores[0, ord("M")] == float("-inf")
    assert scores[0, ord("n")] == 0.0
    # Unrelated tail: nothing banned (no single char token completes it).
    scores = proc(torch.tensor([[ord("z")]]), _scores(tok))
    assert torch.isfinite(scores).all()


def test_string_ban_blocks_every_segmentation():
    # The live q-fullv45 leak: bad_words_ids banned ['card','amom'], so the
    # sampler rerouted through ['card','am','om'] and emitted the exact banned
    # string. The string-level processor blocks BOTH paths and the
    # whole-phrase single token.
    tok = _WordTok()
    proc = BannedStringsProcessor(tok, ["cardamom"], prompt_len=0)
    v = tok.VOCAB.index

    # After 'card': 'amom' completes; 'cardamom'-in-one-token banned always.
    scores = proc(torch.tensor([[v("card")]]), _scores(tok))
    assert scores[0, v("amom")] == float("-inf")
    assert scores[0, v("am")] == 0.0          # not yet completing: allowed
    # After 'card','am': 'om' AND 'oms' both complete -> both banned.
    scores = proc(torch.tensor([[v("card"), v("am")]]), _scores(tok))
    assert scores[0, v("om")] == float("-inf")
    assert scores[0, v("oms")] == float("-inf")
    assert scores[0, v("n")] == 0.0
    # Empty tail: the whole-phrase token is still banned (k=0 remainder).
    scores = proc(torch.tensor([[v("x")]]), _scores(tok))
    assert scores[0, v("cardamom")] == float("-inf")


def test_string_ban_only_inspects_generated_tail_not_prompt():
    # The prompt itself CONTAINS the exclusion list; a phrase prefix that ends
    # the prompt must not poison the first generated step.
    tok = _CharTok()
    proc = BannedStringsProcessor(tok, ["cardamom"], prompt_len=7)
    ids = torch.tensor([[ord(c) for c in "cardamo"]])   # all prompt
    scores = proc(ids, _scores(tok))
    # k=0 empty-prefix ban only bans tokens starting the WHOLE phrase — no
    # single char does — so completing 'm' stays legal at step 0.
    assert scores[0, ord("m")] == 0.0


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


def test_generate_batch_passes_string_ban_processor():
    b = _stub_base()
    out = b.generate_batch(["hi"], temp=0.7, banned_strings=["No"])
    kwargs = b.model.calls[0]
    procs = kwargs["logits_processor"]
    assert procs is not None and len(procs) == 1
    assert isinstance(procs[0], BannedStringsProcessor)
    assert "no" in procs[0].phrases and "nos" in procs[0].phrases
    assert procs[0].prompt_len == 2
    # Masking never changes the decode bookkeeping.
    assert out[0].completion_tokens == [ord("x"), ord("!")]
    assert out[0].text == "x"


def test_generate_and_batch_default_to_no_ban():
    b = _stub_base()
    b.generate("hi", temp=0.7)
    b.generate_batch(["hi", "yo"], temp=0.7)
    assert all(c["logits_processor"] is None for c in b.model.calls)
