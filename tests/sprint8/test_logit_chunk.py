"""Sprint 8 — chunked+checkpointed completion_logprobs equivalence.

The chunked path must return the same values AND the same gradients as the
plain [T, V] path (the head carries no trainable params, so checkpointing it
is pure rematerialization). Tested on a tiny synthetic model shaped like
mlx-lm's Qwen3 ``Model`` (inner ``.model`` returning hidden states + a head,
both tied and untied variants) — no weights on disk needed."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from twin.models.base import TwinBase

V, H = 29, 8


class _Inner(nn.Module):
    """Stands in for Qwen3Model: embeds ids, mixes them causally-ish."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, H)
        self.proj = nn.Linear(H, H)

    def __call__(self, ids, cache=None, input_embeddings=None):
        h = self.embed_tokens(ids)
        # cheap position mixing so different prefixes give different states
        return self.proj(h) + 0.01 * mx.cumsum(h, axis=1)


class _Untied(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = SimpleNamespace(tie_word_embeddings=False)
        self.model = _Inner()
        self.lm_head = nn.Linear(H, V, bias=False)

    def __call__(self, ids, cache=None, input_embeddings=None):
        return self.lm_head(self.model(ids))


class _Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = SimpleNamespace(tie_word_embeddings=True)
        self.model = _Inner()

    def __call__(self, ids, cache=None, input_embeddings=None):
        return self.model.embed_tokens.as_linear(self.model(ids))


def _base(model) -> TwinBase:
    tb = TwinBase.__new__(TwinBase)
    tb.model = model
    return tb


_PROMPT = [1, 2, 3, 4]
_COMPLETION = [5, 6, 7, 8, 9, 10, 11]     # 7 tokens: chunk=3 -> 3,3,1


# Tolerances: Metal picks different matmul kernels for different row counts
# (a 1-row chunk is a gemv, the full [T, H] a gemm), whose reduction orders
# round differently — measured ~1e-4 relative on this tiny model, larger on
# the quantized 8B. Logic errors (wrong rows, wrong mask) produce O(1) diffs,
# so tolerance-based equivalence still catches everything we care about.
_TOL = dict(atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("model_cls", [_Untied, _Tied])
@pytest.mark.parametrize("chunk", [1, 3, 7, 100])
def test_forward_equivalence(model_cls, chunk):
    mx.random.seed(0)
    tb = _base(model_cls())
    plain = tb.completion_logprobs(_PROMPT, _COMPLETION)
    chunked = tb.completion_logprobs(_PROMPT, _COMPLETION, logit_chunk=chunk)
    assert plain.shape == chunked.shape == (len(_COMPLETION),)
    assert mx.allclose(plain, chunked, **_TOL).item()


@pytest.mark.parametrize("model_cls", [_Untied, _Tied])
def test_gradient_equivalence(model_cls):
    """Gradients w.r.t. everything UPSTREAM of the head must match. The head
    itself is frozen — as in real training, where only the LoRA tree is
    trainable — because mx.checkpoint treats the closed-over head weights as
    constants (documented caveat in completion_logprobs).

    Runs on the CPU stream: checkpoint rematerialization on Metal can pick
    different matmul kernels in backward (shape-dependent rounding, sub-1%
    relative), which would force sloppy tolerances; on CPU the two paths are
    identical to ~1e-7, which is the actual mathematical claim."""
    from mlx.utils import tree_flatten

    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(0)
        tb = _base(model_cls())
        if isinstance(tb.model, _Untied):
            tb.model.lm_head.freeze()
        else:                              # tied: the embedding IS the head
            tb.model.model.embed_tokens.freeze()

        def loss_plain(m):
            return -tb.completion_logprobs(_PROMPT, _COMPLETION).sum()

        def loss_chunked(m):
            return -tb.completion_logprobs(_PROMPT, _COMPLETION, logit_chunk=3).sum()

        l0, g0 = nn.value_and_grad(tb.model, loss_plain)(tb.model)
        l1, g1 = nn.value_and_grad(tb.model, loss_chunked)(tb.model)
        assert mx.allclose(l0, l1, atol=1e-5).item()
        flat0, flat1 = dict(tree_flatten(g0)), dict(tree_flatten(g1))
        assert flat0.keys() == flat1.keys()
        for k in flat0:
            assert mx.allclose(flat0[k], flat1[k], atol=1e-6).item(), k
    finally:
        mx.set_default_device(prev)


def test_zero_and_none_chunk_take_plain_path():
    tb = _base(_Untied())
    a = tb.completion_logprobs(_PROMPT, _COMPLETION, logit_chunk=None)
    b = tb.completion_logprobs(_PROMPT, _COMPLETION, logit_chunk=0)
    assert mx.allclose(a, b).item()


def test_empty_completion():
    tb = _base(_Untied())
    out = tb.completion_logprobs(_PROMPT, [], logit_chunk=3)
    assert out.shape == (0,)
