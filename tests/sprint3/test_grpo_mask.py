"""Sprint 3 — GRPO loss masking for inline ReAct rollouts. Fast: toy LM, no base.

Injected tool-observation tokens are part of ``completion_ids`` (so the forward
pass conditions on them) but carry ``loss_mask==0`` and must NOT contribute to
the policy-gradient or KL terms. These tests pin that on the exact
``grpo_update`` path the trainer uses."""

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from twin.rl import Trajectory, grpo_update


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 8, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab, bias=False)

    def __call__(self, ids):
        return self.out(self.emb(ids))


def _score(model, prompt_ids, completion_ids):
    seq = list(prompt_ids) + list(completion_ids)
    logits = model(mx.array(seq)[None])[0].astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    start = len(prompt_ids) - 1
    n = len(completion_ids)
    sel = logp[start : start + n]
    tgt = mx.array(list(completion_ids))[:, None]
    return mx.take_along_axis(sel, tgt, axis=-1)[:, 0]


def _params_snapshot(model):
    return {k: mx.array(v) for k, v in tree_flatten(model.parameters())}


def test_pg_counts_only_unmasked_positions():
    mx.random.seed(0)
    model = TinyLM()
    mx.eval(model.parameters())
    prompt, comp, mask, A = [0], [1, 2, 3, 4], [1, 0, 1, 0], 1.5

    logp = _score(model, prompt, comp)            # pre-update, full completion
    mx.eval(logp)
    # loss normalizes by TRAINED tokens (2 here); pg = -A * sum(masked-in logp) / 2
    expected_pg = float((-A) * (logp[0] + logp[2])) / 2
    unmasked_pg = float((-A) * logp.sum()) / 4

    m = grpo_update(
        model, optim.AdamW(learning_rate=0.0), [
            Trajectory(prompt, comp, advantage=A, loss_mask=mask)
        ], _score, kl_beta=0.0, grad_clip=None,
    )
    assert m["n_tokens"] == 2                      # trained-token count, not 4
    assert abs(m["pg"] - expected_pg) < 1e-4       # only positions 0 and 2
    assert abs(m["pg"] - unmasked_pg) > 1e-3       # genuinely different from full


def test_kl_is_finite_despite_huge_ref_at_masked_position():
    mx.random.seed(1)
    model = TinyLM()
    mx.eval(model.parameters())
    prompt, comp, mask = [0], [1, 2, 3], [1, 0, 1]
    # A pathological ref at the MASKED position would blow up k3 (exp(delta));
    # index-selecting trained positions must keep it out entirely.
    ref = mx.array([-1.0, 1e9, -1.0])
    m = grpo_update(
        model, optim.AdamW(learning_rate=0.05), [
            Trajectory(prompt, comp, advantage=1.0, ref_logprobs=ref, loss_mask=mask)
        ], _score, kl_beta=0.1, grad_clip=None,
    )
    assert math.isfinite(m["kl"]) and math.isfinite(m["loss"])


def test_fully_masked_trajectory_is_noop():
    mx.random.seed(2)
    model = TinyLM()
    mx.eval(model.parameters())
    before = _params_snapshot(model)
    m = grpo_update(
        model, optim.AdamW(learning_rate=0.5), [
            Trajectory([0], [1, 2], advantage=1.0, loss_mask=[0, 0])
        ], _score, kl_beta=0.0, grad_clip=None,
    )
    assert m["n_traj"] == 0 and m["n_tokens"] == 0
    after = _params_snapshot(model)
    for k in before:
        assert float(mx.max(mx.abs(before[k] - after[k]))) == 0.0
