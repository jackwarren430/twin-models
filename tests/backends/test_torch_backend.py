"""Torch backend parity — SKIPPED unless torch + transformers are installed, so
it is a no-op on the Mac and runs on the DGX Spark.

Exercises the framework-specific pieces that need no 8B weights: the toy-LM GRPO
step (mirrors tests/sprint3/test_grpo.py so the torch update is proven to move
params in the advantage direction), the k3 KL estimator, mask exclusion, and the
LoRA-target leaf mapping. The full model path (generation, PEFT adapters,
scoring) is covered by the 2-iter end-to-end smoke on the Spark (DGX_SPARK.md)."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")  # torch_backend imports it at module load
import torch.nn as nn  # noqa: E402

from twin.backends.torch_backend import (  # noqa: E402
    TorchAdapters,
    _k3_kl,
    torch_grpo_update,
)
from twin.rl.core import Trajectory  # noqa: E402


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 8, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab, bias=False)

    def forward(self, ids):  # ids [1, T] -> [1, T, V]
        return self.out(self.emb(ids))


def _score(model, prompt_ids, completion_ids):
    seq = torch.tensor([list(prompt_ids) + list(completion_ids)])
    logits = model(seq)[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    start = len(prompt_ids) - 1
    n = len(completion_ids)
    sel = logp[start : start + n]
    tgt = torch.tensor(list(completion_ids)).unsqueeze(-1)
    return sel.gather(-1, tgt).squeeze(-1)


def test_k3_kl_nonnegative_and_zero_at_equality():
    lp = torch.tensor([-1.0, -2.0, -0.5])
    assert float(_k3_kl(lp, lp).abs().max()) < 1e-6
    other = torch.tensor([-1.5, -1.0, -2.0])
    assert float(_k3_kl(lp, other).min()) >= -1e-6


def test_grpo_step_moves_logprobs_toward_advantage():
    torch.manual_seed(0)
    model = TinyLM()
    prompt, c_plus, c_minus = [0], [1, 2], [3, 4]

    def total(c):
        with torch.no_grad():
            return float(_score(model, prompt, c).sum())

    before_plus, before_minus = total(c_plus), total(c_minus)
    opt = torch.optim.AdamW(model.parameters(), lr=0.2, weight_decay=0.0)
    last = None
    for _ in range(40):
        batch = [
            Trajectory(prompt, c_plus, advantage=1.0),
            Trajectory(prompt, c_minus, advantage=-1.0),
        ]
        last = torch_grpo_update(model, opt, batch, _score, kl_beta=0.0, grad_clip=None)

    after_plus, after_minus = total(c_plus), total(c_minus)
    assert after_plus > before_plus       # rewarded completion's logprob rises
    assert after_minus < before_minus     # penalized one's falls
    assert last["n_traj"] == 2 and last["n_tokens"] == 4
    assert math.isfinite(last["loss"]) and math.isfinite(last["grad_norm"])


def test_grpo_empty_batch_is_noop():
    model = TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    m = torch_grpo_update(model, opt, [], _score, kl_beta=0.0)
    assert m["n_traj"] == 0 and m["grad_norm"] == 0.0


def test_grpo_masked_positions_excluded():
    torch.manual_seed(0)
    model = TinyLM()
    prompt, comp, mask = [0], [1, 2, 3], [1, 0, 1]  # middle token injected
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    batch = [Trajectory(prompt, comp, advantage=1.0, loss_mask=mask)]
    m = torch_grpo_update(model, opt, batch, _score, kl_beta=0.0, grad_clip=1.0)
    assert m["n_tokens"] == 2  # only the two mask==1 positions are trained


def test_grpo_kl_reported_when_ref_present():
    torch.manual_seed(1)
    model = TinyLM()
    prompt, comp = [0], [1, 2]
    with torch.no_grad():
        ref = _score(model, prompt, comp)  # ref == current policy
    batch = [Trajectory(prompt, comp, advantage=1.0, ref_logprobs=ref)]
    m = torch_grpo_update(model, torch.optim.AdamW(model.parameters(), lr=0.05),
                          batch, _score, kl_beta=0.1, grad_clip=None)
    assert m["kl"] >= -1e-6 and math.isfinite(m["kl"])


def test_lora_leaf_mapping():
    # module-prefixed keys collapse to leaf names for PEFT target_modules
    assert TorchAdapters._leaves(["self_attn.q_proj", "mlp.down_proj"]) == [
        "down_proj", "q_proj"]
    assert TorchAdapters._leaves(None) == TorchAdapters._DEFAULT_LEAVES
