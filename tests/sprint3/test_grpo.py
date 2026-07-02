"""Sprint 3 — GRPO core: advantages, KL, grad norm/clip, and a real optimizer
step on a tiny toy LM. Fast: no base model.

The toy-LM step is the important one — it exercises the exact
``grpo_update`` -> ``nn.value_and_grad`` -> ``optimizer.update`` path used by the
trainer, just over an 8-token vocab instead of the 6 GB base, so it runs in
milliseconds while proving the machinery moves params in the advantage direction.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from twin.rl import (
    Trajectory,
    clip_by_global_norm,
    grad_global_norm,
    group_advantages,
    grpo_update,
    k3_kl,
)


# ----- pure helpers --------------------------------------------------------
def test_group_advantages_centers():
    # Default is the Dr.GRPO mean baseline (Sprint 6): A = R − mean(R).
    advs = group_advantages([0.0, 1.0, 2.0, 3.0])
    assert abs(sum(advs)) < 1e-6                      # zero mean
    assert advs[0] < 0 < advs[-1]                     # ordered
    assert advs == [-1.5, -0.5, 0.5, 1.5]             # magnitude preserved


def test_group_advantages_std_mode_standardizes():
    advs = group_advantages([0.0, 1.0, 2.0, 3.0], mode="std")
    assert abs(sum(advs)) < 1e-6                      # zero mean
    assert advs[0] < 0 < advs[-1]                     # ordered


def test_group_advantages_zero_variance_is_zero():
    assert group_advantages([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]
    assert group_advantages([2.0, 2.0, 2.0], mode="std") == [0.0, 0.0, 0.0]
    assert group_advantages([]) == []


def test_k3_kl_nonnegative_and_zero_at_equality():
    lp = mx.array([-1.0, -2.0, -0.5])
    assert float(mx.max(mx.abs(k3_kl(lp, lp)))) < 1e-6
    other = mx.array([-1.5, -1.0, -2.0])
    assert float(mx.min(k3_kl(lp, other))) >= -1e-6   # k3 is >= 0


def test_grad_global_norm_and_clip():
    grads = {"w": mx.array([3.0, 4.0])}               # norm 5
    assert abs(grad_global_norm(grads) - 5.0) < 1e-5
    clipped, pre = clip_by_global_norm(grads, 1.0)
    assert abs(pre - 5.0) < 1e-5
    assert abs(grad_global_norm(clipped) - 1.0) < 1e-3
    # below the threshold -> untouched
    same, _ = clip_by_global_norm(grads, 100.0)
    assert abs(grad_global_norm(same) - 5.0) < 1e-5


# ----- a real GRPO step on a toy LM ---------------------------------------
class TinyLM(nn.Module):
    def __init__(self, vocab: int = 8, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab, bias=False)

    def __call__(self, ids):                          # ids [1, T] -> [1, T, V]
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


def test_grpo_step_moves_logprobs_toward_advantage():
    mx.random.seed(0)
    model = TinyLM()
    mx.eval(model.parameters())
    prompt, c_plus, c_minus = [0], [1, 2], [3, 4]

    def total(c):
        return float(_score(model, prompt, c).sum())

    before_plus, before_minus = total(c_plus), total(c_minus)

    opt = optim.AdamW(learning_rate=0.2, weight_decay=0.0)
    last = None
    for _ in range(40):
        batch = [
            Trajectory(prompt, c_plus, advantage=1.0),
            Trajectory(prompt, c_minus, advantage=-1.0),
        ]
        last = grpo_update(model, opt, batch, _score, kl_beta=0.0, grad_clip=None)

    after_plus, after_minus = total(c_plus), total(c_minus)
    # rewarded completion's logprob rises; penalized one's falls
    assert after_plus > before_plus
    assert after_minus < before_minus
    assert last["n_traj"] == 2 and last["n_tokens"] == 4
    assert math.isfinite(last["loss"]) and math.isfinite(last["grad_norm"])


def test_grpo_step_empty_batch_is_noop():
    model = TinyLM()
    opt = optim.AdamW(learning_rate=0.1)
    m = grpo_update(model, opt, [], _score, kl_beta=0.0)
    assert m["n_traj"] == 0 and m["grad_norm"] == 0.0


def test_grpo_kl_penalty_reported_when_ref_present():
    mx.random.seed(1)
    model = TinyLM()
    mx.eval(model.parameters())
    prompt, comp = [0], [1, 2]
    ref = _score(model, prompt, comp)                 # ref == current policy
    mx.eval(ref)
    batch = [Trajectory(prompt, comp, advantage=1.0, ref_logprobs=ref)]
    m = grpo_update(model, optim.AdamW(learning_rate=0.05), batch, _score,
                    kl_beta=0.1, grad_clip=None)
    # pre-update policy == ref, so reported KL is ~0 but defined and finite
    assert m["kl"] >= -1e-6 and math.isfinite(m["kl"])
