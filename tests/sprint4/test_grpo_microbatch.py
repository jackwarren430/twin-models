"""GRPO gradient-accumulation (microbatching) must be numerically identical to
scoring the whole batch in one graph — it only bounds peak memory (Sprint-4
scaling lever; mini-01 run0 OOM'd at a 4096 token budget with the whole batch in
one autograd graph). Same advantages, KL refs and loss masks, three chunk sizes,
all must land on the same loss / grad-norm / post-step parameters."""

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
    sel = logp[start:start + n]
    tgt = mx.array(list(completion_ids))[:, None]
    return mx.take_along_axis(sel, tgt, axis=-1)[:, 0]


def _init_params():
    mx.random.seed(0)
    m = TinyLM()
    mx.eval(m.parameters())
    return {k: mx.array(v) for k, v in tree_flatten(m.parameters())}


def _build(init):
    m = TinyLM()
    m.load_weights([(k, mx.array(v)) for k, v in init.items()])
    mx.eval(m.parameters())
    return m


def _make_batch(model):
    prompt = [0]
    comps = [[1, 2, 3], [4, 5], [6, 7, 1], [2, 3]]
    advs = [1.0, -1.0, 0.5, -0.5]
    masks = [None, [1, 0], None, [0, 1]]   # exercise the masked-index gather path
    batch = []
    for c, a, msk in zip(comps, advs, masks):
        ref = _score(model, prompt, c)
        mx.eval(ref)
        batch.append(Trajectory(prompt, c, advantage=a, ref_logprobs=ref, loss_mask=msk))
    return batch


def _run(init, microbatch_size):
    model = _build(init)
    batch = _make_batch(model)
    m = grpo_update(model, optim.AdamW(learning_rate=0.1, weight_decay=0.0),
                    batch, _score, kl_beta=0.1, grad_clip=1.0,
                    microbatch_size=microbatch_size)
    params = {k: float(mx.sum(v).item()) for k, v in tree_flatten(model.parameters())}
    return m, params


def test_microbatch_matches_full_batch():
    init = _init_params()
    base_m, base_p = _run(init, 0)            # whole batch in one graph
    assert base_m["n_tokens"] == 8            # 3 + 1(masked) + 3 + 1(masked)
    for mb in (1, 2, 3):
        m, p = _run(init, mb)
        assert m["n_tokens"] == base_m["n_tokens"]
        assert abs(m["loss"] - base_m["loss"]) < 1e-5
        assert abs(m["pg"] - base_m["pg"]) < 1e-5
        assert abs(m["kl"] - base_m["kl"]) < 1e-5
        assert abs(m["grad_norm"] - base_m["grad_norm"]) < 1e-5
        assert max(abs(p[k] - base_p[k]) for k in base_p) < 1e-5


def test_microbatch_larger_than_batch_is_single_chunk():
    init = _init_params()
    base_m, base_p = _run(init, 0)
    m, p = _run(init, 999)                     # chunk bigger than batch == one chunk
    assert abs(m["loss"] - base_m["loss"]) < 1e-5
    assert max(abs(p[k] - base_p[k]) for k in base_p) < 1e-5
