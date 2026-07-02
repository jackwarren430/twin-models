"""Minimal GRPO, single-inner-step, implemented directly in MLX (DESIGN.md §8).

For a group of responses to one prompt with scalar rewards ``R_1..R_G`` the
group-relative advantage is (Dr.GRPO-style mean baseline, the default)::

    A_g = R_g − mean(R)

or, in the legacy ``mode="std"`` kept for ablation::

    A_g = (R_g − mean(R)) / (std(R) + eps)

Because we take exactly **one** optimizer step per batch, ``π_new == π_old`` at
update time, the PPO ratio is 1 and the clip is inactive — so the per-token
policy loss is just the REINFORCE-with-baseline form, plus a KL-to-reference
penalty (k3 estimator) toward the frozen base (zeroed adapter)::

    L_pg = − (1/N_tok) · Σ_{g,t}  A_g · logπ_θ(token_{g,t})
    kl   =   (1/N_tok) · Σ_{g,t}  exp(δ) − δ − 1     with δ = logπ_ref − logπ_θ
    loss = L_pg + β · kl

This module is **model-agnostic**: it consumes :class:`Trajectory` objects and a
``score_fn(model, prompt_ids, completion_ids) -> mx.array`` that returns
per-completion-token log-probs (for the real loop this is
``TwinBase.completion_logprobs``; tests inject a tiny toy LM). Advantages are
computed by the caller (the trainer groups solver attempts per-problem and
creator suites per-prompt — DESIGN.md §5) and stored on each trajectory before
the update, so the update itself is group-structure-agnostic.

The whole batch is scored in a single value_and_grad graph. That keeps the code
simple and is fine for the tiny default config; chunking / gradient accumulation
across micro-batches is the Sprint-4 scaling lever, not needed to bring the loop
up.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

# score_fn(model, prompt_ids, completion_ids) -> per-token logprobs [len(completion)]
ScoreFn = Callable[[object, list[int], list[int]], mx.array]


@dataclass
class Trajectory:
    """One scored rollout: a prompt, the exact completion ids that were sampled,
    its scalar reward, and (filled in by the trainer/update) its group-relative
    advantage and the reference log-probs used for the KL term.

    ``loss_mask`` (parallel to ``completion_ids``; ``1`` = policy-sampled, ``0`` =
    injected) marks which completion tokens the policy actually produced. Inline
    ReAct rollouts splice tool observations (``<obs>...</obs>``) into the
    completion so the forward pass conditions on them, but those tokens are NOT
    policy samples and must be excluded from the policy-gradient and KL terms.
    ``None`` => every token is policy-sampled (the single-turn default)."""

    prompt_ids: list[int]
    completion_ids: list[int]
    reward: float = 0.0
    advantage: float = 0.0
    ref_logprobs: Optional[mx.array] = None   # [len(completion_ids)] or None (no KL)
    loss_mask: Optional[list[int]] = None     # [len(completion_ids)] or None (all 1)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.loss_mask is not None and len(self.loss_mask) != len(self.completion_ids):
            raise ValueError(
                f"loss_mask length {len(self.loss_mask)} != completion length "
                f"{len(self.completion_ids)}"
            )

    @property
    def n_tokens(self) -> int:
        return len(self.completion_ids)

    @property
    def n_train_tokens(self) -> int:
        """Number of policy-sampled (trained) tokens — the ones that contribute
        to the loss. Equals ``n_tokens`` when there is no mask."""
        if self.loss_mask is None:
            return len(self.completion_ids)
        return int(sum(1 for b in self.loss_mask if b))

    def train_indices(self) -> Optional[list[int]]:
        """Indices into the completion of the trained (mask==1) positions, or
        ``None`` when every position is trained."""
        if self.loss_mask is None:
            return None
        return [i for i, b in enumerate(self.loss_mask) if b]


def group_advantages(
    rewards: list[float], *, mode: str = "mean", eps: float = 1e-6
) -> list[float]:
    """Group-relative advantages within one GRPO group.

    ``mode="mean"`` (default): ``A = R − mean(R)`` — Dr.GRPO-style mean
    baseline, no ÷std. For the small groups this project runs (G_c 2-4, K=4)
    standardization is degenerate: it maps *any* non-tie to ±1, so a
    0.7889-vs-0.7738 reward gap (solve-rate noise) trains exactly as hard as a
    1.6-vs-−1.0 one. Mean-centering preserves magnitude — near-tie groups give
    near-zero gradients, real gaps proportionally strong ones (DESIGN.md §8).

    ``mode="std"``: legacy standardized form ``(R − mean)/(std + eps)``, kept
    as an ablation arm (config ``train.adv_mode``).

    In both modes a group with no reward variance yields all-zero advantages
    (nothing to learn), which is the correct, stable behaviour."""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    centered = [r - mean for r in rewards]
    if mode == "mean":
        return centered
    if mode == "std":
        var = sum(c * c for c in centered) / n
        std = var ** 0.5
        return [c / (std + eps) for c in centered]
    raise ValueError(f"unknown advantage mode: {mode!r} (expected 'mean' or 'std')")


def all_zero_advantages(trajs: list["Trajectory"], *, tol: float = 1e-8) -> bool:
    """True when every trajectory's advantage is (numerically) zero — a fully
    tied batch. The policy-gradient term then vanishes exactly, and the only
    thing a GRPO step would compute is the pure-KL gradient, which is ≈0 right
    after the reference pass anyway — at the full cost of reference pass +
    policy pass + backward (minutes on a saturated 4096-token iteration). The
    trainer uses this to skip the update entirely (DESIGN.md §11 Sprint 6).
    Empty batches are trivially all-zero. ``tol`` absorbs float residue from
    mean-centering identical rewards."""
    return all(abs(t.advantage) <= tol for t in trajs)


def grad_global_norm(grads) -> float:
    """L2 norm of the flattened gradient tree (for logging / clipping)."""
    total = mx.array(0.0)
    for _, v in tree_flatten(grads):
        total = total + mx.sum(v.astype(mx.float32) ** 2)
    return float(total) ** 0.5


def clip_by_global_norm(grads, max_norm: float | None):
    """Scale ``grads`` so their global L2 norm is at most ``max_norm``. Returns
    ``(grads, pre_clip_norm)``. ``max_norm`` falsy => no clipping."""
    norm = grad_global_norm(grads)
    if max_norm and norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        grads = tree_map(lambda g: g * scale, grads)
    return grads, norm


def k3_kl(logp_theta: mx.array, logp_ref: mx.array) -> mx.array:
    """Per-token k3 KL estimator of KL(π_θ ‖ π_ref): ``exp(δ) − δ − 1`` with
    ``δ = logp_ref − logp_theta``. Non-negative, unbiased, low-variance."""
    delta = logp_ref - logp_theta
    return mx.exp(delta) - delta - 1.0


def grpo_update(
    model,
    optimizer,
    batch: list[Trajectory],
    score_fn: ScoreFn,
    *,
    kl_beta: float = 0.0,
    grad_clip: float | None = 1.0,
    microbatch_size: int = 0,
) -> dict:
    """One GRPO optimizer step over ``batch`` (advantages already assigned).

    ``model`` must have the policy adapter active; ``optimizer`` is that
    adapter's own optimizer (separate AdamW state per adapter). Trajectories
    whose ``ref_logprobs`` is set contribute the KL penalty when ``kl_beta>0``.

    ``microbatch_size`` bounds how many trajectories are scored in a single
    autograd graph: with ``0`` the whole batch goes through one
    ``value_and_grad`` (simplest, highest peak memory); with ``m > 0`` the batch
    is split into chunks of ``m``, each chunk's gradient is computed and
    **summed** into an accumulator, and a single optimizer step is taken at the
    end. Because the gradient is linear, ``Σ_chunk ∇(Σ_{t∈chunk} loss_t) =
    ∇(Σ_all loss_t)`` — the result is identical to the single-graph version, but
    peak memory is bounded by one chunk (a 4096-token completion alone
    materializes a ~2.5GB [T,V] logits tensor, so chunking is what keeps a dozen
    long trajectories from OOMing the backward pass). The accumulator is realized
    between chunks (``mx.eval``) so each chunk's graph is freed before the next.

    Returns a metrics dict (loss, pg, kl, grad_norm, n_traj, n_tokens — where
    ``n_tokens`` counts *trained* tokens). A batch with no trained tokens is a
    no-op. Per trajectory, injected (mask==0) positions are dropped from both the
    PG and KL sums by index-selecting the trained positions before reducing —
    the mask is static at trace time, so this is a clean differentiable gather
    and avoids the ``inf*0 -> nan`` that a ``k3_kl * mask`` multiply would hit at
    masked positions."""
    batch = [t for t in batch if t.n_train_tokens > 0]
    total_tokens = sum(t.n_train_tokens for t in batch)
    if total_tokens == 0:
        return {"loss": 0.0, "pg": 0.0, "kl": 0.0, "grad_norm": 0.0,
                "n_traj": 0, "n_tokens": 0}

    chunk_size = microbatch_size if microbatch_size and microbatch_size > 0 else len(batch)

    def make_loss_fn(chunk, aux):
        # Unnormalized chunk loss (summed over its trajectories); we divide the
        # accumulated gradient by total_tokens once, after all chunks.
        def loss_fn(model):
            pg = mx.array(0.0)
            kl = mx.array(0.0)
            for t in chunk:
                logp = score_fn(model, t.prompt_ids, t.completion_ids)  # [n], grad
                ref = t.ref_logprobs
                idx = t.train_indices()
                if idx is not None:                   # drop injected positions
                    sel = mx.array(idx)
                    logp = logp[sel]
                    ref = ref[sel] if ref is not None else None
                pg = pg + (-t.advantage) * logp.sum()
                if kl_beta and ref is not None:
                    kl = kl + k3_kl(logp, ref).sum()
            aux["pg"] = pg
            aux["kl"] = kl
            return pg + kl_beta * kl
        return loss_fn

    acc_grads = None
    pg_total = 0.0
    kl_total = 0.0
    for start in range(0, len(batch), chunk_size):
        chunk = batch[start : start + chunk_size]
        aux: dict[str, mx.array] = {}
        _, grads = nn.value_and_grad(model, make_loss_fn(chunk, aux))(model)
        if acc_grads is None:
            acc_grads = grads
        else:
            acc_grads = tree_map(lambda a, g: a + g, acc_grads, grads)
        # Realize the running accumulator + this chunk's metrics so the chunk's
        # autograd graph (and its big [T,V] logits) is released before the next.
        mx.eval(acc_grads, aux["pg"], aux["kl"])
        pg_total += float(aux["pg"])
        kl_total += float(aux["kl"])

    grads = tree_map(lambda g: g / total_tokens, acc_grads)
    grads, raw_norm = clip_by_global_norm(grads, grad_clip)
    mx.eval(grads)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return {
        "loss": (pg_total + kl_beta * kl_total) / total_tokens,
        "pg": pg_total / total_tokens,
        "kl": kl_total / total_tokens,
        "grad_norm": raw_norm,
        "n_traj": len(batch),
        "n_tokens": total_tokens,
    }
