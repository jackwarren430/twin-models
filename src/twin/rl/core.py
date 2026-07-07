"""Backend-neutral GRPO pieces: the :class:`Trajectory` container and the pure
advantage math. NO framework dependency (no mlx, no torch).

These live apart from :mod:`twin.rl.grpo` (which holds the mlx-specific update)
so the training loop and the torch backend can import ``Trajectory`` /
``group_advantages`` / ``all_zero_advantages`` on a machine that has no mlx
installed. :mod:`twin.rl.grpo` re-exports all three, and ``from twin.rl import
Trajectory`` resolves here — so every historical import path is unchanged."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Trajectory:
    """One scored rollout: a prompt, the exact completion ids that were sampled,
    its scalar reward, and (filled in by the trainer/update) its group-relative
    advantage and the reference log-probs used for the KL term.

    ``ref_logprobs`` is a **backend-native tensor** (``mx.array`` on MLX,
    ``torch.Tensor`` on the torch backend) of shape ``[len(completion_ids)]``,
    or ``None`` (no KL).

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
    ref_logprobs: Optional[Any] = None   # [len(completion_ids)] backend tensor or None
    loss_mask: Optional[list[int]] = None  # [len(completion_ids)] or None (all 1)
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
