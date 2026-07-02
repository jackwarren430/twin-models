"""GRPO RL core (DESIGN.md §8): group advantages, k3 KL, and the single-step
update. Model-agnostic — consumes :class:`Trajectory` + a ``score_fn``."""

from twin.rl.grpo import (
    Trajectory,
    all_zero_advantages,
    clip_by_global_norm,
    grad_global_norm,
    group_advantages,
    grpo_update,
    k3_kl,
)

__all__ = [
    "Trajectory",
    "all_zero_advantages",
    "group_advantages",
    "grpo_update",
    "k3_kl",
    "grad_global_norm",
    "clip_by_global_norm",
]
