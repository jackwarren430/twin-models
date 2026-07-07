"""GRPO RL core (DESIGN.md §8): group advantages, k3 KL, and the single-step
update. Model-agnostic — consumes :class:`Trajectory` + a ``score_fn``.

The pure pieces (``Trajectory``, ``group_advantages``, ``all_zero_advantages``)
come from :mod:`twin.rl.core` and import with no framework dependency. The
mlx-specific pieces (``grpo_update``, ``k3_kl``, ``grad_global_norm``,
``clip_by_global_norm``) live in :mod:`twin.rl.grpo` and are loaded *lazily* via
module ``__getattr__`` — so ``from twin.rl import Trajectory`` works on a machine
with no mlx (the torch backend / DGX Spark), while ``from twin.rl import
grpo_update`` still pulls in the mlx implementation on demand."""

from twin.rl.core import Trajectory, all_zero_advantages, group_advantages

_LAZY = {"grpo_update", "k3_kl", "grad_global_norm", "clip_by_global_norm"}


def __getattr__(name):
    # PEP 562: defer the mlx import until an mlx-backed symbol is actually used.
    if name in _LAZY:
        from twin.rl import grpo
        return getattr(grpo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Trajectory",
    "all_zero_advantages",
    "group_advantages",
    "grpo_update",
    "k3_kl",
    "grad_global_norm",
    "clip_by_global_norm",
]
