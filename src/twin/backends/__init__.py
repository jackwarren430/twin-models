"""Compute backends: one uniform interface, two implementations.

The self-play loop (``twin.train.loop``) and entry script (``scripts/train.py``)
were written directly against MLX / Apple silicon. This package factors the
handful of framework-specific operations behind a small :class:`Backend`
interface so the SAME loop runs on either:

  * ``"mlx"``  — Apple silicon, mlx-lm (the historical default; unchanged).
  * ``"torch"`` — NVIDIA DGX Spark / CUDA, PyTorch + transformers + PEFT
    (GB10 Grace Blackwell, 128GB unified memory). See DGX_SPARK.md.

Selection is one config field, ``compute.backend`` (:class:`twin.config.ComputeConfig`).

The interface deliberately owns only what is genuinely framework-specific:
model+adapter construction, the optimizer, RNG seeding, array *realization*
(``mx.eval`` has no torch analog — it's a no-op there), the one GRPO step, and
peak-memory probing. Everything above it — rewards, verifiers, tools, prompts,
problem parsing, curve analysis — is pure Python and backend-agnostic.

Importing this package pulls in NEITHER mlx nor torch: the concrete backends
import their framework lazily (inside :func:`get_backend` / method bodies), so a
Mac without torch can still ``import twin.backends`` and a Spark without mlx can
still select the torch backend.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """The framework-specific surface the training loop needs. Both
    :class:`~twin.backends.mlx_backend.MlxBackend` and
    :class:`~twin.backends.torch_backend.TorchBackend` implement it."""

    name: str

    def load_base(self, model_cfg, compute_cfg) -> Any:
        """Load the frozen base model + tokenizer, returning a ``TwinBase``-like
        object (render / generate / generate_batch / generate_react /
        completion_logprobs)."""

    def build_adapters(self, base, lora_cfg) -> Any:
        """Build the two LoRA adapters ('A','B') + zeroed 'base' over ``base``,
        returning an ``Adapters``-like object (activate / using / capture /
        save / load / global_norm / snapshot / drift_from / num_params / NAMES)."""

    def make_optimizer(self, adapters, name: str, lr: float) -> Any:
        """A fresh AdamW (weight_decay=0) for adapter ``name``'s LoRA params.
        MLX binds params at update time and ignores ``adapters``/``name``; the
        torch backend must construct the optimizer over that adapter's params."""

    def seed(self, n: int) -> None:
        """Seed the backend's global sampling RNG."""

    def realize(self, *xs) -> None:
        """Force any lazily-computed arrays in ``xs`` to be materialized. This
        is ``mx.eval`` on MLX; on eager PyTorch it is a no-op (tensors are
        already realized). ``None`` args are ignored."""

    def no_grad(self):
        """Context manager under which the reference (KL) pass runs. ``no_grad``
        on torch (so the ref logprobs carry no autograd graph); a null context
        on MLX (grad there comes only from ``nn.value_and_grad`` tracing)."""

    def grpo_update(
        self, model, optimizer, batch, score_fn, *,
        kl_beta: float = 0.0, grad_clip: float | None = 1.0,
        microbatch_size: int = 0,
    ) -> dict:
        """One GRPO optimizer step over ``batch`` (advantages pre-assigned).
        Returns the metrics dict (loss/pg/kl/grad_norm/n_traj/n_tokens)."""

    def peak_memory_gb(self) -> float | None:
        """Peak device memory used so far, in GB, or ``None`` if unavailable."""

    def reset_peak_memory(self) -> None:
        """Reset the peak-memory counter (no-op if unavailable)."""

    def clear_cache(self) -> None:
        """Release the framework's cached-but-unused device buffers back to the
        OS (``mx.clear_cache`` on MLX, ``torch.cuda.empty_cache`` on CUDA).
        Called at iteration boundaries so a large unquantized base does not let
        the buffer cache creep into swap and get OOM-killed across iterations."""


def get_backend(name: str) -> Backend:
    """Instantiate the backend named ``name`` ('mlx' | 'torch').

    The framework import happens here, not at package import, so selecting one
    backend never requires the other to be installed."""
    key = (name or "mlx").lower()
    if key == "mlx":
        from twin.backends.mlx_backend import MlxBackend
        return MlxBackend()
    if key == "torch":
        from twin.backends.torch_backend import TorchBackend
        return TorchBackend()
    raise ValueError(
        f"unknown compute.backend {name!r} (expected 'mlx' or 'torch')"
    )


__all__ = ["Backend", "get_backend"]
