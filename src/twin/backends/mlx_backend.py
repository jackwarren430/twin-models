"""MLX / Apple-silicon backend — a thin adapter over the existing code.

This carries no new behaviour: it delegates to the same
``twin.models.TwinBase`` / ``twin.models.Adapters`` / ``twin.rl.grpo_update``
that ran before the backend split, so a run with ``compute.backend: mlx`` (the
default) is bit-identical to the historical loop. mlx is imported lazily inside
the methods so a machine without mlx can still ``import twin.backends``."""

from typing import Any


class MlxBackend:
    name = "mlx"

    def load_base(self, model_cfg, compute_cfg=None) -> Any:
        from twin.models import TwinBase
        # TwinBase owns the mlx-lm load; compute_cfg carries no mlx levers today
        # (max_kv_size etc. live on model_cfg and are applied at generate time).
        # load_strict / eos_token_ids default to the historical Qwen3 behaviour;
        # the gemma4 twentyq config sets them (see ModelConfig).
        return TwinBase(
            model_cfg.path,
            strict=getattr(model_cfg, "load_strict", True),
            eos_token_ids=getattr(model_cfg, "eos_token_ids", None),
        )

    def build_adapters(self, base, lora_cfg) -> Any:
        from twin.models import Adapters
        return Adapters.from_config(base.model, lora_cfg)

    def make_optimizer(self, adapters, name: str, lr: float) -> Any:
        import mlx.optimizers as optim
        # mlx optimizers bind to params at update time (optimizer.update(model,
        # grads)), so adapters/name are unused here — the signature matches the
        # torch backend, where the optimizer must be constructed over that
        # adapter's params. weight_decay=0: decaying LoRA toward zero would
        # fight the RL signal.
        return optim.AdamW(learning_rate=lr, weight_decay=0.0)

    def seed(self, n: int) -> None:
        import mlx.core as mx
        mx.random.seed(n)

    def no_grad(self):
        # mlx has no ambient grad mode (gradients come from nn.value_and_grad
        # tracing), so the reference pass needs no special context.
        import contextlib
        return contextlib.nullcontext()

    def realize(self, *xs) -> None:
        import mlx.core as mx
        vals = [x for x in xs if x is not None]
        if vals:
            mx.eval(*vals)

    def grpo_update(self, model, optimizer, batch, score_fn, *,
                    kl_beta: float = 0.0, grad_clip=1.0,
                    microbatch_size: int = 0) -> dict:
        from twin.rl import grpo_update
        return grpo_update(
            model, optimizer, batch, score_fn,
            kl_beta=kl_beta, grad_clip=grad_clip, microbatch_size=microbatch_size,
        )

    def peak_memory_gb(self) -> float | None:
        import mlx.core as mx
        get_peak = getattr(mx, "get_peak_memory", None) or getattr(
            getattr(mx, "metal", None), "get_peak_memory", None)
        return None if get_peak is None else get_peak() / 1e9

    def reset_peak_memory(self) -> None:
        import mlx.core as mx
        reset = getattr(mx, "reset_peak_memory", None) or getattr(
            getattr(mx, "metal", None), "reset_peak_memory", None)
        if reset is not None:
            reset()

    def active_memory_gb(self) -> float | None:
        import mlx.core as mx
        get_active = getattr(mx, "get_active_memory", None) or getattr(
            getattr(mx, "metal", None), "get_active_memory", None)
        return None if get_active is None else get_active() / 1e9

    def clear_cache(self) -> None:
        import mlx.core as mx
        clear = getattr(mx, "clear_cache", None) or getattr(
            getattr(mx, "metal", None), "clear_cache", None)
        if clear is not None:
            clear()
