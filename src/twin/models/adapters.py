"""Adapters: two independent LoRA trees over ONE shared frozen base.

This is the memory-saving core of the project (DESIGN.md §3). We call
``linear_to_lora_layers`` exactly once to wrap the base's linear layers with
trainable LoRA modules. The trainable parameters form a small "tree". We keep
three such trees in Python and swap the active one into the (single, shared)
model with ``model.update(tree)``:

    "A"    -> Model A's adapter
    "B"    -> Model B's adapter
    "base" -> an all-zero tree => LoRA contribution is 0 => exactly the base
              model. Used as the knowledge oracle and the GRPO KL reference.

The 6.2 GB quantized base weights are never copied; only the tiny LoRA trees
(tens of MB) are duplicated. Swapping is a pointer update, not a data copy.
"""

from contextlib import contextmanager
from typing import Iterator

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm.tuner.utils import linear_to_lora_layers


def _copy_tree(tree):
    """Deep copy a parameter tree into independent arrays."""
    return tree_map(lambda a: mx.array(a), tree)


class Adapters:
    NAMES = ("A", "B")

    def __init__(
        self,
        model,
        *,
        num_layers: int,
        rank: int,
        scale: float = 20.0,
        dropout: float = 0.0,
        keys: list[str] | None = None,
    ):
        # Freeze the base, then convert the last `num_layers` blocks to LoRA.
        # LoRALinear keeps the base weights frozen and exposes lora_a/lora_b as
        # the only trainable params.
        model.freeze()
        cfg: dict = {"rank": rank, "scale": scale, "dropout": dropout}
        if keys is not None:
            cfg["keys"] = keys
        linear_to_lora_layers(model, num_layers, cfg)
        self.model = model

        # Snapshot the freshly-initialised LoRA tree. lora_b initialises to
        # zeros, so at init ALL of {A, B, base} are behaviourally identical to
        # the base model; they diverge only as A/B are trained.
        init = model.trainable_parameters()
        self.trees: dict[str, dict] = {
            "A": _copy_tree(init),
            "B": _copy_tree(init),
        }
        # Reference / oracle tree: all LoRA params zero => zero delta => base.
        self._zero = tree_map(lambda a: mx.zeros_like(a), init)
        mx.eval(self.trees["A"], self.trees["B"], self._zero)

        self.active: str | None = None
        self.activate("A")

    # ----- activation ------------------------------------------------------
    def activate(self, name: str) -> None:
        """Make ``name`` ('A' | 'B' | 'base') the live adapter on the model."""
        tree = self._zero if name == "base" else self.trees[name]
        self.model.update(tree)
        self.active = name

    @contextmanager
    def using(self, name: str) -> Iterator[None]:
        """Temporarily activate ``name``, restoring the previous adapter after.

        Use ``using('base')`` for oracle / reference passes."""
        prev = self.active
        self.activate(name)
        try:
            yield
        finally:
            if prev is not None:
                self.activate(prev)

    def capture(self, name: str) -> None:
        """Read the model's current LoRA params back into tree ``name``.

        Call this after an optimizer step on the active adapter so the updated
        weights are retained in our Python-side store."""
        if name not in self.trees:
            raise KeyError(name)
        self.trees[name] = self.model.trainable_parameters()

    # ----- explicit cross-model blending (optional ablation, DESIGN §4) ----
    def blend_into(self, dst: str, src: str, rate: float) -> None:
        """dst <- (1-rate)*dst + rate*src. Off by default (rate=0). Naive LoRA
        averaging can be destructive; this exists only for ablations."""
        if rate <= 0.0:
            return
        a, b = self.trees[dst], self.trees[src]
        fa = dict(tree_flatten(a))
        fb = dict(tree_flatten(b))
        blended = {k: (1.0 - rate) * fa[k] + rate * fb[k] for k in fa}
        self.trees[dst] = tree_unflatten(list(blended.items()))
        mx.eval(self.trees[dst])

    # ----- persistence -----------------------------------------------------
    def save(self, name: str, path: str) -> None:
        flat = dict(tree_flatten(self.trees[name]))
        mx.save_safetensors(path, flat)

    def load(self, name: str, path: str) -> None:
        flat = mx.load(path)
        self.trees[name] = tree_unflatten(list(flat.items()))
        if self.active == name:
            self.activate(name)

    # ----- introspection ---------------------------------------------------
    def num_params(self, name: str = "A") -> int:
        return sum(v.size for _, v in tree_flatten(self.trees[name]))
