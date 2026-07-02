"""Sprint 5 — SelfPlayTrainer seeds MLX's global RNG. Fast: no model.

`train.seed` used to seed only the Python RNG (domain/theme schedule); MLX's
sampler RNG was never seeded, so no two runs generated the same tokens. The
trainer now seeds `mx.random` at construction. Uses a stub adapters object —
the base model is untouched during __init__.
"""

import mlx.core as mx

from twin.config import Config
from twin.train import SelfPlayTrainer


class _StubAdapters:
    NAMES = ("A", "B")

    def snapshot(self, name):
        return {"w": mx.zeros(1)}


def _make_trainer(seed: int) -> SelfPlayTrainer:
    cfg = Config.from_dict({"train": {"seed": seed}})
    return SelfPlayTrainer(None, _StubAdapters(), cfg)


def test_trainer_seeds_mx_random():
    _make_trainer(123)
    a = mx.random.uniform(shape=(8,))
    mx.random.seed(123)
    b = mx.random.uniform(shape=(8,))
    assert mx.allclose(a, b)


def test_explicit_seed_overrides_config():
    cfg = Config.from_dict({"train": {"seed": 0}})
    SelfPlayTrainer(None, _StubAdapters(), cfg, seed=77)
    a = mx.random.uniform(shape=(8,))
    mx.random.seed(77)
    b = mx.random.uniform(shape=(8,))
    assert mx.allclose(a, b)


def test_different_seeds_diverge():
    _make_trainer(1)
    a = mx.random.uniform(shape=(8,))
    _make_trainer(2)
    b = mx.random.uniform(shape=(8,))
    assert not mx.allclose(a, b)
