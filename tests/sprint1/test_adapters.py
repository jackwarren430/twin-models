"""Sprint 1 — adapter-swap invariants against the REAL base model.

Heavy: loads ~6 GB. Marked ``model`` and skipped unless TWIN_RUN_MODEL_TESTS=1
(set automatically by ``scripts/run_tests.py --model``), so the default run stays
fast. The standalone scripts/smoke_test.py covers the same ground interactively.
"""

import os

import pytest

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        os.environ.get("TWIN_RUN_MODEL_TESTS") != "1",
        reason="set TWIN_RUN_MODEL_TESTS=1 to run model-loading tests",
    ),
]


@pytest.fixture(scope="module")
def base_and_adapters():
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from twin.config import Config
    from twin.models import Adapters, TwinBase

    cfg = Config.from_yaml("configs/tiny.yaml")
    base = TwinBase(cfg.model.path)
    ad = Adapters(base.model, num_layers=2, rank=4, scale=16.0)
    return base, ad, mx, tree_flatten, tree_unflatten


def _ids(base):
    return base.tokenizer.encode(base.render("What is 2+2?"))


def test_init_adapters_equal_base(base_and_adapters):
    base, ad, mx, *_ = base_and_adapters
    ids = _ids(base)
    ad.activate("base"); lp_base = base.token_logprobs(ids)
    ad.activate("A"); lp_a = base.token_logprobs(ids)
    ad.activate("B"); lp_b = base.token_logprobs(ids)
    assert float(mx.max(mx.abs(lp_a - lp_base))) < 1e-4
    assert float(mx.max(mx.abs(lp_b - lp_base))) < 1e-4


def test_perturbing_A_does_not_touch_B(base_and_adapters):
    base, ad, mx, tree_flatten, tree_unflatten = base_and_adapters
    ids = _ids(base)
    ad.activate("base"); lp_base = base.token_logprobs(ids)
    flat = dict(tree_flatten(ad.trees["A"]))
    for k in list(flat):
        if k.endswith("lora_b"):
            flat[k] = mx.random.normal(flat[k].shape) * 0.05
    ad.trees["A"] = tree_unflatten(list(flat.items()))
    mx.eval(ad.trees["A"])
    ad.activate("A"); lp_a = base.token_logprobs(ids)
    ad.activate("B"); lp_b = base.token_logprobs(ids)
    assert float(mx.max(mx.abs(lp_a - lp_base))) > 1e-3   # A moved
    assert float(mx.max(mx.abs(lp_b - lp_base))) < 1e-4   # B untouched
