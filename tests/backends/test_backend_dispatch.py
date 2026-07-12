"""Backend selection + config dispatch. Runs on any host — no torch required.

Confirms the config carries the ``compute`` section, ``get_backend`` returns the
right implementation, and selecting the torch backend on a torch-less machine
fails with a clear, actionable error rather than an import crash."""

import pytest

from twin.backends import Backend, get_backend
from twin.config import Config


def test_default_backend_is_mlx():
    assert Config().compute.backend == "mlx"


def test_compute_config_parses_torch_knobs():
    cfg = Config.from_dict({"compute": {
        "backend": "torch", "device": "cuda", "dtype": "bfloat16",
        "attn_impl": "flash_attention_2", "compile": True, "tf32": False,
        "matmul_precision": "medium", "grad_checkpointing": True,
    }})
    c = cfg.compute
    assert c.backend == "torch" and c.attn_impl == "flash_attention_2"
    assert c.compile is True and c.tf32 is False and c.grad_checkpointing is True


def test_unknown_compute_key_raises():
    with pytest.raises(ValueError):
        Config.from_dict({"compute": {"bogus": 1}})


def test_get_backend_mlx_builds_and_matches_protocol():
    b = get_backend("mlx")
    assert b.name == "mlx"
    assert isinstance(b, Backend)  # runtime-checkable: has the required methods


def test_get_backend_default_none_is_mlx():
    assert get_backend(None).name == "mlx"


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("tensorflow")


def test_mlx_backend_realize_tolerates_none_and_no_grad_is_context():
    pytest.importorskip("mlx.core")  # mlx backend behaviour; skip on torch-only hosts (DGX Spark)
    b = get_backend("mlx")
    b.realize(None)  # None args are ignored, no error
    with b.no_grad():
        pass


def test_get_backend_torch_dispatch():
    """On a torch host -> a TorchBackend; on a torch-less host (this Mac) ->
    a clear ImportError naming torch, not a bare ModuleNotFoundError."""
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError) as ei:
            get_backend("torch")
        assert "torch" in str(ei.value).lower()
    else:
        assert get_backend("torch").name == "torch"
