"""Config loader tests (no model)."""

import pytest

from twin.config import Config


def test_tiny_yaml_loads_with_defaults_filled(tmp_path):
    cfg = Config.from_yaml("configs/tiny.yaml")
    # explicit values from tiny.yaml
    assert cfg.lora.num_layers == 4
    assert cfg.game.n_problems == 3
    assert cfg.train.iters == 20
    # defaults filled for omitted sections
    assert cfg.rewards.w_gradient == 1.0
    assert cfg.oracle.enabled is True
    assert cfg.gen.top_p == 0.95


def test_base_yaml_loads():
    cfg = Config.from_yaml("configs/base.yaml")
    assert cfg.game.domains == ["math", "coding"]
    assert cfg.roles.injection_rate == 0.0
    assert cfg.lora.keys is None


def test_from_dict_nested_override():
    cfg = Config.from_dict({"lora": {"rank": 32}, "train": {"iters": 5}})
    assert cfg.lora.rank == 32
    assert cfg.lora.num_layers == 16  # default preserved
    assert cfg.train.iters == 5


def test_unknown_top_level_key_raises():
    with pytest.raises(ValueError, match="Unknown config keys"):
        Config.from_dict({"loraa": {"rank": 8}})


def test_unknown_nested_key_raises():
    with pytest.raises(ValueError, match="Unknown config keys"):
        Config.from_dict({"lora": {"ranks": 8}})
