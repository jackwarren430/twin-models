"""Sprint 1 — config loader. Fast: no model."""

import pytest

from twin.config import DEFAULT_LORA_KEYS, Config, LoraConfig, ToolsConfig


def test_tiny_yaml_loads_with_defaults_filled(tmp_path):
    cfg = Config.from_yaml("configs/tiny.yaml")
    # explicit values from tiny.yaml
    assert cfg.lora.num_layers == 4
    assert cfg.lora.rank == 64
    assert cfg.lora.alpha == 128.0
    assert cfg.game.n_problems == 3
    assert cfg.train.iters == 20
    # tools section: small ReAct budgets, cas_timeout default filled
    assert cfg.tools.creator_tools == ["solve", "calc"]
    assert cfg.tools.creator_tool_rounds == 2
    assert cfg.tools.cas_timeout_s == 3.0
    # defaults filled for omitted sections
    assert cfg.rewards.w_gradient == 1.0
    assert cfg.oracle.enabled is True
    assert cfg.gen.top_p == 0.95


def test_tools_config_defaults():
    t = ToolsConfig()
    assert t.creator_tools == ["solve", "calc"]
    assert t.creator_max_tool_calls == 4
    assert t.creator_tool_rounds == 4
    assert t.cas_timeout_s == 3.0


def test_tools_config_override_via_dict():
    cfg = Config.from_dict({"tools": {"creator_tools": ["solve"], "creator_tool_rounds": 1}})
    assert cfg.tools.creator_tools == ["solve"]
    assert cfg.tools.creator_tool_rounds == 1
    assert cfg.tools.creator_max_tool_calls == 4  # default preserved


def test_base_yaml_loads():
    cfg = Config.from_yaml("configs/base.yaml")
    assert cfg.game.domains == ["math", "coding"]
    assert cfg.roles.injection_rate == 0.0
    # keys omitted in YAML -> all 7 linear projections
    assert cfg.lora.keys == DEFAULT_LORA_KEYS
    assert cfg.lora.rank == 64 and cfg.lora.alpha == 128.0


def test_default_lora_targets_all_seven_projections():
    keys = LoraConfig().keys
    assert keys == DEFAULT_LORA_KEYS
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert any(k.endswith(name) for k in keys), name


def test_effective_scale_is_alpha_over_rank():
    assert LoraConfig(rank=64, alpha=128.0).effective_scale == 2.0
    assert LoraConfig(rank=16, alpha=32.0).effective_scale == 2.0
    # explicit scale override wins over alpha/rank
    assert LoraConfig(rank=64, alpha=128.0, scale=20.0).effective_scale == 20.0


def test_keys_null_in_yaml_means_autodiscover():
    cfg = Config.from_dict({"lora": {"keys": None}})
    assert cfg.lora.keys is None      # falls back to mlx-lm auto-discovery


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
