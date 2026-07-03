"""Sprint 7 — config knobs: back-compatible defaults (a pre-Sprint-7 YAML
must behave exactly as before) and the mini4.yaml run config. Fast."""

from pathlib import Path

from twin.config import Config

REPO = Path(__file__).resolve().parents[2]


def test_defaults_are_backward_compatible():
    cfg = Config.from_dict({})
    assert cfg.game.creator_mode == "suite"
    assert cfg.game.condition_on_previous is True
    assert cfg.game.personas is False
    assert cfg.game.require_tool_use is False
    assert cfg.tools.protocol == "react"


def test_mini3_yaml_still_loads_with_v1_semantics():
    cfg = Config.from_yaml(REPO / "configs" / "mini3.yaml")
    assert cfg.game.creator_mode == "suite"
    assert cfg.game.personas is False
    assert cfg.tools.protocol == "react"


def test_mini4_yaml_is_the_sprint7_config():
    cfg = Config.from_yaml(REPO / "configs" / "mini4.yaml")
    assert cfg.game.creator_mode == "per_problem"
    assert cfg.game.condition_on_previous is True
    assert cfg.game.personas is True
    assert cfg.game.require_tool_use is False      # log-first; flip if faking persists
    assert cfg.tools.protocol == "native"
    assert cfg.game.n_problems == 5
    assert cfg.game.solver_attempts == 8
    assert cfg.game.creator_group == 4
    assert cfg.game.domains == ["math"]
    assert cfg.rewards.w_brevity == 0.25
    assert cfg.rewards.w_brevity < cfg.rewards.w_solve
    assert (cfg.rewards.target_hi, cfg.rewards.target_lo) == (0.9, 0.1)
    assert cfg.gen.solver_temp == 0.8              # decision: no hotter sampling
    assert cfg.model.enable_thinking is True
    assert cfg.train.grpo_microbatch == 1
