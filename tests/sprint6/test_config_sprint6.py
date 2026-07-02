"""Sprint 6 — new config knobs and the mini-02 run config."""

from pathlib import Path

from twin.config import Config, RewardsConfig, TrainConfig

REPO = Path(__file__).resolve().parents[2]


def test_defaults():
    # mean baseline is the new code-wide default (a fix, not an ablation)...
    assert TrainConfig().adv_mode == "mean"
    # ...but the target band defaults to the original ramp (ablation opt-in)
    assert RewardsConfig().target_hi == 1.0
    assert RewardsConfig().target_lo == 0.0


def test_mini2_yaml_is_the_sprint6_run():
    cfg = Config.from_yaml(REPO / "configs" / "mini2.yaml")
    assert cfg.game.creator_group == 4            # G_c 2 -> 4
    assert cfg.rewards.target_hi == 0.9           # interior band
    assert cfg.rewards.target_lo == 0.1
    assert cfg.train.adv_mode == "mean"           # Dr.GRPO mean baseline
    # everything else matches mini-01 (spot-check the load-bearing knobs)
    mini = Config.from_yaml(REPO / "configs" / "mini.yaml")
    assert cfg.game.n_problems == mini.game.n_problems == 3
    assert cfg.game.solver_attempts == mini.game.solver_attempts == 4
    assert cfg.train.grpo_microbatch == mini.train.grpo_microbatch == 1
    assert cfg.roles.swap_interval == mini.roles.swap_interval == 8
    assert cfg.model.enable_thinking and mini.model.enable_thinking
