"""Sprint 4 — the scale config turns thinking on. Fast: no model.

Locks in the DESIGN §12-Q4 decision: thinking mode is OFF in tiny.yaml (bring the
loop up fast) and ON in base.yaml (the Sprint-4 scale config), with generation
budgets sized to leave the <think> block room.
"""

from twin.config import Config


def test_base_config_enables_thinking():
    cfg = Config.from_yaml("configs/base.yaml")
    assert cfg.model.enable_thinking is True


def test_base_config_budgets_have_room_for_thinking():
    cfg = Config.from_yaml("configs/base.yaml")
    # bigger than tiny so the think block doesn't crowd out the answer/JSON
    assert cfg.gen.creator_max_tokens >= 2048
    assert cfg.gen.solver_max_tokens >= 1024
    assert cfg.gen.oracle_max_tokens >= 1024


def test_tiny_config_keeps_thinking_off_for_speed():
    cfg = Config.from_yaml("configs/tiny.yaml")
    assert cfg.model.enable_thinking is False


def test_base_config_is_the_bigger_loop():
    tiny = Config.from_yaml("configs/tiny.yaml")
    base = Config.from_yaml("configs/base.yaml")
    assert base.lora.num_layers > tiny.lora.num_layers
    assert base.game.n_problems >= tiny.game.n_problems
    assert base.game.creator_group >= tiny.game.creator_group
    assert base.train.iters >= tiny.train.iters
