"""Typed configuration for twin-models, with a tolerant YAML loader.

Every section is a dataclass with sensible defaults so that a partial YAML
(e.g. configs/tiny.yaml) still produces a fully-populated Config. Unknown keys
in the YAML raise, to catch typos early.
"""

# NOTE: deliberately NOT using `from __future__ import annotations` — _build()
# inspects real field types at runtime to recurse into nested dataclasses, which
# would break if annotations were stringized. Requires Python >=3.11 for the
# `X | None` annotations below to evaluate at class-definition time.

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/.cache/lm-studio/models/mlx-community/Qwen3-8B-6bit"
)


@dataclass
class ModelConfig:
    path: str = DEFAULT_MODEL_PATH
    enable_thinking: bool = False  # Qwen3 thinking mode; off for v1 throughput
    max_kv_size: int | None = None


@dataclass
class LoraConfig:
    num_layers: int = 16          # convert the last N transformer blocks
    rank: int = 16
    scale: float = 20.0           # LoRA alpha-equivalent (mlx-lm uses `scale`)
    dropout: float = 0.0
    keys: list[str] | None = None  # None => all linear layers in those blocks


@dataclass
class GenConfig:
    """Sampling settings per call-site."""
    creator_temp: float = 0.9
    solver_temp: float = 0.8
    oracle_temp: float = 0.2
    top_p: float = 0.95
    creator_max_tokens: int = 1024
    solver_max_tokens: int = 768
    oracle_max_tokens: int = 512


@dataclass
class GameConfig:
    n_problems: int = 5           # N: problems per suite (difficulty ramp)
    creator_group: int = 4        # G_c: candidate suites per creator prompt (GRPO group)
    solver_attempts: int = 4      # K: solver attempts per problem (GRPO group)
    domains: list[str] = field(default_factory=lambda: ["math", "coding"])


@dataclass
class RewardsConfig:
    w_gradient: float = 1.0
    w_consistency: float = 0.5
    w_oracle: float = 0.1         # creator oracle tax (per call)
    w_valid: float = 0.1
    w_solve: float = 1.0
    w_oracle_solver: float = 0.05  # solver oracle tax (per call)
    mse_beta: float = 4.0         # sharpness of exp(-beta*MSE) gradient reward
    clip: float = 10.0


@dataclass
class OracleConfig:
    enabled: bool = True
    cost_per_query: float = 1.0   # counted; multiplied by reward weight elsewhere
    max_calls_per_turn: int = 4


@dataclass
class RolesConfig:
    swap_interval: int = 100      # iterations between role flips
    warmup_iterations: int = 0    # no swaps before this
    injection_rate: float = 0.0   # explicit cross-model LoRA blend at swap (0 = off)


@dataclass
class TrainConfig:
    iters: int = 1000
    learning_rate: float = 1.0e-5
    kl_beta: float = 0.02
    grad_accumulation_steps: int = 1
    seed: int = 0
    checkpoint_every: int = 100
    log_every: int = 1


@dataclass
class PathsConfig:
    checkpoints: str = "checkpoints"
    runs: str = "runs"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    gen: GenConfig = field(default_factory=GenConfig)
    game: GameConfig = field(default_factory=GameConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    roles: RolesConfig = field(default_factory=RolesConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return _build(cls, data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return _build(cls, data)


def _build(dc_type: type, data: dict[str, Any]):
    """Recursively build a (possibly nested) dataclass from a dict, applying
    defaults for absent keys and raising on unknown keys."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping for {dc_type.__name__}, got {type(data).__name__}")
    known = {f.name: f for f in fields(dc_type)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown config keys for {dc_type.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[name] = _build(f.type, value)
        else:
            kwargs[name] = value
    return dc_type(**kwargs)
