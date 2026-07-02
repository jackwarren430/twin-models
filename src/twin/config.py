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


# Every linear projection in a Qwen3 transformer block, by module path relative
# to the block. Adapting all 7 (attention q/k/v/o + MLP gate/up/down) is the
# "full" LoRA target set; this is also what mlx-lm's auto-discovery (keys=None)
# selects, but we name them explicitly so the target set is visible and locked.
DEFAULT_LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


@dataclass
class LoraConfig:
    num_layers: int = 16          # convert the last N transformer blocks
    rank: int = 64                # LoRA rank r
    alpha: float = 128.0          # LoRA alpha; effective mlx-lm scale = alpha / rank
    dropout: float = 0.0
    # Which projections to adapt. Default = all 7 (DEFAULT_LORA_KEYS). Set to
    # `null` in YAML to fall back to mlx-lm auto-discovery (same 7 for Qwen3).
    keys: list[str] | None = field(default_factory=lambda: list(DEFAULT_LORA_KEYS))
    # Raw mlx-lm scale override. Leave null to use the PEFT-standard alpha/rank;
    # set a number only if you want to bypass alpha and pin the scale directly.
    scale: float | None = None

    @property
    def effective_scale(self) -> float:
        """The multiplier mlx-lm applies to the LoRA delta (``y + scale·BA·x``).
        PEFT convention: ``alpha / rank``, unless a raw ``scale`` is pinned."""
        return self.scale if self.scale is not None else self.alpha / self.rank


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
    # Brevity bonus for CORRECT solver attempts only: w_brevity * (1 - L/L_max)
    # where L = completion tokens, L_max = gen.solver_max_tokens. Gated on
    # solved so it can never reward truncating into a wrong answer, and capped
    # at w_brevity so a correct answer always out-scores an incorrect one.
    # Default 0.0 (off) — enable per-run as a measured ablation (DESIGN §6.2).
    w_brevity: float = 0.0
    mse_beta: float = 4.0         # sharpness of exp(-beta*MSE) gradient reward
    clip: float = 10.0
    # Target solve-rate ramp endpoints (easy rank -> hard rank). The default
    # 1.0 -> 0.0 ramp makes the endpoint ranks' solver K-groups zero-variance
    # (all-solved / all-failed) at creator-optimum; an interior band like
    # 0.9 -> 0.1 keeps outcome variance at every rank. Changing the band
    # changes the game's incentives — run it as an ablation (DESIGN §11 S6).
    target_hi: float = 1.0
    target_lo: float = 0.0


@dataclass
class OracleConfig:
    enabled: bool = True
    cost_per_query: float = 1.0   # counted; multiplied by reward weight elsewhere
    max_calls_per_turn: int = 4


@dataclass
class ToolsConfig:
    """Inline tool use for the CREATOR rollout (DESIGN.md §9). The creator may
    call these tools mid-generation (ReAct); results are spliced back as <obs>
    and masked out of the GRPO loss. The solver path has no inline tools in v1.

    The JUDGE (verifier fallback, frozen base) also gets the CAS tool so it can
    recompute answers instead of eyeballing them — judge tool use is untaxed and
    never scored (it's a verification activity, not a trained rollout)."""
    creator_tools: list[str] = field(default_factory=lambda: ["solve", "calc"])
    creator_max_tool_calls: int = 4   # ToolHarness budget per creator rollout
    creator_tool_rounds: int = 4      # max ReAct rounds (generate->tool->continue)
    cas_timeout_s: float = 3.0        # best-effort wall-clock guard for `solve`
    judge_tools: list[str] = field(default_factory=lambda: ["solve", "calc"])
    judge_max_tool_calls: int = 4     # ToolHarness budget per judge call
    judge_tool_rounds: int = 4        # max ReAct rounds for the judge


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
    grad_clip: float = 1.0        # global-norm gradient clip per GRPO step (0 = off)
    grad_accumulation_steps: int = 1
    # Max trajectories scored per GRPO backward chunk; grads are accumulated
    # across chunks for one optimizer step (identical math, bounded peak memory).
    # 0 = whole batch in one graph (fine for tiny configs / short token budgets).
    # Set >0 when long token budgets * many trajectories would OOM the backward
    # pass — a 4096-token completion alone materializes a ~2.5GB [T,V] logits
    # tensor, so a dozen at once blows past 32GB.
    grpo_microbatch: int = 0
    # Group-advantage normalization (twin.rl.group_advantages). "mean" (the
    # default) is Dr.GRPO-style A = R − mean(R): for the small groups this
    # project runs (G_c 2-4, K=4) the classic ÷std maps ANY non-tie to ±1 —
    # a 0.015 reward gap (solve-rate noise) trains as hard as a 2.6 one —
    # while "mean" preserves magnitude (near-ties -> near-zero gradients).
    # "std" is the legacy standardized form, kept for ablation.
    adv_mode: str = "mean"
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
    tools: ToolsConfig = field(default_factory=ToolsConfig)
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
