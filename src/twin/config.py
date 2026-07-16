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
    # Backend-specific weights. The MLX backend wants an mlx-community MLX dir
    # (the 6-bit Qwen3-8B default below); the torch backend wants a HuggingFace
    # checkpoint — a repo id ("Qwen/Qwen3-8B") or a local HF dir — loaded in
    # bf16. Set this alongside compute.backend (see configs/spark.yaml).
    path: str = DEFAULT_MODEL_PATH
    enable_thinking: bool = False  # Qwen3 thinking mode; off for v1 throughput
    # Creator-rollout override for thinking (2026-07-04): null = follow
    # enable_thinking. mini-04(a) measured the creator simulating the CAS
    # inside <think> on 40/40 rollouts and truncating in hard-rank
    # deliberation spirals — denying the creator its thinking space (while
    # the solver keeps it) is the structural fallback if prompt-level
    # pressure can't produce real tool calls, and an ablation lever besides.
    creator_enable_thinking: bool | None = None
    max_kv_size: int | None = None
    # Weight-loading overrides (2026-07-11, gemma4). load_strict=False drops
    # checkpoint tensors absent from the mlx-lm model: gemma4-E2B ships per-layer
    # k/v for its 20 KV-shared layers (15-34) that mlx-lm reuses from earlier
    # layers, so they're vestigial and MUST be dropped or load raises. Qwen3
    # keeps the default True (its checkpoint matches exactly, so this is a no-op
    # safety net there). eos_token_ids overrides the stop-token set: mlx-lm reads
    # config.json's scalar eos_token_id (gemma4 = <eos> only), missing that
    # gemma4 declares its turn terminator <turn|>=106 ONLY in generation_config
    # ([1,106,50]) — without it generation never stops and pads to max_tokens.
    load_strict: bool = True
    eos_token_ids: list[int] | None = None


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
    # Sprint 8 — batched solver generation: sample the K solver attempts at a
    # problem in continuous-batching chunks of this size (they share one
    # prompt, the ideal batching case). 1 = the v1 sequential path, unchanged.
    # >1 changes the RNG *stream* (not the sampling distribution) and trades
    # KV-cache memory (~0.6GB/sequence worst-case at a 4096 budget) for
    # decode throughput. K=8 at batch 8 is the intended mini-05 setting.
    solver_batch: int = 1


@dataclass
class GameConfig:
    n_problems: int = 5           # N: problems per suite (difficulty ramp)
    creator_group: int = 4        # G_c: candidate suites per creator prompt (GRPO group)
    solver_attempts: int = 4      # K: solver attempts per problem (GRPO group)
    domains: list[str] = field(default_factory=lambda: ["math", "coding"])
    # Sprint 7 — creator generation architecture.
    # "suite": one rollout emits the whole N-problem suite (v1 behaviour).
    # "per_problem": N separate rollouts, one problem each — each problem gets
    #   its own thinking budget, trajectories are shorter (smaller GRPO
    #   backward), and the suite reward is broadcast to all N trajectories.
    creator_mode: str = "suite"
    # per_problem only: show problem k the JSONs of problems 1..k-1 (never
    # their thinking) so the suite stays coherent (distinct problems, real
    # ramp). Prompt-injection only, so it's cheap to ablate.
    condition_on_previous: bool = True
    # Named personas (alpha/omega, glued to adapters A/B) + fair-competition
    # framing injected into creator AND solver prompts. The judge and the
    # held-out benchmark always stay neutral.
    personas: bool = False
    # Strict tool gate: void a problem (consistency=False) when its creator
    # rollout contained no successful tool call — the anti "guess what the
    # tool would return" measure. Off = log-only (per-problem tool usage and
    # answer-appears-in-obs are always recorded in the suite summary).
    require_tool_use: bool = False
    # Sprint 8 — creator credit assignment in per_problem mode.
    # "broadcast": all N rank trajectories share the suite reward (Sprint 7
    #   default — one advantage per suite, as in v1).
    # "per_problem": rank i earns its OWN calibration fit + consistency flag
    #   + oracle tax (suite validity stays shared) via
    #   RewardEngine.creator_problem_rewards — the TRR++ direction
    #   (RELATED_WORK.md §1): a rank that nailed its target is not dragged by
    #   a sibling. Ignored in suite creator_mode (one trajectory per suite).
    credit: str = "broadcast"
    # Sprint 9 — SPICE-style target grounding: path to a theme-weights JSON
    # ({domain: [[theme, weight], ...]}), usually built by
    # scripts/build_grounded_themes.py from a held-out bench result, biasing
    # the per-iteration theme draw toward measured failure topics. null =
    # uniform over the static prompts.THEMES pool (v1 behaviour).
    theme_weights: str | None = None


@dataclass
class TwentyQConfig:
    """21-questions mode (twentyq/DESIGN.md). Read only by TwentyQTrainer;
    the single-step self-play loop ignores this section entirely."""
    n_secrets: int = 3            # N: secrets per iteration (difficulty ramp)
    episodes_per_secret: int = 4  # K: episodes per secret (the GRPO group)
    max_turns: int = 10           # T: guesser turn budget per episode
    categories: list[str] = field(default_factory=lambda: [
        "animal", "food", "household object", "place", "occupation",
    ])
    # Parsed creator secrets retained across iterations and role rotations.
    # Only entries from the active category are shown to the creator.  The
    # current iteration's earlier picks are always excluded separately, so 0
    # disables only the cross-iteration rolling window.
    recent_secret_window: int = 128
    # Credit granularity (DESIGN §2.3). "broadcast": every turn of episode k
    # carries the episode advantage (v1). "per_turn": reward-to-go from judge
    # closeness deltas + terminal reward (Sprint Q7 ablation; needs a judge
    # call per turn instead of per episode).
    # "terminal": terminal reward-to-go with turn-index baselines, but no dense
    # scorer. This is the clean no-ensemble control for credit="ensemble": the
    # grouping and terminal assignment are identical and only dense reward is
    # removed.
    # "ensemble": dense per-turn shaping from an ensemble of FROZEN base LLMs
    # scoring the secret's log-prob given the Q/A history (r_t = gamma*score_t -
    # score_{t-1}, score(history_0) = empty-history baseline), paid alongside the
    # sparse terminal reward. Replaces the judge closeness Φ signal (no per-turn
    # judge call). See games/twentyq/ensemble_reward.py + twentyq-ensemble-reward.
    credit: str = "broadcast"
    gamma: float = 1.0            # per-turn discount (per_turn / ensemble credit)
    # Ensemble-reward knobs (credit="ensemble" only). Empty models list = the
    # whole ENSEMBLE_MODELS registry. w_ensemble scales the per-turn shaping
    # delta (the terminal reward is never scaled); 1.0 = the bare spec formula.
    ensemble_models: list[str] = field(default_factory=list)
    ensemble_device: str = "cuda"
    ensemble_dtype: str = "bfloat16"
    w_ensemble: float = 1.0
    # Per-role thinking, all OFF by default (q-shakeout-01/02 findings). A
    # thinking budget truncates Qwen3 mid-<think> into unusable output: the
    # guesser spent 512 tokens thinking and never wrote a QUESTION line (every
    # turn a format-fail); the creator brainstormed candidates past a 1024
    # budget and emitted no JSON (0/2 secrets parsed). Prompt constraints don't
    # bind inside <think>, so v1 emits contract output directly and lets the
    # REWARD (not visible reasoning) train calibration. Turning a role's
    # thinking back ON is only safe with a GENEROUS budget and is a Sprint-Q8
    # ablation (esp. creator calibration — the mini-04b caution that reward
    # alone may under-ratchet difficulty for a non-thinking creator).
    creator_thinking: bool = False
    guesser_thinking: bool = False
    answerer_thinking: bool = False
    # The twentyq judge (validity / closeness Φ) is pure
    # NL judgment with no arithmetic, so it runs thinking OFF by default: a
    # thinking budget truncated the closeness trace before the CLOSENESS line
    # (q-shakeout-03, Φ None on every episode → dead shaping signal). ON (with
    # a large oracle budget) is a judge-quality Q8 ablation.
    judge_thinking: bool = False
    # Solver episode reward (DESIGN §2.5). Keep w_close < w_guess so a
    # near-miss never outearns a win, and w_efficiency gated on success so
    # rushing can never beat guessing right.
    w_guess: float = 1.0
    w_efficiency: float = 0.3
    w_close: float = 0.5
    w_format: float = 0.5         # penalty when the guesser broke the contract
    # Secret validity gate (judge_secret_validity, DESIGN §2.4). gemma4-E2B
    # often ignores the vetting instruction and role-plays the guesser, ending
    # its turn before any VERDICT line; the pre-fix fail-CLOSED default then
    # voided valid, guessable secrets (Axolotl/Salmon/Octopus in q-fullv2-rot),
    # discarding all their episodes — the same data-loss pathology that retired
    # the answer audit. Modes:
    #   "fail_open"   (default, new strategy) — no parseable verdict => VALID;
    #                 only a clear "VERDICT: INVALID" voids.
    #   "fail_closed" — unparseable => INVALID (pre-fix posture; the v2 CONTROL
    #                 arm uses this so it matches the rotation arm that already ran).
    #   "off"         — skip the gate entirely; every parsed secret is played.
    secret_validity: str = "fail_open"
    # NOTE (2026-07-13): the LLM truthfulness-audit / consistency-void mechanism
    # was removed — the creator is assumed to answer truthfully. On gemma4-E2B
    # the audit false-flagged truthful episodes and abstained on many others,
    # voiding good training data (see twentyq/DESIGN.md). The former
    # `audit_void_fraction` knob is gone; judge_answer_audit() is retained but
    # no longer wired into the training loop.
    # Per-turn generation budgets (history grows linearly in turns — keep
    # these tight; mini-05's uncapped-thinking lesson applies per turn here).
    secret_max_tokens: int = 512
    question_max_tokens: int = 256
    answer_max_tokens: int = 128
    # Repeat handling for creator secrets (twentyq/DESIGN.md §6.5). The v4
    # arm-A run showed the prompt-only exclusion list LOSES to the reward
    # gradient: GRPO net-reinforced one per-category attractor (Okapi +3.5
    # cumulative creator advantage over 18 iterations) and by iter 11 the
    # creator repeated it 8/10 ranks with the exclusion list in-prompt. Modes:
    #   "off"   — prompt-and-measurement only (the v4 arm-A behaviour).
    #   "void"  — a parsed secret that matches the exclusion list it was shown
    #             (repeat_matches: normalized + bare-plural + edit-distance-1
    #             for 5+ chars, catching ban-evasion misspellings like
    #             'Wasbi'/'Wasabi') is voided: NO episodes, fixed
    #             repeat_gate_reward in the creator GRPO group.
    #   "retry" — "void" plus ONE masked resample: the same prompt is decoded
    #             again at creator_temp with every excluded secret banned at
    #             the logits level (the decoder takes the next-most-likely
    #             non-excluded continuation — no blind resampling). A retry
    #             that parses non-repeat plays the rank's episodes and trains
    #             the creator with its real game reward; a retry that still
    #             repeats (tokenization-variant slip) voids the rank.
    repeat_handling: str = "off"
    # Fixed reward for a voided repeat rollout: well-formed but disallowed, so
    # below every honest secret (~1.3-1.5) yet above a parse failure (-1.0).
    repeat_gate_reward: float = 0.0
    # Lockstep player generation: active sibling episodes at the same turn are
    # decoded together in chunks of this size. 1 preserves the serial path.
    # This changes the RNG stream, not the per-prompt sampling distribution.
    generation_batch_size: int = 1
    # Frozen-ensemble history scoring batch. Histories are flattened across the
    # K sibling episodes for one secret and scored in chunks per member model.
    # 1 preserves the historical one-forward-per-history path.
    ensemble_batch_size: int = 1
    # Stationary evaluation. 0 disables it; otherwise the launch script runs a
    # pre-training step-0 baseline and post-update validation after every X
    # completed training iterations. The
    # fixed, versioned set is played by BOTH adapters as guesser against the
    # frozen base answerer, with greedy decoding so checkpoints see identical
    # prompts and validation does not consume the training sampling stream.
    validation_every: int = 0
    validation_secret_set: str = "data/twentyq-validation-v1.json"
    validation_max_turns: int | None = None  # null => use max_turns
    # Write a compact <run>.rewards.jsonl sidecar containing terminal, dense
    # ensemble, combined-immediate, and turn-zero-return aggregates for every
    # training iteration and validation pass. The main JSONL remains complete.
    reward_log: bool = False


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
    # Sprint 9 — cross-suite repetition penalty (the R-Zero-style diversity
    # mechanism; RELATED_WORK.md §2/§5). Per suite: mean nearest-neighbour
    # word-bigram Jaccard of its problems vs the OTHER suites in the same
    # iteration (twin.analysis.diversity), subtracted from the suite reward
    # scaled by this weight. 0 = off (telemetry still always logged as
    # problem_similarity / per-suite repetition). Run as a measured ablation.
    w_diversity: float = 0.0
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
    # Tool-call protocol for the CREATOR rollout (Sprint 7).
    # "react":  legacy text protocol — <tool>name(arg)</tool> / <obs>...</obs>.
    #   Qwen3 was never trained on it; with thinking ON it plans calls inside
    #   <think> and hallucinates the results (mini-03b: ZERO real calls).
    # "native": Qwen3 function calling — tools declared via the chat template
    #   (tools=...), model emits <tool_call>{json}</tool_call>, harness splices
    #   a <tool_response> user turn back (masked out of the GRPO loss).
    # The judge stays on "react" either way (migration queued for Sprint 8).
    protocol: str = "react"
    cas_timeout_s: float = 3.0        # best-effort wall-clock guard for `solve`
    # Wall-clock limit for the sandboxed `run_python` creator tool (Sprint 8
    # coding pipeline). The sandbox also enforces CPU/memory/file limits.
    run_python_timeout_s: float = 5.0
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
    # Sprint 8 — chunked+checkpointed LM-head in completion_logprobs: the
    # transformer runs once for [T, H] hidden states, then the head +
    # log-softmax run over the completion in mx.checkpoint-ed chunks of this
    # many positions (backward rematerializes one [chunk, V] at a time
    # instead of pinning two fp32 [T, V] monsters — the mini-02 probe's 55GB
    # worst case at 4096 budgets). Numerically identical gradients (the head
    # carries no LoRA params). 0/None = plain v1 path; 256 is a sane setting.
    logit_chunk: int = 0
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
    # Also write the INPUT prompts to the raw-text transcript (system prompts
    # once per iteration — they change at rotation when personas are on —
    # and each rollout's user prompt; the K solver attempts share one prompt,
    # logged once per problem). Off by default: prompts are large and mostly
    # templated, so this roughly doubles transcript size. No effect on the
    # JSONL log or when the transcript sink is disabled.
    log_prompts: bool = False


@dataclass
class ComputeConfig:
    """Which compute backend runs the model, plus that backend's native
    performance levers (twin.backends). The default is MLX / Apple silicon —
    the platform the loop was developed on — so an unspecified ``compute``
    section reproduces the historical behaviour exactly.

    ``backend="torch"`` selects the NVIDIA DGX Spark path (GB10 Grace
    Blackwell, sm_121, 128GB unified memory): PyTorch + HuggingFace
    transformers + PEFT LoRA, base loaded bf16. All the ``torch`` fields below
    are read ONLY by the torch backend; the MLX backend ignores them (they're
    parsed either way so a single config can carry both). See DGX_SPARK.md."""

    backend: str = "mlx"          # "mlx" (Apple silicon) | "torch" (DGX Spark / CUDA)
    # ----- torch-only performance levers -----
    device: str = "cuda"          # torch device ("cuda" | "cpu" | "mps")
    dtype: str = "bfloat16"       # base-weight compute dtype ("bfloat16" | "float16" | "float32")
    # Attention kernel: "sdpa" (torch fused, always available), "flash_attention_2"
    # (needs the flash-attn wheel; fastest on Blackwell), or "eager" (reference).
    attn_impl: str = "sdpa"
    compile: bool = False         # torch.compile the scoring forward (opt-in; slow first step)
    tf32: bool = True             # allow TF32 matmul/cuDNN (Ampere+; big speedup, tiny precision cost)
    matmul_precision: str = "high"  # torch.set_float32_matmul_precision ("highest"|"high"|"medium")
    grad_checkpointing: bool = False  # HF gradient checkpointing on the scoring backward


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
    twentyq: TwentyQConfig = field(default_factory=TwentyQConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    roles: RolesConfig = field(default_factory=RolesConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
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
