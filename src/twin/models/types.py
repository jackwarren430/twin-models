"""Backend-neutral rollout value types + token bookkeeping.

These are plain-Python dataclasses and a pure function with NO framework
dependency (no mlx, no torch). They live here — rather than in
``twin.models.base``, which imports mlx at module top — so the torch backend
(``twin.backends.torch_backend``) can produce/consume the same ``GenResult`` /
``ReactResult`` shapes on a machine that has no mlx installed.

``twin.models.base`` re-exports all three, so ``from twin.models.base import
GenResult`` / ``from twin.models import GenResult`` keep working unchanged."""

from dataclasses import dataclass


@dataclass
class GenResult:
    text: str
    prompt_tokens: list[int]
    completion_tokens: list[int]


@dataclass
class ReactResult:
    """A multi-turn (ReAct) rollout: the model's text interleaved with injected
    tool observations, plus the token bookkeeping GRPO needs. ``completion_tokens``
    is the FULL spliced sequence (policy tokens + injected ``<obs>`` tokens), so
    the forward pass conditions on the observations; ``loss_mask`` marks which of
    those are policy-sampled (1) vs injected (0)."""

    text: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
    loss_mask: list[int]
    n_rounds: int = 0
    n_tool_calls: int = 0


def assemble_react(segments: list[tuple[list[int], bool]]) -> tuple[list[int], list[int]]:
    """Flatten ``(tokens, is_policy)`` segments into ``(completion_ids,
    loss_mask)``. Pure bookkeeping, factored out so the masking is unit-testable
    without loading the model."""
    ids: list[int] = []
    mask: list[int] = []
    for tokens, is_policy in segments:
        ids.extend(tokens)
        mask.extend([1 if is_policy else 0] * len(tokens))
    return ids, mask
