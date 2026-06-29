"""``oracle`` tool: a base-only factual query, **counted and taxed**.

The oracle is the frozen base (zeroed adapter), so it never reflects the policy
being trained. Every call is counted; the reward engine applies the per-call tax
(DESIGN.md §6/§9). The point of training is *self-reliance*, so leaning on the
oracle should cost something.

``OracleTool`` wraps a :class:`~twin.models.base.TwinBase` + ``Adapters`` and
guarantees the base adapter is active for the duration of the call. For unit
tests (no model) build one from a plain callable via :meth:`from_callable`.
"""

from typing import Callable


class OracleTool:
    name = "oracle"

    def __init__(self, query_fn: Callable[[str], str], *, max_calls: int | None = None):
        self._query_fn = query_fn
        self.max_calls = max_calls
        self.n_calls = 0

    # ----- construction ----------------------------------------------------
    @classmethod
    def from_model(cls, base, adapters, *, max_tokens: int = 512, temp: float = 0.2,
                   max_calls: int | None = None) -> "OracleTool":
        """Bind to the real model: activate the zeroed ('base') adapter around
        each query so the response is pure base behaviour."""
        def _query(question: str) -> str:
            with adapters.using("base"):
                return base.oracle(question, max_tokens=max_tokens, temp=temp)
        return cls(_query, max_calls=max_calls)

    @classmethod
    def from_callable(cls, fn: Callable[[str], str], *, max_calls: int | None = None) -> "OracleTool":
        return cls(fn, max_calls=max_calls)

    # ----- use -------------------------------------------------------------
    def __call__(self, question: str) -> str:
        if self.max_calls is not None and self.n_calls >= self.max_calls:
            return "error: oracle call limit reached for this turn"
        self.n_calls += 1
        try:
            return self._query_fn(question)
        except Exception as e:  # noqa: BLE001 - tool boundary
            return f"error: {type(e).__name__}: {e}"

    def reset(self) -> None:
        """Zero the call counter (call at the start of each rollout/turn)."""
        self.n_calls = 0
