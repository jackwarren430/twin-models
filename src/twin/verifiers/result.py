"""The single result type every verifier returns (DESIGN.md §7)."""

from dataclasses import dataclass


@dataclass
class VerificationResult:
    correct: bool
    score: float          # 0..1 (binary verifiers use 0.0/1.0)
    detail: str           # human-readable reason, for logging
    method: str           # "sympy" | "exec" | "judge" | "unverifiable"

    @classmethod
    def ok(cls, method: str, detail: str = "", score: float = 1.0) -> "VerificationResult":
        return cls(correct=True, score=score, detail=detail, method=method)

    @classmethod
    def fail(cls, method: str, detail: str = "", score: float = 0.0) -> "VerificationResult":
        return cls(correct=False, score=score, detail=detail, method=method)
