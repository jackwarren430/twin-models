"""Reward engine (DESIGN.md §6).

Pure arithmetic: it turns already-measured quantities (realized solve rates,
consistency flags, oracle-call counts) into scalar creator/solver rewards. It
does **not** run the model or the verifiers — the trainer (Sprint 3) gathers the
measurements and feeds them here, which keeps the reward maths trivial to test.

Creator (per candidate suite):
    R = w_grad·R_gradient + w_cons·R_consistency − w_oracle·n_oracle + w_valid·R_valid
Solver (per attempt):
    R = w_solve·solved − w_oracle_s·n_oracle
"""

import math
from dataclasses import dataclass, field

from twin.config import RewardsConfig
from twin.problems.schema import ProblemSuite


@dataclass
class CreatorReward:
    total: float
    r_gradient: float          # raw, 0..1 (curve fit)
    r_consistency: float       # raw, 0..1 (fraction consistent)
    r_valid: float             # raw, 0/1 (well-formed suite)
    n_oracle_calls: int
    solve_rates_by_rank: list[float] = field(default_factory=list)
    target_curve: list[float] = field(default_factory=list)


@dataclass
class SolverReward:
    total: float
    solved: bool
    n_oracle_calls: int


class RewardEngine:
    def __init__(self, cfg: RewardsConfig | None = None, *, parse_gate_reward: float = -1.0):
        self.cfg = cfg or RewardsConfig()
        # Fixed low reward for an unparseable creator rollout (DESIGN §6.3).
        self.parse_gate_reward = parse_gate_reward

    # ----- solver ----------------------------------------------------------
    def solver_reward(self, solved: bool, n_oracle_calls: int = 0) -> SolverReward:
        c = self.cfg
        total = c.w_solve * (1.0 if solved else 0.0) - c.w_oracle_solver * n_oracle_calls
        return SolverReward(total=self._clip(total), solved=bool(solved),
                            n_oracle_calls=n_oracle_calls)

    def solver_parse_gate(self) -> SolverReward:
        """Unparseable solver attempt -> reward 0 (DESIGN §6.3)."""
        return SolverReward(total=0.0, solved=False, n_oracle_calls=0)

    # ----- creator ---------------------------------------------------------
    def creator_reward(
        self,
        suite: ProblemSuite,
        solve_rates: list[float],
        consistent_flags: list[bool],
        n_oracle_calls: int = 0,
        *,
        valid: bool | None = None,
    ) -> CreatorReward:
        """``solve_rates`` and ``consistent_flags`` are parallel to
        ``suite.problems`` (NOT pre-sorted). The gradient term internally
        reorders solve rates easy->hard before comparing to the target curve."""
        n = len(suite.problems)
        if len(solve_rates) != n or len(consistent_flags) != n:
            raise ValueError(
                f"expected {n} solve_rates/consistent_flags, got "
                f"{len(solve_rates)}/{len(consistent_flags)}"
            )
        c = self.cfg

        rates_by_rank = self._rates_by_rank(suite, solve_rates)
        target = ProblemSuite.target_curve(n)
        r_gradient = self._gradient(rates_by_rank, target)
        r_consistency = (sum(1 for f in consistent_flags if f) / n) if n else 0.0
        is_valid = suite.is_valid() if valid is None else bool(valid)
        r_valid = 1.0 if is_valid else 0.0

        total = (
            c.w_gradient * r_gradient
            + c.w_consistency * r_consistency
            - c.w_oracle * n_oracle_calls
            + c.w_valid * r_valid
        )
        return CreatorReward(
            total=self._clip(total),
            r_gradient=r_gradient,
            r_consistency=r_consistency,
            r_valid=r_valid,
            n_oracle_calls=n_oracle_calls,
            solve_rates_by_rank=rates_by_rank,
            target_curve=target,
        )

    def creator_parse_gate(self) -> CreatorReward:
        """Unparseable creator rollout -> fixed low reward, not fed to solver."""
        return CreatorReward(
            total=self._clip(self.parse_gate_reward),
            r_gradient=0.0, r_consistency=0.0, r_valid=0.0, n_oracle_calls=0,
        )

    # ----- helpers ---------------------------------------------------------
    @staticmethod
    def _rates_by_rank(suite: ProblemSuite, solve_rates: list[float]) -> list[float]:
        order = sorted(range(len(suite.problems)), key=lambda i: suite.problems[i].difficulty)
        return [float(solve_rates[i]) for i in order]

    def _gradient(self, rates_by_rank: list[float], target: list[float]) -> float:
        if not rates_by_rank:
            return 0.0
        mse = sum((p - t) ** 2 for p, t in zip(rates_by_rank, target)) / len(rates_by_rank)
        return math.exp(-self.cfg.mse_beta * mse)

    def _clip(self, x: float) -> float:
        lo, hi = -self.cfg.clip, self.cfg.clip
        return max(lo, min(hi, float(x)))


def solve_rate(attempt_flags: list[bool]) -> float:
    """Realized solve rate p = (#solved)/K for one problem's attempts."""
    if not attempt_flags:
        return 0.0
    return sum(1 for f in attempt_flags if f) / len(attempt_flags)
