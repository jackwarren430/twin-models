"""Reward engine (DESIGN.md §6).

Pure arithmetic: it turns already-measured quantities (realized solve rates,
consistency flags, oracle-call counts) into scalar creator/solver rewards. It
does **not** run the model or the verifiers — the trainer (Sprint 3) gathers the
measurements and feeds them here, which keeps the reward maths trivial to test.

Creator (per candidate suite):
    R = w_grad·R_gradient + w_cons·R_consistency − w_oracle·n_oracle + w_valid·R_valid
Solver (per attempt):
    R = w_solve·solved + w_brevity·(1 − L/L_max)·solved − w_oracle_s·n_oracle
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
    r_brevity: float = 0.0     # raw, 0..1 (1 = empty completion, 0 = at budget)


class RewardEngine:
    def __init__(self, cfg: RewardsConfig | None = None, *, parse_gate_reward: float = -1.0):
        self.cfg = cfg or RewardsConfig()
        # Fixed low reward for an unparseable creator rollout (DESIGN §6.3).
        self.parse_gate_reward = parse_gate_reward

    # ----- solver ----------------------------------------------------------
    def solver_reward(
        self,
        solved: bool,
        n_oracle_calls: int = 0,
        *,
        n_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> SolverReward:
        """``n_tokens``/``max_tokens`` (completion length vs generation budget)
        feed the brevity bonus: w_brevity * (1 - L/L_max), paid ONLY on solved
        attempts so brevity can never compensate for being wrong — a correct
        answer's reward stays >= w_solve while an incorrect one stays <= 0.
        Either arg ``None`` (or w_brevity == 0) disables the term."""
        c = self.cfg
        r_brevity = 0.0
        if solved and c.w_brevity and n_tokens is not None and max_tokens:
            r_brevity = min(1.0, max(0.0, 1.0 - n_tokens / max_tokens))
        total = (
            c.w_solve * (1.0 if solved else 0.0)
            + c.w_brevity * r_brevity
            - c.w_oracle_solver * n_oracle_calls
        )
        return SolverReward(total=self._clip(total), solved=bool(solved),
                            n_oracle_calls=n_oracle_calls, r_brevity=r_brevity)

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
        scored_mask: list[bool] | None = None,
        expected_n: int | None = None,
        target_by_problem: list[float] | None = None,
    ) -> CreatorReward:
        """``solve_rates`` and ``consistent_flags`` are parallel to
        ``suite.problems`` (NOT pre-sorted). The gradient term internally
        reorders solve rates easy->hard before comparing to the target curve.

        ``scored_mask`` (parallel to the problems) marks which problems actually
        have a measured solve rate. Void problems — those that failed the
        consistency check, whose stated answer we don't trust enough to grade
        the solver against (DESIGN.md §6.3) — are excluded from the gradient
        curve so they can't be farmed as "hard" problems; they still drag
        ``R_consistency`` down. ``None`` => every problem scored (back-compat).

        The gradient term is **scaled by the scored fraction** (Sprint 5): the
        curve fit is measured over the scored subset against a re-stretched
        target ramp, so without the scaling a suite with ONE consistent easy
        problem fits its [1.0] target perfectly and nearly out-earns an honest
        fully-consistent ramp (mini-01 showed exactly this: reward 1.27-1.43
        for mostly-void suites vs 1.6 for the true ideal). Scaling by
        n_scored/n makes voiding problems strictly unprofitable.

        ``expected_n`` (Sprint 7 per-problem mode, audit 2026-07-03) is the
        number of problems the suite was *supposed* to have. A rank whose
        rollout failed to parse is absent from ``suite.problems`` entirely, so
        the Sprint-5 n_scored/n scaling never saw it — a 3/5-parsed suite
        fitting its re-stretched 3-rank ramp out-earned an honest 5/5 (repro:
        1.598 vs 1.503). With ``expected_n`` the scored fraction is measured
        against the intended size, making parse-dropping strictly
        unprofitable too. ``target_by_problem`` (parallel to
        ``suite.problems``) pins each problem's target to the solve rate its
        rank was PROMPTED with, instead of re-stretching a fresh ramp over
        whatever subset parsed — keeping "prompt targets == reward targets"
        true under partial parses. Both default to ``None`` == pre-audit
        behaviour."""
        n = len(suite.problems)
        if len(solve_rates) != n or len(consistent_flags) != n:
            raise ValueError(
                f"expected {n} solve_rates/consistent_flags, got "
                f"{len(solve_rates)}/{len(consistent_flags)}"
            )
        if scored_mask is not None and len(scored_mask) != n:
            raise ValueError(f"expected {n} scored_mask entries, got {len(scored_mask)}")
        if target_by_problem is not None and len(target_by_problem) != n:
            raise ValueError(
                f"expected {n} target_by_problem entries, got {len(target_by_problem)}")
        if expected_n is not None and expected_n < n:
            raise ValueError(f"expected_n={expected_n} < {n} problems in suite")
        c = self.cfg

        rates_by_rank, target = self._rates_and_targets_by_rank(
            suite, solve_rates, scored_mask, target_by_problem)
        r_gradient = self._gradient(rates_by_rank, target)
        denom = expected_n if expected_n else n
        if (scored_mask is not None or expected_n is not None) and denom:
            n_scored = (sum(1 for m in scored_mask if m)
                        if scored_mask is not None else n)
            r_gradient *= n_scored / denom
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

    def creator_problem_rewards(
        self,
        suite: ProblemSuite,
        solve_rates: list[float],
        consistent_flags: list[bool],
        *,
        valid: bool | None = None,
        scored_mask: list[bool] | None = None,
        target_by_problem: list[float] | None = None,
        n_oracle_by_problem: list[int] | None = None,
    ) -> list[float]:
        """Per-problem creator rewards (Sprint 8 credit decomposition, the
        TRR++ direction — RELATED_WORK.md §1): rank i earns its OWN
        calibration fit ``w_grad·exp(-β·(p_i - t_i)²)`` (0 if unscored), its
        own consistency flag, its own oracle tax, and the shared suite-level
        validity. Used in per-problem creator mode instead of broadcasting
        the suite reward to all N trajectories, so a rank that nailed its
        target is not dragged by a sibling that missed. Same weights and clip
        as :meth:`creator_reward`; parallel to ``suite.problems``."""
        n = len(suite.problems)
        if len(solve_rates) != n or len(consistent_flags) != n:
            raise ValueError(
                f"expected {n} solve_rates/consistent_flags, got "
                f"{len(solve_rates)}/{len(consistent_flags)}"
            )
        for name, xs in (("scored_mask", scored_mask),
                         ("target_by_problem", target_by_problem),
                         ("n_oracle_by_problem", n_oracle_by_problem)):
            if xs is not None and len(xs) != n:
                raise ValueError(f"expected {n} {name} entries, got {len(xs)}")
        c = self.cfg
        if target_by_problem is None:
            ranked = sorted(range(n), key=lambda i: suite.problems[i].difficulty)
            curve = ProblemSuite.target_curve(n, hi=c.target_hi, lo=c.target_lo)
            targets = [0.0] * n
            for rank, i in enumerate(ranked):
                targets[i] = curve[rank]
        else:
            targets = [float(t) for t in target_by_problem]
        is_valid = suite.is_valid() if valid is None else bool(valid)
        rewards: list[float] = []
        for i in range(n):
            scored = scored_mask[i] if scored_mask is not None else True
            r_grad = (
                math.exp(-c.mse_beta * (float(solve_rates[i]) - targets[i]) ** 2)
                if scored else 0.0
            )
            n_oracle = n_oracle_by_problem[i] if n_oracle_by_problem else 0
            rewards.append(self._clip(
                c.w_gradient * r_grad
                + c.w_consistency * (1.0 if consistent_flags[i] else 0.0)
                - c.w_oracle * n_oracle
                + c.w_valid * (1.0 if is_valid else 0.0)
            ))
        return rewards

    def creator_parse_gate(self) -> CreatorReward:
        """Unparseable creator rollout -> fixed low reward, not fed to solver."""
        return CreatorReward(
            total=self._clip(self.parse_gate_reward),
            r_gradient=0.0, r_consistency=0.0, r_valid=0.0, n_oracle_calls=0,
        )

    # ----- helpers ---------------------------------------------------------
    @staticmethod
    def _rates_by_rank(
        suite: ProblemSuite,
        solve_rates: list[float],
        scored_mask: list[bool] | None = None,
    ) -> list[float]:
        idx = [
            i for i in range(len(suite.problems))
            if scored_mask is None or scored_mask[i]
        ]
        idx.sort(key=lambda i: suite.problems[i].difficulty)
        return [float(solve_rates[i]) for i in idx]

    def _rates_and_targets_by_rank(
        self,
        suite: ProblemSuite,
        solve_rates: list[float],
        scored_mask: list[bool] | None,
        target_by_problem: list[float] | None,
    ) -> tuple[list[float], list[float]]:
        """Scored solve rates sorted easy->hard, paired with the target each
        problem is graded against: its own prompted target when
        ``target_by_problem`` is given, else the re-stretched ramp."""
        idx = [
            i for i in range(len(suite.problems))
            if scored_mask is None or scored_mask[i]
        ]
        idx.sort(key=lambda i: suite.problems[i].difficulty)
        rates = [float(solve_rates[i]) for i in idx]
        if target_by_problem is not None:
            return rates, [float(target_by_problem[i]) for i in idx]
        return rates, ProblemSuite.target_curve(
            len(rates), hi=self.cfg.target_hi, lo=self.cfg.target_lo)

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
