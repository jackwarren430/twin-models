"""Role assignment and rotation (DESIGN.md §4).

Adapters are owned by *models* (A keeps θ_A, B keeps θ_B); the role manager only
decides, per iteration, which model **creates** and which **solves**, and flips
that assignment every ``swap_interval`` iterations. Rotation IS the injection
mechanism: when a model that has been solving becomes the creator it carries its
just-trained adapter into the creating role — no weight surgery (DESIGN §12 Q1).

Explicit cross-model LoRA blending at a swap stays available as a Sprint-4
ablation (``injection_rate > 0`` -> :meth:`maybe_blend`), but is off by default.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Assignment:
    creator: str       # "A" | "B"
    solver: str        # the other one
    iteration: int
    n_swaps: int       # how many role flips have happened by this iteration


class RoleManager:
    def __init__(
        self,
        *,
        swap_interval: int = 100,
        warmup_iterations: int = 0,
        injection_rate: float = 0.0,
        names: tuple[str, str] = ("A", "B"),
    ):
        self.swap_interval = swap_interval
        self.warmup_iterations = warmup_iterations
        self.injection_rate = injection_rate
        self.names = names

    def n_swaps(self, iteration: int) -> int:
        """Number of role flips that have occurred by ``iteration``. No swaps
        during warmup; afterwards one every ``swap_interval`` iterations.
        ``swap_interval <= 0`` disables swapping entirely."""
        if self.swap_interval <= 0 or iteration < self.warmup_iterations:
            return 0
        return (iteration - self.warmup_iterations) // self.swap_interval

    def assignment(self, iteration: int) -> Assignment:
        swaps = self.n_swaps(iteration)
        a, b = self.names
        creator, solver = (a, b) if swaps % 2 == 0 else (b, a)
        return Assignment(creator=creator, solver=solver,
                          iteration=iteration, n_swaps=swaps)

    def is_swap_iteration(self, iteration: int) -> bool:
        """True if the assignment at ``iteration`` differs from ``iteration-1``
        (i.e. a flip lands exactly here). Iteration 0 is never a swap."""
        if iteration <= 0:
            return False
        return self.n_swaps(iteration) != self.n_swaps(iteration - 1)

    def maybe_blend(self, adapters, assignment: Assignment) -> bool:
        """At a swap, optionally blend the outgoing solver's adapter into the
        incoming creator's (DESIGN §4 ablation). No-op unless
        ``injection_rate > 0``. Returns whether a blend happened.

        ``assignment`` is the *new* (post-swap) assignment; the incoming creator
        receives a fraction of what is now the solver's adapter."""
        if self.injection_rate <= 0.0:
            return False
        adapters.blend_into(assignment.creator, assignment.solver, self.injection_rate)
        return True
