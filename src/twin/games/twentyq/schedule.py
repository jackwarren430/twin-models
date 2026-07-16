"""Small, model-free scheduling helpers for the TwentyQ launcher."""


def validation_steps(start_iter: int, stop_iter: int, every: int) -> tuple[int, ...]:
    """Scheduled completed-step labels, including a fresh run's step 0."""
    if every <= 0:
        return ()
    steps = [0] if start_iter == 0 else []
    steps.extend(
        step for step in range(start_iter + 1, stop_iter + 1)
        if step % every == 0
    )
    return tuple(steps)
