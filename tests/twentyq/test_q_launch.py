"""TwentyQ launch scheduling (model-free)."""

from twin.games.twentyq.schedule import validation_steps


def test_fresh_sixty_iteration_run_validates_at_zero_and_tens():
    assert validation_steps(0, 60, 10) == (0, 10, 20, 30, 40, 50, 60)


def test_resume_does_not_repeat_the_checkpoint_or_step_zero():
    assert validation_steps(30, 60, 10) == (40, 50, 60)
    assert validation_steps(34, 60, 10) == (40, 50, 60)


def test_zero_disables_validation():
    assert validation_steps(0, 60, 0) == ()
