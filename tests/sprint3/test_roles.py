"""Sprint 3 — role assignment, rotation, warmup, and optional blend. Fast: no model."""

from twin.roles import Assignment, RoleManager


def test_assignment_flips_every_swap_interval():
    rm = RoleManager(swap_interval=4, warmup_iterations=0)
    a0 = rm.assignment(0)
    assert (a0.creator, a0.solver) == ("A", "B")
    assert rm.assignment(3).creator == "A"            # still first block
    assert rm.assignment(4).creator == "B"            # flipped
    assert rm.assignment(8).creator == "A"            # flipped back
    assert isinstance(a0, Assignment)


def test_is_swap_iteration():
    rm = RoleManager(swap_interval=4, warmup_iterations=0)
    assert not rm.is_swap_iteration(0)
    assert not rm.is_swap_iteration(3)
    assert rm.is_swap_iteration(4)
    assert not rm.is_swap_iteration(5)
    assert rm.is_swap_iteration(8)


def test_warmup_delays_first_swap():
    rm = RoleManager(swap_interval=2, warmup_iterations=4)
    for it in range(6):                               # 0..5 all still creator A
        assert rm.assignment(it).creator == "A"
    assert rm.assignment(6).creator == "B"            # first flip at warmup+interval
    assert rm.is_swap_iteration(6)


def test_swap_disabled():
    rm = RoleManager(swap_interval=0)
    assert rm.assignment(1000).creator == "A"
    assert rm.n_swaps(1000) == 0


def test_maybe_blend_off_by_default():
    calls = []

    class FakeAdapters:
        def blend_into(self, dst, src, rate):
            calls.append((dst, src, rate))

    rm = RoleManager(swap_interval=2, injection_rate=0.0)
    assert rm.maybe_blend(FakeAdapters(), rm.assignment(2)) is False
    assert calls == []

    rm2 = RoleManager(swap_interval=2, injection_rate=0.25)
    assign = rm2.assignment(2)                         # creator flipped to B
    assert rm2.maybe_blend(FakeAdapters(), assign) is True


def test_maybe_blend_passes_rate_and_targets():
    captured = {}

    class FakeAdapters:
        def blend_into(self, dst, src, rate):
            captured.update(dst=dst, src=src, rate=rate)

    rm = RoleManager(swap_interval=2, injection_rate=0.3)
    assign = rm.assignment(2)
    rm.maybe_blend(FakeAdapters(), assign)
    assert captured == {"dst": assign.creator, "src": assign.solver, "rate": 0.3}
