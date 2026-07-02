"""Sprint 3 — `assemble_react` token/mask bookkeeping. Fast: no model.

The masking logic for inline ReAct rollouts is factored out of the
model-touching ``generate_react`` so it can be tested directly: policy segments
get mask 1, injected (tool-observation) segments get mask 0."""

from twin.models import assemble_react


def test_concatenates_and_masks():
    ids, mask = assemble_react([([1, 2, 3], True), ([9, 9], False), ([4], True)])
    assert ids == [1, 2, 3, 9, 9, 4]
    assert mask == [1, 1, 1, 0, 0, 1]


def test_all_policy():
    ids, mask = assemble_react([([5, 6], True)])
    assert ids == [5, 6]
    assert mask == [1, 1]


def test_empty():
    assert assemble_react([]) == ([], [])


def test_lengths_align():
    ids, mask = assemble_react([([1], True), ([2, 3], False), ([4, 5, 6], True)])
    assert len(ids) == len(mask)
