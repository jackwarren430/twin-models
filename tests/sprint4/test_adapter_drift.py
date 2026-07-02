"""Sprint 4 — adapter norm / drift maths. Fast: no model.

``Adapters.global_norm`` / ``drift_from`` delegate to the pure tree helpers
``tree_global_norm`` / ``tree_l2_distance``, which operate on plain nested dicts
of mx arrays. We test those directly with synthetic trees so the drift curve's
arithmetic is verified without loading the 6 GB base (the real-model wiring is
covered by the Sprint-4 e2e model test).
"""

import math

import mlx.core as mx

from twin.models import tree_global_norm, tree_l2_distance


def _tree(values):
    """A small nested parameter tree, mirroring an mlx adapter tree's shape."""
    return {
        "layers": {
            "0": {"lora_a": mx.array(values[0]), "lora_b": mx.array(values[1])},
        }
    }


def test_global_norm_matches_euclidean():
    tree = _tree([[3.0, 0.0], [4.0, 0.0]])  # one [3,0], one [4,0]
    assert math.isclose(tree_global_norm(tree), 5.0, rel_tol=1e-6)


def test_global_norm_of_zero_tree_is_zero():
    tree = _tree([[0.0, 0.0], [0.0, 0.0]])
    assert tree_global_norm(tree) == 0.0


def test_l2_distance_is_symmetric_and_correct():
    a = _tree([[0.0, 0.0], [0.0, 0.0]])
    b = _tree([[1.0, 2.0], [2.0, 0.0]])  # diff vector (1,2,2,0) -> norm 3
    d = tree_l2_distance(a, b)
    assert math.isclose(d, 3.0, rel_tol=1e-6)
    assert math.isclose(d, tree_l2_distance(b, a), rel_tol=1e-6)


def test_l2_distance_to_self_is_zero():
    a = _tree([[1.5, -2.0], [0.25, 9.0]])
    assert tree_l2_distance(a, a) == 0.0


def test_drift_grows_as_tree_moves_from_snapshot():
    init = _tree([[0.0, 0.0], [0.0, 0.0]])
    near = _tree([[0.1, 0.0], [0.0, 0.0]])
    far = _tree([[1.0, 1.0], [1.0, 1.0]])
    assert tree_l2_distance(init, near) < tree_l2_distance(init, far)
