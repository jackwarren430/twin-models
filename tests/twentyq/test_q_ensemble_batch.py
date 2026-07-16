"""Model-free aggregation/chunking tests for ensemble history batching."""

from twin.games.twentyq.ensemble_reward import (
    EnsembleReward,
    MemberScore,
)


class _FakeMember:
    def __init__(self, name, offset):
        self.name = name
        self.offset = offset
        self.batch_sizes = []

    def score_batch(self, users, answers, *, system=None, scaffold="ANSWER:"):
        self.batch_sizes.append(len(users))
        return [MemberScore(
            self.name, self.offset + float(answer), 1,
            token_logprobs=[self.offset + float(answer)])
            for answer in answers]


def test_ensemble_batch_chunks_each_member_and_preserves_order():
    a, b = _FakeMember("a", 0.0), _FakeMember("b", 2.0)
    ensemble = EnsembleReward([a, b])

    scores = ensemble.score_batch(
        ["u0", "u1", "u2", "u3", "u4"],
        ["0", "1", "2", "3", "4"],
        batch_size=2,
    )

    assert [score.score for score in scores] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert a.batch_sizes == [2, 2, 1]
    assert b.batch_sizes == [2, 2, 1]
    assert [list(score.per_model) for score in scores] == [["a", "b"]] * 5


def test_score_histories_validates_parallel_inputs():
    ensemble = EnsembleReward([_FakeMember("a", 0.0)])
    try:
        ensemble.score_histories([[]], ["1", "2"], batch_size=2)
    except ValueError as exc:
        assert "users vs" in str(exc)
    else:
        raise AssertionError("expected mismatched histories/answers to fail")
