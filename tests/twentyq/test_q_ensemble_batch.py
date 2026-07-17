"""Model-free aggregation/chunking tests for ensemble history batching."""

from twin.games.twentyq.ensemble_reward import (
    DEFAULT_INSTRUCTION,
    DEFAULT_SYSTEM,
    EnsembleMember,
    EnsembleReward,
    MemberScore,
    PINNED_DATE_SHORT,
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


# ----- render stationarity + naming (DESIGN §6.8) ---------------------------

class _KwargCaptureTokenizer:
    """Stands in for a tokenizer whose template accepts the optional extras."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, msgs, **kwargs):
        self.calls.append(kwargs)
        return "rendered"


class _StrictTokenizer(_KwargCaptureTokenizer):
    """Rejects the optional extras like an older/stricter tokenizer would."""

    def apply_chat_template(self, msgs, **kwargs):
        self.calls.append(kwargs)
        if "enable_thinking" in kwargs or "date_string" in kwargs:
            raise TypeError("unexpected keyword argument")
        return "rendered"


def _member_with(tokenizer):
    member = EnsembleMember.__new__(EnsembleMember)
    member.name = "fake"
    member.tokenizer = tokenizer
    return member


def test_apply_template_pins_thinking_and_date():
    tok = _KwargCaptureTokenizer()
    assert _member_with(tok)._apply_template([]) == "rendered"
    (kwargs,) = tok.calls
    assert kwargs["enable_thinking"] is False
    assert kwargs["date_string"] == PINNED_DATE_SHORT
    # SmolLM3's template calls strftime_now directly; the kwarg shadows the
    # Jinja global, so it must render a constant whatever today is.
    assert kwargs["strftime_now"]("%d %b %Y") == PINNED_DATE_SHORT
    assert "January" in kwargs["strftime_now"]("%d %B %Y")
    assert kwargs["add_generation_prompt"] is False


def test_apply_template_falls_back_when_extras_rejected():
    tok = _StrictTokenizer()
    assert _member_with(tok)._apply_template([]) == "rendered"
    assert len(tok.calls) == 2
    assert "date_string" not in tok.calls[1]
    assert "enable_thinking" not in tok.calls[1]


def test_scoring_prompts_name_the_real_game():
    # The deduction game is "20 Questions" in the frozen scorers' pretraining
    # data; "21 questions" names a different game and miscues the ensemble.
    for text in (DEFAULT_SYSTEM, DEFAULT_INSTRUCTION):
        assert "20 Questions" in text
        assert "21" not in text
