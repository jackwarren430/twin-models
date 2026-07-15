"""Sprint 3 — JSONL run logging round-trip. Fast: no model."""

from twin.log import JsonlLogger, read_jsonl


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "sub" / "run.jsonl"          # parent created on demand
    with JsonlLogger(path, meta={"config": "tiny.yaml", "run": "t"}) as log:
        log.log({"iter": 0, "creator_reward_mean": 0.5})
        log.log({"iter": 1, "creator_reward_mean": -0.2})

    records = read_jsonl(path)
    assert len(records) == 3
    assert records[0]["type"] == "meta"
    assert records[0]["config"] == "tiny.yaml"
    assert [r["type"] for r in records[1:]] == ["iteration", "iteration"]
    assert records[1]["iter"] == 0 and records[2]["creator_reward_mean"] == -0.2
    assert all("time" in r for r in records)


def test_jsonl_handles_nonserializable(tmp_path):
    path = tmp_path / "run.jsonl"
    with JsonlLogger(path) as log:
        log.log({"obj": object()})                 # default=str keeps it alive
    records = read_jsonl(path)
    assert len(records) == 1 and "obj" in records[0]


def test_jsonl_validation_record_type(tmp_path):
    path = tmp_path / "run.jsonl"
    with JsonlLogger(path) as log:
        log.log_validation({"step": 5, "adapters": {"A": {"guess_rate": 0.5}}})
    records = read_jsonl(path)
    assert records[0]["type"] == "validation"
    assert records[0]["step"] == 5
