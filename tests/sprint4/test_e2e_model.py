"""Sprint 4 — very short end-to-end runs against the REAL base model.

Heavy: loads ~6 GB and generates. Marked ``model`` and skipped unless
TWIN_RUN_MODEL_TESTS=1 (set by ``scripts/run_tests.py --model`` /
``--only-model``), so the default fast suite never loads the model. This is the
"separate suite that loads the model and runs very short e2e runs" — it proves
the Sprint-4 instrumentation (adapter drift, oracle/tool usage), the analysis
module, and thinking mode all hold together on a real two-iteration loop.

Run:  conda run -n twin-models python scripts/run_tests.py --only-model --sprint 4
"""

import math
import os
import tempfile
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        os.environ.get("TWIN_RUN_MODEL_TESTS") != "1",
        reason="set TWIN_RUN_MODEL_TESTS=1 to run model-loading tests",
    ),
]


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _log_summary(transcript, records) -> None:
    """Append the end-of-run analysis (parse errors, solver accuracy, creator
    problem-creation accuracy) to the bottom of the transcript section."""
    if transcript is None:
        return
    from twin.analysis import format_run_summary

    transcript.section("RUN SUMMARY")
    transcript.block(format_run_summary(records))


@pytest.fixture
def transcript(request):
    """Raw-text transcript sink for the e2e run, enabled when TWIN_E2E_LOG points
    at a file (set by ``scripts/run_tests.py --log``). Appends, so every model
    test in the suite writes into the one file; ``None`` (the default) is a no-op
    that leaves the trainer's behaviour unchanged."""
    path = os.environ.get("TWIN_E2E_LOG")
    if not path:
        yield None
        return
    from twin.log import TranscriptLogger

    tr = TranscriptLogger(path, meta={"test": request.node.name})
    try:
        yield tr
    finally:
        tr.close()


@pytest.fixture(scope="module")
def loaded():
    """Load the base + adapters once and shrink the config to the bare minimum
    so a couple of iterations finish quickly."""
    import mlx.core as mx

    from twin.config import Config
    from twin.models import Adapters, TwinBase

    cfg = Config.from_yaml("configs/tiny.yaml")
    cfg.game.n_problems = 2
    cfg.game.creator_group = 2
    cfg.game.solver_attempts = 1
    # Generous budgets: a higher cap is free when the model emits EOS early (it
    # only bites when generation would otherwise truncate). At 256/128 the
    # creator JSON and solver answer truncated mid-token (parse-gate failed,
    # wrong answer extracted) and the loop never actually exercised create->solve.
    cfg.gen.creator_max_tokens = 2048
    cfg.gen.solver_max_tokens = 1024
    # The judge now runs a ReAct loop (recompute via the CAS `solve` tool, then
    # rule), so it needs room for reasoning + a tool call + the VERDICT line.
    cfg.gen.oracle_max_tokens = 1024

    base = TwinBase(cfg.model.path)
    adapters = Adapters.from_config(base.model, cfg.lora)

    # Baseline base-adapter log-probs, to confirm training never touches the base.
    ids = base.tokenizer.encode(base.render("What is 2+2? Answer with a number."))
    adapters.activate("base")
    base_lp0 = base.token_logprobs(ids)
    mx.eval(base_lp0)
    return {"mx": mx, "cfg": cfg, "base": base, "adapters": adapters,
            "ids": ids, "base_lp0": base_lp0}


def test_short_run_logs_new_metrics_and_analysis_runs(loaded, transcript):
    from twin.analysis import load_iterations, summarize_run, time_series
    from twin.log import JsonlLogger, read_jsonl
    from twin.train import SelfPlayTrainer

    base, adapters, cfg, mx = (loaded["base"], loaded["adapters"],
                               loaded["cfg"], loaded["mx"])

    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "run.jsonl"
        with JsonlLogger(log, meta={"test": "sprint4_e2e"}) as logger:
            trainer = SelfPlayTrainer(base, adapters, cfg,
                                      logger=logger, transcript=transcript)
            rec0 = trainer.run_iteration(0)
            rec1 = trainer.run_iteration(1)
        _log_summary(transcript, [rec0, rec1])
        records = read_jsonl(log)

        # the new Sprint-4 instrumentation is present and well-typed
        for rec in (rec0, rec1):
            for key in ("creator_oracle_calls", "solver_oracle_calls",
                        "creator_tool_calls", "r_gradient_mean",
                        "adapter_norm", "adapter_drift"):
                assert key in rec, key
            assert set(rec["adapter_drift"]) == {"A", "B"}
            for name in ("A", "B"):
                d = rec["adapter_drift"][name]
                assert _finite(d) and d >= 0.0
            cu, su = rec["creator_update"], rec["solver_update"]
            for m in (cu, su):
                assert _finite(m["loss"]) and _finite(m["kl"]) and _finite(m["grad_norm"])
                assert -1e-6 <= m["kl"] < 50.0

        # the analysis module consumes the real log end-to-end
        loaded_recs = load_iterations(log)
        assert len(loaded_recs) == 2
        ts = time_series(loaded_recs)
        assert len(ts["adapter_drift_A"]) == 2
        summary = summarize_run(loaded_recs)
        import json
        json.dumps(summary)  # serializable
        assert summary["n_iterations"] == 2

    # base invariant survives training A/B
    adapters.activate("base")
    base_lp1 = base.token_logprobs(loaded["ids"])
    mx.eval(base_lp1)
    md = float(mx.max(mx.abs(base_lp1 - loaded["base_lp0"])))
    assert md < 1e-4, f"base adapter moved: maxdiff={md:.2e}"


def test_thinking_mode_renders_and_runs(loaded, transcript):
    """Thinking mode is plumbed through render() and a full iteration runs with
    it on (bounded KL, finite metrics, base untouched)."""
    from twin.log import JsonlLogger
    from twin.train import SelfPlayTrainer

    base, adapters, cfg, mx = (loaded["base"], loaded["adapters"],
                               loaded["cfg"], loaded["mx"])

    # The chat template must actually differ when thinking is toggled.
    user = "Solve x + 1 = 3. End with ANSWER: <n>."
    assert base.render(user, enable_thinking=True) != base.render(user, enable_thinking=False)

    cfg.model.enable_thinking = True
    # Thinking mode emits a long <think> block BEFORE the JSON / ANSWER, so the
    # budget must cover both. At 384 the reasoning consumed the whole budget and
    # no JSON was ever produced (every suite parse-failed). Give it ample room.
    cfg.gen.creator_max_tokens = 3072
    cfg.gen.solver_max_tokens = 2048
    # Thinking judge: <think> block + CAS tool round + VERDICT all share the cap.
    cfg.gen.oracle_max_tokens = 2048
    try:
        with tempfile.TemporaryDirectory() as td:
            with JsonlLogger(Path(td) / "think.jsonl", meta={"think": True}) as logger:
                trainer = SelfPlayTrainer(base, adapters, cfg,
                                          logger=logger, transcript=transcript)
                rec = trainer.run_iteration(0)
        _log_summary(transcript, [rec])
    finally:
        cfg.model.enable_thinking = False

    assert _finite(rec["creator_reward_mean"]) and _finite(rec["solver_reward_mean"])
    for m in (rec["creator_update"], rec["solver_update"]):
        assert _finite(m["kl"]) and -1e-6 <= m["kl"] < 50.0

    adapters.activate("base")
    base_lp1 = base.token_logprobs(loaded["ids"])
    mx.eval(base_lp1)
    md = float(mx.max(mx.abs(base_lp1 - loaded["base_lp0"])))
    assert md < 1e-4, f"base adapter moved under thinking run: maxdiff={md:.2e}"
