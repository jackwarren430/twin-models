"""Sprint 8 — the oracle is finally WIRED (§14 gap: OracleTool was never
instantiated, so the per-call tax taxed nothing). Listing "oracle" in
tools.creator_tools now builds a real OracleTool per rollout (zeroed adapter
around each query), and taxable calls are counted from the HARNESS — the
legacy text count would double-count executed calls under the react
protocol."""

from contextlib import contextmanager

from twin.config import Config
from twin.tools import tool_schemas
from twin.tools.native import parse_native_tool_calls
from twin.tools.protocol import ToolCall, ToolResult
from twin.train.loop import SelfPlayTrainer


class _FakeAdapters:
    def __init__(self):
        self.log = []

    @contextmanager
    def using(self, name):
        self.log.append(f"+{name}")
        try:
            yield
        finally:
            self.log.append(f"-{name}")


class _FakeBase:
    def oracle(self, question, *, max_tokens, temp):
        return f"the answer to {question!r} is 42"


def _trainer(creator_tools):
    t = SelfPlayTrainer.__new__(SelfPlayTrainer)
    t.cfg = Config.from_dict({"tools": {"creator_tools": creator_tools}})
    t.base = _FakeBase()
    t.adapters = _FakeAdapters()
    return t


def test_oracle_schema_exposed():
    names = [s["function"]["name"] for s in tool_schemas(["oracle"])]
    assert names == ["oracle"]
    calls = parse_native_tool_calls(
        '<tool_call>{"name": "oracle", "arguments": {"question": '
        '"capital of France?"}}</tool_call>')
    assert calls[0].name == "oracle" and calls[0].arg == "capital of France?"


def test_oracle_runs_under_base_adapter_and_is_counted():
    t = _trainer(["solve", "oracle"])
    runner, harness = t._build_tool_runner(
        ["solve", "oracle"], max_tool_calls=4, native=True)
    obs = runner('<tool_call>{"name": "oracle", "arguments": '
                 '{"question": "capital of France?"}}</tool_call>')
    assert obs is not None and "42" in obs
    # the zeroed adapter was activated for the query and restored after
    assert t.adapters.log == ["+base", "-base"]
    # taxable count comes from the harness, not the rollout text
    assert t._oracle_count("no legacy markup here", harness) == 1


def test_oracle_count_does_not_double_count_executed_calls():
    t = _trainer(["oracle"])
    harness_like = type("H", (), {"calls": [
        ToolResult(ToolCall("oracle", "q"), "a", ok=True),
    ]})()
    # legacy text of the SAME executed call must not add a second unit
    text = "<tool>oracle(q)</tool>\n<obs>a</obs>"
    assert t._oracle_count(text, harness_like) == 1


def test_oracle_count_legacy_semantics_without_the_tool():
    t = _trainer(["solve", "calc"])          # oracle NOT wired
    harness_like = type("H", (), {"calls": []})()
    text = "<tool>oracle(what is 2+2?)</tool>"
    assert t._oracle_count(text, harness_like) == 1


def test_oracle_call_limit_respected():
    t = _trainer(["oracle"])
    runner, harness = t._build_tool_runner(["oracle"], max_tool_calls=8, native=True)
    call = ('<tool_call>{"name": "oracle", "arguments": {"question": "q"}}'
            "</tool_call>")
    limit = t.cfg.oracle.max_calls_per_turn
    for _ in range(limit + 1):
        runner(call)
    outputs = [res.output for res in harness.calls]
    assert sum("42" in o for o in outputs) == limit
    assert any("limit" in o for o in outputs)
