"""Sprint 2 — tool protocol & harness. Fast: no model (calc/oracle stubbed)."""

from twin.tools import OracleTool, ToolHarness, calc, parse_tool_calls


def test_parse_single_call():
    calls = parse_tool_calls("let me compute <tool>calc(1 + 2)</tool> ok")
    assert len(calls) == 1
    assert calls[0].name == "calc"
    assert calls[0].arg == "1 + 2"


def test_parse_preserves_inner_parens():
    calls = parse_tool_calls("<tool>python(print((1 + 2) * 3))</tool>")
    assert len(calls) == 1
    assert calls[0].arg == "print((1 + 2) * 3)"


def test_parse_multiple_in_order():
    text = "<tool>calc(1+1)</tool> then <tool>oracle(what is pi)</tool>"
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["calc", "oracle"]


def test_unbalanced_call_skipped():
    assert parse_tool_calls("<tool>calc(1 + 2</tool>") == []


def test_harness_executes_calc():
    harness = ToolHarness({"calc": calc}, max_tool_calls=4)
    results = harness.run("<tool>calc(6*7)</tool>")
    assert len(results) == 1
    assert results[0].output == "42"
    assert results[0].ok
    obs = ToolHarness.format_observations(results)
    assert obs == "<obs>42</obs>"


def test_harness_budget_enforced():
    harness = ToolHarness({"calc": calc}, max_tool_calls=1)
    results = harness.run("<tool>calc(1+1)</tool><tool>calc(2+2)</tool>")
    assert len(results) == 2
    assert results[0].ok
    assert not results[1].ok
    assert "budget" in results[1].output


def test_harness_counts_oracle_calls():
    oracle = OracleTool.from_callable(lambda q: "42")
    harness = ToolHarness({"oracle": oracle, "calc": calc}, max_tool_calls=4)
    harness.run("<tool>oracle(meaning of life)</tool><tool>calc(1+1)</tool>")
    assert harness.n_oracle_calls == 1
    assert harness.n_calls == 2


def test_oracle_respects_its_own_limit():
    oracle = OracleTool.from_callable(lambda q: "ok", max_calls=2)
    assert oracle("a") == "ok"
    assert oracle("b") == "ok"
    assert oracle("c").startswith("error:")
    assert oracle.n_calls == 2
    oracle.reset()
    assert oracle.n_calls == 0


def test_unknown_tool_is_error():
    harness = ToolHarness({"calc": calc})
    results = harness.run("<tool>nope(x)</tool>")
    assert not results[0].ok
    assert "unknown tool" in results[0].output
