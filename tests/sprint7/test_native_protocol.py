"""Sprint 7 — Qwen3 native function-calling protocol: <tool_call> JSON
parsing, the <tool_response> splice glue, harness integration (budget,
parse-error feedback), and the tool schemas. Fast: no model.

The glue strings are additionally verified against apply_chat_template ground
truth in the model-gated suite (test_native_model.py) so chat-template drift
is caught."""

from twin.tools import (
    NATIVE_STOP,
    PARSE_ERROR_NAME,
    ToolHarness,
    format_tool_responses,
    parse_native_tool_calls,
    tool_schemas,
)
from twin.tools.protocol import ToolCall, ToolResult


def _call(text):
    return f"<tool_call>\n{text}\n</tool_call>"


def test_stop_string():
    assert NATIVE_STOP == "</tool_call>"


def test_parses_standard_call():
    text = 'reasoning...\n' + _call('{"name": "solve", "arguments": {"expression": "2*x + 3 = 11"}}')
    calls = parse_native_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "solve"
    assert calls[0].arg == "2*x + 3 = 11"


def test_parses_string_arguments():
    calls = parse_native_tool_calls(_call('{"name": "calc", "arguments": "7*8 + 3"}'))
    assert calls[0].name == "calc" and calls[0].arg == "7*8 + 3"


def test_parses_double_encoded_arguments():
    calls = parse_native_tool_calls(
        _call('{"name": "solve", "arguments": "{\\"expression\\": \\"x + 1 = 2\\"}"}'))
    assert calls[0].arg == "x + 1 = 2"


def test_single_unknown_key_falls_back_to_its_value():
    calls = parse_native_tool_calls(
        _call('{"name": "solve", "arguments": {"equation": "x - 4 = 0"}}'))
    assert calls[0].arg == "x - 4 = 0"


def test_malformed_json_yields_parse_error_sentinel():
    calls = parse_native_tool_calls(_call('{"name": "solve", oops'))
    assert len(calls) == 1
    assert calls[0].name == PARSE_ERROR_NAME


def test_missing_name_yields_parse_error_sentinel():
    calls = parse_native_tool_calls(_call('{"arguments": {"expression": "1+1"}}'))
    assert calls[0].name == PARSE_ERROR_NAME


def test_multiple_calls_in_order():
    text = _call('{"name": "solve", "arguments": {"expression": "a"}}') + \
        "\n" + _call('{"name": "calc", "arguments": {"expression": "b"}}')
    calls = parse_native_tool_calls(text)
    assert [c.name for c in calls] == ["solve", "calc"]


def test_glue_format_single_response():
    r = ToolResult(ToolCall("solve", "2*x = 8"), "[4]")
    glue = format_tool_responses([r])
    assert glue == (
        "<|im_end|>\n<|im_start|>user\n<tool_response>\n[4]\n</tool_response>"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def test_glue_format_multiple_responses_one_user_turn():
    rs = [ToolResult(ToolCall("solve", "a"), "1"),
          ToolResult(ToolCall("solve", "b"), "2")]
    glue = format_tool_responses(rs)
    assert glue.count("<|im_start|>user") == 1
    assert glue.count("<tool_response>") == 2
    assert glue.endswith("<|im_start|>assistant\n")


def test_harness_with_native_parser_dispatches_and_budgets():
    seen = []
    harness = ToolHarness(
        {"solve": lambda a: (seen.append(a) or f"ok:{a}")},
        max_tool_calls=2,
        parser=parse_native_tool_calls,
    )
    out = harness.run(_call('{"name": "solve", "arguments": {"expression": "x=1"}}'))
    assert out[0].ok and out[0].output == "ok:x=1" and seen == ["x=1"]
    harness.run(_call('{"name": "solve", "arguments": {"expression": "x=2"}}'))
    out3 = harness.run(_call('{"name": "solve", "arguments": {"expression": "x=3"}}'))
    assert not out3[0].ok and "budget" in out3[0].output


def test_harness_feeds_format_error_back_for_malformed_call():
    harness = ToolHarness({"solve": lambda a: "unused"},
                          max_tool_calls=4, parser=parse_native_tool_calls)
    out = harness.run(_call('{"name": "solve", broken'))
    assert len(out) == 1 and not out[0].ok
    assert "malformed tool call" in out[0].output
    assert '"arguments"' in out[0].output   # tells the model the right shape


def test_tool_schemas_known_and_unknown():
    schemas = tool_schemas(["solve", "calc", "python"])  # python: no native schema yet
    names = [s["function"]["name"] for s in schemas]
    assert names == ["solve", "calc"]
    solve_schema = schemas[0]
    assert solve_schema["type"] == "function"
    assert "expression" in solve_schema["function"]["parameters"]["properties"]
    assert solve_schema["function"]["parameters"]["required"] == ["expression"]
