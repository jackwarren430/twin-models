"""Sprint 7 — the hardcoded native-protocol glue must match what the Qwen3
chat template actually renders for a tool-response turn, or the spliced
context silently diverges from the model's training distribution.

Loads ONLY the tokenizer (transformers AutoTokenizer on the local model dir —
a few MB, no weights), so it's safe to run alongside other model workloads;
skipped when the model directory or transformers is unavailable."""

import os

import pytest

from twin.config import DEFAULT_MODEL_PATH
from twin.tools import format_tool_responses, tool_schemas
from twin.tools.protocol import ToolCall, ToolResult

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.expanduser(DEFAULT_MODEL_PATH)),
    reason="local Qwen3 model directory not present",
)


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(
        os.path.expanduser(DEFAULT_MODEL_PATH))


def test_glue_matches_chat_template_ground_truth(tokenizer):
    messages = [
        {"role": "user", "content": "Compute 2x = 8."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {"name": "solve",
                             "arguments": {"expression": "2*x = 8"}},
            }],
        },
        {"role": "tool", "content": "[4]"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True, tools=tool_schemas(["solve"]),
    )
    # Everything the template emits after the model's `</tool_call>` is what
    # our runner must splice.
    ground_truth = rendered.split("</tool_call>")[-1]
    ours = format_tool_responses([ToolResult(ToolCall("solve", "2*x = 8"), "[4]")])
    assert ours == ground_truth


def test_template_declares_tools_and_call_format(tokenizer):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
        tools=tool_schemas(["solve", "calc"]),
    )
    assert "<tools>" in rendered and "</tools>" in rendered
    assert '"solve"' in rendered and '"calc"' in rendered
    assert "<tool_call>" in rendered      # the format instruction block
