"""Fail closed when a named or required Chat Completions tool_choice yields no valid call.

With `tool_choice: "required"` or a named function, vLLM's Chat Completions
serving module (vllm/entrypoints/openai/chat_completion/serving.py) serializes
whatever calls the tool parser returned. When generation stops before a
complete call, for example because reasoning uses up `max_tokens`, the
response is HTTP 200 with `finish_reason: "length"` and an empty `tool_calls`
list (FujitsuPolycom/sparkring#217).

This layer adds the tool-result policy of integrations/vllm/tool_choice_contract
(contract.py, unchanged) as vLLM's `sparkring_tool_choice_contract` module and
appends two statements to the serving module that install it on
`OpenAIServingChat` when `SPARKRING_TOOL_CHOICE_CONTRACT=1`. The serving module
runs them on import, so every Python API server process reads the same
setting. The installer sets the variable to 1 for every installer profile
unless the profile's environment sets it (runtime/common/installer_image.py).
A named or required request without a complete call to a declared (and, when
named, the selected) function then returns HTTP 500 `ToolChoiceContractError`
whose message names the engine's finish reason, or, when streaming, an SSE
error event followed by `[DONE]`. Responses with complete calls, and requests
with `auto`, `none` or no tool_choice, are unchanged. With the variable unset
or 0 the image serves as its parent does.

The parent's serving.py must have SHA-256 ea1f7607...; the result has SHA-256
5f29e4eb..., and the added module has SHA-256 f5939ac5... (all pinned below).
/opt/sparkring/receipts/derived-tool-choice-contract.json records both files.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.images.derived_layer import SITE, Layer, main, swap  # noqa: E402
from runtime.images.feature_extension import pinned_bytes  # noqa: E402

CHAT = SITE + "vllm/entrypoints/openai/chat_completion/"
SERVING = CHAT + "serving.py"
MODULE = CHAT + "sparkring_tool_choice_contract.py"
SOURCE = Path(__file__).resolve().parents[2] / "integrations/vllm/tool_choice_contract/contract.py"
INHERITED = "ea1f76074a9587c8054f54d30a6ba748b6a5fc4d90dd82c801504505c1da1f92"
RESULT = "5f29e4ebd6744816c316ccc5be76a360da3fcfc10fe824a44faf985d18fea2f2"
MODULE_SHA256 = "f5939ac5cb7579106c40cc41e964a2ce65f737011e83bd7c5b10fdd26c1cdb97"
# The serving module's last statement; the install statements follow it.
END = "        return ChatCompletionLogProbs(content=logprobs_content)\n"
INSTALL = END + """

# SparkRing: with SPARKRING_TOOL_CHOICE_CONTRACT=1, a named or required
# tool_choice without a complete declared call fails with
# ToolChoiceContractError instead of returning an empty tool_calls list.
from vllm.entrypoints.openai.chat_completion import (  # noqa: E402
    sparkring_tool_choice_contract,
)

sparkring_tool_choice_contract.install_from_environment(OpenAIServingChat)
"""


def replace(read, receipt, source=SOURCE):
    return {SERVING: swap(read(SERVING).decode("utf-8"), END, INSTALL).encode("utf-8"),
            MODULE: pinned_bytes(source.read_bytes(), MODULE_SHA256)}


LAYER = Layer(
    name="tool-choice-contract",
    purpose=("A named or required Chat Completions tool_choice without a complete call to a declared "
             "function fails with ToolChoiceContractError instead of HTTP 200 with empty tool_calls "
             "when SPARKRING_TOOL_CHOICE_CONTRACT=1 (FujitsuPolycom/sparkring#217)"),
    replace=replace,
    provenance="/opt/sparkring/receipts/derived-tool-choice-contract.json",
    pins={SERVING: (INHERITED, RESULT), MODULE: (None, MODULE_SHA256)},
)

if __name__ == "__main__":
    main(LAYER)
