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
named, the selected) function then fails with a message that names the
engine's finish reason: HTTP 400 `BadRequestError` for `max_tokens` when the
token limit ended generation before a call or before complete arguments, and
HTTP 500 `ToolChoiceContractError` otherwise. Streamed, the same error is an
SSE event followed by `[DONE]`. Responses with complete calls, and requests
with `auto`, `none` or no tool_choice, are unchanged. With the variable unset
or 0 the image serves as its parent does.

The parent's serving.py must have SHA-256 ea1f7607...; the result has SHA-256
3f48398a..., and the added module has SHA-256 667f88db... (all pinned below).
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
RESULT = "3f48398a5a750e4955efdf8b654f646a2d70e64d3404cd189d0f0802d05339a9"
MODULE_SHA256 = "667f88dbe6e25857474544084eb9967e56ab85a2ecaec7c43177629215fb094e"
# The serving module's last statement; the install statements follow it.
END = "        return ChatCompletionLogProbs(content=logprobs_content)\n"
INSTALL = END + """

# SparkRing: with SPARKRING_TOOL_CHOICE_CONTRACT=1, a named or required
# tool_choice without a complete declared call returns an error (HTTP 400 when
# max_tokens ended generation, otherwise 500) instead of empty tool_calls.
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
             "function fails with HTTP 400 (token limit) or 500 instead of HTTP 200 with empty tool_calls "
             "when SPARKRING_TOOL_CHOICE_CONTRACT=1 (FujitsuPolycom/sparkring#217)"),
    replace=replace,
    provenance="/opt/sparkring/receipts/derived-tool-choice-contract.json",
    pins={SERVING: (INHERITED, RESULT), MODULE: (None, MODULE_SHA256)},
)

if __name__ == "__main__":
    main(LAYER)
