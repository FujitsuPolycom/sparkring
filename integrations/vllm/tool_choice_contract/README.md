# Chat Completions tool-result contract

Status: implemented; CPU source replay covers all three admitted serving
modules. Serving qualification is required for each image before adoption.

This optional API policy addresses [issue #217](https://github.com/FujitsuPolycom/sparkring/issues/217):
named or required tool requests can finish with an empty, malformed, or
incorrectly named tool result. The published DeepSeek native image and the
SparkRing LIL R37 source both serialize parser results without these checks.

The policy checks the final parser output:

- `required` must produce at least one call to a function declared in `tools`.
- A named choice must call the selected function. Multiple calls to that same
  function are allowed unless `parallel_tool_calls` is `false`.
- Each call must contain a complete JSON object in `function.arguments`.
- Every requested completion choice must finish. A complete call at the token
  limit is allowed; `finish_reason: "length"` remains visible, including when
  vLLM's streaming tool detection would otherwise replace it with `tool_calls`.

It does not validate an argument object's JSON Schema, infer missing arguments,
repair malformed calls, disable reasoning, or retry generation. A parser that
repairs an incomplete call into valid JSON can still conceal missing semantic
information; this policy cannot recover parser data it never receives.
`auto`, `none`, existing engine errors, and unconstrained text are unchanged.

Nonstreaming violations return HTTP 500 with `ToolChoiceContractError`.
Streaming violations emit an SSE error with that type, followed by `[DONE]`.
Streaming HTTP headers may already say 200, and previously emitted deltas
cannot be retracted. Clients must wait for successful completion before
executing tool calls and must handle SSE errors even on HTTP 200 responses.

## Local deployment

Mount this complete directory read-only at `/opt/sparkring-tool-contract` in the
API container. Set `SPARKRING_TOOL_CHOICE_CONTRACT=1`, and replace the API
entrypoint with:

```bash
/opt/venv/bin/python /opt/sparkring-tool-contract/serve.py serve <existing-serving-arguments>
```

Keep the image, model path, tensor-parallel peers, environment and serving
arguments unchanged. Peer workers do not need the API wrapper. The variable
defaults to `0`; setting it without this entrypoint has no effect.

The bootstrap hashes the complete installed Chat Completions serving module
against three reviewed inputs in [serve.py](serve.py): the published DeepSeek
native parent, its API source upgrade, and SparkRing LIL R37. Unknown source is
rejected. Installed sources, native libraries, model files and source contracts
are unchanged. The bootstrap wraps the Python class in memory.

Only one Python API frontend with data-parallel size one is supported. TP2 and
TP4 are compatible with that frontend boundary. Multiple API processes, the
Rust frontend, multi-port data-parallel supervision and gRPC are rejected.
Keep public image pins unchanged until the packaged entrypoint has passed
serving tests; this directory is not a published image identity.

## Verification

CPU checks need neither vLLM nor hardware:

```bash
python -m pytest integrations/vllm/tool_choice_contract -q
python integrations/vllm/tool_choice_contract/source_replay.py \
  /path/to/vllm/entrypoints/openai/chat_completion/serving.py \
  --output /path/to/private-source-replay.json
```

The source replay executes the exact admitted full and streaming methods with
fixture engine, parser and response interfaces. It compares unwrapped behavior
with the optional policy for empty, malformed, wrong-name, valid and
complete-at-length outputs. It is not an HTTP or model test.

The [API probe](api_probe.py) sends eight sequential requests to an explicitly
chosen `/v1` endpoint: named and required tools, streamed and nonstreamed, with
128-token positive and one-token truncation budgets. It verifies the positive
function name and arguments, and records raw bodies privately. Supply
`VLLM_API_KEY` through the environment if the endpoint requires authentication.

```bash
python integrations/vllm/tool_choice_contract/api_probe.py \
  --base-url http://<api-host>:<port>/v1 --model <served-model-name> \
  --policy enabled --output /path/to/new-private-evidence-directory
```

Use `--policy disabled` against the baseline to reproduce successful empty
outputs. A one-token budget is not a deterministic parser fault injector; if a
model emits a complete valid call or fails earlier, inspect the recorded body
and retain the exact source replay as the deterministic contract regression.
These requests do not qualify model quality, all tool schemas, or transport
performance.
