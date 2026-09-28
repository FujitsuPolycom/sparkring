# Chat Completions tool-result contract

Status: **implemented**; CPU source replay covers all four admitted serving
modules. The [Qwen TP4 evidence](serving-evidence.json) and
[DeepSeek TP2 record](../../../performance/records/deepseek-v4-flash/api-ipc-tp2-k5-20260917.md)
qualify bounded empty-result checks on their identified images with the
entrypoint wrapper. The installer image layer is **implemented** without a
built image or hardware result. Other images and model parsers require their
own serving qualification.

This optional API policy addresses [issue #217](https://github.com/FujitsuPolycom/sparkring/issues/217):
named or required tool requests can finish with an empty, malformed, or
incorrectly named tool result. The published DeepSeek native image, the
SparkRing LIL R37 source and the vLLM of the installer images serialize parser
results without these checks. On the installer image
`dev-20260927-mimovision-cuda1342-nccl2323-status032`, a `required` or named
`get_weather` request with `max_tokens` 16 at temperature 0 returned HTTP 200,
`finish_reason: "length"` and no tool call on MiMo-V2.6-Flash-RL (four Sparks)
and GLM-5.3-Flash (two Sparks); with `max_tokens` 400 both returned the call.

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

Nonstreaming violations return HTTP 500 with `ToolChoiceContractError`,
`param: "tool_choice"` and a message that names the engine's finish reason, for
example `Named or required tool_choice produced no tool calls; finish_reason:
length.` A `length` finish means the token budget ended before a complete call;
raise `max_tokens` or reduce reasoning. Streaming violations emit an SSE error
with that type, followed by `[DONE]`. Streaming HTTP headers may already say
200, and deltas already sent cannot be retracted. Clients must wait for
successful completion before executing tool calls and must handle SSE errors
even on HTTP 200 responses.

`SPARKRING_TOOL_CHOICE_CONTRACT` selects the policy: `1` enables it, `0` or an
unset variable leaves responses unchanged, and any other value stops the API
server at startup.

## Installer images

[derive_tool_choice_contract.py](../../../runtime/images/derive_tool_choice_contract.py)
derives an installer image from `dev-20260927-mimovision-cuda1342-nccl2323-status032`
in one layer. It adds [contract.py](contract.py), unchanged, as
`vllm/entrypoints/openai/chat_completion/sparkring_tool_choice_contract.py` and
appends two statements to that package's `serving.py` that call
`install_from_environment(OpenAIServingChat)` when vLLM imports the module.
Every Python API server process therefore reads the variable itself, and
vLLM's Anthropic Messages handler, a subclass of `OpenAIServingChat`, inherits
the policy for its `any` and `tool` choices; its streaming converter reports a
violation as a generic `internal_error` event. The layer pins the inherited and
resulting `serving.py` and the added module by SHA-256, and the image's
`verify` checks both files through the re-recorded receipts.

The installer container of every installer profile sets
`SPARKRING_TOOL_CHOICE_CONTRACT=1` unless the profile's `environment` sets the
variable (`installer_image.adapt`); a profile opts out with `0`. Only an image
built by this layer reads the variable. The default installer image lock
selects `dev-20260927-mimovision-cuda1342-nccl2323-status032`, which does not
carry the layer, so installer deployments keep vLLM's unchecked responses until
a release built by this layer becomes the default lock
([release procedure](../../../docs/development/releases.md)). The Rust
frontend and gRPC do not use `OpenAIServingChat` and are not covered.

## Entrypoint wrapper for other images

Mount this complete directory read-only at `/opt/sparkring-tool-contract` in the
API container. Set `SPARKRING_TOOL_CHOICE_CONTRACT=1`, and replace the API
entrypoint with:

```bash
/opt/venv/bin/python /opt/sparkring-tool-contract/serve.py serve <existing-serving-arguments>
```

Keep the image, model path, tensor-parallel peers, environment and serving
arguments unchanged. Peer workers do not need the API wrapper. Setting the
variable without this entrypoint has no effect on such an image.

The bootstrap hashes the complete installed Chat Completions serving module
against the reviewed inputs in [serve.py](serve.py): the published DeepSeek
native parent, its API source upgrade, SparkRing LIL R37 and the installer
software layer. Unknown source is rejected. Installed sources, native libraries,
model files and source contracts are unchanged. The bootstrap wraps the Python
class in memory. Installer images start vLLM through their toolchain entrypoint,
which this wrapper would bypass; they use the image layer instead.

Only one Python API frontend with data-parallel size one is supported. TP2 and
TP4 are compatible with that frontend boundary. Multiple API processes, the
Rust frontend, multi-port data-parallel supervision and gRPC are rejected.
Keep public image pins unchanged until the packaged entrypoint has passed
serving tests; this directory is not a published image identity.

## Verification

CPU checks need neither vLLM nor hardware:

```bash
python -m pytest integrations/vllm/tool_choice_contract runtime/images/test_derived_layer.py -q
python integrations/vllm/tool_choice_contract/source_replay.py \
  /path/to/vllm/entrypoints/openai/chat_completion/serving.py \
  --output /path/to/private-source-replay.json
```

The source replay executes the exact admitted full and streaming methods with
fixture engine, parser and response interfaces. It compares unwrapped behavior
with the optional policy for empty, malformed, wrong-name, valid and
complete-at-length outputs. It is not an HTTP or model test. With the installer
software layer's `serving.py` (SHA-256 `ea1f7607…`), all 40 cases passed; the
unwrapped empty and truncated cases returned no error with
`finish_reason: "length"`, as on hardware.

The [API probe](api_probe.py) sends eight sequential requests to an explicitly
chosen `/v1` endpoint: named and required tools, streamed and nonstreamed, with
a positive and a truncated `max_tokens` budget (defaults 128 and 1). It verifies
the positive function name and arguments, and records raw bodies privately.
`--thinking off`, the default, sends `enable_thinking: false`; `--thinking
default` leaves the served chat template's reasoning setting. Supply
`VLLM_API_KEY` through the environment if the endpoint requires authentication.

```bash
python integrations/vllm/tool_choice_contract/api_probe.py \
  --base-url http://<api-host>:<port>/v1 --model <served-model-name> \
  --policy enabled --output /path/to/new-private-evidence-directory
```

Use `--policy disabled` against the baseline to reproduce successful empty
outputs. A truncated budget is not a deterministic parser fault injector; if a
model emits a complete valid call or fails earlier, inspect the recorded body
and retain the exact source replay as the deterministic contract regression.
These requests do not qualify model quality, all tool schemas, or transport
performance.

### Installer image build and hardware check

On Node A, which holds the parent image
`sha256:8e4de5f05f0287c4d0326f3a6ed5d25d3248ec4d482f369308a08a36a2f893bf`,
from a checkout of this repository:

```bash
python3 runtime/images/derive_tool_choice_contract.py prepare \
  --parent-lock runtime/releases/dev-20260927-mimovision-cuda1342-nccl2323-status032/installer-image.json \
  --output CONTEXT
python3 runtime/images/derive_tool_choice_contract.py build --context CONTEXT \
  --tag sparkring:RELEASE --name RELEASE --output LOCK
```

`prepare` reads the parent's receipts and `serving.py` through a network-less
container and refuses a parent whose `serving.py` differs from the pin. `build`
runs the installer's admission, including the image's `verify`, for every
profile of the lock. The written lock names the local image ID, so
`sudo sparkring install --profile PROFILE --image-lock LOCK` streams the image
from Node A to the other Sparks
([image distribution](../../../docs/operations/install-reference.md#image-distribution-and-caches)).
Run the probe against the installed API with the measured budgets, for
`mimo-v26-flash-rl-tp4` and `glm53-flash-nvfp4-spark-tp2`:

```bash
python integrations/vllm/tool_choice_contract/api_probe.py \
  --base-url http://NODE_A:PORT/v1 --model SERVED_MODEL_NAME --policy enabled \
  --truncated-max-tokens 16 --positive-max-tokens 400 --thinking default \
  --output /path/to/new-private-evidence-directory
```

Every truncated request must return `ToolChoiceContractError` (HTTP 500
nonstreamed; an SSE error and `[DONE]` streamed) and every positive request one
`lookup` call with arguments `{"key": "cedar"}`. The same command with
`--policy disabled` against a profile installed on the default lock, whose
image lacks the layer, records the baseline: HTTP 200 without a tool call for
the truncated requests.

## Evidence

The recorded Qwen test reproduced HTTP 200 with empty calls in all four
one-token baseline requests. The wrapper returned two HTTP 500 full-response
errors and two explicit SSE errors under HTTP 200. All four positive tool
requests and an ordinary arithmetic request passed. Existing model, image,
transport and cache settings were retained; this result does not qualify a
different image or external-cache restoration.
