# DeepSeek-V4-Flash-0731 runtime for GB10

## API reasoning and required-tool policies

Status: implemented in the source overlay; rebuilt-image qualification is
outstanding. The published native derivative at digest
`sha256:827a8e8c5749b78529cc0015dd174e1b19a0accc116bc142282f8b75428f98bd`
does not contain these API changes.

The Responses request and response models accept `reasoning.effort: "max"` in addition to
`none`, `minimal`, `low`, `medium`, `high`, and `xhigh`. They retain the OpenAI
SDK reasoning model's other fields and validation. Unknown effort values remain
invalid. Streaming `response.created`, `response.in_progress`, and
`response.completed` events preserve the requested effort when validating and
serializing the response. Model sampling and generation code are unchanged.

Set `SPARKRING_REJECT_EMPTY_REQUIRED_TOOL_CALLS=1` in the API container
environment to reject nonstreaming Chat Completions whose named or required
tool choice produces no parsed tool calls. The response is an HTTP 500
`ToolChoiceContractError`, rather than a successful empty tool-call array.
The default remains disabled. Valid nonempty parser results retain their
serialization, and automatic tool choice is unaffected.

The guard checks only whether parsed calls exist. It does not validate argument
JSON, function names, schemas, or streaming results.
A streaming HTTP response may already have started before a missing tool
call is detectable; streaming requires a separate event-level policy.
Clients should bound retries and may increase the output-token budget when
reasoning exhausts the available generation length.

The runtime patch and per-file hashes are bound by `runtime-contract.json`.
The image label `local-inference.vllm.gb10-runtime-overlay` is
`sparse-row-clamp+dsml-recovery+responses-max+empty-required-tool-error-v2`. No model,
cache-identity, or transport setting changes. Enabling the variable in an
image without this overlay has no effect.

### API source validation

Status: implemented with CPU regression coverage. The
[source replay receipt](api-contract-source-replay.json) identifies all twelve
source transformations based on vLLM commit
`e2666d9a65f41fc376607531453cbd57c4c71016`. It records matching preimage/result
hashes, Python compilation, and repeat application leaving the result unchanged.

The CPU regressions run under Python 3.12 with OpenAI SDK 2.29.0 and Pydantic
2.12.5. They cover patch identity, request/response validation, all three SSE
event models, and the optional empty-tool serialization guard. The
[Responses model probe](responses_streaming_probe.py) applies the packaged
patch to the complete, hash-verified upstream protocol. It executes the actual
request, response, SDK event classes, and JSON serialization. Its
[receipt](api-responses-streaming-replay.json) records 27 serialized events
and 20 invalid-reasoning rejections. A regression restores the narrow response
annotation and reproduces the `max` failure while keeping `xhigh` valid.
The [fixture manifest](upstream/sources.json) pins both upstream protocol files
to the same vLLM revision; their [license](upstream/LICENSE) is included.

Run this probe without a vLLM installation or network access:

```bash
python runtime/deepseek0731-gb10/responses_streaming_probe.py
```

The [full-method probe](tool_choice_method_probe.py) executes the patched
`chat_completion_full_generator` with fixture engine/parser results and response
interfaces; its [receipt](api-tool-method-replay.json) records 25 passing cases.
Run the probe against the patched serving source:

```bash
python runtime/deepseek0731-gb10/tool_choice_method_probe.py \
  /path/to/vllm/entrypoints/openai/chat_completion/serving.py
```

The Responses probe substitutes rendering, sampling, and Harmony interfaces;
their optional request fields are outside its coverage. These checks do not
execute HTTP routing, a model, streaming generation, or the installed ARM64
image. Rebuilt-image qualification requires the named/required
tool-result and Responses-effort reproductions against an image carrying the
combined overlay label above. The published native derivative remains unchanged.

## Published native runtime

Status: **research-only**. The builder is implemented; its recorded diagnostic
used four tensor-parallel ranks (TP4) and five speculative tokens.
This directory derives one ARM64 image from the exact published runtime
`ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028`.
It does not contain model weights. The published child image requires exact
two-rank (TP2) and TP4 live replays with five speculative tokens.

Published image:

```text
ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:827a8e8c5749b78529cc0015dd174e1b19a0accc116bc142282f8b75428f98bd
```

The pinned `base_image` in `runtime-contract.json` is shared with the GLM
runtime and provides rollback.
The published derivative includes this repository's `LICENSE`,
`THIRD_PARTY_NOTICES.md`, and source-revision label. It does not claim one
combined OCI license expression for every inherited component; that label
requires a separate audit of the exact base image.

## Runtime contracts

`runtime-contract.json` defines these source and build invariants:

- attests `sparse_mla.py`, whose fixed-capacity sparse-attention rows preserve
  physical buffer stride;
- attests `attention.py`, which dispatches through `indexer_op`;
- clamps negative sparse-row lengths in the Triton and PyTorch producers;
- recovers complete malformed-DSML calls for declared tools while preserving
  the streaming parser's drop-token and skip-tool behavior;
- patches the same Python/parser bytes in the installed package and retained
  source tree; and
- by default rebuilds only `_C_stable_libtorch` from retained vLLM commit
  `e2666d9a65f41fc376607531453cbd57c4c71016`, component-installs one shared
  object, and copies only that component into the final runtime.

The [native top-k patch](patches/0002-pr431-native-topk-row-contract.patch)
retains its historical upstream-series references to r29 and the `0003` producer
patch. The [Python runtime patch](patches/0001-gb10-deepseek-runtime-hardening.patch)
owns the producer change in this builder.

`runtime-contract.json` pins source preimages, patched files, native build
inputs, and the required native library size. Source and ABI mismatches stop
the build. The rebuilt library digest is compared with the reference artifact
for reporting; it may differ. Final installation must match the digest and
ABI fields recorded for that build.

## Build

The default target includes the compiled sparse top-k defenses described in
the runtime contracts above:

```bash
bash runtime/deepseek0731-gb10/build-image.sh \
  sparkring/deepseek-v4-flash-0731-gb10-hardened:local
```

Build only the Python/parser diagnostic image with:

```bash
SPARKRING_DEEPSEEK_GB10_TARGET=thin \
  bash runtime/deepseek0731-gb10/build-image.sh \
  sparkring/deepseek-v4-flash-0731-gb10-hardened:thin
```

The retained native graph rebuilds two objects and relinks one library. A
cached ARM64 measurement completed that build step in 15.03 seconds. The
recorded local base-image size is approximately 30.84 GB. Neither figure
establishes a cold-build or registry-transfer duration.

The build compares the old and rebuilt ELF machine, dynamic dependencies,
RPATH/RUNPATH, GNU version requirements, exported symbol set, and Build ID.
It also verifies the exact `CMakeCache.txt` and `build.ninja`, component install
contents, final shared-object hash, and final retained native headers.

## Launch boundary

The runtime supports DeepSeek-V4-Flash-0731 with TP2 or TP4 and five
speculative tokens. The TP4 seven-token NVFP4 configuration is a separate
research arm. Each exact image, topology, and speculative depth still
requires startup, strict streamed tool-call, long-context, health, and
performance evidence.

Do not rely on the inherited image entrypoint. The DeepSeek quickstart must
continue to override it with `/opt/venv/bin/vllm`, pass
`--device /dev/infiniband` for multi-host NCCL, and retain:

```text
LD_PRELOAD=/usr/local/cuda/compat/libcuda.so.1:/opt/sparkring/nccl/libnccl.so.2
```

Before a live launch, run the explicit verifier under the resolved launch
environment. For the default native image:

```bash
docker run --rm \
  --entrypoint python3 \
  --env 'LD_PRELOAD=/usr/local/cuda/compat/libcuda.so.1:/opt/sparkring/nccl/libnccl.so.2' \
  <image-id> \
  /opt/sparkring-deepseek-gb10/verify_image.py \
  --expect-native --require-launch-env
```

The verifier does not load a checkpoint or replace a live service. Final
qualification must use the exact TP2 or TP4 environment and launch command.
