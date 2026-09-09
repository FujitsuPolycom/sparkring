# Hardened DeepSeek V4 runtime for Fujitsu GB10

## API reasoning and required-tool policies

Status: implemented in the source overlay; rebuilt-image qualification is
outstanding. The published native derivative at digest
`sha256:827a8e8c5749b78529cc0015dd174e1b19a0accc116bc142282f8b75428f98bd`
does not contain these API changes.

The Responses request model accepts `reasoning.effort: "max"` in addition to
`none`, `minimal`, `low`, `medium`, `high`, and `xhigh`. It retains the OpenAI
SDK reasoning model's other fields and validation. Unknown effort values remain
invalid. This request-model compatibility change addresses issue #184 and does
not change model sampling or generation behavior.

Set `SPARKRING_REJECT_EMPTY_REQUIRED_TOOL_CALLS=1` in the API container
environment to reject nonstreaming Chat Completions whose named or required
tool choice produces no parsed tool calls. The response is an HTTP 500
`ToolChoiceContractError`, rather than a successful empty tool-call array.
The default remains disabled. Valid nonempty parser results retain their
serialization, and automatic tool choice is unaffected.

This check covers the empty-result failure in SparkRing issue #217. It does
not validate argument JSON, function names, schemas, or streaming results.
A streaming HTTP response may already have started before a missing tool
call is detectable; streaming requires a separate event-level policy.
Clients should bound retries and may increase the output-token budget when
reasoning exhausts the available generation length.

The runtime patch and per-file hashes are bound by `runtime-contract.json`.
The image overlay label is
`sparse-row-clamp+dsml-recovery+responses-max+empty-required-tool-error-v1`. No model,
cache-identity, or transport setting changes. Enabling the variable in an
image without this overlay has no effect.

### API source validation

Status: implemented with CPU regression coverage. The
[source replay receipt](api-contract-source-replay.json) identifies all twelve
source files at vLLM commit `e2666d9a65f41fc376607531453cbd57c4c71016`.
The complete patch matches every preimage and result hash, compiles as Python,
and is a no-op when applied again to the complete result set.

Twelve focused tests pass under Python 3.12 with OpenAI SDK 2.29.0 and Pydantic
2.12.5. They cover patch identity, request-model validation, and the optional
empty-tool serialization guard. The
[full-method probe](tool_choice_method_probe.py) executes the patched
`chat_completion_full_generator` with fixture engine/parser results and response
interfaces; its [receipt](api-tool-method-replay.json) records 25 passing cases.
Run the probe against the patched serving source:

```bash
python runtime/deepseek0731-gb10/tool_choice_method_probe.py \
  /path/to/vllm/entrypoints/openai/chat_completion/serving.py
```

These checks do not execute HTTP routing, a model, streaming generation, or the
installed ARM64 image. Rebuilt-image qualification requires the named/required
tool-result and Responses-effort reproductions against an image carrying the
combined overlay label above. The published native derivative remains unchanged.

## Published native runtime

Status: **research-only; builder implemented with a TP4/K5 diagnostic run**.
This directory derives one ARM64 image from the exact published runtime
`ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028`.
It does not contain model weights. The published digest still needs exact
TP2/K5 and TP4/K5 live replays.

Published image:

```text
ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:827a8e8c5749b78529cc0015dd174e1b19a0accc116bc142282f8b75428f98bd
```

The generic `6fc26f...` image remains unchanged as the rollback and GLM base.
The published derivative includes this repository's `LICENSE`,
`THIRD_PARTY_NOTICES.md`, and source-revision label. It does not claim one
combined OCI license expression for every inherited component; that label
requires a separate audit of the exact base image.

## Runtime contracts

The GB10 contract:

- attests without changing `sparse_mla.py`, whose fixed-capacity C128A rows
  already preserve physical buffer stride;
- attests without changing `attention.py`, because the captured
  all-candidates shortcut fixed by vLLM pull request 52492 is absent;
- clamps negative sparse-row lengths in both the Triton producer and the
  native two-dimensional Python producer;
- semantically ports malformed-DSML recovery from vLLM pull request 52645,
  preserving the GB10 streaming parser's drop-token and skip-tool behavior;
- accepts `reasoning.effort: "max"` on the Responses API while preserving the
  other fields and validation supplied by the OpenAI-compatible reasoning
  object;
- patches the same Python/parser bytes in the installed package and retained
  source tree; and
- by default rebuilds only `_C_stable_libtorch` from retained vLLM commit
  `e2666d9a65f41fc376607531453cbd57c4c71016`, component-installs one shared
  object, and copies only that component into the final runtime.

The native `0002` patch retains its upstream-series commit message, which
names the r29 `0003` producer patch. In this GB10 builder, that Python/Triton
producer change is carried by `0001-gb10-deepseek-runtime-hardening.patch`.

`runtime-contract.json` records every preimage, result, build input, source
header, and native artifact hash. Any drift stops the build.

## Build

The default target includes the compiled pull request 431 top-k defenses:

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
cached ARM64 measurement completed that build step in 15.03 seconds. A cold
build is dominated by pulling the approximately 30.84 GB local base image;
this directory does not establish a cold registry-transfer duration.

The build compares the old and rebuilt ELF machine, dynamic dependencies,
RPATH/RUNPATH, GNU version requirements, exported symbol set, and Build ID.
It also verifies the exact `CMakeCache.txt` and `build.ninja`, component install
contents, final shared-object hash, and final retained native headers.

## Launch boundary

One content-addressed image can serve the primary plain-0731 profiles as TP2
K5 or TP4 K5; topology and speculative depth are launch inputs, not separate
compiled runtimes. TP4 K7 belongs to a separate, unqualified NVFP4 research
arm and is not implied by this builder. Each exact image/topology/depth still
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
