# Hardened DeepSeek V4 runtime for Fujitsu GB10

Status: **research-only; builder implemented with a TP4/K5 diagnostic run**.
This directory derives one ARM64 image from the exact published runtime
`ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028`.
It does not contain model weights. The published digest still needs exact
TP2/K5 and TP4/K5 live replays.

Published image:

```text
ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:1574ba87fe4a0ad38c25a30087929ad549d823730be83b33e91fe4745b7a6571
```

The generic `6fc26f...` image remains unchanged as the rollback and GLM base.
The first published derivative does not add explicit OCI revision or combined
license labels. Treat the digest and this repository as its provenance record;
a later rebuild should add those labels only after auditing the inherited
image's complete license expression.

## What changes

The GB10 contract:

- attests without changing `sparse_mla.py`, whose fixed-capacity C128A rows
  already preserve physical buffer stride;
- attests without changing `attention.py`, because the captured
  all-candidates shortcut fixed by vLLM pull request 52492 is absent;
- clamps negative sparse-row lengths in both the Triton producer and the
  native two-dimensional Python producer;
- semantically ports malformed-DSML recovery from vLLM pull request 52645,
  preserving the GB10 streaming parser's drop-token and skip-tool behavior;
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
