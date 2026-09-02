# GLM-5.3 Flash GB10 operator image

This directory builds and runs one Linux/ARM64 image for GLM-5.3 Flash on four
NVIDIA GB10 systems. The image combines Local Inference Lab's GLM-specific
vLLM work, BF16 DFlash2 speculation, B12X kernels, switchless NCCL,
fastsafetensors, and SparkCache. The pinned Local Inference Lab source line is
named `Jovian Judgement Community R10` in [`pins.json`](pins.json). One image
supports TP4 with DCP1, DCP2, or DCP4.

Local Inference Lab supplies the model quantization and the primary runtime
work that makes this profile practical:

- [`local-inference-lab/GLM-5.3-Flash-NVFP4`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4)
  is the target checkpoint;
- [`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement)
  supplies the GLM runtime and scheduler work;
- [`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x)
  supplies the GB10 kernel integration.

The external BF16 draft is
[`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
Exact revisions and source-tree hashes are in [`pins.json`](pins.json).

## Use the runtime

Follow the
[`GLM-5.3 GB10 quickstart`](../../docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md)
to obtain or build the image, distribute it once through the direct fabric,
and start four ranks. [`runtime.env.example`](runtime.env.example) exposes the
model paths, image identity, DCP degree, context limit, scheduler budget, KV
allocation, speculation, cache limits, network interfaces, and ports.

The recommended source-built profile uses:

| Setting | Value |
|---|---:|
| topology | TP4/DCP4 |
| maximum model length | 1,048,576 tokens |
| batched-token budget | 8,192 tokens |
| prefill scheduler interval | 2 |
| sequences | 16 |
| multimodal requests | up to four images and one video per request |
| FP8 KV allocation | 24 GiB for the default DCP4 profile; 26 GiB for DCP1; 30 GiB for DCP2 |
| DFlash2 depth | 7 |
| SparkCache publication | flat copy-on-write page tails (`tail-cow-v2`) |
| shared GPU-prefix retention | up to 300 seconds |
| modalities | images and video (`MULTIMODAL_INPUTS=1`); text-only mode available |

DCP1 resolves to one-token KV interleaving without full-CKV gather. DCP2 and
DCP4 resolve to four-token KV interleaving with full-CKV gather. Operators can
change every value in the environment file without rebuilding the image.
`SPARKCACHE_ENABLED=0` omits the persistent connector while retaining vLLM's
GPU prefix cache; `SPARKCACHE_ENABLED=1` enables both layers.

The launcher accepts images and video by default. The target checkpoint ships the
GLM-5.3 vision tower (`Glm5NextForConditionalGeneration`, 347 BF16 tensors,
1.05 GiB), and the runtime registers it. `MULTIMODAL_INPUTS=1` admits up to
`MAX_IMAGES_PER_PROMPT` images and `MAX_VIDEOS_PER_PROMPT` videos per request.
SparkCache binds media identity and placeholder geometry into persistent
context digests. `MULTIMODAL_INPUTS=0` passes `--language-model-only`, so the
vision tower is not loaded and media content is rejected before inference.

With the connector enabled, `SPARKCACHE_ACCESS_MODE=read-write` restores and
publishes persistent entries. `restore-only` reuses compatible entries but
does not capture or publish new prompt state. Missing entries are computed by
vLLM normally. `store-only` and `disabled` are diagnostic modes.

### Choose the persistent publication format

`SPARKCACHE_PUBLICATION_SCHEMA` controls how SparkCache writes reusable
manager-page state. Each value has a distinct cache identity. Entries from one
format cannot be mistaken for entries from another.

| Value | What it writes | Use |
|---|---|---|
| `snapshot-v1` | One complete immutable object for every published context | Published-image rollback and the simplest storage layout |
| `tail-cow-v1` | An immutable base plus changed page objects | Compatibility testing for the first page-tail format |
| `tail-cow-v2` | One authenticated base plus a flat descriptor chain of changed page objects | Recommended source-built TP4/DCP4 profile for growing conversations |

`tail-cow-v2` captures only the changed physical pages after a reusable base.
The publication worker encodes those sparse pages directly instead of
reconstructing and comparing another complete snapshot. Restore resolves the
flat descriptors onto the authenticated base, which keeps lookup depth
bounded as the conversation grows. A damaged or incompatible object is
rejected and vLLM computes the missing prompt state normally.

`SPARKCACHE_CACHE_NAMESPACE` selects rank-local persistent-context storage.
Use `glm53-flash-dcp4-page-tail-cow-v2` with the recommended profile and
`glm53-flash-dcp4-snapshot-v1` with the published rollback. The directory name
is not part of SparkCache's content identity or stored format, but separate
directories make rollback and inspection unambiguous.

Compilation artifacts use the independent, source-bound
`JIT_CACHE_NAMESPACE`. Changing or clearing a SparkCache data namespace does
not discard Triton, TorchInductor, B12X, or vLLM compilation caches. Each rank
keeps its own persistent copy under `CACHE_HOST_ROOT`; the four ranks do not
write to one network-shared compilation directory.

Set `SPARKCACHE_ASYNC_PAGE_CAPTURE=1` to capture manager pages through the
bounded CUDA ring. `SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES` defaults to 8 GiB for
DCP1, 5 GiB for DCP2, and 3 GiB for DCP4. The DCP4 profile uses two 3 GiB
capture slots, so the background publisher can consume one completed capture
while a later capture uses the other. Restore separately pipelines bounded
NVMe reads and CUDA placement through two 256 MiB mapped arenas. A third arena
is not part of the profile because the two-stage pipeline has no measured
arena wait that would justify more unified-memory pressure.

The rank-0 launcher can run `warmup-glm53-dflash.py` before it returns success.
The default environment template enables C1/C2/C4/C8/C16 warmup and prompt
spans covering the DFlash Triton `BLOCK_SIZE` specializations through 256.
Do not admit normal traffic until the rank-0 launcher returns. A failed or
timed-out warmup makes launch fail instead of leaving an apparently healthy
API in front of a wedged engine.
The failure-shaped replay and remaining causal limitation are recorded in
[`dflash-jit-readiness-validation.json`](dflash-jit-readiness-validation.json).

The published registry image below contains the complete-snapshot
implementation. It does not contain `tail-cow-v2`. Build the pinned source in
this directory before selecting `tail-cow-v2`.

### Read SparkCache telemetry

SparkCache presents the aggregate worker state as three short INFO lines:

```text
sparkcache: capacity ranks=4 entries=12 used=1.2/160.0GiB healthy=yes
sparkcache: publications count=12 payload=1.2GiB unique=1.2GiB
sparkcache: writes staged=1.2GiB dedup=0B aborted=0B failed=0B
```

`capacity` describes visible entries and configured storage. `publications`
compares the logical state represented by committed manifests with newly
stored immutable bytes. `writes` reports submitted storage traffic,
deduplication, and bytes from aborted or failed attempts. The `/metrics`
endpoint retains the individual numeric counters for monitoring and analysis.

## Build from pinned source

The builder accepts clean checkouts at the exact vLLM and SparkCache commits
recorded in `pins.json`. It rejects a different commit, Git tree, package tree,
parent image, retained runtime file, or CUDA library. Build the SparkCache
snapshot library on an ARM64 CUDA 13 host before invoking the image builder:

| Input | Source of truth |
|---|---|
| ARM64 parent image and retained compiled extensions | `pins.json` `parent` and `vllm` records |
| vLLM source checkout | `pins.json` `vllm.commit`, tree, and package tree |
| SparkCache source checkout | `pins.json` `sparkcache.commit`, tree, package tree, and source hash |
| CUDA placement and snapshot libraries | SparkCache source plus the SHA-256 values in `pins.json` |
| Short KV-metrics logger transform | [`patch_kv_metrics_logging.py`](patch_kv_metrics_logging.py) and its exact vLLM preimage |

```bash
cmake -S /source/sparkcache/sparkcache/native \
  -B /source/sparkcache/sparkcache/native/build-cuda \
  -G Ninja -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build /source/sparkcache/sparkcache/native/build-cuda \
  --target spark_cache_snapshot
```

The image builder verifies commits, trees, package subtrees, runtime files,
the parent image, retained compiled extensions, and both SparkCache CUDA
libraries before producing an image. It also applies the exact-preimage vLLM
metrics formatter and records that transform in the source receipt.

```bash
python runtime/glm53-flash-jj-r8-gb10/build_image.py \
  --vllm-source /source/vllm \
  --sparkcache-source /source/sparkcache \
  --snapshot-library /source/sparkcache/sparkcache/native/build-cuda/libspark_cache_snapshot.so \
  --output-image sparkring-glm53-sparkcache:page-tail-v2-local \
  --receipt ./glm53-build-receipt.json
```

The source-built image has no published registry digest. The build does not
include model checkpoints, site addresses, SSH credentials, or persistent
cache data. Record its local image ID with:

```bash
docker image inspect sparkring-glm53-sparkcache:page-tail-v2-local \
  --format '{{.Id}}'
```

The locally built page-tail image has ID
`sha256:1c98731de7e3963a609aa7e30582cabbaf59c7d3a59e88d704f21319fa3e0daa`.
Its embedded-source and native-library checks are recorded in
[`page-tail-v2-local-image-receipt.json`](page-tail-v2-local-image-receipt.json).
This is a local image identity, not a pullable registry digest.

## Complete-snapshot rollback image

The immutable Linux/ARM64 image below remains the pullable rollback. It uses
complete `snapshot-v1` publication and does not contain the source-built
`tail-cow-v2` implementation.

```text
ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:3c377f1e4136285ebf66c32c36c3d01fd929f8aba0836cd0a16ed63cfd7e1762
```

Its local Docker image ID is
`sha256:d1a07147c9e25f3d3e0af6b1499c4988b1ae61138e327aa05c9ad9dc568e39a9`.
Construction, direct-fabric distribution, profile smoke tests, historical
deep-context evidence, and limitations are recorded in
[`multimodal-lease300-image-receipt.json`](multimodal-lease300-image-receipt.json),
[`async-capture-image-receipt.json`](async-capture-image-receipt.json),
[`ASYNC_CAPTURE_IMAGE_VALIDATION.md`](ASYNC_CAPTURE_IMAGE_VALIDATION.md), and the
[`deep-context record`](../../performance/records/glm53-flash/dcp1-deep-context-boundary-20260831.md).
The
[`scheduler-cadence record`](../../performance/records/glm53-flash/scheduler-cadence-20260902.md)
compares intervals two and eight on simultaneous 6K-token requests.

Run the offline contracts with:

```bash
python -m pytest runtime/glm53-flash-jj-r8-gb10 -q
```
