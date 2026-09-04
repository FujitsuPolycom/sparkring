# GLM-5.3 Flash GB10 operator image

This directory builds and runs one Linux/ARM64 image for GLM-5.3 Flash on four
NVIDIA GB10 systems. The runtime combines Local Inference Lab's GLM-specific
vLLM work, BF16 DFlash2 speculation, B12X kernels, patched NCCL,
fastsafetensors, and SparkCache. The operator image also embeds the source-bound
SIRCL Python overlay and ARM64 native library. The pinned Local Inference Lab
source line is named
`Jovian Judgement Community R10` in [`pins.json`](pins.json). One image supports
TP4 with DCP1, DCP2, or DCP4.

The path `runtime/glm53-flash-jj-r8-gb10/` and JSON schema names beginning
with `sparkring-glm53-jj-r8-gb10` are stable compatibility locators. Their
`r8` component identifies the filesystem and JSON interface family; it does
not identify the embedded vLLM source composition. The exact vLLM commit and
the `community_release` field in [`pins.json`](pins.json) define that source
composition.

Local Inference Lab supplies the model quantization and the primary runtime
work that makes this profile practical:

- [`local-inference-lab/GLM-5.3-Flash-NVFP4`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4)
  is the target checkpoint;
- [`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement)
  supplies the upstream GLM runtime and scheduler work. The builder consumes
  the public
  [`sparkring-glm53-flash-gb10-e02b1746`](https://github.com/FujitsuPolycom/vllm/tree/sparkring-glm53-flash-gb10-e02b1746)
  tag in the FujitsuPolycom vLLM fork; that tag resolves to commit
  `e02b174693e13859de61811b5e8cd13d5308e259` in `pins.json`;
- [`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x)
  is the upstream B12X project. The builder installs commit `9ae41c5c` from the
  exact source fork recorded as
  [`voipmonitor/b12x`](https://github.com/voipmonitor/b12x) in `pins.json`.

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

The recommended profile uses:

| Setting | Value |
|---|---:|
| operator image ID | `sha256:5e32aaa1bbe3559e81db7706ed4286248f18d27cfdb186f6b851bf786eb43075` |
| topology | TP4/DCP4 |
| collective transport | SIRCL with capability and health checks; patched NCCL fallback |
| compute and quantization | BF16 compute with ModelOpt mixed quantization |
| maximum model length | 1,048,576 tokens |
| batched-token budget | 8,192 tokens |
| prefill scheduler interval | 2 |
| sequences | 16 |
| scheduler | asynchronous with chunked prefill and prefix caching |
| graph mode | `FULL_AND_PIECEWISE` |
| CUDA graph capture sizes | 8, 16, 32, 64, and 128 |
| model kernels | B12X attention, KDA prefill, MoE, and linear |
| collective/RMSNorm fusion | disabled |
| FlashInfer autotuning | disabled |
| model loader | fastsafetensors with queue size 1 |
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

### Preferred DCP4 transport: SIRCL with capability and health checks

**Status: four-rank functionally qualified.** The base
[`runtime.env.example`](runtime.env.example) keeps patched NCCL enabled because
SIRCL requires rank-specific peer addresses and RDMA devices. For DCP4, append
[`sircl-fused.env.example`](sircl-fused.env.example) to select the preferred
SIRCL graph-native and fused eager paths. The image supplies the Python and
native bundle, so only the fabric inputs vary by rank.

#### Embedded bundle identity

The image builder regenerates the allowlisted Python overlay from the checked
out SparkRing revision. It accepts only the ARM64
`libspark_transport_capi.so` whose SHA-256 is recorded by
[`sircl-public-build-receipt.json`](sircl-public-build-receipt.json). Build that
native input from the same clean revision on an ARM64 CUDA host:

```bash
cmake -S spark_transport -B build/spark-transport \
  -G Ninja \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DSPARK_TP4_ENABLE_FUSED_STREAM_SWITCH_SMOKE=ON
cmake --build build/spark-transport --parallel
ctest --test-dir build/spark-transport --output-on-failure
```

The
[`sircl-public-build-receipt.json`](sircl-public-build-receipt.json) receipt
binds the public `spark_transport` Git tree, overlay specification, generated
manifest, native-library SHA-256, toolchain, and native test result. It
establishes a content-addressed native build, not a four-rank serving result.
The image builder does not compile this library. Byte-for-byte reproducibility
has not been established, so `--sircl-library` must name the preserved artifact
from the receipt or a rebuild that happens to match its recorded SHA-256.

The builder rejects a different SparkRing transport tree, overlay
specification, generated manifest, build receipt, or native-library digest.
The resulting image carries the complete bundle at `/opt/spark-sircl` and the
launcher records its native-library and overlay-manifest hashes as container
labels.

Developers can set `SIRCL_BUNDLE_HOST_ROOT` to an absolute directory containing
a complete bundle. The launcher validates it and mounts it read-only over the
embedded bundle. Normal deployments leave that setting empty.

#### Configure the four ranks

Start with the base runtime environment, then append the fused SIRCL overlay:

```bash
cp runtime/glm53-flash-jj-r8-gb10/runtime.env.example "$HOME/glm53-flash.env"
cat runtime/glm53-flash-jj-r8-gb10/sircl-fused.env.example >> "$HOME/glm53-flash.env"
${EDITOR:-vi} "$HOME/glm53-flash.env"
```

Replace every `REPLACE` value in the combined file. The secondary values select
the second RDMA device function on each existing cabled ring edge; the topology
requires neither additional cables nor diagonal rank links. The launcher
rejects incomplete or repeated peer/device assignments, inconsistent modes,
invalid GIDs, and invalid port ranges before Docker starts.

The launcher sets `VLLM_SPARK_TP4_MODE=custom`, enables the width-4096 graph
adapter and shared capture stream, and disables the width-6144 Q1/Q40 graph
paths. It fixes `SPARK_TP4_FLIGHT_RECORDER=0`. Effective collective routing is:

| Collective | Implementation |
|---|---|
| Captured contiguous TP4 BF16 `[Q, 4096]`, Q=8/16/32/64/128 | graph-native SIRCL with direct doorbells |
| Eager contiguous TP4 BF16 `[Q, 4096]`, Q128 through Q8192 | fused dual-rail SIRCL |
| Eager contiguous TP4 BF16 `[Q, 4096]`, Q1 through Q127 | NCCL |
| Unsupported signatures, DCP, and non-TP collectives | NCCL |

The launcher passes `--disable-custom-all-reduce` to disable vLLM's built-in
custom all-reduce. SIRCL remains active when `SIRCL_ENABLED=1`.

The fused session owns four persistent QPs and two operation slots. Each slot
has a 67,109,888-byte mapped arena (64 MiB plus 1 KiB of control storage), for
134,219,776 mapped bytes per rank. Per-slot CUDA completion events permit
successive operations to use different caller streams. The fused proxy is
pinned to CPU 12; graph submission and progress are pinned to CPUs 10 and 11.

The Q8192 session derives its ports from the configured base pairs: primary
ports 19006/19007 and secondary ports 19106/19107. The derivation reserves two
ports for each admitted capacity Q1024/Q2048/Q4096/Q8192.

SIRCL's two transport slots are independent from SparkCache's two 3-GiB
asynchronous page-capture slots and two 256-MiB restore arenas.

#### Recorded functional evidence

The exact public image passed four-rank DCP4 startup with the embedded bundle.
Every rank accepted the capability vote; a 32,768-token SparkCache entry
restored after restart; a 129K-class request followed by eight concurrent
4K-class requests returned scheduler and cache ownership to idle. Test-only
builds also forced fused-device poison and a rank-2 proxy exit: no API became
ready and all four worker groups stopped without a collective hang. The
[`public-image receipt`](glm53-dcp4-sircl-public-image-receipt.json) records
these checks and their limits. They establish functional qualification, not a
broad transport-performance comparison.

The launcher enables a rank-wide capability vote before native construction.
Every rank reports the native and overlay identities, shared protocol geometry,
and local RDMA device/GID availability over vLLM's CPU process group. Any
rank-specific failure or shared mismatch stops all ranks before a SIRCL session
is created.

After vLLM's existing output synchronization, a host-only native health check
prevents synchronous or asynchronous model output from leaving an unhealthy
worker. The check does not synchronize CUDA. When one worker reports a SIRCL
error, vLLM's multiprocess monitor terminates every peer worker. Four-rank
fault injection must show that every worker terminates without a collective
hang before that image and profile receive functional qualification.

### NCCL fallback

Patched NCCL 2.30.7 handles every collective outside the SIRCL
supported-signature table. The launcher selects the ring algorithm over
RoCE/IB, the
`LL,LL128,Simple` protocol set, four minimum and maximum channels, cross-NIC
routing, and subnet-aware routing. `NCCL_SWITCHLESS_RING_ONLY=1` rejects a
topology that cannot use the direct cycle. cuMem is disabled, and the P2P level
is `SYS`. The default HCA pair is `rocep1s0f0,rocep1s0f1` with GID index 3;
operators replace those defaults when their primary interface names differ.

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
SparkCache translates the operator setting `tail-cow-v2` to the cache-identity
wire value `page-tail-cow-v2`; the longer name appears in the DCP4
storage-directory name so an operator can see which stored identity it
contains.
The publication worker encodes those sparse pages directly instead of
reconstructing and comparing another complete snapshot. Restore resolves the
flat descriptors onto the authenticated base, which keeps lookup depth
bounded as the conversation grows. A damaged or incompatible object is
rejected and vLLM computes the missing prompt state normally.

`SPARKCACHE_CACHE_NAMESPACE` selects rank-local persistent-context storage.
The directory name is not part of SparkCache's content identity, so the
source-built defaults include the runtime sources that determine manager-page
meaning. Use these complete values:

| Profile | Source-bound namespace |
|---|---|
| DCP1 snapshot | `glm53-flash-vllm-e02b1746-b12x-9ae41c5c-dcp1-snapshot-v1` |
| DCP2 snapshot | `glm53-flash-vllm-e02b1746-b12x-9ae41c5c-dcp2-snapshot-v1` |
| DCP4 page tails | `glm53-flash-vllm-e02b1746-b12x-9ae41c5c-dcp4-page-tail-cow-v2` |

These names prevent vLLM `e02b1746` with B12X `9ae41c5c` from discovering
entries written by a different source composition. Other directories remain
on disk, but operators must not rename or copy them into this source-bound
root. Recompute the prompt with vLLM `e02b1746` and B12X `9ae41c5c` to populate
the matching root. The complete-snapshot recovery image retains its assigned
`glm53-flash-dcp4-snapshot-v1` directory and must use the launcher named by its
receipt.

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

The image entrypoint runs `warmup_dflash.py` before Docker reports rank 0 as
healthy.
The default environment template enables C1/C2/C4/C8/C16 warmup and prompt
spans covering the DFlash Triton `BLOCK_SIZE` specializations through 256.
Do not admit normal traffic until the rank-0 launcher returns. A failed or
timed-out warmup makes launch fail instead of leaving an apparently healthy
API in front of a wedged engine.
The failure-shaped replay and remaining causal limitation are recorded in
[`dflash-jit-readiness-validation.json`](dflash-jit-readiness-validation.json).

The page-tail registry image contains `tail-cow-v2`. The complete-snapshot
image remains available in the rollback section below.

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

### Distinguish readiness from scheduler liveness

Rank zero exposes scheduler liveness on the configured
`SPARKRING_LIVENESS_PORT`, which defaults to 8016. API `/health` proves that
the HTTP process is ready; it does not prove that the scheduler can admit a
waiting request.

`GET /liveness` returns HTTP 503 after the scheduler has zero running requests
and at least one waiting request for 60 seconds. It also returns 503 when
SparkCache reports uncertain capture-page ownership. `GET /metrics` on the
same port exports the liveness state and blocked duration.

Idle KV retention is warning-only. The default 330-second warning interval is
longer than the GLM profile's 300-second shared-prefix lease, so an intentional
lease is not treated as a dead scheduler.

## Build from pinned source

The builder accepts clean checkouts at the exact vLLM, B12X, and SparkCache
commits recorded in `pins.json`. It rejects a different commit, Git tree,
package tree, parent image, retained runtime file, or CUDA library. The B12X
checkout replaces the inherited Python package as a complete tree; its source
manifest is verified inside the image. Build the SparkCache snapshot library
on an ARM64 CUDA 13 host before invoking the image builder:

| Input | Source of truth |
|---|---|
| ARM64 parent image and retained compiled extensions | `pins.json` `parent` and `vllm` records |
| vLLM source checkout | `pins.json` `vllm.commit`, tree, and package tree |
| B12X source checkout | `pins.json` `b12x.repository`, commit, tree, and package tree |
| SparkCache source checkout | `pins.json` `sparkcache.commit`, tree, package tree, and source hash |
| CUDA placement and snapshot libraries | SparkCache source plus the SHA-256 values in `pins.json` |
| SIRCL Python overlay | This checkout plus `runtime/public-overlay-files.json` |
| SIRCL ARM64 native library | This checkout plus `sircl-public-build-receipt.json` and `pins.json` `sircl` hashes |
| Short KV-metrics logger transform | [`patch_kv_metrics_logging.py`](patch_kv_metrics_logging.py) and its exact vLLM preimage |

```bash
cmake -S /source/sparkcache/sparkcache/native \
  -B /source/sparkcache/sparkcache/native/build-cuda \
  -G Ninja -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build /source/sparkcache/sparkcache/native/build-cuda \
  --target spark_cache_snapshot
```

The image builder verifies commits, trees, package subtrees, runtime files, the
parent image, retained compiled extensions, both SparkCache CUDA libraries, and
the SIRCL overlay and native identities before producing an image. It records
the public vLLM commits that supply the B12X KDA prefill path, workspace
isolation, sparse MLA and DSA backends, C4 indexer binding, and replay-safe
per-token cache lengths. It also applies the exact-preimage vLLM metrics
formatter and records that transform in the source receipt.

```bash
python runtime/glm53-flash-jj-r8-gb10/build_image.py \
  --vllm-source /source/vllm \
  --b12x-source /source/b12x \
  --sparkcache-source /source/sparkcache \
  --snapshot-library /source/sparkcache/sparkcache/native/build-cuda/libspark_cache_snapshot.so \
  --sircl-library ./build/spark-transport/libspark_transport_capi.so \
  --output-image sparkring-glm53-sparkcache:page-tail-v2-local \
  --receipt ./glm53-build-receipt.json
```

Building from source produces a local image without publishing it. The build
does not include model checkpoints, site addresses, SSH credentials, or
persistent cache data. Record its local image ID with:

```bash
docker image inspect sparkring-glm53-sparkcache:page-tail-v2-local \
  --format '{{.Id}}'
```

The image recorded by
[`async-store-completion-public-image-receipt.json`](async-store-completion-public-image-receipt.json)
does not contain the embedded SIRCL bundle. That immutable receipt describes
only its named artifact; `glm53-dcp4-sircl-public-image-receipt.json` describes
the DCP4 operator image documented here.

## Page-tail operator image

Pull the immutable Linux/ARM64 image used by the recommended DCP4 profile:

```text
ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f
```

Its local image ID is
`sha256:5e32aaa1bbe3559e81db7706ed4286248f18d27cfdb186f6b851bf786eb43075`.
See the
[`public image receipt`](glm53-dcp4-sircl-public-image-receipt.json)
for source identities, registry-pull verification, DCP4 startup and restore,
concurrent ownership drain, failure containment, and limitations.

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
