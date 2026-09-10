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

The launcher also accepts `TARGET_MODEL_VARIANT=nvfp4-spark` and
`SPECULATION_METHOD=mtp`. `TARGET_MODEL_VARIANT=nvidia-nvfp4` selects
[`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
at revision `423acf37`, NVIDIA's official plain-ModelOpt NVFP4 export; the
launcher switches to `--quantization modelopt` and derives an `--hf-overrides`
entry that keeps its BF16 native-MTP layer unquantized. Use `LOAD_FORMAT=safetensors`
and `DFLASH_WARMUP_TIMEOUT_SECONDS=1500` with it (the managed mesh renderer
sets both): its ~6 GB shards overflow GB10 unified memory under fastsafetensors,
and the host-mmap load needs ~8 min. That variant is research-only and has no
published receipt. The separate
[native-MTP3 mesh profile](../glm53-spark-mtp3-mesh/README.md) supplies its
target revision, depth-three graph sizes, transport bundle, and cache identity.
That profile requires no external draft checkpoint. Its hardware-forwarded
mesh and native-MTP cache namespace are research-only, not covered by the
DFlash/SIRCL functional qualification described below.

## Use the runtime

Follow the
[`GLM-5.3 GB10 quickstart`](../../docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md)
to obtain or build the image, distribute it once through the direct fabric,
and start four ranks. [`runtime.env.example`](runtime.env.example) exposes the
model paths, image identity, DCP degree, context limit, scheduler budget, KV
allocation, speculation, cache limits, network interfaces, ports, and an
optional chat-template override (`CHAT_TEMPLATE_HOST_PATH`, bind-mounted
read-only and passed as `--chat-template`; empty serves the checkpoint's own
template).

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
| CUDA graph capture sizes | every eight-row DFlash request-batch shape from 8 through 128 |
| model kernels | B12X attention, KDA prefill, MoE, and linear |
| collective/RMSNorm fusion | disabled |
| FlashInfer autotuning | disabled |
| model loader | fastsafetensors with queue size 1 |
| multimodal requests | up to four images and one video per request |
| FP8 KV allocation | 24 GiB per rank for DCP1, DCP2, and DCP4; explicit byte override supported |
| DFlash2 depth | 7 |
| SparkCache publication | flat copy-on-write page tails (`tail-cow-v2`) |
| shared GPU-prefix retention | up to 300 seconds |
| modalities | images and video (`MULTIMODAL_INPUTS=1`); text-only mode available |

DCP1 resolves to one-token KV interleaving without full-CKV gather. DCP2 and
DCP4 resolve to four-token KV interleaving with full-CKV gather. Operators can
change every value in the environment file without rebuilding the image.
Read model-wide KV capacity from vLLM's startup output. The quickstart lists
the reference capacity measurements and their memory allocations.
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
asynchronous page-capture slots and sixteen 256-MiB restore arenas (two for
each of eight load lanes).

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
The environment template sets `SPARKCACHE_ASYNC_PAGE_CAPTURE=auto`: capture
is enabled for `read-write` and `store-only`, and disabled for `restore-only`,
`disabled`, or `SPARKCACHE_ENABLED=0`. Explicit `1` still rejects a mode that
cannot publish. Explicit `0` uses synchronous publication in publishing modes.
The launcher defaults to `0` when no capture setting is supplied.

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

Set `SPARKCACHE_ASYNC_PAGE_CAPTURE=auto` to capture manager pages through the
bounded CUDA ring. `SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES` defaults to 8 GiB for
DCP1, 5 GiB for DCP2, and 3 GiB for DCP4. The DCP4 profile uses two 3 GiB
capture slots, so the background publisher can consume one completed capture
while a later capture uses the other. Restore separately pipelines bounded
NVMe reads and CUDA placement through two 256 MiB mapped arenas **per load
lane**. Eight lanes reserve 4 GiB per rank for restore payloads. With the 6 GiB
capture ring, the DCP4 profile configures 10 GiB per rank (40 GiB across TP4),
in addition to the 24 GiB per-rank KV allocation. Restore-only configures
4 GiB per rank and no capture slots. These figures describe payload capacities;
control arrays, Python objects, shared bases, transport, model weights and
allocator overhead require additional memory. `SPARKCACHE_LOAD_THREADS` defaults
to eight; throughput and memory-pressure effects need hardware measurements.

### Bounded SparkCache source profile

Status: **implemented**, with CPU launcher-contract coverage. The explicit
`tp4-dcp1-mtp3-sparkcache` source-image profile selects GLM NVFP4-Spark,
TP4/DCP1/PP1, native MTP3, 512-token blocks, coalescing, and mHC prefill sharding.
Its image receipt and source lock must identify the installed sources and native
libraries. Selecting the profile does not qualify a rebuilt image or enable
SparkCache on a TP2 profile.

The renderer must set `SPARKCACHE_SOURCE_LEASE_CONTRACT` to
`/usr/local/lib/python3.12/dist-packages/sparkcache/runtime_patches/vllm-connector-jobs-source-contract.json`.
The image verifier binds that complete contract to the installed vLLM files.
The launcher rejects another path and rejects this override outside the named
profile. Capture uses `connector-jobs`, the snapshot library at
`/opt/sparkcache-native/libspark_cache_snapshot.so`, and the profile's exact
snapshot and placement hashes. Cache load failures retain `recompute` behavior.

| Profile capacity | Required value per rank |
|---|---:|
| GPU KV allocation | 24 GiB |
| Capture slots | 2 × 512 MiB |
| Restore lanes / I/O workers | 2 / 2 |
| Restore arenas | 2 × 64 MiB per lane; 256 MiB total |
| Capture plus restore payload budget | 1,280 MiB |
| Disk budget / low watermark | 8 GiB / 6 GiB |
| Capture span minimum / maximum | 4,096 / 65,536 tokens |
| Model context limit | 1,048,576 tokens |

The profile requires read/write asynchronous capture, two pending restores,
and `tail-cow-v2` publication. Periodic page snapshots are disabled. Larger
operator-default buffers or alternate package overlays are rejected. Existing
profiles retain their contract and allocation defaults. All explicit source
profiles pass their name to the image entrypoint and enable unbuffered logging.

### Inspect configured memory before launch

Status: **implemented**. The launcher can print a JSON allocation plan without
Docker, GPUs, checkpoint files, or cache directories. It sources the same trusted
shell configuration as a launch and resolves its byte counts and capture mode:

```bash
SPARKRING_PRINT_MEMORY_PLAN=1 bash runtime/glm53-flash-jj-r8-gb10/launch-rank.sh \
  0 runtime/glm53-flash-jj-r8-gb10/runtime.env.example
```

Set `SPARKCACHE_BUFFER_BUDGET_BYTES` in the configuration to reject restore
and capture payload capacities above that per-rank ceiling before host checks
or Docker access. Zero, the default, disables the ceiling. For example,
`10737418240` admits the 10 GiB DCP4 read-write configuration exactly. The
report includes per-rank and topology totals; DCP shards state but does not
reduce the physical TP rank count. A normal launch also logs the plan.

This ceiling does not cap total process memory or predict whether serving fits.
It excludes model weights, KV allocation, transient reads, retained shared bases,
CUDA control arrays, compilation and transport buffers, and allocator overhead.
The report lists KV separately. A passing offline plan does not qualify a CUDA
allocation or a serving performance result.

When `DFLASH_WARMUP=1`, the readiness entrypoint runs `warmup_dflash.py` before
Docker reports rank 0 as healthy. Source builds then run six explicit streaming
sampler requests through `serve_with_warmup.py`, each with thinking enabled and
`min_p=0`:

| Request | Temperature | top_k | top_p | Explicit seed |
|---|---:|---:|---:|---:|
| Unfiltered | 1.0 | -1 | 1.0 | absent |
| Temperature scaling | 0.7 | -1 | 1.0 | absent |
| Top-k only | 1.0 | 40 | 1.0 | absent |
| Top-p only | 1.0 | -1 | 0.9 | absent |
| Top-k and top-p | 1.0 | 40 | 0.9 | absent |
| Seeded top-k and top-p | 0.7 | 40 | 0.9 | 0 |

If `DFLASH_WARMUP_CONCURRENCIES` includes a value of at least two, the sampler
recipe then runs three homogeneous pairs: top-k only, top-p only, and both
filters. Each paired request uses temperature one, no explicit seed,
`max_tokens=min_tokens=128`, and `ignore_eos=true`. Both requests must complete
all 128 tokens with finish reason `length`, and their HTTP intervals must
overlap. The pairs run one stage at a time under the same startup deadline.
A C1-only configuration skips these pairs and records `coverage=limited-c1-only`.

Explicit neutral filters prevent a checkpoint's generation defaults from choosing
an unintended arm. In the pinned vLLM source, [sampling state](https://github.com/FujitsuPolycom/vllm/blob/e02b174693e13859de61811b5e8cd13d5308e259/vllm/v1/worker/gpu/sample/states.py#L40-L100)
normalizes disabled top-k, omits neutral top-k/top-p tensors, and skips temperature
scaling when every temperature is zero or one. The [sampler](https://github.com/FujitsuPolycom/vllm/blob/e02b174693e13859de61811b5e8cd13d5308e259/vllm/v1/worker/gpu/sample/sampler.py#L298-L321)
uses its Gumbel fallback for unfiltered or explicitly seeded requests. These
branches justify the recipe; they do not prove every speculative execution path
or compiled kernel was reached by an HTTP request.

Status: **implemented**, with CPU request and readiness tests. Each arm requests
`stream=true` and final usage. Its SSE stream must contain one choice at index
zero, finish with `stop` or `length`, report positive integer prompt/completion
usage with a consistent total, and terminate with `[DONE]`. A final usage-only
chunk after the finish is accepted, as are SSE comments and blank lines. Token
counts come from usage, not the number of chunks. Malformed, failed, truncated,
oversized or unfinished streams withhold readiness. The `sampling_warmup` log
records `stream=true`, each arm's usage, finish reason and elapsed time with
`coverage=request-recipe-complete` when all concurrent stages run, and
`jit_coverage_verified=false`. API readiness, shape batches and sampler requests
share `DFLASH_WARMUP_TIMEOUT_SECONDS`; each operation receives the remaining
budget, stream consumption checks the deadline, and an expired budget prevents
the readiness marker. Shape-request
nonces precede repeated prompt text and vary between runs to avoid prefix reuse.

The [published child image](hotfix/README.md) contains the earlier single
temperature-one/thinking request, not this six-arm recipe. Its
[receipt](hotfix/public-image.json) records 22 installed readiness/liveness
tests; those results do not qualify a rebuilt image. The operator image in
`pins.json` also remains unchanged.

Cold-cache full-model sampler coverage, filtered concurrent GPU specializations,
mixed long/short prefill coverage, and all recurrent KDA specializations remain
unqualified. The concurrent stages record HTTP overlap, which does not establish
that both requests shared a GPU batch or identify each worker's sampler path.
The [bounded sampler observation](../../performance/records/glm53-flash/sampler-concurrency-20260909.md)
records why C1-only warmup was insufficient on one native-MTP3 runtime.
In particular, the
reported several-4K-prefills-behind-long-decode case requires an identified image,
tokenized request lengths, actual overlapping execution and per-rank JIT evidence.
The short shape sweep below does not establish that case. Do not gate readiness
on guessed Triton cache filenames: cache presence alone does not prove that the
serving process initialized every required compiled variant.
The pinned sampler can select FlashInfer, Gumbel or a speculative path. JIT
monitor events are not per-worker execution receipts for all of those paths.
A backend coverage gate needs source-bound reports from every worker after its
required sampler paths complete; this HTTP warmup does not provide them.
The default environment template warms every concurrency from C1 through C16
and prompt spans covering the DFlash Triton `BLOCK_SIZE` specializations
through 256. DFlash depth seven verifies eight target rows per active request,
so the launcher captures every eight-row request-batch shape from 8 through
128. Each supported concurrency therefore has an exact CUDA graph instead of
padding intermediate request counts to a larger graph.
Do not admit normal traffic until the rank-0 launcher returns. A failed or
timed-out warmup makes launch fail instead of leaving an apparently healthy
API in front of a wedged engine.
The failure-shaped replay and remaining causal limitation are recorded in
[`dflash-jit-readiness-validation.json`](dflash-jit-readiness-validation.json).
The measured effect of exact request-batch graphs is recorded in
[`dflash2-exact-concurrency-graphs-20260904.md`](../../performance/records/glm53-flash/dflash2-exact-concurrency-graphs-20260904.md).

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

### Verify host memory before loading the model

Long-uptime GB10 systems can retain plenty of available RAM while exposing too
few large contiguous blocks for pinned loader staging. The recognizable
pattern is a fastsafetensors load that stops advancing on one shard, repeated
kernel compaction, very low available memory during the attempt, or an NVIDIA
`NV_ERR_NO_MEMORY` allocation failure. A degraded service can also show high
GPU utilization with unusually low power and falling decode throughput.

Run the read-only cluster preflight before starting the four ranks. The
companion
`scripts/config/glm53-flash-tp4-site.example.yaml` profile
checks both total available RAM and Normal-zone buddy blocks using the page
size reported by each kernel. It reports cumulative compaction counters for
diagnosis but does not use lifetime totals as a pass/fail threshold.

If memory headroom fails, inspect the explicit recovery plan generated by
`scripts/prepare_launch_memory.py`. Execution requires all configured serving
ports to be free and the confirmation token printed in the plan. The command
releases clean page-cache pages, requests kernel compaction, and reruns the
read-only memory checks. Reboot a rank if the configured headroom does not
recover. Neither preflight nor the model launcher performs this host mutation
automatically.

The
[`GLM-5.3 memory-preflight validation`](glm53-memory-preflight-live-validation.json)
records the configured rejection, unsuccessful online compaction, post-reboot
recovery, complete preflight pass, and public-image relaunch on four GB10 ranks.

### Distinguish readiness from scheduler liveness

Rank zero exposes scheduler liveness on the configured
`SPARKRING_LIVENESS_PORT`, which defaults to 8016. API `/health` proves that
the HTTP process is ready; it does not prove that the scheduler can admit a
waiting request.

`GET /liveness` returns HTTP 503 after the scheduler has zero running requests
and at least one waiting request for 60 seconds. It also returns 503 when
SparkCache reports uncertain capture-page ownership. `GET /metrics` on the
same port exports the liveness state and blocked duration.

Status: implemented; the output-stall detector has offline regression coverage
and still needs validation under live TP4 cache-restore traffic.
The source policy below is not included in the immutable
[published hotfix image](hotfix/README.md), which uses output-counter movement
only. Using this source policy requires a rebuilt, source-verified image;
changing a timeout alone does not install it.
Running requests return HTTP 503 with reason `engine_output_stall` after
`SPARKRING_LIVENESS_OUTPUT_SECONDS` (default 300 seconds) without output-counter
movement, an increase in the optional prompt-token counter, or KV allocation
above its high-water mark. Fresh HTTP scrapes do not reset that inactivity timer.
The high-water marks belong to the interval without output; falling KV usage,
request-count changes, and repeated allocation below the prior maximum do not
renew it. Output-counter movement or an observed idle period starts a new interval.
Missing output metrics while requests are running eventually report
`metrics_unavailable`, rather than certifying progress.

This is an allocation/activity heuristic, not an engine-step heartbeat. The
pinned vLLM reports output iterations and prompt totals with output-bearing
batches, so neither is assumed to advance after every prefill chunk. Increasing
KV allocation can keep a long prefill healthy even while both counters stay flat.
Allocation is not proof that GPU computation finished. A fully preallocated
prefill or restore can also remain flat while doing legitimate work.
Set the timeout above the longest measured interval without these observable
signals, with margin. At 2500 tokens/s, 1,048,576 prompt tokens take about 420
seconds before the first output, so 300 seconds is insufficient without
intermediate signals.
For example, 900 seconds provides margin for that single-request case; concurrent
load still requires measurement. The monitor sums metrics for the single-engine TP4 deployment;
it does not detect one stalled engine hidden by another progressing engine.
The JSON preserves `output_stalled_seconds` and `output_iterations`. It adds
`progress_stalled_seconds`, `last_progress_signal`, `kv_allocation_high_water`,
and optional `prompt_tokens`. The raw output gap can exceed the configured
timeout while allocation progresses. The monitor exports both
`sparkring:engine_output_stalled_seconds` and
`sparkring:engine_progress_stalled_seconds`; the latter controls the stall rule.

An unhealthy result does not restart the cluster. Use the deployment's
coordinated stop/recovery procedure after collecting all worker-thread stacks.
See [the executor-stall investigation](../../docs/ISSUE224_ENGINE_STALL.md).
For isolation of the pinned B12X indexer's histogram-publication race, the
launcher accepts `B12X_FUSED_INDEXER=0` and forwards it to the container. Use a
separate JIT cache namespace and coordinated restart on all ranks. This bypass
can change throughput and has not been qualified on the affected cluster.
The image builder applies the GPU-tested publication barrier and compile-cache
revision through `patch_indexer_barrier.py` before generating the B12X source
manifest. A [published child image](hotfix/README.md) includes this correction,
sampling/reasoning readiness warmup, and output-stall detection. The default
image pins remain unchanged. See [the source trace and fix](../../docs/ISSUE224_INDEXER_BARRIER.md).

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
| B12X histogram publication barrier | [`patch_indexer_barrier.py`](patch_indexer_barrier.py), with checked source and result hashes |

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
