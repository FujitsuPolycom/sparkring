# Run GLM-5.3 Flash on four GB10 systems

This guide starts one TP4 service across four NVIDIA GB10 systems. The same
Linux/ARM64 image supports DCP1, DCP2, and DCP4. The default request limit is
1,048,576 tokens and the default prefill scheduler budget is 8,192 tokens.
The operator can enable persistent SparkCache or use vLLM's GPU prefix cache
alone without changing the image.

The preferred launch is TP4/DCP4 with 24 GiB of FP8 KV per rank, SIRCL with
capability and health checks, scheduler interval two, BF16 DFlash2 at depth
seven, and SparkCache's flat copy-on-write page tails. Patched NCCL is the
fallback transport and handles collective signatures outside SIRCL's admission
rules. Growing conversations write changed pages instead of another complete
cached context. An earlier complete-snapshot image remains available as a
recovery artifact.

The image does not contain model checkpoints. It mounts the exact
[`local-inference-lab/GLM-5.3-Flash-NVFP4`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4)
target and the
[`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
BF16 draft. Local Inference Lab's
[`vLLM GLM development`](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement)
and [`B12X`](https://github.com/local-inference-lab/b12x) GB10 kernels provide
the model-specific runtime and performance foundation. The pinned vLLM source
line is named `Jovian Judgement Community R10` in the image contract.

## Choose the image

### Pullable page-tail image

The recommended DCP4 profile uses this immutable Linux/ARM64 image:

```bash
image='ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:e34aa58fda32c2cc63bc70de680b50c5f2bb69c1e0ad3c5bce0782c6501f7d34'
expected_image_id='sha256:058b17b49ee3b5ffd805fa4a17e4d9efcb885f92349b98a8c8623bd7f0f96dd4'
docker pull "${image}"
test "$(docker image inspect "${image}" --format '{{.Id}}')" = "${expected_image_id}"
```

The published tag is
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache:20260903-async-store-completion`.
Use the digest above for reproducible deployment.

The exact source composition and local build command remain in
[`runtime/glm53-flash-jj-r8-gb10/`](../runtime/glm53-flash-jj-r8-gb10/README.md).

This published digest predates the embedded SIRCL bundle. Keep
`SIRCL_ENABLED=0` with this exact digest, or set `SIRCL_BUNDLE_HOST_ROOT` to a
complete developer bundle. An image built from the current source composition
contains the receipt-bound bundle and needs no host bundle path. This repository
does not yet record a published, verified digest for that derived composition.

### Complete-snapshot recovery artifact

The rollback uses complete `snapshot-v1` publication:

```text
registry: ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:3c377f1e4136285ebf66c32c36c3d01fd929f8aba0836cd0a16ed63cfd7e1762
local image ID: sha256:d1a07147c9e25f3d3e0af6b1499c4988b1ae61138e327aa05c9ad9dc568e39a9
platform: linux/arm64
```

The recovery image predates the readiness entrypoint used by this guide and is
not compatible with the launcher in this checkout. Its matching SparkRing
source revision is `a150c98ccfdc4b655679860121f24712490dd9ee`; the
[`recovery image receipt`](../runtime/glm53-flash-jj-r8-gb10/multimodal-lease300-image-receipt.json)
records its exact launch contract. The remaining commands in this guide use
the page-tail image selected above.

## Download the checkpoints once

Run on rank 0:

```bash
target_model=/srv/models/glm53-target
draft_model=/srv/models/glm53-dflash2-bf16

hf download local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --revision 520de24eabf507659eaef7c70f14fd584527facc \
  --local-dir "${target_model}"
hf download incoai/GLM-5.3-Flash-DFlash2 \
  --revision dc77ff1c99eeb2df044ee3d4f0094eb033fee410 \
  --local-dir "${draft_model}"
```

Copy each immutable directory to the same absolute path on the three follower
ranks. Use direct-link addresses where the site permits SSH over the 200 Gb/s
fabric:

```bash
followers=(operator@rank1.example.net operator@rank2.example.net operator@rank3.example.net)
for host in "${followers[@]}"; do
  ssh "${host}" mkdir -p "${target_model}" "${draft_model}"
  rsync -aH --partial --info=progress2 "${target_model}/" "${host}:${target_model}/"
  rsync -aH --partial --info=progress2 "${draft_model}/" "${host}:${draft_model}/"
done
```

The launcher verifies the target config/index and draft config/weights before
starting Docker.

## Distribute the image once through the direct fabric

Create one compressed archive from the selected image on rank 0:

```bash
archive_dir=/var/tmp/sparkring-images/glm53-flash
archive_name=sparkring-glm53-flash-arm64.tar.zst
mkdir -p "${archive_dir}"
docker image save "${image}" | zstd -T0 -3 -o "${archive_dir}/${archive_name}"
archive_sha256=$(sha256sum "${archive_dir}/${archive_name}" | awk '{print $1}')
```

Serve that directory on a trusted private address reachable from rank 0. Keep
this process running while the fan-out command executes:

```bash
python3 -m http.server 18080 \
  --bind '<rank-0-private-address>' \
  --directory "${archive_dir}"
```

In another rank-0 shell, forward the exact archive through the three direct
links, import it, and verify the image ID on every rank:

```bash
python scripts/fanout_image_archive.py \
  --site /secure/site.yaml \
  --source-url "http://<rank-0-private-address>:18080/${archive_name}" \
  --archive-name "${archive_name}" \
  --expected-sha256 "${archive_sha256}" \
  --target-directory /var/tmp/sparkring-images/glm53-flash \
  --image "${image}" \
  --expected-image-id "${expected_image_id}" \
  --execute --confirmation FANOUT_IMAGE_ARCHIVE \
  --output ./glm53-flash-image-fanout.json
```

See the [fan-out reference](DIRECT_FABRIC_IMAGE_ARCHIVE_FANOUT.md) for site
format, planning, resumption, verification, and interruption behavior.

## Configure each rank

Copy the environment template on every rank. It contains the page-tail image,
DCP4 geometry, and persistent-cache defaults used by this guide:

```bash
cp runtime/glm53-flash-jj-r8-gb10/runtime.env.example "$HOME/glm53-flash.env"
${EDITOR:-vi} "$HOME/glm53-flash.env"
```

Replace these five site values:

- `HOST_IP`: the address used by this rank;
- `MASTER_ADDR`: rank 0's address, identical on every rank;
- `TARGET_MODEL_HOST_PATH`: the target checkpoint directory;
- `DFLASH_MODEL_HOST_PATH`: the BF16 draft directory;
- `CACHE_HOST_ROOT`: a writable rank-local JIT and SparkCache directory.

The base environment leaves SIRCL disabled because RoCE peer addresses and
device names are rank-specific. In that form, patched NCCL is the complete
fallback. For an image built from the current source composition, append the
transport settings for the preferred DCP4 path:

```bash
cat runtime/glm53-flash-jj-r8-gb10/sircl-fused.env.example >> "$HOME/glm53-flash.env"
${EDITOR:-vi} "$HOME/glm53-flash.env"
```

Replace every additional `REPLACE` value with that rank's primary and secondary
peer addresses and RDMA devices. Leave `SIRCL_BUNDLE_HOST_ROOT` empty to use
the bundle inside the image. The overlay sets
`SIRCL_ENABLED=1`, direct graph doorbells, dual-rail fused exposure, the graph
CPU assignments, control-port bases, and timeouts. The runtime guide specifies
the resulting SIRCL/NCCL routing and mapped-memory allocation.

Developers can point `SIRCL_BUNDLE_HOST_ROOT` at an absolute directory that
contains the complete Python overlay, generated manifest, and native library.
The launcher validates the files and mounts that override read-only.

Before constructing native sessions, all ranks exchange the SIRCL artifact and
protocol identities and report their local RDMA device and GID availability. A
missing capability or shared mismatch stops all ranks. After vLLM synchronizes
model output, a host-only check stops output from an unhealthy SIRCL session;
the check does not synchronize CUDA.

**Status:** the capability and health checks are implemented, but live
qualification of the exact public bundle on four ranks is still pending. The
published DCP4 results in this guide used patched NCCL and are not SIRCL
performance evidence. To use the fallback, do not append the overlay and keep
`SIRCL_ENABLED=0`; patched NCCL then handles every collective.

The default OpenAI-compatible model name is `glm-5.3-flash`. Override
`SERVED_MODEL_NAME` only when the site needs a distinct routing name.

The profile accepts up to four images and one video per request. Set
`MAX_IMAGES_PER_PROMPT` or `MAX_VIDEOS_PER_PROMPT` to zero to disable that
modality. SparkCache binds media identity and placeholder geometry into the
persistent context digest, so different media cannot share an entry merely
because their placeholder tokens have the same shape.

The server binds `0.0.0.0` and serves without authentication by default. To
require an OpenAI-compatible bearer token, point `API_KEYS_FILE` at a mode-0600
rank-local file holding one accepted key per line; the launcher refuses to
start if the file is missing, empty, world- or group-readable, or contains
whitespace in a key.

This is host-level access control, not secret management. vLLM receives the
keys in its process arguments, which remain visible to an administrator who
can inspect the container or host process.

Choose the DCP degree with one line. DCP4 uses the environment template as
written:

```bash
DECODE_CONTEXT_PARALLEL_SIZE=4  # change to 1 or 2
```

When SparkCache is enabled with DCP1 or DCP2, use complete snapshots until
those layouts have matching asynchronous page-capture evidence:

```bash
DECODE_CONTEXT_PARALLEL_SIZE=1  # or 2
SPARKCACHE_PUBLICATION_SCHEMA='snapshot-v1'
SPARKCACHE_CACHE_NAMESPACE='glm53-flash-dcp1-snapshot-v1'  # use dcp2 for DCP2
SPARKCACHE_ASYNC_PAGE_CAPTURE=0
```

When `SPARKCACHE_ENABLED=0`, only the DCP value needs to change.

The launcher selects the matching GLM KV geometry automatically:

| DCP | KV interleave | Full-CKV prefill gather | Default FP8 KV per rank | Approx. logical KV capacity |
|---:|---:|---:|---:|---:|
| 1 | 1 token | disabled | 26 GiB | 1.30M tokens |
| 2 | 4 tokens | enabled | 30 GiB | 2.90M tokens |
| 4 | 4 tokens | enabled | 24 GiB | 4.32M tokens |

The capacity column is the model-wide value reported by vLLM. Do not multiply
it by the four physical ranks.

The recorded DCP4 deployment used 24 GiB per rank and completed exact 900K and
1M needle restores. The DCP1 profile completed a 942,898-token needle
retrieval under the 1M request limit. Set `KV_CACHE_MEMORY_BYTES` to a positive
byte count to override the topology-aware `auto` policy.

Choose persistent SparkCache or vLLM's GPU prefix cache alone without changing
the image:

```bash
SPARKCACHE_ENABLED=1  # persistent SparkCache plus vLLM prefix caching
SPARKCACHE_ENABLED=0  # vLLM prefix caching only
```

When SparkCache is enabled, choose whether the connector may publish:

```bash
SPARKCACHE_ACCESS_MODE=read-write   # restore existing entries and publish new ones
SPARKCACHE_ACCESS_MODE=restore-only # restore existing entries; never capture new prompts
```

The GLM-5.3 profile retains a verified shared GPU prefix for up to five
minutes so one restore can serve an extended request queue:

```bash
SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS=300
```

vLLM may release the prefix earlier when active requests need its KV blocks.
Reduce the value when large retained prefixes compete with the required
context length or concurrency.

Restore-only mode is useful for reuse-heavy serving or performance tests where
one-off prompt publication would add GPU-to-host capture work. A restore miss
is computed by vLLM normally.

The template's `SPARKCACHE_CACHE_NAMESPACE` value selects rank-local
persistent-context storage. It is not part of SparkCache's content identity or
stored format. Changing it selects a different root and therefore a different
set of discoverable entries.

`JIT_CACHE_NAMESPACE` independently selects persistent Triton,
TorchInductor, B12X, and vLLM compilation data. Keep its source-bound default
when changing or clearing SparkCache storage. Every rank keeps a local copy;
do not point all four ranks at one network-shared compilation directory.

The image supports three persistent publication formats:

| Value | Stored state | Intended use |
|---|---|---|
| `snapshot-v1` | A complete immutable context for every publication | DCP1/DCP2 persistent-cache profile and simple storage inspection |
| `tail-cow-v1` | An immutable base with changed page objects | Compatibility testing for the first page-tail format |
| `tail-cow-v2` | An authenticated base with a flat chain of changed-page descriptors | Recommended DCP4 profile for growing conversations |

The publication format is part of cache identity. An incompatible entry is a
miss, and vLLM computes the prompt normally. Keep each format in a separately
named directory so storage inspection and rollback remain obvious.

The environment template already selects `tail-cow-v2` and its separate DCP4
storage directory. To use a locally rebuilt image, override only `IMAGE_REF`
and `IMAGE_ID`; keep the page-tail settings unchanged.

The recommended DCP4 profile enables bounded asynchronous page capture:

```bash
SPARKCACHE_ASYNC_PAGE_CAPTURE=1
SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES=auto
SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT=2
```

The `auto` slot policy selects 8 GiB for DCP1, 5 GiB for DCP2, or 3 GiB for
DCP4. Two capture slots let the background publisher consume one completed
capture while a later capture uses the other. Restore separately overlaps
bounded NVMe reads and CUDA placement through two 256 MiB mapped arenas. More
restore arenas are not part of this profile because measured arena waits did
not justify the additional unified-memory pressure. DCP1 and DCP2 page-tail
capture have no matching live record; use complete snapshots or test those
layouts separately.

The environment template enables `DFLASH_WARMUP=1`. Rank 0 waits for the API,
then exercises C1/C2/C4/C8/C16 and scheduled prompt spans covering DFlash's
Triton block-size specializations. Treat completion of the rank-0 launch
command—not an early `/health` response—as service readiness.
The engine-level failure and readiness replay are recorded in the
[`DFlash readiness validation`](../runtime/glm53-flash-jj-r8-gb10/dflash-jit-readiness-validation.json).

Disabling SparkCache omits the external KV connector and all persistent
publication and restore work. `--enable-prefix-caching` remains enabled. The
published image passed a live semantic request in this mode.

`MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`,
`KV_CACHE_MEMORY_BYTES`, speculation depth, ports, SparkCache capacity, and
network settings are ordinary environment values. Changing them does not
require an image rebuild.

To avoid loading the vision tower for a text-only deployment, set:

```bash
MULTIMODAL_INPUTS=0
```

Text-only mode passes `--language-model-only` and rejects media content before
inference. It does not change SparkCache identity or stored entries.

## Start TP4

Start all four ranks within the rendezvous timeout. Rank 0 uses argument `0`:

```bash
bash runtime/glm53-flash-jj-r8-gb10/launch-rank.sh 0 "$HOME/glm53-flash.env"
```

Use arguments `1`, `2`, and `3` on the corresponding follower systems:

```bash
bash runtime/glm53-flash-jj-r8-gb10/launch-rank.sh 1 "$HOME/glm53-flash.env"
bash runtime/glm53-flash-jj-r8-gb10/launch-rank.sh 2 "$HOME/glm53-flash.env"
bash runtime/glm53-flash-jj-r8-gb10/launch-rank.sh 3 "$HOME/glm53-flash.env"
```

The launcher expands to the complete `docker run` invocation and verifies the
configured local image ID before it creates a container. Tail rank 0 with:

```bash
docker logs --follow --tail 120 glm53-flash-gb10-r0
```

Check the OpenAI-compatible API after rank 0 reports readiness:

```bash
curl --fail http://rank0.example.net:8015/v1/models
```

API `/health` is a readiness check. It can remain healthy when the scheduler
cannot admit waiting requests. Use the separate rank-zero liveness endpoint
for routing and operator alerts:

```bash
curl --fail http://rank0.example.net:8016/liveness
curl --fail http://rank0.example.net:8016/metrics
```

The liveness endpoint returns HTTP 503 when zero running requests and one or
more waiting requests persist for 60 seconds, when its vLLM metrics sample is
stale, or when SparkCache cannot prove capture-page ownership. Nonzero idle KV
is warning-only until it remains unchanged beyond the configured 330-second
interval.

Before directing normal traffic to the service, run a concurrent prompt gate.
The requests disable model thinking, put a unique nonce at the front of every
prompt, and require request, capture, and KV usage to return to their measured
idle baseline:

```bash
python scripts/glm53_liveness_gate.py \
  --endpoint http://rank0.example.net:8015 \
  --model glm-5.3-flash \
  --concurrency 4 \
  --prompt-words 100000 \
  --cycles 3 \
  --output ./glm53-liveness-gate.json
```

Add `--api-key-file /secure/api-keys` when the API requires authentication.
Use `--duration-seconds 900` for a 15-minute soak.

When SparkCache is enabled, INFO logs summarize the four ranks in three short
lines:

```text
sparkcache: capacity ranks=4 entries=12 used=1.2/160.0GiB healthy=yes
sparkcache: publications count=12 payload=1.2GiB unique=1.2GiB
sparkcache: writes staged=1.2GiB dedup=0B aborted=0B failed=0B
```

The `/metrics` endpoint retains the individual counters. `payload` is the
logical state represented by committed entries; `unique` and `staged` make
the physical storage cost visible.

## Recover with the complete-snapshot image

The complete-snapshot artifact listed above requires SparkRing source revision
`a150c98ccfdc4b655679860121f24712490dd9ee` and the launch values in its
receipt. Do not combine it with this checkout's launcher: the page-tail image
adds a readiness entrypoint that the recovery artifact does not contain.
Its `glm53-flash-dcp4-snapshot-v1` cache directory remains separate from the
page-tail directory, so recovery does not modify page-tail entries.

## Evidence and limits

The pullable rollback artifact passed four-rank TP4/DCP4 startup and API checks
with 24 GiB of FP8 KV per rank, 4,321,618 logical KV tokens, a 300-second
shared-prefix lease, and no SparkCache source bind mounts. A 448×448 solid-red
PNG used 256 multimodal tokens and was identified as red. All ranks loaded and
ran image ID `sha256:d1a07147…`. See the
[`artifact receipt`](../runtime/glm53-flash-jj-r8-gb10/multimodal-lease300-image-receipt.json)
for complete identities and limitations.

SparkCache pull request 52 separately tested the exact embedded SparkCache
source with different image and video contents, persistent publication, and
restart restore. The built-image smoke did not repeat video input or
persistent multimodal restoration after another process restart.

The operator image embeds SparkCache merge commit `9c6218c`. Asynchronous
manager-page publication reports terminal completion when a worker skips a
store before CUDA submission, allowing vLLM to reclaim the finished request's
KV blocks. A TP4/DCP4 test forced 12 busy-saver skips per rank and one
already-present skip per rank. All requests completed, idle KV usage returned
to 0.0%, three explicit cache resets succeeded, and no preemption, CUDA error,
or NCCL error occurred. See the
[`operator image receipt`](../runtime/glm53-flash-jj-r8-gb10/async-store-completion-public-image-receipt.json)
and the
[`SparkCache validation record`](https://github.com/FujitsuPolycom/sparkcache/blob/9c6218c96f1db233c0d17691dbc32a7d9fb2c0e4/evidence/glm53-flash-dcp4-page-tail-v2/async-store-completion.json).

The unchanged page-tail storage schema completed an exact
131,072 → 262,144 → 524,288 → 921,600-token DCP4 growth sequence. Every
extension remained a page delta, and the final root used a 7,459-byte flat
manifest with three stages. After `docker restart`, the runtime withheld
readiness until DFlash warmup completed, then served two concurrent requests
over the 921,600-token stored prefix with exact responses and no
post-readiness JIT or CUDA error. The same replay passed during image-transfer
pressure. See the
[`page-tail behavior record`](../runtime/glm53-flash-jj-r8-gb10/page-tail-v2-public-image-receipt.json)
and
[`DFlash readiness validation`](../runtime/glm53-flash-jj-r8-gb10/dflash-jit-readiness-validation.json).

The retained vLLM, B12X, NCCL, and CUDA components also have DCP4 evidence
from an earlier SparkCache source composition. That deployment captured a
124,928-token boundary and restored 899,072-token and 999,424-token entries.
Those measurements support the unchanged runtime components; they are not
performance qualification of the registry artifact above. See the
[`scheduler-cadence record`](../performance/records/glm53-flash/scheduler-cadence-20260902.md),
[`asynchronous capture validation`](../runtime/glm53-flash-jj-r8-gb10/ASYNC_CAPTURE_IMAGE_VALIDATION.md)
and the
[`DCP1 deep-context record`](../performance/records/glm53-flash/dcp1-deep-context-boundary-20260831.md)
for exact conditions and limitations.
