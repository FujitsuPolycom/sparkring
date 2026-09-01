# Run GLM-5.3 Flash on four GB10 systems

This guide starts one TP4 service across four NVIDIA GB10 systems. The same
Linux/ARM64 image supports DCP1, DCP2, and DCP4. The default request limit is
1,048,576 tokens and the default prefill scheduler budget is 8,192 tokens.
The operator can enable persistent SparkCache or use vLLM's GPU prefix cache
alone without changing the image.

The preferred launch is TP4/DCP4 with 24 GiB of FP8 KV per rank, scheduler
interval eight, BF16 DFlash2 at depth seven, and SparkCache enabled. The
environment template selects these values and enables bounded asynchronous
SparkCache capture without additional overrides.

The image does not contain model checkpoints. It mounts the exact
[`local-inference-lab/GLM-5.3-Flash-NVFP4`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4)
target and the
[`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
BF16 draft. Local Inference Lab's
[`Jovian Judgement`](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement)
vLLM work and [`B12X`](https://github.com/local-inference-lab/b12x) GB10
kernels provide the model-specific runtime and performance foundation.

## Image identity

```text
registry: ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:bc7d079f16ff4a418669c58c5250f2da52e989a0c5805569ba9429d41b765f65
local image ID: sha256:35f397668c01075d0bdd28bbdb3398afd3744df6086646c6f68bcf7ebe7f918f
platform: linux/arm64
```

Use the `sparkring-glm53-sparkcache` package for this procedure. The
`sparkring-glm53-runtime` and `gb10-vllm-serving` packages are build or
profile inputs and are not substitutes for the operator image below.

Pull and verify the immutable image on rank 0:

```bash
image='ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:bc7d079f16ff4a418669c58c5250f2da52e989a0c5805569ba9429d41b765f65'
expected_image_id='sha256:35f397668c01075d0bdd28bbdb3398afd3744df6086646c6f68bcf7ebe7f918f'
docker pull "${image}"
test "$(docker image inspect "${image}" --format '{{.Id}}')" = "${expected_image_id}"
```

The [`runtime README`](../runtime/glm53-flash-jj-r8-gb10/README.md) also
documents a source build from the pinned public commits.

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

## Distribute one image download through the direct fabric

Create one compressed archive from the verified pull on rank 0:

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

Copy the environment template on every rank:

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

The default OpenAI-compatible model name is `glm-5.3-flash`. Override
`SERVED_MODEL_NAME` only when the site needs a distinct routing name.

Choose the DCP degree with one line:

```bash
DECODE_CONTEXT_PARALLEL_SIZE=4  # change to 1 or 2
```

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

Restore-only mode is useful for reuse-heavy serving or performance tests where
one-off prompt publication would add GPU-to-host capture work. A restore miss
is computed by vLLM normally.

The template's `SPARKCACHE_CACHE_NAMESPACE` value selects rank-local storage
and JIT directories. It is not part of SparkCache's content identity or stored
format. Changing it selects a different root and therefore a different set of
discoverable entries.

The preferred DCP4 profile enables complete-snapshot CUDA capture:

```bash
SPARKCACHE_ASYNC_PAGE_CAPTURE=1
SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES=auto
SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT=2
```

The `auto` slot policy selects 8 GiB for DCP1, 5 GiB for DCP2, or 3 GiB for
DCP4. Two 3 GiB slots are **qualified** for asynchronous DCP4 publication of
the recorded 124,928-token, 231.8 MiB-per-rank snapshot. The same image also
restored retained 900K and 1M entries. Larger asynchronous publication and
DCP1/DCP2 asynchronous capture are not live-qualified; set
`SPARKCACHE_ASYNC_PAGE_CAPTURE=0` outside the recorded publication scope.
Asynchronous capture supports complete `snapshot-v1` publication only.

Disabling SparkCache omits the external KV connector and all persistent
publication and restore work. `--enable-prefix-caching` remains enabled. The
published image passed a live semantic request in this mode.

`MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`,
`KV_CACHE_MEMORY_BYTES`, speculation depth, ports, SparkCache capacity, and
network settings are ordinary environment values. Changing them does not
require an image rebuild.

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

## Evidence and limits

The immutable image above completed a fresh 125,999-token publication with
bounded asynchronous CUDA capture. Every rank stored the 124,928-token,
231.8 MiB boundary. Capture completion was observed 403.7–408.5 ms after
submission per rank, and durable commit took 520.6–567.6 ms per rank.

The same DCP4 service restored 899,072 tokens from an exact 899,998-token
prompt in 2.123–2.180 seconds per rank, then restored 999,424 tokens from an
exact 1,000,000-token prompt in 2.276–2.395 seconds per rank. Both requests
returned their exact needles without an identical prompt being sent after
startup to prime either entry. Startup inventory checked 29 manifests and
rejected none.

The live checks overrode the template's semantic storage-root name with the
durable evidence identifier `jj-r10-async-ab-v1`. The operator default
`glm53-flash-dcp4-snapshot-v1` was not measured. The name difference changes
which directory is searched; it does not change `CacheIdentity` or the stored
snapshot format.

This evidence covers asynchronous complete-snapshot publication at the
124,928-token boundary and persistent restore for the recorded DCP4 profile.
It does not establish larger asynchronous publication, long-duration serving,
concurrent deep-context publication, or asynchronous capture at DCP1 or DCP2.
See the
[`asynchronous capture validation`](../runtime/glm53-flash-jj-r8-gb10/ASYNC_CAPTURE_IMAGE_VALIDATION.md)
and the
[`DCP1 deep-context record`](../performance/records/glm53-flash/dcp1-deep-context-boundary-20260831.md)
for exact conditions and limitations.
