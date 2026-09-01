# Run GLM-5.3 Flash R8 on four GB10 systems

This guide starts one TP4 service across four NVIDIA GB10 systems. The same
Linux/ARM64 image supports DCP1, DCP2, and DCP4. The default request limit is
1,048,576 tokens and the default prefill scheduler budget is 8,192 tokens.
The operator can enable persistent SparkCache or use vLLM's GPU prefix cache
alone without changing the image.

The preferred launch is TP4/DCP4 with 24 GiB of FP8 KV per rank, scheduler
interval eight, BF16 DFlash2 at depth seven, and SparkCache enabled. The
environment template selects these values without additional overrides.

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
registry: ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:380283a506aeb8f9d486a3c64cd738e44268c3cc21590913ea9e4685869f256a
local image ID: sha256:b3a13d8003e7de30d7737fd33c8307404e506ba570240819ec7eb4f5c611400f
platform: linux/arm64
```

Use the `sparkring-glm53-sparkcache` package for this procedure. The
`sparkring-glm53-runtime` and `gb10-vllm-serving` packages are build or
profile inputs and are not substitutes for the operator image below.

Pull and verify the immutable image on rank 0:

```bash
image='ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:380283a506aeb8f9d486a3c64cd738e44268c3cc21590913ea9e4685869f256a'
expected_image_id='sha256:b3a13d8003e7de30d7737fd33c8307404e506ba570240819ec7eb4f5c611400f'
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
archive_dir=/var/tmp/sparkring-images/glm53-r8
archive_name=sparkring-glm53-r8-arm64.tar.zst
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
  --target-directory /var/tmp/sparkring-images/glm53-r8 \
  --image "${image}" \
  --expected-image-id "${expected_image_id}" \
  --execute --confirmation FANOUT_IMAGE_ARCHIVE \
  --output ./glm53-r8-image-fanout.json
```

See the [fan-out reference](DIRECT_FABRIC_IMAGE_ARCHIVE_FANOUT.md) for site
format, planning, resumption, verification, and interruption behavior.

## Configure each rank

Copy the environment template on every rank:

```bash
cp runtime/glm53-flash-jj-r8-gb10/runtime.env.example "$HOME/glm53-r8.env"
${EDITOR:-vi} "$HOME/glm53-r8.env"
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

The published image's DCP1 profile completed a 942,898-token needle retrieval
under the 1M request limit. The DCP2 and DCP4 validation records used 30 GiB
per rank. The DCP4 operator default uses 24 GiB per rank to retain more
host-memory headroom under concurrent serving. Set `KV_CACHE_MEMORY_BYTES` to
a positive byte count to override the topology-aware `auto` policy.

Choose persistent SparkCache or vLLM's GPU prefix cache alone without changing
the image:

```bash
SPARKCACHE_ENABLED=1  # persistent SparkCache plus vLLM prefix caching
SPARKCACHE_ENABLED=0  # vLLM prefix caching only
```

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
bash runtime/glm53-flash-jj-r8-gb10/launch-rank.sh 0 "$HOME/glm53-r8.env"
```

Use arguments `1`, `2`, and `3` on the corresponding follower systems:

```bash
bash runtime/glm53-flash-jj-r8-gb10/launch-rank.sh 1 "$HOME/glm53-r8.env"
bash runtime/glm53-flash-jj-r8-gb10/launch-rank.sh 2 "$HOME/glm53-r8.env"
bash runtime/glm53-flash-jj-r8-gb10/launch-rank.sh 3 "$HOME/glm53-r8.env"
```

The launcher expands to the complete `docker run` invocation and verifies the
configured local image ID before it creates a container. Tail rank 0 with:

```bash
docker logs --follow --tail 120 glm53-jj-r8-gb10-r0
```

Check the OpenAI-compatible API after rank 0 reports readiness:

```bash
curl --fail http://rank0.example.net:8015/v1/models
```

## Evidence and limits

The published image completed SparkCache publication and process-replacement
restore checks at DCP1, DCP2, and DCP4. DCP2 and DCP4 used 30 GiB of FP8 KV
per rank. The DCP1 deep-context profile uses 26 GiB per rank, exposes
1,303,701 KV tokens, and completed a 942,898-token needle request in 473.4
seconds.

The retained restart-restore checks stored spans from 8,192 through 14,336
tokens. The deep-context run published 942,592 tokens but did not replay that
snapshot after process replacement. The evidence does not establish
concurrent large-context restore, long-duration behavior, or general
throughput. See
[`PUBLIC_IMAGE_VALIDATION.md`](../runtime/glm53-flash-jj-r8-gb10/PUBLIC_IMAGE_VALIDATION.md)
and the
[`deep-context record`](../performance/records/glm53-flash/dcp1-deep-context-boundary-20260831.md)
for the exact conditions and results.
