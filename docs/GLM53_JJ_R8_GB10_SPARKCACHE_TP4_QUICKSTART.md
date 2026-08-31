# Run GLM-5.3 Flash R8 with SparkCache on four GB10 systems

This guide starts one TP4 service across four NVIDIA GB10 systems. The same
Linux/ARM64 image supports DCP1, DCP2, and DCP4. The default request limit is
1,048,576 tokens and the default prefill scheduler budget is 8,192 tokens.

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
local tag: sparkring-glm53-jj-r8-sparkcache:d803c6b-55969c-65895c8-arm64
local image ID: sha256:77da063d1d51fa181eb39e519dda7c5ae4eb59a47e169cb4c33bd2cd42120225
registry digest: UNAVAILABLE_UNTIL_PUBLICATION
archive SHA-256: 51b1aece26dad833ac2b2727a88429642d38b8c1b48b00f6d4b28214f7d840fc
archive bytes: 8467812978
```

Until a registry digest or archive URL is published, either import a copy of
the exact archive or build the image from source as described in the
[`runtime README`](../runtime/glm53-flash-jj-r8-gb10/README.md). Do not replace
the image ID with the ID of a merely similar image.

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

When the archive has a public URL, replace the single placeholder below. The
command downloads on rank 0, verifies the archive, forwards it across three
direct links, imports it on every rank, and verifies the local image ID.

```bash
python scripts/fanout_image_archive.py \
  --site /secure/site.yaml \
  --source-url '<ARCHIVE_URL_AFTER_PUBLICATION>' \
  --archive-name glm53-jj-r8-sparkcache-arm64.tar.zst \
  --expected-sha256 51b1aece26dad833ac2b2727a88429642d38b8c1b48b00f6d4b28214f7d840fc \
  --target-directory /var/lib/sparkring/images/glm53-r8 \
  --image sparkring-glm53-jj-r8-sparkcache:d803c6b-55969c-65895c8-arm64 \
  --expected-image-id sha256:77da063d1d51fa181eb39e519dda7c5ae4eb59a47e169cb4c33bd2cd42120225 \
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

Choose the DCP degree with one line:

```bash
DECODE_CONTEXT_PARALLEL_SIZE=1  # change to 2 or 4
```

The launcher selects the matching GLM KV geometry automatically:

| DCP | KV interleave | Full-CKV prefill gather |
|---:|---:|---:|
| 1 | 1 token | disabled |
| 2 | 4 tokens | enabled |
| 4 | 4 tokens | enabled |

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

The exact image completed SparkCache publication and process-replacement
restore checks at DCP1, DCP2, and DCP4. DCP2 and DCP4 used 30 GiB of FP8 KV
per rank. A DCP1 capacity sweep selected 41 GiB per rank and reported
2,056,272 KV tokens while retaining 11–13 GiB of available host memory.

The retained checks stored spans from 8,192 through 14,336 tokens. They do not
establish one-million-token request completion, concurrent large-context
restore, long-duration behavior, or general throughput. See
[`LIVE_VALIDATION.md`](../runtime/glm53-flash-jj-r8-gb10/LIVE_VALIDATION.md)
for the exact conditions and results.
