# Run the qualified GLM-5.3 split-page SparkCache image on four DGX Sparks

Status: **qualified** for exact output, eight external restores, and
authenticated shared-base reads in one C8 × 16K cohort. The local image has no
published OCI digest. These
commands are immediately usable only on ranks that already retain image ID
`sha256:becf556650dff79a9959aef371ea861187db248bd0f46c3ebfbd26759e458818`.

The source repositories and model checkpoints are public. The SparkCache and
vLLM composition commits remain review artifacts until their draft branches
are published. Outside operators therefore cannot reconstruct the exact
qualified image from the repositories yet, and a later rebuild will require
its own qualification.

The GLM runtime derives primarily from Local Inference Lab's Jovian Judgement
[`vLLM` PR 535 source at `ead9d8a4`](https://github.com/local-inference-lab/vllm/commit/ead9d8a4e21b3818b21ec6f4d4d94564dd60c3f8).
[B12X at `b1d541f9`](https://github.com/local-inference-lab/b12x/commit/b1d541f9e71a35f030d45fae437630fff7507c2a)
supplies the Blackwell backend. This exact recipe uses
[`GLM-5.3-Flash-NVFP4@520de24e`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/520de24eabf507659eaef7c70f14fd584527facc)
with the BF16
[`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410).
The external draft is not Local Inference Lab's separate
[MXFP8 DFlash2 checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8).

## Configuration

| Setting | Qualified value |
|---|---|
| Hardware and topology | four NVIDIA DGX Spark systems; TP4/DCP1/PP1 |
| Target | `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc` |
| Draft | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, BF16, seven speculative tokens |
| KV cache | FP8; 20 GiB per rank |
| Limits | 262,144-token context; 16 sequences; 4,096 batched tokens |
| Page geometry | 256-token logical blocks; 512-token target and recurrent pages |
| SparkCache restore | eight load lanes; eight pending restores; 256 MiB CUDA arena per lane |
| Qualified containers | `glm53-pr535-sc59ac-c8-01-r{0..3}` |
| Retained rollback containers | `glm53-pr535-sc78-hotpatch-c8-qualified-r{0..3}` |

The draft model is licensed CC BY-NC-ND 4.0. Review its model card before
downloading or serving it.

## Obtain the public model inputs

Download each immutable revision to the same path on every rank. The image
contains runtime code, not model weights.

```bash
hf download local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --revision 520de24eabf507659eaef7c70f14fd584527facc \
  --local-dir /REPLACE/TARGET_MODEL_HOST_PATH

hf download incoai/GLM-5.3-Flash-DFlash2 \
  --revision dc77ff1c99eeb2df044ee3d4f0094eb033fee410 \
  --local-dir /REPLACE/DFLASH_MODEL_HOST_PATH
```

The launch script binds the target cache identity
`a35e6bf2875c1875609b8deaec404c07c6cc80259e4222fc0b51e649498bd6b9`
and draft weights identity
`b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`.

## Verify the retained image

Set the four SSH targets in rank order. This read-only check must print the
same expected image ID four times.

```bash
ranks=(
  operator@rank0.example.net
  operator@rank1.example.net
  operator@rank2.example.net
  operator@rank3.example.net
)
image='sparkring-glm53-sparkcache:pr535-6da4865-sc59ac0b0-c8-exact-arm64'
expected='sha256:becf556650dff79a9959aef371ea861187db248bd0f46c3ebfbd26759e458818'

for host in "${ranks[@]}"; do
  actual="$(ssh "${host}" docker image inspect --format '{{.Id}}' "${image}")"
  test "${actual}" = "${expected}"
  printf '%s %s\n' "${host}" "${actual}"
done
```

Stop if any rank lacks the tag or reports another ID. Do not replace the image
reference with a mutable registry tag.

## Configure one rank

Clone the reviewed SparkRing revision on every host. Copy the complete
environment template to a rank-local file and edit it:

```bash
cp runtime/glm53-flash-split-page-sparkcache/qualified.env.example \
  "$HOME/glm53-sparkcache-rank.env"
${EDITOR:-vi} "$HOME/glm53-sparkcache-rank.env"
```

Five values describe the site and have no portable default:

- `HOST_IP`: this rank's routable address;
- `MASTER_ADDR`: rank 0's routable address, identical on all ranks;
- `TARGET_MODEL_HOST_PATH`: target checkpoint directory;
- `DFLASH_MODEL_HOST_PATH`: DFlash checkpoint directory;
- `CACHE_HOST_ROOT`: writable, rank-local cache directory.

The template also exposes the operational settings most often changed by an
operator:

| Group | Variables |
|---|---|
| Image and API | `IMAGE_REF`, `CONTAINER_PREFIX`, `SERVED_MODEL_NAME`, `PORT`, `MASTER_PORT`, `SHM_SIZE` |
| Topology | `TENSOR_PARALLEL_SIZE`, `PIPELINE_PARALLEL_SIZE`, `DECODE_CONTEXT_PARALLEL_SIZE`, `NODE_COUNT` |
| Scheduling | `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS` |
| Device memory | `KV_CACHE_MEMORY_BYTES`, `GPU_MEMORY_UTILIZATION`, `KV_CACHE_DTYPE` |
| Speculation | `SPECULATION_METHOD`, `NUM_SPECULATIVE_TOKENS`, `DRAFT_TENSOR_PARALLEL_SIZE`, `DRAFT_KV_CACHE_DTYPE`, sampling methods |
| Kernels | `ATTENTION_BACKEND`, `MOE_BACKEND`, `LINEAR_BACKEND`, `KDA_PREFILL_BACKEND`, CUDA-graph settings |
| SparkCache | `CACHE_NAMESPACE`, capacity and low watermark, TTL, stored-span limits, load threads, pending restores, I/O workers, CUDA arena bytes |
| Network | `SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, NCCL channel bounds |
| CPU and loading | `OMP_NUM_THREADS`, `TORCHINDUCTOR_COMPILE_THREADS`, `FASTSAFETENSORS_QUEUE_SIZE` |

Byte values are per rank. The launcher validates numeric ranges, path syntax,
rank bounds, cache watermarks, and the exact local image ID. It uses Python's
JSON encoder for the speculative-decoding, compilation, and SparkCache
configuration objects; values are not interpolated into handwritten JSON.

The defaults reproduce the recorded configuration. Any runtime-setting change
adds container label
`org.sparkring.qualification-status=user-modified-unqualified` and records the
changed variable names in `org.sparkring.modified-settings`. Site addresses,
bind-mount paths, container names, and an image alias resolving to the same
verified image ID do not change qualification status. A modified configuration
may be useful, but it does not inherit the recorded result.

`SPECULATION_METHOD=dflash` is the only method accepted by this qualified
launcher. Another speculative method needs a launcher whose configuration has
been exercised with the corresponding image.

## Launch one rank

The launcher refuses to overwrite an existing container and verifies the image
ID before starting. Run it once per rank, passing that rank's edited file:

```bash
bash runtime/glm53-flash-split-page-sparkcache/launch-qualified-rank.sh \
  0 "$HOME/glm53-sparkcache-rank.env"
```

The same file can instead be selected through `SPARKRING_CONFIG_FILE`:

```bash
export SPARKRING_CONFIG_FILE="$HOME/glm53-sparkcache-rank.env"
bash runtime/glm53-flash-split-page-sparkcache/launch-qualified-rank.sh 0
```

Use rank arguments `1`, `2`, and `3` on the other hosts. Each host needs its
own `HOST_IP`; all four use rank 0's `MASTER_ADDR`. Start all ranks within the
runtime's distributed startup window.

The cache root is rank-local and writable. The one-shot cache-clear marker in
the profile removes this profile's cache once and then consumes itself after a
successful clear. Repeated service starts do not repeatedly clear the cache.

## Observe readiness

Tail rank 0:

```bash
ssh operator@rank0.example.net \
  'docker logs --follow --tail 120 glm53-pr535-sc59ac-c8-01-r0 2>&1'
```

Health and model identity are authoritative; a particular startup log line is
not:

```bash
api_endpoint='http://rank0.example.net:8015'
until curl --fail --silent "${api_endpoint}/health" >/dev/null; do sleep 5; done
curl --fail --silent "${api_endpoint}/v1/models"
```

HTTP health does not prove that the scheduler has received the persistent
manifest inventory discovered by each worker. Complete one tiny inference
before expecting an external cache hit:

```bash
served_model='glm-5.3-flash-pr535-sc78-tp4'
curl --fail --silent "${api_endpoint}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${served_model}\",\"prompt\":\"Reply OK.\",\"max_tokens\":1,\"temperature\":0}" \
  > scheduler-inventory-warmup.json
```

This request executes a real model step so worker inventory reaches the
scheduler. `/health` alone is insufficient for a persistent-hit assertion.

## Qualified C8 boundary

The recorded cohort sent eight distinct 16,384-token stored roots with one
common stored 8,192-token base. All eight responses were exact and all eight
used external restore. On each rank, the eight requests shared one physical
read of the authenticated 100,868,258-byte base and avoided seven duplicate
reads.

A diagnostic cohort sent immediately after HTTP health, before a model step
populated scheduler inventory, restored seven requests and safely recomputed
one. That diagnostic result establishes the readiness requirement; it is not
the main qualification result.

The exact result and limitations are in the
[machine receipt](../performance/receipts/glm53-flash/split-page-shared-base-c8-20260830/validation.json)
and [evidence record](../performance/records/glm53-flash/split-page-shared-base-c8-20260830.md).
Do not describe the single cohort as a general latency or throughput claim.

## Roll back to the retained containers

Rollback stops serving and must be applied to all four ranks. Review the
commands first. Never run a mixed generation across the TP4 service.

On each rank:

```bash
bash runtime/glm53-flash-split-page-sparkcache/rollback-rank.sh 0
```

Use that rank's argument. After all four commands complete, tail the rollback
rank-zero container and check the API:

```bash
ssh operator@rank0.example.net \
  'docker logs --follow --tail 120 glm53-pr535-sc78-hotpatch-c8-qualified-r0 2>&1'
curl --fail --silent 'http://rank0.example.net:8015/health'
```

The helper preserves both container generations. It does not delete images,
model files, JIT caches, SparkCache objects, or evidence.

## Provenance

[`qualified-artifact.json`](../runtime/glm53-flash-split-page-sparkcache/qualified-artifact.json)
binds the image ID, source digest, source commits, CUDA placement library,
model identities, serving settings, and rollback names. SparkCache issues
belong at <https://github.com/FujitsuPolycom/sparkcache/issues>; SparkRing
runtime and operator issues belong at
<https://github.com/FujitsuPolycom/sparkring/issues>.
