# Qwen3.8-Flash-Next NVFP4 QAD on two Sparks

[Qwen3.8-Flash-Next NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/60215d26cf5e42c2db6128774032d57fc62678da)
(QAD checkpoint step 5500, revision `60215d26cf5e`) on two DGX Sparks, with MTP
speculative decoding and 262K context. Node A serves the API on port 8000 as
`Qwen3.8-Flash-Next-NVFP4-QAD-TP2`, with no API key. Status: Development.

On the Spark connected to your network:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile qwen38-flash-next-tp2
```

[Install SparkRing](../../docs/operations/install.md) covers requirements,
logs and recovery. To run the same containers with Docker Compose, see
[Compose](compose/README.md).

## Settings

| Setting | Installer profile ([config.json](config.json)) | SparkCache profile ([sparkcache.json](sparkcache.json)) |
|---|---|---|
| Image | `dev-20260925-qwendecode-cuda1342-nccl2323-status031` | `shared-2026.09.3` |
| Checkpoint | Step 5500, revision `60215d26cf5e` (branch `qad-step5500-ple1000`) | Step 4000, revision `629bc3218833` (branch `qad-step-4000`) |
| Parallelism | TP2/DCP1; one p0-to-p0 cable, both PCIe functions of p0 | Same |
| Context / sequences / batch | 262144 / 16 / 8192; no YaRN | Same |
| KV | 24 GiB FP8 per rank | 24 GiB FP8 per rank; separate from host cache buffers |
| Loading / speculation | Managed B12X / MTP3; the draft's MXFP8 experts run on the `humming` MoE backend; drafts sampled from the draft distribution (`"draft_sample_method": "probabilistic"`) | Managed B12X / MTP3; the draft's NVFP4 experts run on B12X; greedy drafts |
| Decode weights | BF16 target LM head; MXFP8 hyper-connection down/injection projections for batches of at most 16 rows; fused rotary-embedding op | BF16 LM head and hyper-connection projections |
| Prefill | Hyper-connection token-row ownership (`VLLM_QWEN3_8_HC_PREFILL_MODE=shard`) with the image's `qwen-collectives` and `qwen4-prefill` features | Hyper-connection row sharding off |
| Decode collectives | All-reduces of up to 64 rows on RoCEnante (`QWEN_DISPATCH_AR_BYTES=327680`) | No Qwen dispatch setting |
| Media | Three images, one video; 16 configured video frames | Same |
| Caching | vLLM native prefix cache; SparkCache off | 4 GiB disk per rank, reclaim toward 3 GiB; about 1.25 GiB data buffers per rank |
| API | Port 8000, model `Qwen3.8-Flash-Next-NVFP4-QAD-TP2`, no API key | Same |

## Performance

One pair, step 5500 ([record](../../performance/records/images/dev-20260925-qwendecode-qwen-step5500-20260926.md)). Single stream, 512 tokens at
temperature 0; prefill is one cold prompt:

| Decode prose / code / JSON (tokens/s) | Prefill 16K / 64K (tokens/s) |
|---|---|
| 49.3 / 77.4 / 84.8 | 4,077 / 3,762 |

Throughput matrix ([llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench), temperature 1.0), total
tokens/s across streams:

| Context | Prefill | 1 stream | 8 streams | 16 streams |
|---|---:|---:|---:|---:|
| 0 | — | 56.2 | 196.6 | 283.3 |
| 8K | 3,692 | 44.3 | 163.8 | 232.6 |
| 16K | 3,883 | 47.2 | 172.2 | 238.9 |
| 32K | 3,820 | 49.3 | 160.9 | 237.9 |
| 64K | 3,666 | 43.3 | 162.4 | 233.1 |

- At temperature 1.0, decode rates move by up to 20% between runs with how many
  draft tokens the model accepts; the step rate stays within 3%.
- Step 4000 (revision `629bc3218833`) decodes 14–21% faster on the same
  probe (59.5 / 88.4 / 101.0).
- The first start after installation compiles kernels for about 9.5
  minutes.

Tuning measurements: [installer tuning record](../../performance/records/qwen38-flash-next/installer-tuning-20260925.md).

## SparkCache profile: manual setup

The SparkCache profile, `qwen38-flash-next-tp2-sparkcache`, adds a disk KV
cache and runs on the [shared-2026.09.3](../../runtime/releases/shared-2026.09.3/README.md)
image. `sparkring install` does not deploy it; these commands do, as does
[`sparkring compose`](../../docs/operations/compose.md). Complete the
[host preparation](../../docs/operations/host-preparation.md) and the
[pair network procedure](../../docs/operations/pair-network.md) first.

### Image and checkpoint

On both Sparks, in Bash from the same SparkRing checkout:

```bash
set -euo pipefail
REPO=$PWD
PROFILE_ID=qwen38-flash-next-tp2-sparkcache
mkdir -p .sparkring
python3 scripts/sparkring.py setup show "$PROFILE_ID" --format shell \
  > .sparkring/selection.env
cat .sparkring/selection.env
source .sparkring/selection.env
PROFILE="$PROFILE_CONFIG"
CONTAINER_PREFIX=qwen-flash-next-sparkcache-tp2
MODEL_DIR="/srv/models/${MODEL_REPO##*/}/${MODEL_REV}"
CACHE_DIR="/srv/cache/${PROFILE_ID}/${RELEASE}"

docker pull --platform linux/arm64 "$IMAGE_REF"
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")
test "$IMAGE_ID" = "$EXPECTED_IMAGE_ID"
mkdir -p "$CACHE_DIR"
```

Download the checkpoint (about 106 GB) on each Spark, or copy checked files
over the fabric, then check it:

```bash
# Skip the download when MODEL_DIR already holds the checkpoint.
"$HOME/.venvs/sparkring-download/bin/hf" download "$MODEL_REPO" \
  --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
(cd "$MODEL_DIR" && sha256sum --check "$REPO/profiles/qwen38-flash-next-tp2-sparkcache/SHA256SUMS")
```

`SHA256SUMS` lists the 48 files the revision needs. The model directory is
mounted read-only; `CACHE_DIR` must be outside it.

### Plan and create

Set each Spark's rank, fabric address and interface. Both ranks use rank 0's
address as `MASTER_ADDR`; on rank 1 set `RANK=1` and `HOST_IP=198.18.20.2`.

```bash
RANK=0
MASTER_ADDR=198.18.20.1
HOST_IP=198.18.20.1
INTERFACE=enp1s0f0np0
launch_rank() {
  python3 runtime/common/qwen_flash_next.py "$1" \
    --profile "$PROFILE" --rank "$RANK" \
    --master "$MASTER_ADDR" --host-ip "$HOST_IP" --interface "$INTERFACE" \
    --image "$IMAGE_ID" --model "$MODEL_DIR" --cache "$CACHE_DIR"
}
launch_rank plan
```

`plan` checks the checkpoint metadata and prints the `docker create` command.
Review both plans, then run `launch_rank create` on each rank; it verifies the
image and refuses an existing container name. RDMA uses `rocep1s0f0` and
`roceP2p1s0f0`, the two PCIe functions of port p0.

### Controlled startup

Stop other GPU workloads first. On each rank, save its inputs for later shells:

```bash
for key in RANK MASTER_ADDR HOST_IP INTERFACE PROFILE IMAGE_ID MODEL_DIR CACHE_DIR CONTAINER_PREFIX; do
  printf '%s=%q\n' "$key" "${!key}"
done > .sparkring/qwen-pair-session.env
declare -f launch_rank >> .sparkring/qwen-pair-session.env
```

Start rank 1 first, then rank 0:

```bash
docker start "${CONTAINER_PREFIX}-r${RANK}"
docker logs --follow --tail 100 "${CONTAINER_PREFIX}-r${RANK}"
```

From rank 0, or a client that reaches its address:

```bash
curl --fail "http://${MASTER_ADDR}:8000/health"
curl --fail "http://${MASTER_ADDR}:8000/v1/models"
curl --fail "http://${MASTER_ADDR}:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next-NVFP4-QAD-TP2","messages":[{"role":"user","content":"Reply only READY"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

- The API has no key: restrict it to trusted clients or an authenticated
  gateway.
- The `SPARKRING STARTUP AUDIT` banner lists configuration warnings before the
  API is ready.
- Docker limits each container to 108 GiB of memory (112 GiB with swap), which
  does not cover every GB10 GPU allocation; watch the host's `MemAvailable`.
- Containers do not start at boot.

#### Restart existing containers

Do not rerun `create`. On each rank, in a fresh shell from the same checkout,
inspect and source the saved inputs and stop the container; then start rank 1
and rank 0 as above. The disk cache is kept.

```bash
source .sparkring/qwen-pair-session.env
docker stop --timeout 30 "${CONTAINER_PREFIX}-r${RANK}"
```

#### Cache-disabled alternative

The installer profile `qwen38-flash-next-tp2` serves this checkpoint without
SparkCache; install it with the command at the top of this page. The manual
launcher does not create its containers.

### SparkCache limits

- Disk entries are not isolated by request `cache_salt`; use separate
  deployments and cache directories for tenant isolation.
- A prefix snapshot that does not fit a 512 MiB capture slot is not cached,
  even below the 65,536-token span limit; the request is still served.
- Capacity overrides are not accepted.
