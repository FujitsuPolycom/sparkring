# Qwen3.8-27B EXL3 K5/K6 two-Spark quickstart

This setup was tested on two directly cabled DGX Sparks at TP2/DCP1. The
temperature-one results are included below.

This profile serves
`malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated@ab3a91a13813df8096cb4c1d560ed3669035d0cf`
with Qwen's official static-YaRN 1M configuration, probabilistic MTP depth 3,
and the scheduler envelope chosen for the two-Spark comparison. The
machine-readable contract is
[`recipes/qwen38-27b-exl3-k5k6-pair.json`](../recipes/qwen38-27b-exl3-k5k6-pair.json).

## Serving contract

| Setting | Value |
|---|---|
| Hardware | two directly cabled NVIDIA DGX Sparks |
| Parallelism | TP2/DCP1, one rank per Spark, multi-node `mp` executor |
| Advertised request limit | 1,048,576 tokens via static YaRN factor 4 over native 262,144 |
| Maximum sequences | 32 |
| Scheduler budget | 8,192 tokens |
| Key-value cache | explicit generic FP8, 0.70 unified-memory utilization |
| Prefix reuse | native vLLM prefix cache with mamba alignment |
| External cache | disabled; no LMCache or SparkCache |
| Speculation | Qwen MTP depth 3, probabilistic draft sampling, standard rejection sampling |
| Model-card thinking guidance | temperature 1.0, top-p 0.95, top-k 20; benchmark requests are described separately |
| Decode | full-decode CUDA graphs |
| Collective transport | patched NCCL over one direct RoCEv2 link |
| SIRCL | unsupported for model width 5,120 |

The 8,192-token scheduler budget is intentionally cache-free. The Qwen
LMCache connector's recurrent-state guard requires a budget below 3,200; do
not attach LMCache by deleting or weakening that guard.

Static YaRN can shift short-context output distributions relative to the
checkpoint's native 262,144-token range. Both normalized SparkRing Qwen
profiles use the same 1,048,576-token static-YaRN contract so pair and cycle
measurements use the same model-length policy.

## 1. Prepare the pair

Complete [`docs/PREREQUISITES.md`](PREREQUISITES.md) for a directly cabled
pair. Both ranks need one active 200 Gb/s ConnectX-7 link, a point-to-point
IPv4 subnet, RoCEv2, Docker with the NVIDIA runtime, passwordless SSH for
operator coordination, and enough disk for the runtime, model and JIT cache.

Record these site values:

- rank-0 and rank-1 SSH targets;
- the direct-link interface and IPv4 address on each rank;
- the RoCE device attached to that link; and
- the RoCEv2 GID index encoding the interface's IPv4 address.

`show_gids` or the files under
`/sys/class/infiniband/<device>/ports/1/gid_attrs/` identify the GID. An empty
GID entry is an artifact-lifecycle failure; do not retry the launch against a
guessed index.

## 2. Build and distribute the runtime

Follow the build and receipt procedure in
[`runtime/qwen38/README.md`](../runtime/qwen38/README.md). Build once on one
ARM64 Spark from a clean checkout:

```bash
BASE_IMAGE='nvcr.io/nvidia/cuda@sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d'
docker pull "$BASE_IMAGE"
BASE_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")
BASE_IMAGE_ID="$BASE_IMAGE_ID" \
IMAGE='sparkring-qwen38:arm64-sm121' \
  bash ./runtime/qwen38/build-image.sh
```

Retain `/ws/runtime/source-receipt.json`, its SHA-256, the image ID and a
`docker image inspect` capture. Distribute the identical OCI image to the
other Spark with `docker save`, `sha256sum`, `scp`, and `docker load`; verify
both ranks report the same image ID.

## 3. Download and verify the checkpoint

Download the pinned revision once, then replicate the exact directory to the
other rank:

```bash
IMAGE=sparkring-qwen38:arm64-sm121
MODEL_PARENT="$HOME/qwen38/model"
MODEL_DIR="$MODEL_PARENT/Qwen3.8-27B-EXL3-K5K6-hydrated"
mkdir -p "$MODEL_DIR"

docker run --rm --network host \
  -v "$MODEL_PARENT:/models" \
  --entrypoint /ws/venv/bin/hf \
  "$IMAGE" download malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated \
  --revision ab3a91a13813df8096cb4c1d560ed3669035d0cf \
  --local-dir /models/Qwen3.8-27B-EXL3-K5K6-hydrated

test "$(sha256sum "$MODEL_DIR/SHA256SUMS" | cut -d' ' -f1)" = \
  7626d18481e7f995fd1d9ff211083b7fd57f044daba39e107fb29a48207f24c4
(cd "$MODEL_DIR" && sha256sum --check --strict SHA256SUMS)
```

Run the same 16-entry `SHA256SUMS` check on rank 1 after replication.

## 4. Resolve one environment per rank

On each Spark:

```bash
mkdir -p "$HOME/qwen38/config" "$HOME/qwen38/cache" "$HOME/qwen38/logs"
cp scripts/config/qwen38-27b-exl3-k5k6-pair.env.example \
  "$HOME/qwen38/config/rank.env"
```

Replace all four placeholders in each rank's file. The pair template names
one HCA, disables subnet-aware routing, and leaves the NCCL channel count and
algorithm at library defaults. Do not use the four-Spark cycle template: its
two-HCA routing, forced Ring algorithm, four-channel pin and tree skip describe
a different topology.

Review the resolved file and confirm it contains no `REPLACE_WITH_` or angle
bracket placeholders.

## 5. Run the no-start preflight

Set the site values separately on each rank:

```bash
IMAGE=sparkring-qwen38:arm64-sm121
MODEL_DIR="$HOME/qwen38/model/Qwen3.8-27B-EXL3-K5K6-hydrated"
ENV_FILE="$HOME/qwen38/config/rank.env"
CACHE_DIR="$HOME/qwen38/cache"
LOG_DIR="$HOME/qwen38/logs"
RANK=0                         # 1 on the follower
RANK0_RENDEZVOUS_ADDR=<rank-0-direct-link-ip>

docker run --rm --network host --ipc host --shm-size 16g --gpus all \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband \
  -v "$MODEL_DIR:/ws/model/Qwen3.8-27B-EXL3-K5K6-hydrated:ro" \
  -v "$CACHE_DIR:/ws/cache" -v "$LOG_DIR:/ws/logs" \
  -v "$ENV_FILE:/ws/rank.env:ro" \
  -e RANK="$RANK" -e RANK0_RENDEZVOUS_ADDR="$RANK0_RENDEZVOUS_ADDR" \
  --entrypoint /ws/qwen38_dgx2_serve.sh \
  "$IMAGE" --check
```

The gate verifies the runtime receipt, all checkpoint files, patched NCCL,
the single-HCA RoCEv2 mapping, the long-context override, GPU visibility, and
rank-0 ports. Inspect the resolved command. Both ranks must pass before either
starts.

## 6. Launch rank 1, then rank 0

Use a fresh attempt identifier. Start the follower first and rank 0 within the
rendezvous window:

```bash
ATTEMPT_ID=qwen38-tp2-1m-probmtp-001
CONTAINER="qwen38-dgx2-${ATTEMPT_ID}-r${RANK}"

docker run -d --name "$CONTAINER" \
  --network host --ipc host --shm-size 16g --gpus all \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband \
  -v "$MODEL_DIR:/ws/model/Qwen3.8-27B-EXL3-K5K6-hydrated:ro" \
  -v "$CACHE_DIR:/ws/cache" -v "$LOG_DIR:/ws/logs" \
  -v "$ENV_FILE:/ws/rank.env:ro" \
  -e RANK="$RANK" -e RANK0_RENDEZVOUS_ADDR="$RANK0_RENDEZVOUS_ADDR" \
  --label org.sparkring.profile=qwen38-27b-exl3-k5k6-pair \
  --label org.sparkring.attempt="$ATTEMPT_ID" \
  --entrypoint /ws/qwen38_dgx2_serve.sh \
  "$IMAGE" --run
```

Tail rank 0:

```bash
docker logs --follow "qwen38-dgx2-${ATTEMPT_ID}-r0"
```

The ready service must report all of these:

- TP world size 2 and one worker per rank;
- `max_seq_len=1048576` and `kv_cache_dtype=fp8`;
- `draft_sample_method=probabilistic` in the resolved speculative config;
- NCCL connected over the selected RoCEv2 GID;
- `Application startup complete` on rank 0; and
- `/v1/models` advertising `qwen38` with `max_model_len` 1,048,576.

## 7. Run bounded functional checks

From the checkout or another machine that can reach rank 0:

```bash
python scripts/qwen38_smoke.py \
  --endpoint http://<rank-0-management-address>:8000 \
  --model qwen38 \
  --expected-max-model-len 1048576 \
  --timeout 180 \
  --output "$HOME/qwen38/logs/qwen38-smoke-${ATTEMPT_ID}.json"
```

The smoke harness disables thinking for its exact-marker requests. It checks
API health, the advertised limit, repeated arithmetic, tool parsing, data-URL
vision, repeated-prefix equality, and distinct shared-prefix suffixes. It does
not prove 1M-input correctness or a native-prefix cache hit.

For an interactive request that follows the model-card guidance, use
temperature 1.0, top-p 0.95, and top-k 20, then verify that the server's
speculative counters increase. These are application sampling choices, not
the benchmark request policy below.

## 8. Benchmark the normalized unique-context lane

The normalized campaign changes temperature only: it sends temperature 1.0
and leaves top-p and top-k unset. vLLM then applies the pinned checkpoint's
`generation_config.json`: effective top-p 0.95 and top-k 20. Its SHA-256 is
`e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`.
Each machine-readable receipt must bind the harness revision, request and
effective sampling policy, 100% unique per-stream context, request shapes,
measurement clock, and server-accounting authority. The hybrid-KV estimator
must defer admission to the server while retaining queue, underfill, and
request-error gates.

![Two-Spark Qwen benchmark](../performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.png)

Prefill measured 1,274–1,401 tok/s through 32K, 1,050 at 64K, and 785 at
128K. Sustained decode measured 25–30 tok/s at C1, 41–54 at C2, 72–100 at C4,
and 90–154 aggregate tok/s at C8. Coding Peak completed 15/15 requests with a
39.95 tok/s mean. The table and N counts are in the
[full result](../performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.md).

Do not publish a live display value. Retain the final JSON and require zero
request errors, no underfill, no capacity-limit flag, the requested running
count, and client/server token-accounting agreement. A fully shared-prefix
lane is a separate cache-benefit diagnostic and must not replace the
normalized unique-context result.

## 9. Stop or restore another stack

Stopping this profile interrupts serving. Preserve the exact logs and launch
inputs before stopping the two named containers. Do not remove model, image,
JIT cache, or receipt trees to make a later launch pass. If this profile
replaced another stack, restore that stack from its retained container or
recorded launch specification rather than reconstructing it from memory.

## Results and receipts

See the [full benchmark result](../performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.md) and
[sanitized replayable receipts](../performance/receipts/qwen38-27b/temp1/20260823-tp2/).
