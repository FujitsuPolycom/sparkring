# Qwen3.8-27B EXL3 K5/K6 four-Spark quickstart

This quickstart builds and serves
`malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated` as four tensor-parallel ranks on a
directly cabled DGX Spark cycle. It starts from public, immutable source inputs.
No maintainer-held archive or published Qwen image is required.

This setup was tested on four directly cabled DGX Sparks at TP4/DCP1. Results
are included below.

The results use the runtime identified in the benchmark record. Building from
this quickstart creates a new image with the same serving settings.

The machine-readable settings are in
[`recipes/qwen38-27b-exl3-k5k6.json`](../recipes/qwen38-27b-exl3-k5k6.json).
The image builder contract is in
[`runtime/qwen38/README.md`](../runtime/qwen38/README.md).

| Setting | Four-Spark setup |
|---|---|
| Model | `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated@ab3a91a13813df8096cb4c1d560ed3669035d0cf` |
| Topology | direct cycle `0-1-2-3-0` |
| Parallelism | TP4/DCP1, one process per Spark |
| Executor | multi-node `mp` |
| Request limit | 1,048,576 tokens through static YaRN factor 4 over native 262,144 |
| Maximum sequences | 64 |
| Scheduler budget | 8,192 tokens |
| Scheduling | chunked prefill, asynchronous, full-input-length reservation |
| Hybrid block geometry | request 16 attention tokens; runtime aligns effective attention and mamba blocks to 1,600 tokens |
| Key-value dtype | FP8 |
| EXL3 prefill | FP8, reconstruction tile 256 |
| Speculation | Qwen MTP depth 3, probabilistic draft sampling, standard rejection sampling |
| Prefix reuse | native prefix caching, mamba alignment |
| External cache | disabled |
| Collective transport | patched NCCL; SIRCL disabled |

Pass `--block-size 16`; the pinned runtime performs the 1,600-token hybrid
alignment. Do not pass `--block-size 1600` directly. SparkCache and LMCache are
not part of this deployment.

Static YaRN can shift short-context output distributions relative to the
checkpoint's native range. The two-Spark and four-Spark normalized profiles
use the same 1,048,576-token static-YaRN object so their measurements share one
model-length policy.

The benchmark leaves top-p and top-k unset, so vLLM applies the pinned
checkpoint's `generation_config.json`. The file's SHA-256 is
`e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`.

## 1. Prepare the four hosts

Complete [the four-Spark prerequisites](PREREQUISITES.md). Each host needs
Linux ARM64, Docker with NVIDIA GPU access, both ConnectX-7 ports,
`/dev/infiniband`, and writable storage for the image, approximately 22 GB of
model files, and JIT caches. The build host additionally needs substantial
temporary disk space and internet access to the pinned CUDA image, GitHub,
PyPI, and the PyTorch CUDA 13.2 index.

The literal flow also requires these commands on the control/build host or the
rank where each command is shown:

```bash
for command in docker python3 git ssh scp rsync curl timeout sha256sum awk cat \
  ip ss ibdev2netdev show_gids; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 1; }
done
```

On Debian/Ubuntu, Git, OpenSSH client tools, rsync, curl, coreutils, iproute2,
`rdma-core`, and `ibverbs-utils` provide the non-Docker utilities. Install
Docker and the NVIDIA container runtime using the platform's supported DGX
Spark procedure.

Assign stable ranks 0-3 and cable the four fabric edges as `0-1`, `1-2`,
`2-3`, and `3-0`. From the clean SparkRing checkout on rank 0, run the
read-only topology check:

```bash
python scripts/ring_doctor.py \
  --node <user>@<rank0-host> \
  --node <user>@<rank1-host> \
  --node <user>@<rank2-host> \
  --node <user>@<rank3-host> \
  --verify
```

Require zero `ERROR` findings and a reachability matrix in which every pair
passes. Do not continue from a plan that reports missing routes, disabled IPv4
forwarding, a blocked Docker forwarding chain, an unhealthy cable, or an
unreachable rendezvous address.

On every rank, identify the management interface, the two fabric netdevs, the
corresponding RDMA devices, and the RoCEv2 GID entries:

```bash
ip -br -4 address
ibdev2netdev
show_gids
```

Rerun the read-only check with the actual Qwen rendezvous and management
interfaces:

```bash
python scripts/ring_doctor.py \
  --node <user>@<rank0-host> \
  --node <user>@<rank1-host> \
  --node <user>@<rank2-host> \
  --node <user>@<rank3-host> \
  --rendezvous-address <rank0-management-ip> \
  --socket-interface <user>@<rank0-host>=<rank0-management-interface> \
  --socket-interface <user>@<rank1-host>=<rank1-management-interface> \
  --socket-interface <user>@<rank2-host>=<rank2-management-interface> \
  --socket-interface <user>@<rank3-host>=<rank3-management-interface> \
  --verify
```

Require zero rendezvous and socket-interface findings in addition to the full
fabric reachability matrix. The site firewall must permit unrestricted mutual
TCP among the four management addresses because the `mp` executor allocates
dynamic worker ports.

Each `NCCL_IB_HCA` value must name exactly the two RDMA devices connected to
that rank's cycle neighbours. `NCCL_IB_GID_INDEX` is rank-global, so both
devices on one rank must expose their fabric IPv4 RoCEv2 entries at the chosen
index. Do not copy an index from another site.

## 2. Build one public runtime image on rank 0

Build once on rank 0 from its clean SparkRing checkout. The build compiles
NCCL, ExLlamaV3, and vLLM and can take several hours. `MAX_JOBS` bounds build
parallelism when memory is limited.

Pull the immutable CUDA parent and record its local content-addressed image ID:

```bash
IMAGE=sparkring-qwen38:arm64-sm121
PARENT='nvcr.io/nvidia/cuda@sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d'

docker pull "$PARENT"
PARENT_ID=$(docker image inspect --format '{{.Id}}' "$PARENT")
```

Build from the tracked public inputs:

```bash
BASE_IMAGE="$PARENT" \
BASE_IMAGE_ID="$PARENT_ID" \
BASE_IMAGE_LICENSES='LicenseRef-NVIDIA-Deep-Learning-Container' \
IMAGE="$IMAGE" \
MAX_JOBS=8 \
  bash ./runtime/qwen38/build-image.sh
```

The builder fails on parent-image drift, dirty builder inputs, source commit or
tree drift, patch drift, dependency-input drift, a wrong CUDA architecture, or
a patched NCCL binary that does not have SHA-256
`e69a8c240f45d10166bcd901d99db78bb63147adda66e586d8dd505c6d608b54`.
It builds vLLM for `12.0f` and ExLlamaV3 for `12.1`, then import-checks vLLM
and `exllamav3_ext.exl3_gemm` without loading a model.

Record the resulting immutable image ID:

```bash
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
printf '%s\n' "$IMAGE_ID"

docker run --rm \
  -e LD_PRELOAD= -e VLLM_NCCL_SO_PATH= \
  --entrypoint cat "$IMAGE" \
  /ws/runtime/source-receipt.json \
  > qwen38-source-receipt.json
RECEIPT_SHA=$(sha256sum qwen38-source-receipt.json | awk '{print $1}')
LABEL_SHA=$(docker image inspect --format \
  '{{index .Config.Labels "org.sparkring.source-receipt-sha256"}}' "$IMAGE")
test "$RECEIPT_SHA" = "$LABEL_SHA" || { echo "source receipt label mismatch" >&2; exit 1; }
docker image inspect "$IMAGE" > qwen38-image-inspect.json
```

Retain `IMAGE_ID`, `qwen38-source-receipt.json`, its SHA-256, and
`qwen38-image-inspect.json` with any startup or smoke evidence for this build.

The image contains the runtime under `/ws`. It contains no model, rank
configuration, site address, credentials, benchmark result, LMCache, or
SparkCache.

## 3. Distribute the identical image

Export the image once on the build host:

```bash
IMAGE=sparkring-qwen38:arm64-sm121
docker save --output qwen38-runtime.oci.tar "$IMAGE"
sha256sum qwen38-runtime.oci.tar > qwen38-runtime.oci.tar.sha256
```

Copy both files to ranks 1-3 using the site's normal transfer mechanism. For
example:

```bash
ssh <user>@<rank1-host> 'mkdir -p "$HOME/qwen38/staging"'
ssh <user>@<rank2-host> 'mkdir -p "$HOME/qwen38/staging"'
ssh <user>@<rank3-host> 'mkdir -p "$HOME/qwen38/staging"'
scp qwen38-runtime.oci.tar qwen38-runtime.oci.tar.sha256 <user>@<rank1-host>:qwen38/staging/
scp qwen38-runtime.oci.tar qwen38-runtime.oci.tar.sha256 <user>@<rank2-host>:qwen38/staging/
scp qwen38-runtime.oci.tar qwen38-runtime.oci.tar.sha256 <user>@<rank3-host>:qwen38/staging/
```

On every receiving rank, verify before loading:

```bash
cd "$HOME/qwen38/staging"
sha256sum --check qwen38-runtime.oci.tar.sha256
docker load --input qwen38-runtime.oci.tar
docker image inspect --format '{{.Id}}' sparkring-qwen38:arm64-sm121
```

All four hosts must report the same `IMAGE_ID`. Building independently on each
rank is not equivalent to distributing one image identity.

## 4. Download, verify, and distribute the model

On one host, create a model parent directory and use the built image's pinned
Hugging Face client:

```bash
IMAGE=sparkring-qwen38:arm64-sm121
MODEL_PARENT="$HOME/qwen38/model"
MODEL_DIR="$MODEL_PARENT/Qwen3.8-27B-EXL3-K5K6-hydrated"
mkdir -p "$MODEL_PARENT"

docker run --rm \
  -v "$MODEL_PARENT:/models" \
  -e LD_PRELOAD= -e VLLM_NCCL_SO_PATH= \
  --entrypoint /ws/venv/bin/hf \
  "$IMAGE" \
  download malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated \
  --revision ab3a91a13813df8096cb4c1d560ed3669035d0cf \
  --local-dir /models/Qwen3.8-27B-EXL3-K5K6-hydrated
```

Verify the manifest identity and all 16 files:

```bash
docker run --rm \
  -v "$MODEL_DIR:/model:ro" \
  -e LD_PRELOAD= -e VLLM_NCCL_SO_PATH= \
  --entrypoint bash \
  "$IMAGE" -lc \
  'printf "%s  %s\n" \
    7626d18481e7f995fd1d9ff211083b7fd57f044daba39e107fb29a48207f24c4 \
    /model/SHA256SUMS | sha256sum --check --strict - && \
   cd /model && sha256sum --check --strict SHA256SUMS'
```

Copy the complete directory to the same host path on ranks 1-3. `rsync -aH`
preserves filenames, modes, and symlinks without deleting unrelated files:

```bash
ssh <user>@<rank1-host> 'mkdir -p "$HOME/qwen38/model/Qwen3.8-27B-EXL3-K5K6-hydrated"'
ssh <user>@<rank2-host> 'mkdir -p "$HOME/qwen38/model/Qwen3.8-27B-EXL3-K5K6-hydrated"'
ssh <user>@<rank3-host> 'mkdir -p "$HOME/qwen38/model/Qwen3.8-27B-EXL3-K5K6-hydrated"'
rsync -aH --info=progress2 "$MODEL_DIR/" <user>@<rank1-host>:qwen38/model/Qwen3.8-27B-EXL3-K5K6-hydrated/
rsync -aH --info=progress2 "$MODEL_DIR/" <user>@<rank2-host>:qwen38/model/Qwen3.8-27B-EXL3-K5K6-hydrated/
rsync -aH --info=progress2 "$MODEL_DIR/" <user>@<rank3-host>:qwen38/model/Qwen3.8-27B-EXL3-K5K6-hydrated/
```

Run the 16-file verification command on every rank after transfer. The image
identity does not attest separately mounted model bytes.

## 5. Create one rank environment per host

From rank 0's checkout, create its writable directories and local environment,
then install one private template copy on each other rank:

```bash
mkdir -p "$HOME/qwen38/cache" "$HOME/qwen38/logs" "$HOME/qwen38/config"
cp scripts/config/qwen38-27b-exl3-k5k6.env.example \
  "$HOME/qwen38/config/rank.env"
ssh <user>@<rank1-host> 'mkdir -p "$HOME/qwen38/cache" "$HOME/qwen38/logs" "$HOME/qwen38/config"'
ssh <user>@<rank2-host> 'mkdir -p "$HOME/qwen38/cache" "$HOME/qwen38/logs" "$HOME/qwen38/config"'
ssh <user>@<rank3-host> 'mkdir -p "$HOME/qwen38/cache" "$HOME/qwen38/logs" "$HOME/qwen38/config"'
scp scripts/config/qwen38-27b-exl3-k5k6.env.example \
  <user>@<rank1-host>:qwen38/config/rank.env
scp scripts/config/qwen38-27b-exl3-k5k6.env.example \
  <user>@<rank2-host>:qwen38/config/rank.env
scp scripts/config/qwen38-27b-exl3-k5k6.env.example \
  <user>@<rank3-host>:qwen38/config/rank.env
```

Edit `$HOME/qwen38/config/rank.env` locally on each rank; do not reuse rank 0's
resolved file on another rank.

Replace all four placeholders:

- `<RENDEZVOUS_IFNAME>`: the management interface used by vLLM, Gloo, and
  NCCL bootstrap TCP;
- `<RANK_RENDEZVOUS_IP>`: this rank's IPv4 address on that management
  interface;
- `<NCCL_IB_HCA>`: the two cycle-facing RDMA devices discovered with
  `ibdev2netdev`; and
- `<NCCL_IB_GID_INDEX>`: the verified RoCEv2 GID index for both devices on
  this rank.

The management LAN must permit mutual TCP between ranks. vLLM's multi-node
`mp` executor uses dynamic worker ports in addition to rendezvous port 29500.
The environment uses management only for process/bootstrap traffic; model
collectives use the two RoCE devices.

## 6. Run the no-start local preflight

Set these values separately on each host:

```bash
IMAGE=sparkring-qwen38:arm64-sm121
MODEL_PARENT="$HOME/qwen38/model"
MODEL_DIR="$MODEL_PARENT/Qwen3.8-27B-EXL3-K5K6-hydrated"
ATTEMPT_ID=<shared-deployment-id>
RANK=<0-to-3>
RANK0_RENDEZVOUS_ADDR=<rank0-management-ip>
ENV_FILE="$HOME/qwen38/config/rank.env"
CACHE_DIR="$HOME/qwen38/cache"
LOG_DIR="$HOME/qwen38/logs"

case "$ATTEMPT_ID" in
  ''|*[!A-Za-z0-9_.-]*) echo "ATTEMPT_ID must match [A-Za-z0-9_.-]+" >&2; exit 1 ;;
esac
```

Before claiming the GPU, inspect host-level ownership. The container preflight
cannot see a serving process isolated in another container:

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
nvidia-smi
ss -ltn
```

Do not continue while another model stack owns the GPU or rank-0 ports 8000
and 29500. Then run the same container and mounts as the real launch, but pass
`--check`:

```bash
docker run --rm \
  --network host --ipc host --shm-size 16g --gpus all \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband \
  -v "$MODEL_DIR:/ws/model/Qwen3.8-27B-EXL3-K5K6-hydrated:ro" \
  -v "$CACHE_DIR:/ws/cache" \
  -v "$LOG_DIR:/ws/logs" \
  -v "$ENV_FILE:/ws/rank.env:ro" \
  -e RANK="$RANK" \
  -e RANK0_RENDEZVOUS_ADDR="$RANK0_RENDEZVOUS_ADDR" \
  --label org.sparkring.profile=qwen38-27b-exl3-k5k6 \
  --label org.sparkring.attempt="$ATTEMPT_ID" \
  --label org.sparkring.rank="$RANK" \
  --entrypoint /ws/qwen38_dgx4_serve.sh \
  "$IMAGE" --check
```

`--check` does not start vLLM. It rejects unresolved placeholders, wrong
source or patch state, model/template/NCCL hash drift, failed runtime imports,
missing verbs/RDMA devices, incorrect management address binding, invalid
HCA/GID mappings, duplicate vLLM processes, and occupied rank-0 ports. It ends
by printing the resolved command. Require success on all four ranks.

The model verification reads approximately 22 GB per rank. This is deliberate
when no immutable model lifecycle has been recorded. A later restart policy
can cite a separately recorded lifecycle rather than silently removing the
gate.

## 7. Start the four ranks

Starting a container claims the GPU and can interrupt an existing service.
This is a **STOPS SERVING** action when another stack is active. Preserve that
stack's image, mounts, configuration, and logs before stopping it.

Use the same variables and mounts as preflight. Start ranks 1-3 first, then
rank 0, dispatching all four within two minutes. Choose one unique
`ATTEMPT_ID` containing only letters, digits, dots, underscores, or hyphens and
use it on all four hosts. Do not reuse an ID from a preserved container:

```bash
CONTAINER="qwen38-dgx4-${ATTEMPT_ID}-r${RANK}"
ID_FILE="$LOG_DIR/container-id-${ATTEMPT_ID}-r${RANK}"
test ! -e "$ID_FILE" || { echo "attempt ID file already exists: $ID_FILE" >&2; exit 1; }

CONTAINER_ID=$(docker run -d --name "$CONTAINER" \
  --network host --ipc host --shm-size 16g --gpus all \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband \
  -v "$MODEL_DIR:/ws/model/Qwen3.8-27B-EXL3-K5K6-hydrated:ro" \
  -v "$CACHE_DIR:/ws/cache" \
  -v "$LOG_DIR:/ws/logs" \
  -v "$ENV_FILE:/ws/rank.env:ro" \
  -e RANK="$RANK" \
  -e RANK0_RENDEZVOUS_ADDR="$RANK0_RENDEZVOUS_ADDR" \
  --label org.sparkring.profile=qwen38-27b-exl3-k5k6 \
  --label org.sparkring.attempt="$ATTEMPT_ID" \
  --label org.sparkring.rank="$RANK" \
  --entrypoint /ws/qwen38_dgx4_serve.sh \
  "$IMAGE" --run) || exit
printf '%s\n' "$CONTAINER_ID" > "$ID_FILE"
```

Rank 0 listens on port 8000. Ranks 1-3 run headless. The launcher carries the
complete serving command, including TP4/DCP1, the 1,048,576-token request limit,
64 sequences, the 8,192-token scheduler budget, FP8 KV, native aligned prefix
caching, probabilistic Qwen MTP3 with standard rejection, FP8 EXL3 prefill,
and full-decode CUDA graphs.
The tracked
[`scripts/qwen38_dgx4_serve.sh`](../scripts/qwen38_dgx4_serve.sh) is baked at
`/ws/qwen38_dgx4_serve.sh` by the image builder.

## 8. Bound startup and inspect every rank

Allow up to 15 minutes for first-start compilation and graph capture:

```bash
timeout 900 bash -c \
  'until curl -fsS http://<rank0-management-ip>:8000/health >/dev/null; do sleep 5; done'
curl -fsS http://<rank0-management-ip>:8000/v1/models
```

On every rank, require the container to remain running and inspect its log:

```bash
ID_FILE="$LOG_DIR/container-id-${ATTEMPT_ID}-r${RANK}"
CONTAINER_ID=$(cat "$ID_FILE")
test "$(docker inspect --format '{{index .Config.Labels "org.sparkring.attempt"}}' "$CONTAINER_ID")" = "$ATTEMPT_ID" || { echo "container attempt label mismatch" >&2; exit 1; }
test "$(docker inspect --format '{{index .Config.Labels "org.sparkring.rank"}}' "$CONTAINER_ID")" = "$RANK" || { echo "container rank label mismatch" >&2; exit 1; }
docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' \
  "$CONTAINER_ID"
test "$(docker inspect --format '{{.State.Running}}' "$CONTAINER_ID")" = true || { echo "rank container is not running" >&2; exit 1; }
docker logs "$CONTAINER_ID"
```

The combined logs must show four-rank rendezvous, TP4/DCP1, both RoCE devices,
four Ring channels, Qwen MTP depth 3, full-decode graph capture, and the
reported key-value capacity. A dispatch success is not startup success.

If startup fails, preserve each log before cleanup:

```bash
ID_FILE="$LOG_DIR/container-id-${ATTEMPT_ID}-r${RANK}"
CONTAINER_ID=$(cat "$ID_FILE")
test "$(docker inspect --format '{{index .Config.Labels "org.sparkring.attempt"}}' "$CONTAINER_ID")" = "$ATTEMPT_ID" || { echo "container attempt label mismatch" >&2; exit 1; }
test "$(docker inspect --format '{{index .Config.Labels "org.sparkring.rank"}}' "$CONTAINER_ID")" = "$RANK" || { echo "container rank label mismatch" >&2; exit 1; }
docker logs "$CONTAINER_ID" \
  > "$LOG_DIR/startup-${ATTEMPT_ID}-rank-${RANK}.log" 2>&1
docker stop "$CONTAINER_ID"
```

Stop only IDs recorded by the failed attempt after both label checks pass. Keep
the containers and logs, and diagnose the first error. Do not remove evidence,
stop a pre-existing container whose ID was not captured, or weaken a
hash/preflight check to make a later attempt proceed.

## 9. Run the public bounded smoke gate

From a client that can reach rank 0:

```bash
mkdir -p "$HOME/qwen38/logs"
python scripts/qwen38_smoke.py \
  --endpoint http://<rank0-management-ip>:8000 \
  --model qwen38 \
  --timeout 120 \
  --output "$HOME/qwen38/logs/qwen38-smoke-${ATTEMPT_ID}.json"
```

The stdlib-only harness checks `/health`, model identity and the 1,048,576-token
limit in `/v1/models`, repeated deterministic arithmetic, the `multiply(6,7)`
tool call, a tiny data-URL vision request, repeated-prefix output equality,
and divergent shared-prefix suffixes. It
compares stable message fields rather than API IDs or timestamps and returns
nonzero on any failed gate. It reports no timing or cache-hit claim.

After the gate, recheck all four container states, rank-0 health, host
`MemAvailable`, swap, and logs for worker exits or runtime errors. A public build
that reaches API, passes the bounded smoke gate, and remains healthy can be
  retained with the exact image ID and launch inputs.

Run this on every rank after the smoke gate:

```bash
ID_FILE="$LOG_DIR/container-id-${ATTEMPT_ID}-r${RANK}"
CONTAINER_ID=$(cat "$ID_FILE")
test "$(docker inspect --format '{{index .Config.Labels "org.sparkring.attempt"}}' "$CONTAINER_ID")" = "$ATTEMPT_ID" || { echo "container attempt label mismatch" >&2; exit 1; }
test "$(docker inspect --format '{{index .Config.Labels "org.sparkring.rank"}}' "$CONTAINER_ID")" = "$RANK" || { echo "container rank label mismatch" >&2; exit 1; }
test "$(docker inspect --format '{{.State.Running}}' "$CONTAINER_ID")" = true || { echo "rank container exited" >&2; exit 1; }
grep -E '^(MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
docker logs --tail 200 "$CONTAINER_ID"
```

From the client, require rank 0 to remain healthy:

```bash
curl -fsS http://<rank0-management-ip>:8000/health
```

## 10. Stop or restart the exact deployment

Stop ranks 0-3 without removing their containers or logs:

```bash
ID_FILE="$LOG_DIR/container-id-${ATTEMPT_ID}-r${RANK}"
CONTAINER_ID=$(cat "$ID_FILE")
test "$(docker inspect --format '{{index .Config.Labels "org.sparkring.attempt"}}' "$CONTAINER_ID")" = "$ATTEMPT_ID" || { echo "container attempt label mismatch" >&2; exit 1; }
test "$(docker inspect --format '{{index .Config.Labels "org.sparkring.rank"}}' "$CONTAINER_ID")" = "$RANK" || { echo "container rank label mismatch" >&2; exit 1; }
docker stop "$CONTAINER_ID"
```

For a coordinated restart with unchanged mounts and environment, start ranks
1-3 and then rank 0 within two minutes:

```bash
ID_FILE="$LOG_DIR/container-id-${ATTEMPT_ID}-r${RANK}"
CONTAINER_ID=$(cat "$ID_FILE")
test "$(docker inspect --format '{{index .Config.Labels "org.sparkring.attempt"}}' "$CONTAINER_ID")" = "$ATTEMPT_ID" || { echo "container attempt label mismatch" >&2; exit 1; }
test "$(docker inspect --format '{{index .Config.Labels "org.sparkring.rank"}}' "$CONTAINER_ID")" = "$RANK" || { echo "container rank label mismatch" >&2; exit 1; }
docker start "$CONTAINER_ID"
```

Re-run health and the bounded smoke gate after restart. Recreate containers
when image, mounts, rank environment, or serving settings change, and record
the resulting identity.

## Benchmark results

![Four-Spark Qwen benchmark](../performance/records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.png)

Prefill measured 1,855–2,001 tok/s through 32K, 1,616 at 64K, and 1,279 at
128K. Sustained decode measured 30–36 tok/s at C1, 55–66 at C2, 87–121 at C4,
and 138–202 aggregate tok/s at C8. Coding Peak completed 15/15 requests with a
48.46 tok/s mean. The table and N counts are in the
[full result](../performance/records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.md). Sanitized command receipts are in
[`performance/receipts/qwen38-27b/temp1/20260823-tp4/`](../performance/receipts/qwen38-27b/temp1/20260823-tp4/).

## SparkCache

SparkCache is not included in this Qwen setup. External key-value caching is
disabled and is not required to run this quickstart.
