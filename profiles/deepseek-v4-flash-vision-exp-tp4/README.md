# DeepSeek V4 Flash Vision-Exp on a four-Spark cycle

Profile: `deepseek-v4-flash-vision-exp-tp4`. Status: **research-only**. The recipe records configuration and evidence boundaries. Its implementation status does not qualify a rebuilt image.

Inspect its selected defaults with `python scripts/profiles.py resolve deepseek-v4-flash-vision-exp-tp4`.

Status: **research-only**. This profile defines a four-rank launch configuration
for `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`. Contributor-reported serving
results are attributed below; independent reproduction is not claimed.

The model and serving hotfixes come from
[MiaAI-Lab's DSpark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/tree/7440c53c1f0352886e47b1909051784879fa0a24)
and the Anemll serving image identified in the
[runtime contract](../../runtime/deepseek-vision-exp/profile.json).
SparkRing supplies the four-rank cycle configuration and patched NVIDIA NCCL.
There is no SparkCache or external KV connector in this profile.

The [runtime contract](../../runtime/deepseek-vision-exp/profile.json) records
artifact identities. The [serving recipe](../../recipes/deepseek-v4-flash-vision-exp-tp4.json)
summarizes model, topology, and serving settings. Use the
[rank environment template](../../runtime/deepseek-vision-exp/cycle.env.example)
and [Compose overlay](../../runtime/deepseek-vision-exp/compose.override.yml)
with the exact upstream recipe revision below.

## Prepare the four hosts

Connect the ranks in the physical cycle `0-1-2-3-0`, with two directly connected
neighbors per rank. Verify the cable endpoints and assign ranks in cycle order;
an existing pair's labels do not establish the cross-pair cable order.

Complete the four-Spark cycle steps in [PREREQUISITES.md](../../docs/operations/prerequisites.md)
and the [patched NCCL contract](../../spark_transport/nccl/README.md). Each rank
needs both neighbor-facing RoCE interfaces, persistent addresses and routes,
and a verified RoCEv2/IPv4 GID on each selected device. Choose the actual device
names and GID indices from the host inventory.

The management network carries torch rendezvous and NCCL/Gloo bootstrap.
Allow connectivity between all four management addresses, including the
configured rendezvous port and bootstrap sockets. Scope the rank-zero API to
its intended clients. Stop conflicting model workloads before starting this
profile. A planned reboot before installation can help after memory-intensive
GPU workloads; it does not replace the host prerequisite checks.

## Pin the serving recipe and model

From the SparkRing checkout on each host, use an unused deployment directory:

```bash
sparkring_root="$PWD"
recipe_dir="$PWD/deepseek-vision-exp-runtime"

git clone https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark.git "$recipe_dir"
git -C "$recipe_dir" checkout --detach 7440c53c1f0352886e47b1909051784879fa0a24
```

The checkpoint is
[DeepSeek-V4-Flash-Vision-Exp revision 86f746b3](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/tree/86f746b36186f0e567729a5c06a8c918caba82a9).
Download that revision into a standard Hugging Face cache on every rank. For
example, with the Hugging Face CLI installed:

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --revision 86f746b36186f0e567729a5c06a8c918caba82a9
```

Set `HF_CACHE` in each rank environment to the cache root containing `hub/`.
The template's `DSPARK_REVISION` selects this exact model revision at startup.
The upstream recipe directory must contain its pinned Compose file and hotfix
mounts; copying only the Compose file is insufficient.

## Select the image and NCCL library

The profile selects the Anemll 0.1.1 image directly, without an additional
FlashInfer package overlay. Pull its immutable reference and the NCCL donor
image from the runtime contract. The donor container is created only to copy
the library; it is never started.

Run from the SparkRing checkout, retaining `recipe_dir` from the preceding
section. Choose an unused export-container name before running this block:

```bash
profile="$sparkring_root/runtime/deepseek-vision-exp/profile.json"
nccl_library="$recipe_dir/artifacts/nccl/libnccl.so.2.30.7"

profile_value() {
  python3 - "$profile" "$1" <<'PY'
import json
import sys
with open(sys.argv[1]) as source:
    value = json.load(source)
for key in sys.argv[2].split('.'):
    value = value[key]
print(value)
PY
}

(
  set -eu
  image="$(profile_value image.reference)"
  donor="$(profile_value transport.donor_image)"
  donor_path="$(profile_value transport.donor_path)"
  docker pull "$image"
  test "$(docker image inspect "$image" --format '{{.Id}}')" = "$(profile_value image.config_digest)"
  docker pull "$donor"
  test "$(docker image inspect "$donor" --format '{{.Id}}')" = "$(profile_value transport.donor_config_digest)"

  mkdir -p "$(dirname "$nccl_library")"
  chmod 0755 "$(dirname "$nccl_library")"
  export_id="$(docker create --name deepseek-vision-nccl-export "$donor")"
  trap 'docker rm "$export_id" >/dev/null' EXIT
  docker cp "$export_id:$donor_path" "$nccl_library"
  printf '%s  %s\n' "$(profile_value transport.library_sha256)" "$nccl_library" | sha256sum --check
  chmod 0644 "$nccl_library"
)
```

The exit trap removes only the export container created by this block, using
its returned container ID. It also runs if copying or verification fails.
Set `SPARKRING_NCCL_LIBRARY` in the rank environment to the absolute
`nccl_library` path above. Use the same verified image and library identities
on every rank.

The [cycle patch](../../spark_transport/nccl/nccl-2.30.7-switchless-cycle.patch)
uses `NCCL_SWITCHLESS_RING_ONLY=1` to disable Tree/PAT connection setup and
provides subnet-aware selection with eligible listener GIDs. The Compose
overlay mounts the library read-only and sets both `LD_PRELOAD` and
`VLLM_NCCL_SO_PATH`. An unpatched library with the same version number is not
an equivalent input.

Optionally check library loading and the NCCL version function inside the
selected serving image before loading a model. This command needs no GPU
assignment and makes no collective or network connection:

```bash
docker run --rm --network none --entrypoint python3 \
  --mount "type=bind,source=$nccl_library,target=/opt/sparkring-nccl.so,readonly" \
  "$(profile_value image.reference)" -c '
import ctypes
library = ctypes.CDLL("/opt/sparkring-nccl.so")
version = ctypes.c_int()
assert library.ncclGetVersion(ctypes.byref(version)) == 0
assert version.value == 23007, version.value
print("NCCL version:", version.value)
'
```

This checks loadability and the reported version, not RDMA operation or
four-rank serving compatibility. The SHA-256 check identifies the actual patch
artifact.

## Configure each rank

Copy the cycle template to a separate file on each rank:

```bash
cp "$sparkring_root/runtime/deepseek-vision-exp/cycle.env.example" "$recipe_dir/.env.cycle"
chmod 0600 "$recipe_dir/.env.cycle"
```

Edit `.env.cycle` for the local rank. Set `NODE_RANK` to 0, 1, 2, or 3;
`HEADLESS` must be empty on rank zero and `1` on the other ranks. Set the shared
`MASTER_ADDR` to rank zero's management address, `VLLM_HOST_IP` to the local
management address, and the NCCL/TP/Gloo socket interfaces to the management
interface. Set the two RoCE devices and their verified GID index. Configure
`HF_CACHE` as the model-cache root, `DSPARK_TMP_HOST` as an existing absolute
compiler-cache directory, `SPARKRING_NCCL_LIBRARY` as the verified library path,
and `VLLM_API_KEY` as the private API key. For example, create a dedicated
compiler-cache directory with `mkdir -p "$recipe_dir/compiler-cache"` and
`chmod 0700 "$recipe_dir/compiler-cache"`, then use that absolute path for
`DSPARK_TMP_HOST`. Replace every `REPLACE_` value in the copied template.

Keep one assignment per line. These settings must agree across all ranks:

```text
NNODES=4
TP_SIZE=4
DSPARK_MODEL=deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
DSPARK_REVISION=86f746b36186f0e567729a5c06a8c918caba82a9
SERVED_MODEL_NAME=deepseek-v4-flash-vision-exp
MTP_NUM_TOKENS=5
DSPARK_ENABLE_DSPARK_BLOCK_K=1
DSPARK_MAX_INFLIGHT_PREFILLS=2
DSPARK_ENABLE_SP_INDEXER=1
GPU_MEMORY_UTILIZATION=0.80
MAX_NUM_SEQS=48
MAX_NUM_BATCHED_TOKENS=12288
DSPARK_RESTART_POLICY=no
```

Vision-Exp has three MTP stages. The upstream block-k hotfix must be enabled
for the selected five-token DSpark block; the upstream default leaves that
hotfix disabled. Two in-flight prefills are also an explicit override of the
upstream default of one. The served model name above is the API identifier.
The 48-sequence limit is a configuration choice, not a per-request speed
or latency guarantee.

## Render and start

Run Compose from the pinned upstream recipe directory so its relative hotfix
mounts resolve correctly. Validate the merged configuration on every rank:

```bash
set -euo pipefail
cd "$recipe_dir"
rank=REPLACE_WITH_RANK_0_TO_3
compose_args=(
  -p deepseek-vision-cycle
  --env-file "$recipe_dir/.env.cycle"
  -f "$recipe_dir/docker-compose.dspark.yml"
  -f "$sparkring_root/runtime/deepseek-vision-exp/compose.override.yml"
)
docker compose "${compose_args[@]}" config --format json |
  python3 "$sparkring_root/runtime/deepseek-vision-exp/check_compose.py" \
    --rank "$rank" --nccl-library "$nccl_library"
```

Exported shell variables take precedence over `--env-file`. The check validates
the resolved image, model, serving limits, rank, required hotfixes, and NCCL
mount against the profile, so conflicting exports for those checked settings
are rejected. It prints a summary without API credentials. The Python check
reads the resolved JSON and requires the verified library path to be an existing
file; it does not contact Docker or any model host. Other optional
upstream switches and custom hotfix mounts remain outside this check.

Review the rank-specific addresses and device/GID choices; the offline check
does not verify network reachability. Then start ranks 3, 2, and 1 before rank zero:

```bash
docker compose "${compose_args[@]}" up -d
docker compose "${compose_args[@]}" logs --follow vllm-dspark
```

The profile uses `restart: no`; automatic cluster recovery is not configured.

## Stop and recover

Stop application traffic, then run the following on every rank, stopping
workers before rank zero:

```bash
docker compose "${compose_args[@]}" stop --timeout 60
docker compose "${compose_args[@]}" ps -a
```

Confirm all four serving containers are stopped before changing configuration
or recovering a failed rank. After correcting the cause, repeat configuration
validation, start workers 3, 2, and 1 before rank zero, and repeat the serving
checks. Use `up -d --force-recreate` when replacing an exited container with the
same configuration. Do not restart one participating rank beneath live collectives.

## Serving checks

Before directing application traffic to rank zero, check authenticated model
listing and a small text request using `deepseek-v4-flash-vision-exp`. An
unauthenticated model-list request should be rejected when an API key is set.
Verify a known-color image response and a normal completion stop. Confirm the
selected image identity and mapped patched NCCL library on every rank.

Headless workers do not expose an HTTP listener, and the upstream worker
healthcheck returns success without performing a collective. Container health
alone therefore does not establish four-rank readiness. Use successful model
responses and the cycle's fabric checks together. The upstream `/health`
endpoint does not establish completion of every request-shape compilation.

## Contributor-reported observations

The contributor to [SparkRing PR #232](https://github.com/FujitsuPolycom/sparkring/pull/232)
reports one-site TP2/TP4 throughput comparisons, image and text checks,
long-context needle retrieval, and a three-hour concurrent serving soak.
The PR records the reported conditions, including its optional FlashInfer
overlay and sequence-parallel indexer. Those observations describe the
contributor's measured configuration; they are not independent measurements of
every artifact combination permitted by this guide. See the PR for detailed
numbers rather than treating them as throughput or per-request service targets.

The sequence-parallel indexer and block-k changes belong to the pinned
MiaAI-Lab recipe; the serving image belongs to Anemll. SparkRing's integration
selects the cycle transport and launch settings. This guide does not claim that
changing rank count alone proves model correctness or performance. A four-rank
cycle is one failure domain and does not provide the redundancy of two pairs.
