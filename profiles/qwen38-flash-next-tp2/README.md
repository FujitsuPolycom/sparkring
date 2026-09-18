# Qwen3.8-Flash-Next NVFP4 QAD on two Sparks

Status: **Validated for bounded functional checks with SparkCache**. The commands select the published SparkCache profile;
the cache-disabled alternative shares the same procedure. This uses the
QAD checkpoint pinned to revision `629bc3218833a38b475b719f34aa571666f4a03e`
in `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`. Complete the
[host prerequisites](../../docs/operations/prerequisites.md) and prepare one
direct cable on cage p0 before starting; the launcher does not configure networking.

Both variants use [SparkRing shared-2026.09.0](../../runtime/releases/shared-2026.09.0/README.md)
with aligned checkpoints, managed B12X loading, Qwen checkpoint coalescing,
compact MTP and projection overlap. HC row sharding remains off on TP2; the
included ownership implementation requires TP4. Request-boundary caching and
request-salt isolation are not enabled by this selection.

| Setting | Published SparkCache profile |
|---|---|
| Parallelism | TP2/DCP1; one DAC, both Socket Direct functions |
| Context / sequences / batch | 262144 / 16 / 8192; no YaRN |
| KV | 24 GiB FP8 per rank; separate from host cache buffers |
| Loading / speculation | Managed B12X / MTP3 |
| Media | Three images, one video; 16 configured video frames |
| Persistent cache | 4 GiB disk per rank, reclaim toward 3 GiB; about 1.25 GiB data buffers per rank |

## Image and checkpoint

Use the same SparkRing checkout on both nodes. Set these variables in Bash;
`MODEL_DIR` must contain the verified checkpoint, and `CACHE_DIR` must be a
separate writable directory.

```bash
REPO=$PWD
PROFILE=profiles/qwen38-flash-next-tp2/sparkcache.json
CONTAINER_PREFIX=qwen-flash-next-sparkcache-tp2
IMAGE_REF=ghcr.io/fujitsupolycom/sparkring@sha256:8cfcfdaffd91af252c0eef2f46d325765f364d3ac812ee7f76343fa2393dd357
MODEL_DIR=/srv/models/Qwen3.8-Flash-Next-NVFP4-QAD/629bc321
CACHE_DIR=/srv/cache/qwen38-flash-next-qad-tp2-shared-2026090

# No separate parent-image pull is required for native-image verification.
docker pull --platform linux/arm64 "$IMAGE_REF"
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")
mkdir -p "$CACHE_DIR"
```

Reuse an existing verified model copy. Otherwise download the approximately
106 GB checkpoint once per rank, or transfer the verified files over the data fabric:

```bash
# Skip this download when MODEL_DIR already contains the verified checkpoint.
hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
  --revision 629bc3218833a38b475b719f34aa571666f4a03e --local-dir "$MODEL_DIR"
(cd "$MODEL_DIR" && sha256sum --check "$REPO/profiles/qwen38-flash-next-tp2/SHA256SUMS")
```

Directory names do not prove model identity. Model mounts are read-only; do not
put the writable cache inside the model directory.

When switching from plain NVFP4, recreate the serving containers with this
profile; restarting an existing container does not change its model mount.
Preserve the original containers and weights for rollback. QAD uses a distinct
served name, checkpoint identity and persistent-cache namespace. Do not carry
the plain-NVFP4 checkpoint digests into the QAD cache configuration.

## Plan and create

Replace the example addresses/interface on **each node** with the prepared
high-speed fabric IPs and Linux interface, so worker/control traffic also uses
that fabric. Rank1 uses `RANK=1` and its own `HOST_IP`; both ranks use the same
reachable `MASTER_ADDR`. SSH and API clients may use the management network.

```bash
RANK=0
MASTER_ADDR=192.0.2.10
HOST_IP=192.0.2.10
INTERFACE=enp1s0f0np0
launch_rank() {
  python3 runtime/common/qwen_flash_next.py "$1" \
    --profile "$PROFILE" --rank "$RANK" \
    --master "$MASTER_ADDR" --host-ip "$HOST_IP" --interface "$INTERFACE" \
    --image "$IMAGE_ID" --model "$MODEL_DIR" --cache "$CACHE_DIR"
}
launch_rank plan
# Inspect both plans before creating the stopped containers.
launch_rank create
```

The defaults select `rocep1s0f0` and `roceP2p1s0f0`, the two PCI-domain views
of cage p0. Confirm the cable/device mapping. The bootstrap interface is a
separate input, not an RDMA device list. `plan` is offline; `create` verifies
the image and refuses existing names. Neither action stops a running workload.

## Controlled startup

During an authorized test window, stop competing GPU workloads explicitly and
preserve them for rollback. Start rank1, then rank0, on their respective hosts:

```bash
docker start "${CONTAINER_PREFIX}-r1" # rank1 host
docker start "${CONTAINER_PREFIX}-r0" # rank0 host
docker logs --follow --tail 100 "${CONTAINER_PREFIX}-r${RANK}"
```

From rank0 or a client that can reach its bootstrap address:

```bash
curl --fail "http://${MASTER_ADDR}:8000/health"
curl --fail "http://${MASTER_ADDR}:8000/v1/models"
curl --fail "http://${MASTER_ADDR}:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next-NVFP4-QAD-TP2","messages":[{"role":"user","content":"Reply only READY"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

The API is `http://RANK0_ADDRESS:8000/v1`, with model name
`Qwen3.8-Flash-Next-NVFP4-QAD-TP2`. It has no configured authentication: restrict access
to trusted clients or an authenticated gateway. The short completion checks basic
generation, not cache persistence or capacity. The `SPARKRING STARTUP AUDIT`
banner appears before API readiness. Read its warnings: requested flags are not
proof of runtime execution or performance. The bundled SparkCache does not
isolate disk entries by request `cache_salt`; use separate deployments/cache
namespaces when tenant isolation is required.

Monitor host `MemAvailable`; Docker's 108 GiB memory and 112 GiB combined
memory/swap limits do not cover every GB10 GPU allocation. This profile installs
no persistent memory guard, boot service or network changes. Containers do not autostart.

### Restart existing containers

Do not rerun `create`. Stop each rank on its own host, then use the startup
commands above in rank1/rank0 order. Disk cache entries remain intact:

```bash
docker stop --timeout 30 "${CONTAINER_PREFIX}-r${RANK}"
```

### Cache-disabled alternative

Before planning/creating, select these values instead; all other steps are shared:

```bash
PROFILE=profiles/qwen38-flash-next-tp2/config.json
CONTAINER_PREFIX=qwen-flash-next-tp2
# IMAGE_REF and IMAGE_ID remain the same shared release.
```

## Evidence and remaining checks

Configuration is owned by [sparkcache.json](sparkcache.json) and
[config.json](config.json), not this table. Capacity overrides are intentionally
not accepted. The [release qualification](../../runtime/releases/shared-2026.09.0/qualification.json)
records bounded text and synthetic image/video correctness with SparkCache
enabled on the exact QAD runtime. Cache-disabled deployment checks are separate;
shared image selection alone is not additional hardware evidence.

These checks do not qualify arbitrary/high-resolution video, C16 multimedia,
sixteen simultaneous full-context requests or prolonged store-pressure stability.
Sixteen video frames are not sixteen visual tokens. A 512 MiB capture slot can
reject a snapshot below the configured 65536-token span ceiling; serving must
continue without optional cache publication.

[Release sources and rollback](../../runtime/releases/shared-2026.09.0/README.md).
[Generated Compose deployments](../../docs/operations/compose.md) support both
configurations, with and without SparkCache. Historical R37 evidence remains
attached to its original image and is not reattributed to this release.
