# Qwen3.8-Flash-Next NVFP4 QAD on two Sparks

Status: **Experimental**. The commands select the published SparkCache profile;
the cache-disabled alternative shares the same procedure. This uses the
QAD checkpoint pinned to revision `629bc3218833a38b475b719f34aa571666f4a03e`
in `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`. Complete the
[host prerequisites](../../docs/operations/prerequisites.md) and prepare one
direct cable on cage p0 before starting; the launcher does not configure networking.

The published SparkCache image uses **aligned checkpoints**. Complete
request-boundary checkpoint support and its measurements are a separate
composition whose publication is pending; do not attribute those measurements
to the image below. See the [boundary-checkpoint findings](../../performance/records/qwen38-flash-next/BOUNDARY-TP2.md).

[QAD TP2 startup evidence](../../performance/records/qwen38-flash-next/qad-tp2-startup-20260916.json)
covers verified weights, startup, SparkCache initialization and a text smoke check
on that separate request-boundary runtime. The published images selected below
have not been requalified with QAD. Prior non-QAD cache-restore, media and performance
measurements do not establish QAD qualification.

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
BASE_IMAGE=ghcr.io/fujitsupolycom/sparkring@sha256:f5a7e01c6112c8ef85a51b24bfacfd3934ee9cfff06b7e8c72abcf5d90b50270
IMAGE_REF=ghcr.io/fujitsupolycom/sparkring@sha256:de885a8a3f687d1966b918f913ab95b0da33a84422313ed4c10ba5477c66f523
MODEL_DIR=/srv/models/Qwen3.8-Flash-Next-NVFP4-QAD/629bc321
CACHE_DIR=/srv/cache/qwen38-flash-next-qad-tp2-r37

# The parent supports full-inventory verification and native-cache rollback.
docker pull --platform linux/arm64 "$BASE_IMAGE"
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
  -d '{"model":"Qwen3.8-Flash-Next-NVFP4-QAD","messages":[{"role":"user","content":"Reply only READY"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

The API is `http://RANK0_ADDRESS:8000/v1`, with model name
`Qwen3.8-Flash-Next-NVFP4-QAD`. It has no configured authentication: restrict access
to trusted clients or an authenticated gateway. The short completion checks basic
generation, not cache persistence or capacity.

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
IMAGE_REF=$BASE_IMAGE
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")
```

## Evidence and remaining checks

Configuration is owned by [sparkcache.json](sparkcache.json) and
[config.json](config.json), not this table. Capacity overrides are intentionally
not accepted. The following records concern the original non-QAD checkpoint,
not this QAD selection. The [SparkCache record](../../performance/records/qwen38-flash-next/r37-sparkcache.json)
covers bounded publication, process-restart text/media restore, changed-input
misses and corruption recovery. The [native-cache record](../../performance/records/qwen38-flash-next/r37-tp2.json)
covers native prefix reuse, C16 short answers, near-limit text and synthetic media.

These checks do not qualify arbitrary/high-resolution video, C16 multimedia,
sixteen simultaneous full-context requests or prolonged store-pressure stability.
Sixteen video frames are not sixteen visual tokens. A 512 MiB capture slot can
reject a snapshot below the configured 65536-token span ceiling; serving must
continue without optional cache publication.

[Image source/build recipe](../../runtime/images/compositions/lil-r37-cache64/README.md).
[Generated Compose deployments](../../docs/operations/compose.md) support both
configurations, with and without SparkCache. [Bounded TP2 checks](../../performance/records/qwen38-flash-next/compose-tp2.json)
cover startup, shutdown, text responses and persistent-cache restore.
