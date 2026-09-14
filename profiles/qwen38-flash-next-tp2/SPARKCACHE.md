# Qwen TP2 with persistent prefix caching

Status: **research-only**, with bounded text and multimodal persistence checks.
This opt-in configuration uses [sparkcache.json](sparkcache.json), the
[R37 hybrid-cache extension](../../runtime/images/compositions/lil-r37-cache64/README.md),
and the shared Qwen launcher. The default [config.json](config.json) remains
cache-disabled and uses the published R37 image.

Complete the [Qwen checkpoint and host setup](README.md) first. Build the
extension on each node; do not use the base image with the cache-enabled profile.
The launcher rejects an extension whose runtime differs from its pinned parent
or declared replacement files.

| Setting | Value |
|---|---|
| Context / sequences / batch | 262144 / 16 / 8192 |
| KV / loader / speculation | 24 GiB FP8 per rank / managed B12X / MTP3 |
| Parallelism | TP2 / DCP1 |
| Media | Three images, one video; 16 sampled video frames |
| Recurrent retention | Aligned checkpoints |
| Persistent disk limit | 4 GiB per rank; reclaim toward 3 GiB |
| Persisted span range | 4096–65536 tokens; eligible page boundaries only |
| Capture buffers | Two 512 MiB slots per rank |
| Restore buffers | Two lanes, two 64 MiB arenas per lane |

The explicit data buffers total 1.25 GiB per rank, in addition to metadata,
workspace and reclaimable OS file cache. Disk capacity is not a RAM reservation.
The 24 GiB KV pin is separate. Synthetic validation is not a guarantee that
sixteen full-context or multimedia-heavy requests fit simultaneously.
Snapshots that exceed a capture slot are skipped, so the configured span range
does not promise that every prefix can be persisted. Native cache-token counters
alone are not proof of disk publication; check SparkCache commit/restore records.

## Plan, create and start

Run from the same SparkRing checkout on both nodes. Supply rank-local model and
cache paths, rank zero's reachable bootstrap address, this node's address and
its corresponding interface. Use the existing one-cable cage-p0 fabric setup
from the base guide. These commands do not configure networking.

```bash
RANK=0
MASTER_ADDR=192.0.2.10
HOST_IP=192.0.2.10
INTERFACE=eth0
MODEL_DIR=/srv/models/Qwen3.8-Flash-Next-NVFP4/ada4da32
CACHE_DIR=/srv/cache/qwen38-flash-next-r37
IMAGE_ID=$(docker image inspect --format '{{.Id}}' sparkring-cache-extension:r37-cache64)
launch_rank() {
  python3 runtime/common/qwen_flash_next.py "$1" \
    --profile profiles/qwen38-flash-next-tp2/sparkcache.json \
    --rank "$RANK" --master "$MASTER_ADDR" --host-ip "$HOST_IP" \
    --interface "$INTERFACE" --image "$IMAGE_ID" \
    --model "$MODEL_DIR" --cache "$CACHE_DIR"
}
launch_rank plan
launch_rank create
```

Set rank 1's values on its host. `create` verifies the actual extension image and
creates a stopped container; it refuses an existing name. Stop competing model
containers explicitly while preserving them for rollback. Start rank 1 before
rank 0, each on its own host:

```bash
docker start qwen-flash-next-sparkcache-tp2-r1 # rank 1 host
docker start qwen-flash-next-sparkcache-tp2-r0 # rank 0 host
docker logs --follow --tail 100 "qwen-flash-next-sparkcache-tp2-r${RANK}"
```

The API serves `Qwen3.8-Flash-Next-NVFP4` on port 8000 without authentication.
Restrict it to trusted networks or an authenticated gateway. Containers do not
autostart; this profile installs no memory guard or boot service.

For an existing deployment, stop both containers, then start rank 1 and rank 0;
do not rerun `create`. Its persistent namespace is
`$CACHE_DIR/persistent/qwen38-flash-next-r37-cache64`. Restart does not clear disk
entries. Incompatible identities or failed integrity checks must recompute.

## Verification scope

The [evidence record](../../performance/records/qwen38-flash-next/r37-sparkcache.json)
distinguishes the serving trial from reproducible-build validation. Bounded
checks cover two-process text and image/video restore, changed-input misses,
single-rank payload corruption recovery, C16 short answers and near-limit text.
Raw timing observations are not a matched performance benchmark. Long-duration
pressure, arbitrary media and concurrent-media capacity remain unqualified.
