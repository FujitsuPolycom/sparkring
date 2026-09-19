# Qwen3.8-Flash-Next NVFP4 QAD on four Sparks

Status: **qualified for bounded correctness and restart checks**. This guide selects
[SparkRing shared-2026.09.3](../../runtime/releases/shared-2026.09.3/README.md).
The cache-disabled profile and [SparkCache selection](../qwen38-flash-next-qad-tp4-sparkcache/README.md)
use the same immutable image. The [qualification record](../../runtime/releases/shared-2026.09.3/qualification.json)
records bounded short/16K text, finite-score, synthetic media, concurrent-request and retained-restart checks, plus physical cache restore
with SparkCache. The [correctness summary](../../runtime/releases/shared-2026.09.3/correctness.json)
owns case counts and evidence hashes. These checks do not qualify full-context,
C16-pressure stability or performance.

| Setting | Selection |
|---|---|
| Checkpoint | Qwen3.8-Flash-Next NVFP4 QAD, revision `629bc3218833a38b475b719f34aa571666f4a03e` |
| Parallelism | TP4/DCP1 on a four-node ring with hardware-forwarded mesh paths |
| Context / sequences / batch | 262144 / 16 / 8192; no YaRN |
| KV allocation | 24 GiB FP8 per rank; 32-token requested attention blocks |
| Loading / speculation | Managed B12X / MTP3 |
| Collectives | Size-based RoCEnante selection and dual-domain NCCL |
| Prefill | HC fusion and row sharding, checkpoint coalescing, paired QSA scoring and B12X #394 |
| Media | Three images / one video, 16 configured frames |
| SparkCache | Optional; choose its profile below. Native prefix caching stays enabled in both |

The [configuration](config.json) owns these settings. GLM-specific mHC/KDA
switches and SIRCL serving switches are disabled for this Qwen profile.
TP2 uses the [two-Spark quickstart](../qwen38-flash-next-tp2/README.md).

## Prepare image, model and fabric

Complete the [host prerequisites](../../docs/operations/prerequisites.md).
Use the same SparkRing checkout on every host and pull this image on all four:

```bash
IMAGE_REF='ghcr.io/fujitsupolycom/sparkring@sha256:2375f876bc9ea065e85ae10cebad7a8db8a2ec0e6862b4441c269c5bf56365c6'
docker pull --platform linux/arm64 "$IMAGE_REF"
docker image inspect --format '{{.Id}}' "$IMAGE_REF"
```

The image ID must be
`sha256:b13ac9630ecdfbc63c99e952bb23a04c7dc8688ff172cea1944dafd0aed44499`.
No separate R37 parent-image pull is required. The
[publication receipt](../../runtime/releases/shared-2026.09.3/publication.json)
binds the image to its source and installed inventory.

Reuse an existing verified QAD checkpoint. Otherwise download the pinned revision
once and transfer it to the other hosts over the data fabric:

```bash
# Skip download when these verified weights already exist.
hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
  --revision 629bc3218833a38b475b719f34aa571666f4a03e \
  --local-dir /srv/models/Qwen3.8-Flash-Next-NVFP4-QAD/629bc3218833
REPO=$PWD
(cd /srv/models/Qwen3.8-Flash-Next-NVFP4-QAD/629bc3218833 && \
  sha256sum --check "$REPO/profiles/qwen38-flash-next-qad-tp4/SHA256SUMS")
```

The model repository's `main` branch is not a substitute for this QAD revision.
The coordinator verifies metadata, not every weight shard. Mount weights
read-only and keep writable caches outside model directories.

Prepare the [managed ring fabric](../../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md)
and its private mesh site. The [fabric guide](../../spark_transport/fabric/cx7_hairpin_diagonal/README.md)
owns device order, routing and hardware forwarding. The model launcher does not
install those resources. Do not restart fabric controllers under live collectives.

## Render, check and start

Follow the [Compose prerequisites](../../docs/operations/compose.md#prepare-the-hosts).
Create dedicated cache/deployment directories on every host. Copy and edit the
site example; its addresses and fabric hashes are placeholders:

```bash
mkdir -p .sparkring
cp profiles/qwen38-flash-next-qad-tp4/compose/site.example.yaml .sparkring/qwen-qad.site.yaml
# Fill every host, model/cache directory, HCA/GID and prepared fabric identity.
PROFILE=qwen38-flash-next-qad-tp4
# To enable persistence instead:
# PROFILE=qwen38-flash-next-qad-tp4-sparkcache
DEPLOYMENT=.sparkring/deployments/qwen-qad
python3 scripts/sparkring.py compose render "$PROFILE" \
  --site .sparkring/qwen-qad.site.yaml --output "$DEPLOYMENT"
python3 scripts/sparkring.py compose check --deployment "$DEPLOYMENT"
python3 scripts/sparkring.py compose check --deployment "$DEPLOYMENT" --hosts
python3 scripts/sparkring.py compose start --deployment "$DEPLOYMENT"
```

Review the printed plan and repeat `start` with `--approve` and its exact hash.
The coordinator creates stopped containers, starts workers before rank0, and
checks readiness. TP4 host checks require `sudo -n` for root-owned fabric state.
Stop competing GPU workloads only with their owner's approval; the coordinator
does not replace them.

On rank0, use the container name recorded in `rank0/container.json`:

```bash
docker logs --follow --tail 100 sr-qwen-qad-example-r0
curl --fail http://127.0.0.1:8015/health
curl --fail http://127.0.0.1:8015/v1/models
curl --fail http://127.0.0.1:8015/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next-NVFP4-QAD-TP4","messages":[{"role":"user","content":"Reply only READY"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

For a remote client, replace `127.0.0.1` with rank0's reachable address.
The API binds without a configured key: restrict it to trusted clients or an
authenticated gateway. SparkCache request-salt isolation is not implemented in
this image; use separate deployments/cache namespaces for tenant isolation.

Before API readiness, `SPARKRING STARTUP AUDIT` reports source/configuration
checks and warnings without changing flags. `HC_ROUTE_VERIFIED` proves that the
dispatcher is reachable, not that it ran. An eligible real TP4 prefill logs
`QWEN_HC_PREFILL mode=shard` on every rank. A short request or mixed
prefill/decode batch may not exercise that path.

## Local source-image testing

The retained [R37 source-image workflow](../../runtime/images/compositions/lil-r37-qwen-prefill/README.md)
is an explicit developer alternative, not this shared release. Selecting it
requires `--local-source-extension lil-r37-qwen-prefill --local-image-id IMAGE_ID`.
Its adapter selects R37 hooks, transport and cache contracts with a separate
namespace. It does not inherit this release's qualification. Do not use local
overrides to substitute an arbitrary image into the published quickstart.

## Stop and rollback

```bash
python3 scripts/sparkring.py compose stop --deployment "$DEPLOYMENT"
# Review and repeat with the printed --approve hash.
```

The coordinator's `start` is create-only; do not rerun it to restart retained
containers. On each host, inspect the name from its `rankN/container.json` and
verify its image and deployment label against the saved deployment:

```bash
RANK=1 # set to this host's rank
NAME="sr-qwen-qad-example-r${RANK}" # use your site's actual name
CID=$(docker inspect --format '{{.Id}}' "$NAME")
docker inspect --format '{{.Image}} {{json .Config.Labels}}' "$CID"
# After checking the recorded image and io.sparkring.deployment label:
docker start "$CID"
```

Restart workers 1/2/3 before rank0, with no competing GPU workload. Repeat the
health/model checks above. These commands retain the exact containers and caches.
Keep the prior deployment directory, image and cache namespace for rollback.
Restore it only after all four replacement ranks have stopped, without changing
fabric ownership beneath a live model. See [stop and recover](../../docs/operations/compose.md#stop-and-recover).
