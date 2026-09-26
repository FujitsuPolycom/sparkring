# Qwen3.8-Flash-Next NVFP4 QAD on four Sparks

[Qwen3.8-Flash-Next NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/629bc3218833a38b475b719f34aa571666f4a03e) with MTP
speculative decoding and 262K context, served on port 8015 as
`Qwen3.8-Flash-Next-NVFP4-QAD-TP4`. On the Spark connected to your network:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile qwen38-flash-next-qad-tp4
```

The installer applies the [ConnectX driver setting](../../docs/operations/install.md#four-spark-rings)
that four-Spark rings need and repeats it at every boot. [Install SparkRing](../../docs/operations/install.md)
covers requirements, logs and recovery.

| Setting | Installer profile ([config.json](config.json)) | SparkCache profile ([sparkcache.json](sparkcache.json)) |
|---|---|---|
| Image | `dev-20260925-qwendecode-cuda1342-nccl2323-status031` | `shared-2026.09.3` |
| Parallelism | TP4/DCP1 on a four-node ring with hardware-forwarded mesh paths | Same |
| Context / sequences / batch | 262144 / 16 / 8192; no YaRN | Same |
| KV allocation | 24 GiB FP8 per rank; 32-token requested attention blocks | Same |
| Loading / speculation | Managed B12X / MTP3, drafts sampled from the draft distribution (`"draft_sample_method": "probabilistic"`) | Managed B12X / MTP3, greedy drafts |
| Decode weights | MXFP8 target LM head; MXFP8 hyper-connection down/injection projections for batches of at most 16 rows; fused rotary-embedding op | BF16 LM head and hyper-connection projections |
| Collectives | Size-based RoCEnante/NCCL selection with decode all-reduces of up to 64 rows on RoCEnante (`QWEN_DISPATCH_AR_BYTES=327680`); NCCL uses all four ring NIC functions (`NCCL_IB_EXTENDED_IPV4_GIDS=1`) | Size-based selection with decode all-reduces of up to 4 rows on RoCEnante (`QWEN_DISPATCH_AR_BYTES=20480`); extended IPv4 GIDs off |
| Prefill | Hyper-connection token-row ownership (`VLLM_QWEN3_8_HC_PREFILL_MODE=shard`) and checkpoint coalescing | Same |
| Media | Three images / one video, 16 configured frames | Same |
| Caching | vLLM native prefix cache; SparkCache off | SparkCache on; native prefix cache on |
| API | Port 8015, model `Qwen3.8-Flash-Next-NVFP4-QAD-TP4`, no API key | Same |

The [configuration](config.json) owns these settings. GLM-specific mHC/KDA
switches and SIRCL serving switches are disabled for this Qwen profile.
TP2 uses the [two-Spark quickstart](../qwen38-flash-next-tp2/README.md).

## SparkCache profile: manual setup

The [SparkCache selection](../qwen38-flash-next-qad-tp4-sparkcache/README.md),
`qwen38-flash-next-qad-tp4-sparkcache`, adds a disk KV cache and runs on the
[shared-2026.09.3](../../runtime/releases/shared-2026.09.3/README.md) image;
the commands below deploy it with `sparkring compose`.

### Prepare image, model and fabric

The manual commands below render and start the SparkCache profile. Install the
installer profile with `sparkring install`, which also writes each rank's
runtime-binding file and verifies the installer image before any model
downtime.

Complete [setup](../../docs/operations/setup.md) through the host and ring
network steps. Use Bash from the recorded checkout directory on each host.
Use the same SparkRing checkout on every host and pull this image on all four:

```bash
set -euo pipefail
mkdir -p .sparkring
PROFILE=qwen38-flash-next-qad-tp4-sparkcache
python3 scripts/sparkring.py setup show "$PROFILE" --format shell \
  > .sparkring/selection.env
cat .sparkring/selection.env
source .sparkring/selection.env
docker pull --platform linux/arm64 "$IMAGE_REF"
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")
test "$IMAGE_ID" = "$EXPECTED_IMAGE_ID"
```

**Pass:** the pulled image ID equals the publication's generated `EXPECTED_IMAGE_ID`.
No separate R37 parent-image pull is required. The
[publication receipt](../../runtime/releases/shared-2026.09.3/publication.json)
binds the image to its source and installed inventory.

Reuse an existing verified QAD checkpoint, setting `MODEL_DIR` to its actual path.
Otherwise run this download on **each rank**. A separately verified file transfer
can replace a download; verify full shard checksums on every destination:

```bash
# Skip download when these verified weights already exist.
MODEL_DIR="/srv/models/${MODEL_REPO##*/}/${MODEL_REV}"
"$HOME/.venvs/sparkring-download/bin/hf" download "$MODEL_REPO" \
  --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
REPO=$PWD
(cd "$MODEL_DIR" && \
  sha256sum --check "$REPO/profiles/qwen38-flash-next-qad-tp4/SHA256SUMS")
```

The model repository's `main` branch is not a substitute for this QAD revision.
The coordinator verifies metadata, not every weight shard. Mount weights
read-only and keep writable caches outside model directories.

Prepare the [managed ring fabric](../../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md)
and its private mesh site. The [fabric guide](../../spark_transport/fabric/cx7_hairpin_diagonal/README.md)
owns device order, routing and hardware forwarding. The model launcher does not
install those resources. Do not restart fabric controllers under live collectives.

### Render, check and start

Follow the [Compose prerequisites](../../docs/operations/compose.md#prepare-the-hosts).
Create dedicated cache/deployment directories on every host. Copy and edit the
site example; its addresses and fabric hashes are placeholders:

```bash
mkdir -p .sparkring
cp profiles/qwen38-flash-next-qad-tp4/compose/site.example.yaml .sparkring/qwen-qad.site.yaml
# Fill every host, model/cache directory, HCA/GID and prepared fabric identity.
# Keep PROFILE from image selection; fill the site with the actual MODEL_DIR.
PROFILE="$PROFILE_ID"
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

### Local source-image testing

The retained [R37 source-image workflow](../../runtime/images/compositions/lil-r37-qwen-prefill/README.md)
is an explicit developer alternative, not this shared release. It applies to
`qwen38-flash-next-qad-tp4-sparkcache`; the installer profile rejects it,
because its installer image lock selects its image. Selecting it
requires `--local-source-extension lil-r37-qwen-prefill --local-image-id IMAGE_ID`.
Its adapter selects R37 hooks, transport and cache contracts with a separate
namespace. It does not inherit this release's qualification. Do not use local
overrides to substitute an arbitrary image into the published quickstart.

### Stop and rollback

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
