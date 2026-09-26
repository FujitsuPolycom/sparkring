# Qwen3.8-Flash-Next NVFP4 QAD on four Sparks

[Qwen3.8-Flash-Next NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/629bc3218833a38b475b719f34aa571666f4a03e)
(QAD checkpoint, revision `629bc3218833`) on a ring of four DGX Sparks, with
MTP speculative decoding and 262K context. Node A serves the API on port 8015
as `Qwen3.8-Flash-Next-NVFP4-QAD-TP4`, with no API key. Status: Development.

On the Spark connected to your network:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile qwen38-flash-next-qad-tp4
```

The installer applies the [ConnectX driver setting](../../docs/operations/install.md#four-spark-rings)
that four-Spark rings need and repeats it at every boot. [Install SparkRing](../../docs/operations/install.md)
covers requirements, logs and recovery. Per-rank Compose files:
[Compose](compose/README.md).

## Settings

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

## Performance

One four-Spark ring, 512-token single-stream requests at temperature 0;
prefill is one cold prompt.

| Deployment | Decode prose / code / JSON (tokens/s) | Prefill 16K / 64K (tokens/s) |
|---|---|---|
| `sudo sparkring install` | 87.9 / 126.8 / 142.2 | 5,003 / 4,723 |
| `install.sh` | 86.0 / 127.0 / 137.2 | 4,977 / 4,694 |

- At temperature 1.0, the checkpoint's default sampling, prose decodes at
  78–79 tokens/s.
- NCCL uses all four ring NIC functions only with
  `NCCL_IB_EXTENDED_IPV4_GIDS=1`, which raises prefill from 4,659 to 5,261
  tokens/s at 16K and from 4,224 to 4,721 at 64K.
- Sending decode all-reduces of up to 64 rows over RoCEnante instead of NCCL
  lowers the steady-state step at 16 concurrent requests from 101.5 to 94.5 ms.
- With kernels already tuned, the model starts in about 4 minutes
  (234.2–244.6 s).

Measurements: [installer tuning record](../../performance/records/qwen38-flash-next/installer-tuning-20260925.md)
and [installation record](../../performance/records/images/dev-20260925-qwendecode-installer-profiles-20260926.md).

## SparkCache profile: manual setup

The [SparkCache profile](../qwen38-flash-next-qad-tp4-sparkcache/README.md),
`qwen38-flash-next-qad-tp4-sparkcache`, adds a disk KV cache and runs on the
[shared-2026.09.3](../../runtime/releases/shared-2026.09.3/README.md) image.
`sparkring install` does not deploy it; these commands deploy it with
`sparkring compose`.

### Prepare image, model and fabric

Complete [setup](../../docs/operations/setup.md) through the host and ring
network steps. Then, on every host, in Bash from the same SparkRing checkout:

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

Set `MODEL_DIR` to an existing copy of the checkpoint, or download it (about
106 GB) on each rank, and check it:

```bash
# Skip the download when these weights already exist.
MODEL_DIR="/srv/models/${MODEL_REPO##*/}/${MODEL_REV}"
"$HOME/.venvs/sparkring-download/bin/hf" download "$MODEL_REPO" \
  --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
REPO=$PWD
(cd "$MODEL_DIR" && \
  sha256sum --check "$REPO/profiles/qwen38-flash-next-qad-tp4/SHA256SUMS")
```

`SHA256SUMS` lists the 48 files the revision needs; the coordinator itself
checks only checkpoint metadata. A `main`-branch download made after
2026-09-16 20:03 UTC has the same weights but a `config.json` for another
model type; replace that file as the [TP2 Compose page](../qwen38-flash-next-tp2/compose/README.md)
describes. Keep writable caches outside model directories.

The model commands do not set up the fabric. Prepare the
[managed ring fabric](../../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md) and
its private mesh site ([fabric guide](../../spark_transport/fabric/cx7_hairpin_diagonal/README.md)),
and do not restart fabric controllers while a model runs.

### Render, check and start

Follow the [Compose prerequisites](../../docs/operations/compose.md#prepare-the-hosts),
create the cache and deployment directories on every host, and fill in a copy
of the site example (its addresses and hashes are placeholders):

```bash
mkdir -p .sparkring
cp profiles/qwen38-flash-next-qad-tp4/compose/site.example.yaml .sparkring/qwen-qad.site.yaml
# Edit .sparkring/qwen-qad.site.yaml; set each rank's model directory to its MODEL_DIR.
DEPLOYMENT=.sparkring/deployments/qwen-qad
python3 scripts/sparkring.py compose render "$PROFILE" \
  --site .sparkring/qwen-qad.site.yaml --output "$DEPLOYMENT"
python3 scripts/sparkring.py compose check --deployment "$DEPLOYMENT"
python3 scripts/sparkring.py compose check --deployment "$DEPLOYMENT" --hosts
python3 scripts/sparkring.py compose start --deployment "$DEPLOYMENT"
```

`start` prints a plan; repeat it with `--approve` and the printed hash to
create the containers, start the workers, then rank 0, and wait for the API.
Host checks need `sudo -n`. Stop other GPU workloads first; the coordinator
does not replace them.

On rank 0, with the container name from `rank0/container.json`:

```bash
docker logs --follow --tail 100 sr-qwen-qad-example-r0
curl --fail http://127.0.0.1:8015/health
curl --fail http://127.0.0.1:8015/v1/models
curl --fail http://127.0.0.1:8015/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next-NVFP4-QAD-TP4","messages":[{"role":"user","content":"Reply only READY"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

- The API has no key: restrict it to trusted clients or an authenticated
  gateway. Disk cache entries are not isolated by request `cache_salt`; use
  separate deployments for tenant isolation.
- The `SPARKRING STARTUP AUDIT` banner lists configuration warnings before the
  API is ready. `QWEN_HC_PREFILL mode=shard` on every rank shows that sharded
  prefill ran; short requests may not use it.

### Local source-image testing

For developer tests of the [R37 source image](../../runtime/images/compositions/lil-r37-qwen-prefill/README.md),
render the SparkCache profile with
`--local-source-extension lil-r37-qwen-prefill --local-image-id IMAGE_ID`.
The installer profile rejects these options.

### Stop and rollback

```bash
python3 scripts/sparkring.py compose stop --deployment "$DEPLOYMENT"
# Review and repeat with the printed --approve hash.
```

`start` only creates containers. To restart stopped ones, check each
container's image and deployment label on its host, then start it:

```bash
RANK=1 # set to this host's rank
NAME="sr-qwen-qad-example-r${RANK}" # use the name in rankN/container.json
CID=$(docker inspect --format '{{.Id}}' "$NAME")
docker inspect --format '{{.Image}} {{json .Config.Labels}}' "$CID"
# After checking the image and the io.sparkring.deployment label:
docker start "$CID"
```

Start ranks 1, 2 and 3 before rank 0, then repeat the health checks. To roll
back, stop all four ranks first; see
[stop and recover](../../docs/operations/compose.md#stop-and-recover).
