# Qwen3.8-Flash-Next NVFP4 QAD on two Sparks

Status: the installer profile `qwen38-flash-next-tp2` ([config.json](config.json))
is **implemented**. Install it with
`sudo sparkring install --profile qwen38-flash-next-tp2`;
[Install SparkRing](../../docs/operations/install.md) covers the package, host
requirements, downloads and network exposure. The profile serves revision
`629bc3218833a38b475b719f34aa571666f4a03e` (Hugging Face branch
`qad-step-4000`) of `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` on the
installer image `dev-20260925-qwendecode-cuda1342-nccl2323-status031`
(`ghcr.io/fujitsupolycom/sparkring@sha256:451c5e23a90e0df2fc904e8851aab12c3ec9ffdcd1258b6f14cf502222e46b5f`).
On one pair, `sparkring install` from source revision `e75451a671a3`
installed it with these settings, including probabilistic drafting, and its
counting, arithmetic and code checks passed; the
[installer tuning record](../../performance/records/qwen38-flash-next/installer-tuning-20260925.md)
gives that installation's decode and prefill rates and the measurements behind
each setting. Its serving is not qualified.

The SparkCache profile `qwen38-flash-next-tp2-sparkcache`
([sparkcache.json](sparkcache.json)) is **qualified for bounded correctness and
restart checks** on the same checkpoint revision and the
[SparkRing shared-2026.09.3](../../runtime/releases/shared-2026.09.3/README.md)
image. The manual commands in this guide create its containers. The manual
launcher, `runtime/common/qwen_flash_next.py`, plans but refuses to check or
create containers for the installer profile, whose image is admitted only
through the installer image lock; install that profile with
`sparkring install`. Complete the
[host preparation](../../docs/operations/host-preparation.md) and
[pair network procedure](../../docs/operations/pair-network.md) before the
manual commands; the launcher does not configure networking. Prepared hosts
verify and reuse their existing network instead of running fresh-network
configuration.

Both profiles use aligned checkpoints, managed B12X loading, Qwen checkpoint
coalescing, compact MTP and projection overlap. Request-boundary caching and
request-salt isolation are not enabled by either selection.

| Setting | Installer profile ([config.json](config.json)) | SparkCache profile ([sparkcache.json](sparkcache.json)) |
|---|---|---|
| Image | `dev-20260925-qwendecode-cuda1342-nccl2323-status031` | `shared-2026.09.3` |
| Parallelism | TP2/DCP1; one DAC, both Socket Direct functions | Same |
| Context / sequences / batch | 262144 / 16 / 8192; no YaRN | Same |
| KV | 24 GiB FP8 per rank | 24 GiB FP8 per rank; separate from host cache buffers |
| Loading / speculation | Managed B12X / MTP3, drafts sampled from the draft distribution (`"draft_sample_method": "probabilistic"`) | Managed B12X / MTP3, greedy drafts |
| Decode weights | MXFP8 target LM head; MXFP8 hyper-connection down/injection projections for batches of at most 16 rows; fused rotary-embedding op | BF16 LM head and hyper-connection projections |
| Prefill | Hyper-connection token-row ownership (`VLLM_QWEN3_8_HC_PREFILL_MODE=shard`) with the image's `qwen-collectives` and `qwen4-prefill` features | Hyper-connection row sharding off |
| Decode collectives | All-reduces of up to 64 rows on RoCEnante (`QWEN_DISPATCH_AR_BYTES=327680`) | No Qwen dispatch setting |
| Media | Three images, one video; 16 configured video frames | Same |
| Caching | vLLM native prefix cache; SparkCache off | 4 GiB disk per rank, reclaim toward 3 GiB; about 1.25 GiB data buffers per rank |
| API | Port 8000, model `Qwen3.8-Flash-Next-NVFP4-QAD-TP2`, no API key | Same |

## Image and checkpoint

Use the same SparkRing checkout on both nodes. Set these variables in Bash;
`MODEL_DIR` must contain the verified checkpoint, and `CACHE_DIR` must be a
separate writable directory. The block selects the SparkCache profile, whose
image the manual launcher admits. Make the same choice on both ranks before
creating containers.

```bash
REPO=$PWD
set -euo pipefail
PROFILE_ID=qwen38-flash-next-tp2-sparkcache
mkdir -p .sparkring
python3 scripts/sparkring.py setup show "$PROFILE_ID" --format shell \
  > .sparkring/selection.env
cat .sparkring/selection.env
source .sparkring/selection.env
PROFILE="$PROFILE_CONFIG"
if test "$SPARKCACHE_ENABLED" = 1; then
  CONTAINER_PREFIX=qwen-flash-next-sparkcache-tp2
else
  CONTAINER_PREFIX=qwen-flash-next-tp2
fi
MODEL_DIR="/srv/models/${MODEL_REPO##*/}/${MODEL_REV}"
CACHE_DIR="/srv/cache/${PROFILE_ID}/${RELEASE}"

# No separate parent-image pull is required for native-image verification.
docker pull --platform linux/arm64 "$IMAGE_REF"
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")
test "$IMAGE_ID" = "$EXPECTED_IMAGE_ID"
mkdir -p "$CACHE_DIR"
```

Reuse an existing verified model copy. Otherwise download the approximately
106 GB checkpoint once per rank, or transfer the verified files over the data fabric:

```bash
# Skip this download when MODEL_DIR already contains the verified checkpoint.
"$HOME/.venvs/sparkring-download/bin/hf" download "$MODEL_REPO" \
  --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
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

Those addresses match the fresh-pair example. On rank 1 set `RANK=1` and
`HOST_IP=198.18.20.2`; both ranks retain rank 0's `MASTER_ADDR`. For prepared
hosts, substitute their actual primary fabric addresses and discovered interface.
Inspect both plans before running `launch_rank create` on **each rank**.

The defaults select `rocep1s0f0` and `roceP2p1s0f0`, the two PCI-domain views
of cage p0. Confirm the cable/device mapping. The bootstrap interface is a
separate input, not an RDMA device list. `plan` is offline; `create` verifies
the image and refuses existing names. Neither action stops a running workload.

## Controlled startup

During an authorized test window, stop competing GPU workloads explicitly and
preserve them for rollback. Start rank1, then rank0, on their respective hosts:

```bash
# Save rank-local inputs on EACH rank before starting either container.
for key in RANK MASTER_ADDR HOST_IP INTERFACE PROFILE IMAGE_ID MODEL_DIR CACHE_DIR CONTAINER_PREFIX; do
  printf '%s=%q\n' "$key" "${!key}"
done > .sparkring/qwen-pair-session.env
declare -f launch_rank >> .sparkring/qwen-pair-session.env
```

Run on **rank 1 first**, then **rank 0**, in their respective shells:

```bash
docker start "${CONTAINER_PREFIX}-r${RANK}"
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

Do not rerun `create`. From the same checkout in a fresh Bash shell on each rank,
restore the locally generated inputs, then stop the container. Inspect the file
before sourcing it. Use the startup
commands above in rank1/rank0 order. Disk cache entries remain intact:

```bash
source .sparkring/qwen-pair-session.env
docker stop --timeout 30 "${CONTAINER_PREFIX}-r${RANK}"
```

### Cache-disabled alternative

The cache-disabled configuration of this checkpoint is the installer profile
`qwen38-flash-next-tp2`; install it with
`sudo sparkring install --profile qwen38-flash-next-tp2`. The manual launcher
does not create its containers. Changing a variable does not change an existing
container; use the documented stop/create sequence for a new selection.

## Evidence and remaining checks

Configuration is owned by [sparkcache.json](sparkcache.json) and
[config.json](config.json), not this table. Capacity overrides are intentionally
not accepted. The [release qualification](../../runtime/releases/shared-2026.09.3/qualification.json)
records passing bounded short/16K text, finite-score, synthetic image/video,
concurrent-request and retained-restart checks on the shared-2026.09.3 image
for revision `629bc3218833`, both with and without SparkCache. That evidence
covers the shared-2026.09.3 configuration with a BF16 LM head; it does not
transfer to the installer profile, whose image and decode settings differ. The
cache-enabled profile also restored two fixtures on every rank after restart.
The [correctness summary](../../runtime/releases/shared-2026.09.3/correctness.json)
owns exact image identities, case counts and evidence hashes. The installer
profile's evidence is in its `profile.json` evidence scope and the
[installer tuning record](../../performance/records/qwen38-flash-next/installer-tuning-20260925.md).

These checks do not qualify arbitrary/high-resolution video, C16 multimedia,
sixteen simultaneous full-context requests or prolonged store-pressure stability.
Sixteen video frames are not sixteen visual tokens. A 512 MiB capture slot can
reject a snapshot below the configured 65536-token span ceiling; serving must
continue without optional cache publication.

[Release sources and rollback](../../runtime/releases/shared-2026.09.3/README.md).
[Generated Compose deployments](../../docs/operations/compose.md) cover both
profiles; `sparkring install` is the installer profile's supported entry point.
Historical R37 evidence remains attached to its original image and is not
reattributed to this release.
