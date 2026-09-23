# GLM-5.3-Flash on two Sparks

Status: **qualified for bounded correctness and restart/restore checks** on
[SparkRing 2026.09.3](../../runtime/releases/shared-2026.09.3/README.md).
Full-context, arbitrary media and long-duration stability are not qualified.

Run GLM-5.3-Flash with TP2/DCP1, MTP3 and SparkCache. Select NVFP4-Spark or
the LIL NVFP4 QAD checkpoint. Both use B12X loading. Defaults are a configured
1,048,576-token context limit, eight sequences, an 8192-token batch and
7.5 GiB KV per node. Configured limits are not full-limit stability claims.

SparkCache is optional for GLM. This guide enables it on 2026.09.3; the
[cache-disabled alternative](../glm53-flash-spark-tp2-dcp1/README.md) uses a
separately pinned R37 procedure with its own settings and validation limits.

<a id="1-prepare-the-pair"></a>
## Prepare both hosts

Complete [host preparation](../../docs/operations/host-preparation.md) and
[pair networking](../../docs/operations/pair-network.md) on both GB10 nodes.
Use the same recorded checkout revision. For prepared hosts, check and reuse their
settings; skip fresh-network configuration. Keep management access separate from
the data fabric and stop unrelated GPU workloads explicitly before serving.

<a id="3-install-the-memory-guard"></a>
The native profile does not install or opt into a custom memory-kill guard.
Linux memory handling remains unchanged. Avoid image builds or large file copies
while measuring inference, and monitor available host memory.

Run the following commands in Bash from the repository root on each node.

<a id="2-download-the-image-and-model"></a>
## Select the image and checkpoint

```bash
set -euo pipefail
mkdir -p .sparkring
VARIANT=nvfp4-spark
python3 scripts/sparkring.py setup show glm53-flash-spark-tp2-dcp1-sparkcache \
  --variant "$VARIANT" --format shell > .sparkring/selection.env
cat .sparkring/selection.env
source .sparkring/selection.env
docker pull --platform linux/arm64 "$IMAGE_REF"
IMAGE=$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")
test "$IMAGE" = "$EXPECTED_IMAGE_ID"
RECORD=$(mktemp -d "$PWD/.sparkring/native-receipt.XXXXXX")
python3 runtime/common/glm_native_candidate.py \
  --release "$RELEASE" --image-id "$IMAGE" --output "$RECORD/image.json"
```

For QAD, change `VARIANT` to `nvfp4-qad` before the selection block above on
both nodes. The helper obtains the matching checkpoint revision from the release
contract. **Pass:** the image ID matches and the verifier writes `image.json`.
Re-running verification uses a fresh receipt directory; it starts no model.

Download the selected checkpoint on **each rank**. Paths below match the host
preparation defaults. If using other storage, repeat the storage check first:

```bash
MODEL_DIR="/srv/models/${MODEL_REPO##*/}/${MODEL_REV}"
CACHE_DIR="/srv/cache/${PROFILE_ID}/${RELEASE}"
"$HOME/.venvs/sparkring-download/bin/hf" download "$MODEL_REPO" \
  --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
mkdir -p "$CACHE_DIR"
```

Spark is not a Spark-QAD checkpoint. A verified existing snapshot can be reused by
setting `MODEL_DIR` to its directory and skipping download. Never overwrite a
checkpoint mounted by a live server. Create writable directories with appropriate
ownership; do not run serving as root merely to bypass path permissions.

The launcher validates configuration and weight-index hashes. Full shard identity
must also be established by the pinned download or an independently verified
existing snapshot. QAD's MXFP8 MTP experts select Humming automatically; its
target weights still use the B12X loader.

<a id="4-set-private-rank-inputs"></a>
## Configure rank-local networking

Set `RANK` to 0 or 1. `MASTER` is a rank-0 address reachable from both nodes.
Copy the site template and replace its three placeholders with this rank's
address and appropriate rendezvous interfaces:

```bash
RANK=0
MASTER=rank0.example
ENV_FILE="$PWD/.sparkring/glm53-rank${RANK}.env"
cp -n runtime/profiles/glm53-flash-spark-tp2/runtime.env.example "$ENV_FILE"
```

The site file contains `VLLM_HOST_IP`, `NCCL_SOCKET_IFNAME` and
`GLOO_SOCKET_IFNAME`. The profile's data transport selects both Socket Direct
functions for one connected cage/DAC. Verify that this mapping matches the hosts.

<a id="5-plan-create-and-start"></a>
## Plan, create and start

```bash
launch_rank() {
  python3 runtime/common/tp2.py "$1" \
    --rank "$RANK" --master "$MASTER" \
    --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" \
    --env-file "$ENV_FILE" --image "$IMAGE" \
    --runtime-receipt "$RECORD/image.json" \
    --target-model-variant "$VARIANT" --sparkcache
}
launch_rank plan
```

Inspect both plans for the same image/checkpoint, the correct ranks, B12X loading,
the prepared RoCEnante transport and no activation blockers. Run
`launch_rank create` on both nodes to create stopped containers. Then run
`launch_rank start` on rank 1, followed by rank 0. Creation is create-only and
does not replace an existing container.

```bash
NAME="sparkring-${RELEASE}-tp2-dcp1-sparkcache-r${RANK}"
# Save this rank's inputs and function for a fresh-shell restart.
for key in RANK MASTER MODEL_DIR CACHE_DIR ENV_FILE IMAGE RECORD VARIANT RELEASE NAME; do
  printf '%s=%q\n' "$key" "${!key}"
done > .sparkring/glm-pair-session.env
declare -f launch_rank >> .sparkring/glm-pair-session.env
docker logs -f --tail 100 "$NAME"
```

Kernel preparation can take many minutes. API readiness alone does not prove
the selected transport is active; inspect both rank logs for fallback or startup
errors and verify the expected prepared-transport initialization.

<a id="6-check-serving"></a>
## Check the API

From rank 0, or a client that can reach its address:

```bash
curl --fail "http://${MASTER}:8000/health"
curl --fail "http://${MASTER}:8000/v1/models"
if test "$VARIANT" = nvfp4-qad; then
  SERVED_MODEL=GLM-5.3-Flash-NVFP4-QAD-TP2
else
  SERVED_MODEL=GLM-5.3-Flash-NVFP4-Spark-TP2
fi
curl --fail --max-time 180 "http://${MASTER}:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply only READY\"}],\"temperature\":0,\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
```

The served name is `GLM-5.3-Flash-NVFP4-Spark-TP2` or
`GLM-5.3-Flash-NVFP4-QAD-TP2`. Use that exact name in client requests.
The API binds `0.0.0.0:8000` without an API key. Restrict access to a trusted
network or place it behind an authenticated gateway; do not expose it directly
to the public Internet.

## Restart the saved containers

On each node, from the same checkout in a fresh Bash shell, restore the saved
inputs before stopping. This file was generated locally; inspect it before sourcing:

```bash
source .sparkring/glm-pair-session.env
docker stop --time 30 "$NAME"
```

After both are stopped, run `launch_rank start` on rank 1 and then rank 0.
Do not rerun `create` to restart. Preserve the cache directory for restart/restore
tests; changing checkpoint or image selection uses a distinct cache namespace.

The R35/R37 and R33 reproduction procedures retain their own image receipts,
settings and evidence. Do not apply this native-image receipt or QAD selection
to those frozen recipes. GLM cache-disabled selections require their separately
documented evidence; this guide qualifies the SparkCache-enabled configuration.

## SparkCache off

Use the [cache-disabled guide](../glm53-flash-spark-tp2-dcp1/README.md) and its
R37 image receipt. Its status is Experimental. Removing `--sparkcache` from
the 2026.09.3 commands above is not the documented cache-off deployment.

## R35 fallback

Use the [R35 procedure](../../docs/operations/r35-local-launch.md) and
[TP2 evidence](../../performance/records/glm53-flash/r35-tp2-sparkcache.md).
Do not mix its receipt or guard requirements with the native commands above.

## R33 fallback

Use the [R33 TP2 reproduction record](../../performance/records/glm53-flash/r33-image020-tp2-sparkcache-20260911.md).
Its recorded checkpoint requires an existing verified copy or operator archive;
do not substitute another revision into the frozen cache identity.

## Evidence and other builds

The [release qualification](../../runtime/releases/shared-2026.09.3/qualification.json)
and [correctness summary](../../runtime/releases/shared-2026.09.3/correctness.json)
cover Spark and QAD separately. Historical records retain their own image,
checkpoint and workload scope.
