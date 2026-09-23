# GLM-5.3-Flash on four Sparks

Status: **qualified for bounded correctness and restart/restore checks** on
[SparkRing 2026.09.3](../../runtime/releases/shared-2026.09.3/README.md).
Full-context, arbitrary media and long-duration stability are not qualified.

Run GLM-5.3-Flash with TP4/DCP1, MTP3 and SparkCache on a prepared four-Spark
ring. Select NVFP4-Spark or LIL NVFP4 QAD explicitly. Both use the B12X target
loader; the QAD draft uses Humming for its MXFP8 experts. Spark is not QAD.

The configured context limit is 1,048,576 tokens and KV allocation is 24 GiB
per rank. Bounded qualification covers requests through 128K at concurrency
1/2/4/8, basic text/media checks and physical-rank cache restore. It does not
establish full-context or long-duration stability.

SparkCache is optional for GLM. This guide enables it on 2026.09.3; the
[cache-disabled alternative](../glm53-flash-spark-tp4-dcp1/README.md) uses a
separately pinned R37 procedure with its own settings and validation limits.

## 1. Prepare the hosts

Follow [setup](../../docs/operations/setup.md) and the
[four-Spark host procedure](../../docs/GLM53_SPARK_MESH_HOST_SETUP.md), sections 1–7. Use
the same checkout revision on the controller and all four hosts. Keep the
management network separate from the four-cable data ring. Review network
changes during a maintenance window; do not replace another deployment.

## 2. Select the published image

On **rank 0**, the default controller, in Bash from the matching checkout, run:

```bash
set -euo pipefail
mkdir -p .sparkring
VARIANT=nvfp4-spark
python3 scripts/sparkring.py setup show glm53-flash-spark-tp4-dcp1-sparkcache \
  --variant "$VARIANT" --format shell > .sparkring/selection.env
cat .sparkring/selection.env
source .sparkring/selection.env
docker pull --platform linux/arm64 "$IMAGE_REF"
IMAGE=$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")
test "$IMAGE" = "$EXPECTED_IMAGE_ID"
RECORD=$(mktemp -d "$PWD/.sparkring/glm-native-receipt.XXXXXX")
python3 runtime/common/glm_native_candidate.py \
  --release "$RELEASE" --image-id "$IMAGE" --output "$RECORD/image.json"
SPARKRING_RECEIPT="$RECORD/image.json"
```

For QAD, set `VARIANT=nvfp4-qad` before this block. **Pass:** image identity matches
and `image.json` is written. Rank 0 already owns that receipt; no transfer is
needed. An external controller must receive the file and set `SPARKRING_RECEIPT`
to its local absolute path. The publication JSON alone is not a runtime receipt.

## 3. Discover and plan DCP1

Run in Bash on the controller from the repository root. Replace the example
management addresses and SSH aliases with the prepared hosts. Rank order is
the order of the four node arguments.

```bash
sr() { python3 scripts/sparkring.py deploy "$@"; }
STATE="$PWD/.sparkring/glm-tp4-deployment"
sr discover --controller-address 192.0.2.10 \
  --node spark-r0=192.0.2.10 --node spark-r1=192.0.2.21 \
  --node spark-r2=192.0.2.22 --node spark-r3=192.0.2.23 \
  --output "$STATE/inventory.json"
sr plan --inventory "$STATE/inventory.json" --name glm-tp4 \
  --workspace /srv/sparkring/glm-tp4 --preserve-existing-network \
  --image-receipt "$SPARKRING_RECEIPT" \
  --runtime-profile tp4-dcp1-sparkcache --target-model-variant "$VARIANT" \
  --output "$STATE/preparation.json"
sr network-plan --preparation "$STATE/preparation.json" \
  --inventory "$STATE/inventory.json" --output "$STATE/network-plan.json"
```

Keep `VARIANT` from the image-selection block. `setup show` reports its pinned
checkpoint revision. Rank 0's controller address and rank-0 node address are the
same management address in this default arrangement. Revision-specific cache
identities prevent sharing unverified KV state between checkpoints.

If the selected image and checkpoint are already installed on every node, add
`--reuse-existing-image` and four `--existing-model-root /absolute/model/path`
arguments to `sr plan`, in rank order. Each path is resolved on its corresponding
host. Use the complete directory for the selected pinned revision, not a parent
directory containing multiple snapshots. The planner verifies the inputs before
reuse. Do not omit the image receipt or point it at a different checkpoint.

## 4. Apply, stage and start

Continue at “Apply networking, then verify it” in
[the deployment suite](../../docs/operations/deployment-suite.md), keeping this preparation file and image
receipt. Review plans before applying them. Stage the pinned image, checkpoint,
transport and source; create stopped containers; install managed services;
verify native communication; then start through the coordinator.

Use coordinated stop/recover for subsequent operation, not direct rank startup.
Native readiness allows 1800 seconds for cold preparation. Inspect all four
rank logs for the selected image, checkpoint, DCP1, MTP3 and SparkCache connector.
Health alone does not prove cache restore; follow the semantic and cache checks
in [profile validation](../../docs/operations/profile-validation.md).

## 5. Verify the selected configuration

The API listens on rank 0, port 8015, with the model name
`GLM-5.3-Flash-NVFP4-Spark-TP4` or `GLM-5.3-Flash-NVFP4-QAD-TP4`.
It binds all interfaces without an API key. Restrict access to a trusted network
or an authenticated gateway; do not expose it directly to the Internet.

After managed readiness, run on **rank 0**:

```bash
curl --fail http://127.0.0.1:8015/health
curl --fail http://127.0.0.1:8015/v1/models
if test "$VARIANT" = nvfp4-qad; then
  SERVED_MODEL=GLM-5.3-Flash-NVFP4-QAD-TP4
else
  SERVED_MODEL=GLM-5.3-Flash-NVFP4-Spark-TP4
fi
curl --fail --max-time 180 http://127.0.0.1:8015/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply only READY\"}],\"temperature\":0,\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
```

Require the expected model and a successful generation. This is the installation
smoke test; physical cache restore and workload qualification are separate checks.

DCP4 and cache-disabled GLM configurations retain separate image selections
and evidence. They do not inherit this cache-enabled DCP1 qualification.

## DCP4 alternative

Use the separately pinned [R33 DCP4 reproduction procedure](../../performance/records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md#reproduction-overlay-and-quickstart).
It does not qualify DCP4 on 2026.09.3. Do not apply its entrypoint overlay to
the native image or edit a staged DCP1 deployment into DCP4.

For that R33 procedure, select `tp4-dcp4-sparkcache` for persistent caching or
`tp4-dcp4` without it in the private site's `runtime_profile`. Use the R33 image
receipt and contract paths named by the reproduction guide, not this native
image receipt. The deployment-suite planner above remains DCP1-only.

## Validation and results

The [release qualification](../../runtime/releases/shared-2026.09.3/qualification.json)
and [correctness summary](../../runtime/releases/shared-2026.09.3/correctness.json)
cover Spark and QAD separately. Existing [R37](../../performance/records/glm53-flash/r37-tp4-source-upgrade.md)
and [R33](../../performance/records/glm53-flash/r33-image020-tp4-sparkcache-20260911.md)
records retain their own scope. A factory-reset, blank-host deployment has not
been requalified; hardware checks used prepared rings and verified model copies.
