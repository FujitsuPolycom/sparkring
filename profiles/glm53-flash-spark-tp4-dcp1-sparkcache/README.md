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

## 1. Prepare the hosts

Follow [four-Spark host setup](../../docs/GLM53_SPARK_MESH_HOST_SETUP.md) from the repository root. Use
the same checkout revision on the controller and all four hosts. Keep the
management network separate from the four-cable data ring. Review network
changes during a maintenance window; do not replace another deployment.

## 2. Select the published image

On an ARM64 Spark with this checkout, run:

```bash
RELEASE=shared-2026.09.3
IMAGE_REF=$(python3 -c 'import json,sys; print(json.load(open("runtime/releases/"+sys.argv[1]+"/publication.json"))["image_reference"])' "$RELEASE")
docker pull --platform linux/arm64 "$IMAGE_REF"
IMAGE=$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")
mkdir -p .sparkring
RECORD=$(mktemp -d "$PWD/.sparkring/glm-native-receipt.XXXXXX")
python3 runtime/common/glm_native_candidate.py \
  --release "$RELEASE" --image-id "$IMAGE" --output "$RECORD/image.json"
```

Copy the resulting `image.json` to the controller and set `SPARKRING_RECEIPT`
to its controller-local path. The installed inventory receipt authenticates
the image; the publication JSON alone is not a runtime receipt.

## 3. Discover and plan DCP1

Run in Bash on the controller from the repository root. Replace the example
management addresses and SSH aliases with the prepared hosts. Rank order is
the order of the four node arguments.

```bash
sr() { python3 scripts/sparkring.py deploy "$@"; }
STATE="$PWD/.sparkring/glm-tp4-deployment"
VARIANT=nvfp4-spark
sr discover --controller-address 192.0.2.10 \
  --node spark0=192.0.2.20 --node spark1=192.0.2.21 \
  --node spark2=192.0.2.22 --node spark3=192.0.2.23 \
  --output "$STATE/inventory.json"
sr plan --inventory "$STATE/inventory.json" --name glm-tp4 \
  --workspace /srv/sparkring/glm-tp4 --preserve-existing-network \
  --image-receipt "$SPARKRING_RECEIPT" \
  --runtime-profile tp4-dcp1-sparkcache --target-model-variant "$VARIANT" \
  --output "$STATE/preparation.json"
sr network-plan --preparation "$STATE/preparation.json" \
  --inventory "$STATE/inventory.json" --output "$STATE/network-plan.json"
```

Set `VARIANT=nvfp4-qad` for the QAD checkpoint. The target catalog pins Spark
to `a608241037e4c2565356bff7ca293f2133888f88` and QAD to
`175ae8ce3b5af842b0d0140dbeb43e9cfc557c49`. Revision-specific cache identities
prevent sharing unverified KV state between these checkpoints.

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

DCP4 and cache-disabled GLM configurations retain separate image selections
and evidence. They do not inherit this cache-enabled DCP1 qualification.

## DCP4 alternative

Use the separately pinned [R33 DCP4 reproduction procedure](../../performance/records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md#reproduction-overlay-and-quickstart).
It does not qualify DCP4 on 2026.09.3. Do not apply its entrypoint overlay to
the native image or edit a staged DCP1 deployment into DCP4.

## Validation and results

The [release qualification](../../runtime/releases/shared-2026.09.3/qualification.json)
and [correctness summary](../../runtime/releases/shared-2026.09.3/correctness.json)
cover Spark and QAD separately. Existing [R37](../../performance/records/glm53-flash/r37-tp4-source-upgrade.md)
and [R33](../../performance/records/glm53-flash/r33-image020-tp4-sparkcache-20260911.md)
records retain their own scope. A factory-reset, blank-host deployment has not
been requalified; hardware checks used prepared rings and verified model copies.
