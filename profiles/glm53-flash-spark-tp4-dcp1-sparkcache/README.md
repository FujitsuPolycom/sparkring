# GLM-5.3-Flash on four Sparks

[Shared-spec Compose creation](compose/README.md) is available for the R37 path
below. It creates stopped containers and retains managed startup/recovery.
Configuration equivalence is tested; GLM serving through this backend still
requires hardware acceptance.

Run [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark/tree/ec0c3ce05787aa471121235af483b098f11d3570)
with MTP3 on a four-Spark ring. **DCP1 is the default; DCP4 is an alternative.**
Context defaults to 1M tokens. SparkCache is optional. The primary procedure
selects the published **R37 ARM64** image. Status: **Experimental**; bounded
TP4/DCP1 cache-on checks passed, not long-duration stability qualification.

| Selection | KV allocation per rank | KV evidence | Procedure |
|---|---:|---:|---|
| DCP1, with or without SparkCache | 24 GiB | [2.3M sizing reference](../../performance/capacity-references.md) | Deployment suite below |
| DCP4, with or without SparkCache | 24 GiB | 8.4M with SparkCache (R33) | [R33 DCP4 reproduction](#dcp4-alternative) |

Capacity depends on enabled features. The validated results are scoped to the
[image and workload records](#validation-and-results), not every possible configuration.
For a switch-connected fabric, use the [switched setup](../glm53-flash-spark-tp4-switched/README.md).

The [NVIDIA NVFP4 target](../glm53-nvidia-nvfp4.md) is an opt-in Development
adaptation with separate model pins and evidence. NVFP4-Spark stays the default.

## 1. Prepare the hosts

Use a Linux or WSL controller with Python 3, PyYAML, Git, SSH and SCP. Run the
commands from the root of this checkout. Keep every host on the same source
revision; record it with `git rev-parse HEAD`. Do not switch to a retired
integration branch to follow this guide.

Complete [four-Spark host setup](../../docs/GLM53_SPARK_MESH_HOST_SETUP.md),
including Docker/NVIDIA support, independent management access, SSH aliases,
noninteractive sudo, the four-cable ring, and the required RDMA functions.
Use a maintenance window for networking changes; stop dependent containers
and RDMA users first. This setup must not replace an unrelated deployment.

## 2. Select the published image

Use an ARM64 host to record the image, then retain its receipt on the controller.
Set `IMAGE_HOST` to a configured Spark SSH alias from host setup. These commands
pull and verify the image without loading a model; staging later pulls the same
immutable registry digest on each host:

```bash
IMAGE_HOST=spark0
IMAGE_REF='ghcr.io/fujitsupolycom/sparkring@sha256:f5a7e01c6112c8ef85a51b24bfacfd3934ee9cfff06b7e8c72abcf5d90b50270'
RECORD=$(mktemp -d "$HOME/sparkring-r37-receipt.XXXXXX")
ssh "$IMAGE_HOST" "sudo -n docker pull --platform linux/arm64 '$IMAGE_REF'"
IMAGE=$(ssh "$IMAGE_HOST" "sudo -n docker image inspect --format '{{.Id}}' '$IMAGE_REF'")
ssh "$IMAGE_HOST" "sudo -n docker run --rm --network none --pull never --entrypoint cat '$IMAGE' /opt/sparkring/receipts/candidate-installed.json" > "$RECORD/installed.json"
ssh "$IMAGE_HOST" "sudo -n docker run --rm --network none --pull never '$IMAGE' verify" > "$RECORD/verification.json"
python3 runtime/common/candidate.py --composition lil-r37-glm-spark \
  --image-id "$IMAGE" --image-reference "$IMAGE_REF" \
  --installed-receipt "$RECORD/installed.json" \
  --verification "$RECORD/verification.json" --output "$RECORD/image.json"
SPARKRING_RECEIPT="$RECORD/image.json"
```

The runtime receipt selects this image; `publication.json` is distribution
metadata and cannot replace it. [Image building](../../runtime/images/README.md)
is a separate workflow and is not needed for this quickstart.

The human-readable tag is `ghcr.io/fujitsupolycom/sparkring:r37-arm64-beeb32253aa7`.
Use the digest above for reproducibility. For the previous image, use the
[R35 launch procedure](../../docs/operations/r35-local-launch.md); preserve its
separate receipt and cache namespace. Catalog records retain their original
release identities and evidence instead of being rewritten as R37 qualification.

## 3. Discover and plan DCP1

The R37 deployment stages NVFP4-Spark revision
[`ec0c3ce05787aa471121235af483b098f11d3570`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark/tree/ec0c3ce05787aa471121235af483b098f11d3570)
from the [model pins](../glm53-target-variants.json). Its weights match the
recorded `df116c4` checkpoint; the chat template and generation defaults differ.
Download paths and SparkCache identities use the selected revision. Historical
benchmark results remain scoped to their recorded template and settings.

Set the runtime selection to `tp4-dcp1-sparkcache` for SparkCache or
`tp4-dcp1` without it. Replace the controller address, four management addresses
and SSH aliases with your site values. Node order assigns ranks 0–3.

```bash
sr() { python3 scripts/sparkring.py deploy "$@"; }
STATE="$PWD/.sparkring/glm-tp4-deployment"
RUNTIME_PROFILE=tp4-dcp1-sparkcache

sr discover --controller-address 192.0.2.10 \
  --node spark0=192.0.2.20 --node spark1=192.0.2.21 \
  --node spark2=192.0.2.22 --node spark3=192.0.2.23 \
  --output "$STATE/inventory.json"
sr plan --inventory "$STATE/inventory.json" --name glm-tp4 \
  --workspace /srv/sparkring/glm-tp4 --preserve-existing-network \
  --image-receipt "$SPARKRING_RECEIPT" --runtime-profile "$RUNTIME_PROFILE" \
  --output "$STATE/preparation.json"
sr network-plan --preparation "$STATE/preparation.json" \
  --inventory "$STATE/inventory.json" --output "$STATE/network-plan.json"
```

Use a dedicated workspace and a fabric range that does not overlap management
or VPN routes. Discovery reads hosts; the two plan commands are offline.
Inspect the inventory and plans before applying them.

**Keep both selection flags.** Omitting `--image-receipt` chooses another
runtime composition. This CLI currently accepts only the two DCP1 selections;
do not pass `tp4-dcp4` to it. Use the DCP4 procedure below instead.

## 4. Apply, stage and start

Continue in the [deployment suite at “Apply networking, then verify it”](../../docs/operations/deployment-suite.md#apply-networking-then-verify-it),
using the `STATE`, `SPARKRING_RECEIPT` and preparation file from this guide.
The preservation flag retains the endpoint addresses and connection UUIDs configured
by host setup. Planning rejects incomplete addressing, inconsistent cable subnets,
or saved NetworkManager settings that would require connection replacement.
Correct those settings and rediscover before planning again.

Do not repeat the generic guide's receipt-free `sr plan` example.

Follow its stages in order:

1. Apply the reviewed network plan and verify the resulting network.
2. Stage the pinned image, model, transport bundle and tracked source.
3. Create stopped containers and install the managed services.
4. Bring up the mesh and pass the native communication checks.
5. Start the model through the coordinator, then run readiness checks.

Inspect each plan before applying it. The suite provides separate flags for
hardware tests and model actions. It does not start a model during staging.
Do not substitute direct `docker start` for managed startup or alter fabric
settings while queue pairs are active.

## 5. Verify the selected configuration

Inspect the generated rank environments and logs on all four hosts. For the
default selection they must show DCP1, `MAX_MODEL_LEN=1048576` and the chosen
SparkCache setting. After the API is ready, run the
[semantic and serving checks](../../docs/operations/profile-validation.md).
If SparkCache is enabled, check cold requests, prefix reuse and restore;
API health alone does not establish cache operation.

For bounded requests, select reasoning effort explicitly and allow enough output
tokens for the final answer. This checkpoint's template opens a thinking block;
do not assume `enable_thinking=false` disables it. See
[GLM conversation settings](../../docs/operations/r35-local-launch.md#bounded-glm-conversations).

Use the deployment suite's coordinated `stop`/`recover` actions for operation
and recovery. Preserve private site inputs, image receipts and cache roots.

## DCP4 alternative

The procedure below reproduces **R33 DCP4**, with its pinned image, profile-contract
and entrypoint overlay. It does not qualify R37 DCP4. Use the R33 receipt named
by that procedure; do not apply its release overlay to the R37 image.

The deployment-suite planner and staged-source selection are DCP1-only.
For DCP4, use the separately documented
[render-and-launch procedure](../../performance/records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md#reproduction-overlay-and-quickstart)
with an already prepared ring, verified bundle, image receipt and private site.
Choose `tp4-dcp4-sparkcache` or `tp4-dcp4` in the site's `runtime_profile`.
For managed installation, also save the four host-local contract directories in
that private site's `r33_profile_contract_roots`, in rank order:

```json
"r33_profile_contract_roots": [
  "/srv/sparkring/source/runtime/sparkring/jovian-r33/profiles",
  "/srv/sparkring/source/runtime/sparkring/jovian-r33/profiles",
  "/srv/sparkring/source/runtime/sparkring/jovian-r33/profiles",
  "/srv/sparkring/source/runtime/sparkring/jovian-r33/profiles"
]
```

Use each host's actual checkout path and the same source revision. Keep the
sibling `image/entrypoint.py` in that checkout. The renderer puts the selected
path into each rank environment, so managed container verification reproduces
both read-only overlay mounts. A shell export alone does not persist this input.
Re-render before creating the stopped containers and installing services.

That procedure supplies `R33_PROFILE_CONTRACT_HOST_ROOT` on every host and
uses each host's own rendered rank environment.

Do not edit a staged DCP1 environment in place: deployment receipts pin its
hashes. A DCP4 change needs its own rendered inputs and lifecycle preparation.
For managed operation, follow the
[managed installation/startup contract](../../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md#install-on-each-host);
a direct launch from the recorded trial is not a managed-service upgrade.

The [DCP4 profile](../glm53-flash-spark-tp4-dcp4-sparkcache/profile.json) and
[activation receipt](../../runtime/sparkring/jovian-r33/profiles/evidence/tp4-dcp4-sparkcache-activation-20260911.json)
pin the configuration used for the bounded tests. DCP1 remains the default.

Validate a recorded activation from the repository root with:

```bash
python3 runtime/common/verify_activation.py --receipt /path/to/activation.json
```

This checks the receipt's rank, cache, image and source declarations. It does
not run a serving test or replace the workload evidence.

## Validation and results

- [R37 TP4 record](../../performance/records/glm53-flash/r37-tp4-source-upgrade.md): bounded DCP1 cache-on correctness, all-rank restart/restore and matched short performance checks.
- [DCP1 record](../../performance/records/glm53-flash/r33-image020-tp4-sparkcache-20260911.md): bounded serving, cache and restore checks.
- [DCP4 record](../../performance/records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md): 8,364,901-token KV pool, prefix-hit checks and planned/SIGKILL restore.
- [Benchmark summaries](../../performance/benchmarks.md): measurements and their conditions.

R37 cache-off and DCP4 selections do not inherit the cache-on DCP1 test results.
The 1M context setting is distinct from a completed 1M-token test. The full
blank-host deployment procedure has not been requalified from factory-reset
Sparks. See the records for the exact tested configurations.
