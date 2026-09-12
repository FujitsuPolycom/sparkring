# GLM-5.3-Flash on four Sparks

Run [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark)
with MTP3 on a four-Spark ring. **DCP1 is the default; DCP4 is an alternative.**
Context defaults to 1M tokens. SparkCache is optional.

| Selection | KV allocation per rank | KV evidence | Procedure |
|---|---:|---:|---|
| DCP1, with or without SparkCache | 24 GiB | [2.3M sizing reference](../../performance/capacity-references.md) | Deployment suite below |
| DCP4, with or without SparkCache | 24 GiB | 8.4M with SparkCache | [DCP4 setup](#dcp4-alternative) |

Capacity depends on enabled features. The validated results are scoped to the
[image and workload records](#validation-and-results), not every possible configuration.
For a switch-connected fabric, use the [switched setup](../glm53-flash-spark-tp4-switched/README.md).

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

The deployment suite downloads and distributes the image during staging.
The tracked runtime receipt selects
`ghcr.io/fujitsupolycom/sparkring@sha256:1328a4f6f483014021a66a757012793629bd054d28d0fe4d5e581fa4aed776ef`.
Validate it locally before staging:

```bash
SPARKRING_RECEIPT="$PWD/runtime/sparkring/jovian-r33/public-image-receipt.json"
python3 runtime/sparkring/jovian-r33/profiles/verify_profile.py image \
  --receipt "$SPARKRING_RECEIPT"
```

The runtime receipt selects this image; `publication.json` is distribution
metadata and cannot replace it. [Image building](../../runtime/images/README.md)
is a separate workflow and is not needed for this quickstart.

## 3. Discover and plan DCP1

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

For this pinned NVFP4-Spark checkpoint, leave reasoning enabled (omit the request
override or use `chat_template_kwargs: {"enable_thinking": true}`). Its chat
template always opens a thinking block. With this image, setting the flag to
`false` disables reasoning parsing without changing that template and can put
reasoning text and a closing tag in the visible answer. A non-thinking request
needs a separately validated template/parser combination.

Use the deployment suite's coordinated `stop`/`recover` actions for operation
and recovery. Preserve private site inputs, image receipts and cache roots.

## DCP4 alternative

DCP4 uses the **same published image**, plus a profile-contract and entrypoint
overlay. It requires the managed fabric installation. It is not a different
model download or image rebuild.

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

- [DCP1 record](../../performance/records/glm53-flash/r33-image020-tp4-sparkcache-20260911.md): bounded serving, cache and restore checks.
- [DCP4 record](../../performance/records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md): 8,364,901-token KV pool, prefix-hit checks and planned/SIGKILL restore.
- [Benchmark summaries](../../performance/benchmarks.md): measurements and their conditions.

The 1M context setting is distinct from a completed 1M-token test. The full
blank-host deployment procedure has not been requalified from factory-reset
Sparks. See the records for the exact tested configurations.
