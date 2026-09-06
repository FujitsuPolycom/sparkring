# Prepare and operate a four-Spark deployment

Status: **research-only**. The standalone `sparkring deploy` commands prepare
hosts and run the managed GLM-5.3 Flash NVFP4-Spark MTP3 profile. LIL is not
required. Offline tests cover planning, fake hosts, verification, and recovery
ordering; this complete workflow has not been tested on factory-reset Sparks.

## Before starting

- Use Bash and Python on a Linux or WSL controller with Git, SSH, SCP, and
  PyYAML. Run commands from this checkout; deployment source must be tracked
  by Git before staging. Keep private inputs and receipts outside commits.
- Complete [host setup](GLM53_SPARK_MESH_HOST_SETUP.md): first boot, supported
  drivers, Docker/NVIDIA Container Toolkit, SSH enrollment, noninteractive
  sudo, required networking tools, and the documented four-cable ring.
- Use four enrolled SSH aliases, one per rank. Each host needs an independent
  management connection. Do not reconfigure the link carrying SSH.
- Reserve a maintenance window: network changes require stopped containers
  and no RDMA users. Native checks use GPUs/RDMA but do not run a model.
- Provide enough disk space for source, the image archive, extracted image,
  model, and caches. Model downloads may need access approved by its publisher.

**Runtime compatibility:** this suite uses the image and lifecycle code under
[`runtime/glm53-spark-mtp3-mesh/`](../runtime/glm53-spark-mtp3-mesh/).
The preparation document records the lifecycle capabilities found in the
staged source. When that source implements all three memory operations,
startup checks idle hosts, prepares memory, and checks memory readiness before
model arming. Partial memory-operation support is rejected. This checkout's
pinned profile predates those operations; capability negotiation does not add
them to an older image or installer. Match source, image receipt, and installed
lifecycle before replacement. This is not a drop-in upgrade command.

## Discover and review

Replace the example addresses with your management LAN addresses. The
controller address must be reachable from every Spark. Node order assigns
ranks 0–3. Choose a dedicated workspace name; do not reuse an existing model
or cache directory.

```bash
sr() { python3 scripts/sparkring.py deploy "$@"; }
STATE="$PWD/.private/deploy-mesh"

sr discover --controller-address 192.0.2.10 \
  --node spark0=192.0.2.20 --node spark1=192.0.2.21 \
  --node spark2=192.0.2.22 --node spark3=192.0.2.23 \
  --output "$STATE/inventory.json"
sr plan --inventory "$STATE/inventory.json" --name mesh \
  --workspace /srv/sparkring/mesh --fabric-range 198.18.0.0/21 \
  --output "$STATE/preparation.json"
sr network-plan --preparation "$STATE/preparation.json" \
  --inventory "$STATE/inventory.json" --output "$STATE/network-plan.json"
```

Discovery reads hosts; both plan commands are offline. Inspect the resulting
JSON, especially management interfaces, RDMA devices, data addresses, backups,
and proposed changes. The fabric range must not overlap management or VPN
routes. Existing foreign NetworkManager profiles are not silently adopted.

## Apply networking, then verify it

Execution requires the exact reviewed plan's SHA-256. This helper supplies
that value; it does not replace reviewing the plan. Receipts are separate
from plans and record completed, failed, or uncertain actions.

```bash
apply_reviewed() {
  local plan="$1" receipt="$2" digest
  shift 2
  digest=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$plan")
  sr apply-plan --plan "$plan" --receipt "$receipt" \
    --approve-sha256 "$digest" "$@"
}
apply_reviewed "$STATE/network-plan.json" "$STATE/network-execution.json"
```

A plan containing a driver reload additionally requires
`--allow-driver-reload`. It changes one function and stops for rediscovery;
do not loop blindly through reloads. Review the resulting device state.
Network backups remain on each host at the paths in the plan.

After changes, repeat discovery into a **different** inventory filename and
regenerate the network plan. Reconcile any changed device mapping before
applying another plan. Once no changes remain, run:

```bash
sr network-check --preparation "$STATE/preparation.json" \
  --output "$STATE/network-verified.json"
```

This reads all four hosts and verifies configuration, including GIDs and link
state. It does not prove that RDMA traffic works. A completed execution
receipt alone is not network verification.

## Stage the runtime without starting a model

```bash
sr stage --preparation "$STATE/network-verified.json" \
  --state "$STATE/runtime" --execute
PREP="$STATE/runtime/prepared.json"
```

Staging repeats the network check, packages tracked source, downloads the
pinned image and model on rank 0, verifies distributed copies, preserves one
shared mesh key, and renders launch files. Transfers pass through the
controller using SCP; direct fabric fanout is not implemented here. The
download helper runs without GPUs under the staging login user's UID/GID;
no serving process starts. Destination trees are checked for symlinks,
hard-linked files, and special files before writing or mounting them.

The prepared document pins source and rendered launch-file hashes. All four
hosts must produce matching launch files. Runtime commands verify these hashes
with controller-supplied code before importing staged Python; changing a launch
file and its local manifest does not authorize that change. Preparation files
without pinned launch hashes must be staged again before runtime use.
Controller readiness and native checks use a verified source copy in the staging
directory, not imports from a separately edited checkout. Python execution
ignores existing bytecode caches without deleting them. These checks detect
changed inputs; they are not a sandbox against a compromised host administrator.

Host models, caches, source, and artifacts live under the dedicated workspace.
The shared key and receipts are private. Use the same staging directory for a
retry with unchanged inputs; different inputs require a separate directory.

## Create containers, install services, and test the mesh

Generate **one action at a time**, inspect its plan, then apply it. The helper
uses the staged preparation document and never starts anything by itself.

```bash
runtime_plan() {
  sr runtime-plan "$1" --preparation "$PREP" --output "$STATE/$1-plan.json"
}

runtime_plan create
apply_reviewed "$STATE/create-plan.json" "$STATE/create-execution.json"
runtime_plan install
apply_reviewed "$STATE/install-plan.json" "$STATE/install-execution.json"
runtime_plan up
apply_reviewed "$STATE/up-plan.json" "$STATE/up-execution.json"
runtime_plan native-check
apply_reviewed "$STATE/native-check-plan.json" "$STATE/native-check-execution.json" \
  --allow-hardware-tests
```

`create` makes stopped containers. `install` installs managed services and
refuses existing installation targets. `up` starts mesh supervisors, not the
model. `native-check` runs bounded four-rank communication checks and rejects
incomplete or failed results. Inspect failures before serving.

## Start, inspect, and stop serving

```bash
runtime_plan start
apply_reviewed "$STATE/start-plan.json" "$STATE/start-execution.json" \
  --allow-model-actions
runtime_plan ready
apply_reviewed "$STATE/ready-plan.json" "$STATE/ready-execution.json"
runtime_plan status
apply_reviewed "$STATE/status-plan.json" "$STATE/status-execution.json"
runtime_plan logs
apply_reviewed "$STATE/logs-plan.json" "$STATE/logs-execution.json"
```

`start` proves containers are running, not that model loading has finished.
`ready` waits for all four container health checks plus API and liveness
responses. It sends no inference requests and does not test answer quality or
SparkCache restore. Status/log output is stored in the execution receipt.

```bash
runtime_plan stop
apply_reviewed "$STATE/stop-plan.json" "$STATE/stop-execution.json" \
  --allow-model-actions
runtime_plan recover
apply_reviewed "$STATE/recover-plan.json" "$STATE/recover-execution.json" \
  --allow-model-actions
```

`stop` coordinates model shutdown across ranks. `recover` also stops and
cleans the managed mesh, resets its services, and starts mesh supervisors.
It does not restart the model. It uses installed lifecycle tools even if
staged source is damaged. It is not a general network rollback operation.
Use `runtime_plan down` and apply its plan with `--allow-model-actions` to
stop both the model and mesh without starting mesh supervisors again.

## Retries and limits

- Output paths are not overwritten. Readiness checks, including a resumed
  `ready` plan, make fresh observations and preserve each result under an
  unused filename in `ready-results/`. Native-test output locations are
  derived from `controller_launch`; preserve those results before arranging
  another test output location. A historical native-test receipt does not
  prove present serving health.
- Resume an interrupted execution with the **same** plan and receipt by
  appending `--resume` to `apply_reviewed`. Completed actions are rechecked.
  A failed recheck records its failure and revokes the receipt's completion.
  A running or uncertain mutation blocks automatic retry: inspect host state
  before preparing recovery. Do not edit receipts to claim success.
- Local locks prevent concurrent use of one staging directory or receipt.
  Staging additionally locks each remote workspace with a unique operation token.
  Any failure retains remote locks and the controller's `remote-operation.json`:
  timed-out remote work may still be active. Inspect the owning operation and
  all hosts before recovery; do not remove locks merely to retry. These locks
  coordinate staging commands, not arbitrary manual Docker or filesystem changes.
  Tree checks are conservative and may add overhead to large-file inventories.
- Partial staging and installation require inspection; transactional rollback
  and automatic network rollback are not implemented. Backups and model/cache
  files are retained. The suite does not install drivers, flash firmware,
  rewire cables, or replace management-network configuration.

## Remaining integration work

The suite and LIL's image commands are separate interfaces. LIL trials now
verify the installed fabric configuration, but do not enroll their containers
with its supervisor. Managed operation should use this suite's lifecycle;
unified LIL-to-managed lifecycle delegation still needs implementation.

Image/model reuse on staging retry, direct-fabric bulk distribution, adjustable
model-profile settings, and transactional recovery remain incomplete. The source
and image pins must be reconciled with the selected public deployment before a
hardware rehearsal. None of the offline fixes establishes fresh-host readiness.
