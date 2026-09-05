# Operate the four-Spark mesh as a managed service

Status: **implemented** service and CPU contract checks; **research-only**
deployment. Bounded installer, interruption, and recovery observations are
recorded in the [managed functional evidence](../../performance/records/glm53-flash/spark-mtp3-managed-mesh-functional-20260905.md).
The same record qualifies bounded post-recovery model readiness and one
persistent-cache recall under the connection-grace policy. Broader lifecycle
qualification remains pending. This guide does not claim
unattended high availability or complete failure coverage.

The mesh service configures and monitors paths that allow opposite Sparks
to communicate through an intermediate NIC in the ring. It requires all
four ranks to be ready before model startup and stops dependent serving
when the fabric is unhealthy. The operator explicitly initiates recovery.

Small native helper programs, called source markers, install the packet-header
rewrite rules needed by these paths. They are **not packet forwarders**:
intermediate packets traverse the ConnectX-7 ASIC. Endpoint
posting still uses CPU-side transport machinery and GPU-mapped host memory;
this does not add GPUDirect RDMA to DGX Spark.

## Prerequisites and identities

Prepare the target, verified transport bundle, image receipt, and rendered
four-rank launch directory using the
[model quickstart](../../docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md). Use its
model and collective-routing settings. The installer receipt at
`/srv/sparkring/verified-image-receipt.json` in the examples is a staged copy
of the repository's `image-receipt.json`, not `public-image.json`. Pull the
published image on each host before creating containers; Docker cannot
resolve an absent local config ID by pulling it from a registry.
For managed operation, use the
installation and lifecycle commands below instead of bounded marker leases
or direct `docker start` commands.

Before installation:

- Drain and stop all dependent model containers across the four hosts.
- Verify direct-neighbor RoCE, the physical cable cycle, all four local
  RDMA functions, Ethernet MTU 9,000, RoCE MTU 4,096, and GID index 3.
- Verify the [ConnectX-7 hairpin/steering prerequisites](../../docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md#connectx-7-driver-configuration-for-hardware-forwarding):
  four hairpin queues of 1,024 packets per function and the tested `hmfs`
  steering profile. Any required driver reload belongs to a separate
  stopped-all-RDMA-users maintenance step over independent management or a
  console, not to service installation or recovery.
- Use identical source, site, topology, marker binary, and immutable image
  identities on all four ranks. Use the marker executable built with managed
  attachment support; a bounded-only executable is insufficient.
- Reserve RDMA UDP source port 65535 on the selected functions. Its source
  marker is not scoped to a particular model, IP address, or QP number.
- Provide root privileges for installation, systemd, Docker, and exact local
  route/neighbor/TC changes. The cluster coordinator requires noninteractive
  SSH and `sudo -n` on all four topology hosts.
- Restrict TCP port **9975** to the four trusted management addresses and
  authorized operators. The listener binds only to the configured local
  management address. It uses authenticated HTTP, not TLS; do not expose it
  to an untrusted network.

All ranks share a 32-byte random health key and a common 128-bit epoch,
expressed as 32 lowercase hexadecimal characters. Store the key as a
root-owned regular file with mode `0600`; never put it in a repository,
container environment, command argument value, or public receipt. Distribute
the same key file through a trusted channel. Rotate the epoch and key during
a coordinated stopped-stack maintenance window, not one rank at a time.

Generate the key and epoch **once**, on the trusted management host. The
following commands refuse to overwrite either file. They write the secret
directly into a root-only directory; do not print the key or use shell tracing:

```bash
sudo install -d -m 0700 -o root -g root /srv/sparkring/private
sudo sh -c 'umask 077; set -C; openssl rand 32 > /srv/sparkring/private/health.key'
sudo sh -c 'umask 077; set -C; openssl rand -hex 16 > /srv/sparkring/private/epoch.txt'
sudo stat -c '%U %a %s %n' /srv/sparkring/private/health.key
```

Require owner `root`, mode `600`, and size `32` for `health.key`. The epoch
file is a nonsecret deployment identifier; retain and reuse it for all four
installation commands. If a generation command fails, inspect the private
files before proceeding; do not silently replace an existing deployment's key.

Transfer these two files to every Spark over authenticated SSH without
exposing key bytes to terminal output. From a management host separate from
the four Sparks, this example creates absent destination files with root-only
permissions:

```bash
set -o pipefail
for host in spark-r0 spark-r1 spark-r2 spark-r3; do
  ssh "$host" 'sudo -n install -d -m 0700 -o root -g root /srv/sparkring/private'
  sudo cat /srv/sparkring/private/health.key | \
    ssh "$host" 'sudo -n sh -c "umask 077; set -C; cat > /srv/sparkring/private/health.key"' || exit 1
  sudo cat /srv/sparkring/private/epoch.txt | \
    ssh "$host" 'sudo -n sh -c "umask 077; set -C; cat > /srv/sparkring/private/epoch.txt"' || exit 1
  ssh "$host" 'sudo -n stat -c "%U %a %s %n" /srv/sparkring/private/health.key'
done
```

If the management host is itself one Spark, omit it from the transfer loop;
its local files already exist. Use your verified aliases, not an unreviewed
host list. An existing destination file makes the transfer fail rather than
overwrite it. Keep both inputs available until all four installations succeed.

Peer readiness binds the shared epoch, site/topology, service source,
immutable image, and each supervisor's random process generation. Signed
nonce challenges establish response authenticity and freshness. Before model
startup, every rank must report the same four-process generation set as
armed. A restarted peer has a different generation and requires explicit
cluster recovery.

## Install on each host

The first-install helper `managed_install.py` validates an already prepared,
stopped container against the rendered launch and verified image receipt.
Prepare the rank-zero container on its host before invoking the installer:

```bash
SPARKRING_CREATE_ONLY=1 bash /srv/sparkring/mtp3-mesh-launch/launch-rank.sh \
  0 /srv/sparkring/mtp3-mesh-launch/rank0.env
```

Repeat with each host's rank and environment file. The launcher creates but
does not start the container in this mode; it refuses an existing name.
The installer does not create, replace, or start containers. Docker automatic
restart must remain disabled.
The service configuration pins the full container ID and immutable image
ID; matching a container name alone is insufficient.

The installed source root is `/opt/sparkring/managed-mesh`; private
configuration is under `/etc/sparkring/managed-mesh`. Both directories and
both unit-file targets must be absent for first installation. Stage the
verified bundle and extracted marker outside the installed source tree, for
example under `/srv/sparkring/artifacts`. Point the private site's
`bundle_root` and `marker_binary` at those artifact paths; prepopulating the
installed source directory makes installation fail. Runtime status, marker
logs, and ownership journals are under `/run/sparkring-mesh`, which is
boot-scoped. Preserve source and configuration permissions so unprivileged
users cannot replace code, identities, or the health key.

From a checkout on each host, review the installation plan without `--apply`.
For rank zero, with a prepared private key at the path shown:

```bash
MESH_EPOCH=$(sudo cat /srv/sparkring/private/epoch.txt)
sudo python3 runtime/glm53-spark-mtp3-mesh/managed_install.py \
  --launch /srv/sparkring/mtp3-mesh-launch \
  --image-receipt /srv/sparkring/verified-image-receipt.json \
  --rank 0 --epoch "$MESH_EPOCH" \
  --key-file /srv/sparkring/private/health.key
```

After checking the paths, image, rank, and proposed changes, run the same
command with `--apply`. Repeat for ranks 1, 2, and 3 on their corresponding
hosts, with the same epoch and key. Do not overwrite another deployment's
container or installed service without inspecting and explicitly removing
that deployment first.

The installer regenerates the expected launch from the reviewed site and
compares the stopped container's complete model arguments, entrypoint,
environment, bind mounts and access modes, labels, user, and working directory.
It also validates the helper hash and `--managed` support, canonical bundle
hashes, and temperature-one image attestation. Re-render launch files with
the installer checkout before installation; modified rank environments or a
different launcher are rejected. Change the reviewed profile inputs rather
than editing rendered files to bypass this check.

The launcher's `SPARKRING_PRINT_CONTAINER_SPEC=1` mode emits the same Docker
argument array used for creation, after read-only identity checks. It does
not create a container or start a model. The installer invokes this mode in
a controlled environment. Plan mode invokes local read-only Docker/helper
checks and uses a temporary directory for canonical rendering. `--apply` additionally
requires root, copies the source/configuration/key, verifies the unit files,
and reloads systemd. It neither enables nor starts the units.

The [configuration-check receipt](container-validation-receipt.json) records
four-host acceptance checks and twenty rejected argument/environment/mount
mutations. These checks did not install or restart services.

The installed source directory must remain identical across ranks. Source
hashes participate in authenticated readiness; patching one rank breaks the
group identity. To change installed code or an image, stop all models and
mesh services, preserve the installation and pinned containers, and plan an
explicit replacement. The first-install helper is not an in-place updater.

Installation is not model readiness. The managed lifecycle uses two units:

| Unit | Responsibility |
|---|---|
| `sparkring-mesh.service` | Local network ownership, persistent marker attachment, peer checks, failure latch |
| `sparkring-mesh-model.service` | Four-rank authenticated startup gate and the pinned Docker model container |

The model unit requires and binds to the mesh unit. The tested units report
`RuntimeMaxUSec=infinity`: systemd imposes no elapsed-runtime cutoff. Both have
`Restart=no`: failure does not silently reload the model or reconnect a
partially changed fabric. Do not independently enable automatic startup or
Docker restart policies as a substitute for the coordinated procedure.

## Start the fabric and model

Run the coordinator from a management checkout with the private rendered site.
It prints the host/command plan unless `--execute-authorized` is supplied.
Executed actions require a previously absent receipt path. Keep receipts
private because they contain resolved host information.

Create the receipt directory with permissions allowing the coordinator user
to write it. The examples assume `/srv/sparkring/receipts` has been provisioned
for that user. Do not run `up` while another deployment or diagnostic marker
owns the selected functions; preserve and stop only the known owner first.

```bash
python3 runtime/glm53-spark-mtp3-mesh/managed_cluster.py up \
  --site /srv/sparkring/mtp3-mesh-launch/site.json

python3 runtime/glm53-spark-mtp3-mesh/managed_cluster.py up \
  --site /srv/sparkring/mtp3-mesh-launch/site.json \
  --output /srv/sparkring/receipts/mesh-up.json --execute-authorized
```

`up` first confirms that every pinned model container is stopped. It starts
the mesh supervisors and requires four-rank authenticated readiness. Exact
preexisting routes, permanent neighbors, and hardware rules are adopted;
conflicting objects cause failure rather than replacement. Missing planned
objects are created and journaled. No route or qdisc flush is performed.

Before the first model start, run the quickstart's isolated native RC
correctness checks while all model containers remain stopped. A local
hardware rule marked `in_hw` is not proof of end-to-end payload correctness.

```bash
python3 runtime/glm53-spark-mtp3-mesh/managed_cluster.py start-model \
  --site /srv/sparkring/mtp3-mesh-launch/site.json \
  --output /srv/sparkring/receipts/model-start.json --execute-authorized
```

Wait for actual model readiness after starting or restarting the model:

```bash
python3 runtime/glm53-spark-mtp3-mesh/wait_managed_ready.py \
  --launch /srv/sparkring/mtp3-mesh-launch --timeout 900 \
  --output /srv/sparkring/receipts/model-ready.json
```

This read-only tool requires all four expected containers to be running and
Docker-health healthy, plus HTTP 200 from rank zero's API health and scheduler
liveness endpoints. It does not trust systemd active state alone. The receipt
path must not exist and its parent directory must exist. A timeout reports
`NOT READY` without stopping or restarting anything.

The model startup gate checks the exact container identity and the shared
four-rank process-generation view. Inspect completed temperature-one warmup,
rank logs, API health, and scheduler liveness before sending benchmark traffic.
The coordinator's successful start does not mean model loading or warmup
has finished.

On an individual host:

```bash
sudo journalctl -fu sparkring-mesh.service
sudo journalctl -fu sparkring-mesh-model.service
```

For all four hosts' systemd state:

```bash
python3 runtime/glm53-spark-mtp3-mesh/managed_cluster.py status \
  --site /srv/sparkring/mtp3-mesh-launch/site.json \
  --output /srv/sparkring/receipts/mesh-status.json --execute-authorized
```

`status` reports unit state; it is not a fresh end-to-end transport or model
correctness test.

## Planned stop, restart, and recovery

Drain application requests before a planned stop. Use the cluster coordinator,
not individual `docker stop`, `docker start`, or mesh-unit commands:

```bash
python3 runtime/glm53-spark-mtp3-mesh/managed_cluster.py stop-model \
  --site /srv/sparkring/mtp3-mesh-launch/site.json \
  --output /srv/sparkring/receipts/model-stop.json --execute-authorized
```

This quiesces each supervisor's model-start expectation, stops all model
units and pinned containers, and verifies an all-rank model-stop barrier.
The transport remains installed. The stop implementation uses `docker kill`
for containment; it is not a request-draining mechanism. For a same-fabric
model restart, run `start-model` with a distinct receipt path after the
successful stop. Cache-restoration qualification must additionally verify
cache identity, publication, and restoration evidence on every rank.

To stop serving and remove owned transport artifacts:

```bash
python3 runtime/glm53-spark-mtp3-mesh/managed_cluster.py down \
  --site /srv/sparkring/mtp3-mesh-launch/site.json \
  --output /srv/sparkring/receipts/mesh-down.json --execute-authorized
```

`down` crosses the model-stop barrier before stopping mesh units or cleaning
up recorded children. Cleanup removes only exact artifacts created and owned
by the service. Adopted network state remains installed. Changed objects,
ambiguous pending additions, and nonempty owned qdiscs are retained for
operator inspection. A failed phase prevents later phases from running;
inspect the receipt before deciding how to recover.

After correcting the cause of a fault, explicitly recover all four ranks:

```bash
python3 runtime/glm53-spark-mtp3-mesh/managed_cluster.py recover \
  --site /srv/sparkring/mtp3-mesh-launch/site.json \
  --output /srv/sparkring/receipts/mesh-recover.json --execute-authorized
```

Recovery stops the models, verifies the barrier, stops mesh supervisors,
reaps only recorded owned marker identities, cleans owned network state,
resets failure latches, and establishes a fresh authenticated mesh generation.
It does **not** restart the model. Inspect readiness, then explicitly use
`start-model` when ready to load it.

## Detection limits and supervisor loss

### Native helper lifetime contract

`managed_service.py` starts two source-marker processes per rank with
`--managed`. They remain attached until signaled or failed and report
`lifetime_seconds: null`. The service units report `RuntimeMaxUSec=infinity`.
The supervisor, model-stop barrier, and explicit recovery procedure govern
their lifetime; a periodic timer is not used to remove forwarding rules.
The bounded `--run-seconds` mode is for isolated diagnostics only.

### Readiness checks

| Check | Configured interval or bound |
|---|---|
| Child-process and peer checks | 1-second loop; peer HTTP timeout 2 seconds |
| Unavailable peer connection | 4-second grace after the first transport failure; degraded health blocks model startup |
| MAC/IP, Ethernet MTU, sysfs GID/netdev, routes, qdiscs, TC state | 5-second periodic check |
| Full RDMA active-MTU probe | Startup and approximately every 60 seconds |
| Health progress freshness | Readiness rejected after 10 seconds without supervisor progress |
| systemd watchdog | 15 seconds |

A connection timeout or other peer transport error enters a degraded state.
Existing serving is retained during a four-second grace interval measured
from the first observed transport failure. Successful authenticated peer
responses clear that interval. Degraded peer health blocks model startup.
An authentication failure, explicit negative readiness, or changed process
generation does not receive transport-error grace: it triggers failure when
observed. Local marker exits also trigger failure without that grace.

These are polling and timeout settings, not zero-window guarantees. Command
execution, scheduling, management-network delays, and container-stop time
affect detection and containment. In-flight requests can fail; a successful
health response is not numerical validation of their output.

If the supervisor's main process is killed, systemd's `KillMode=process`
deliberately leaves marker children alive so forwarding is not immediately
removed beneath an uncertain container state. The dependent model unit is
bound to the supervisor, and the stop hook attempts to stop its pinned
container. Explicit `recover` performs the all-rank stop barrier and then
reaps recorded orphan markers using PID/start-time/argument identity checks.
Unrecorded markers require operator inspection; cleanup does not kill them
by a broad process-name match.

During an orderly supervisor failure, forwarding is retained until Docker
confirms the dependent model is stopped. If Docker cannot provide that
confirmation, the service reports failure and retains forwarding rather
than assuming teardown is safe. This is containment machinery, not a promise
that every host, NIC, kernel, or Docker failure can be recovered unattended.

Before public recommendation, retain live evidence for clean startup,
planned model and fabric restart, helper loss, supervisor loss, peer failure,
changed hardware rules, explicit recovery, temperature-one model output, and
persistent-cache restoration. CPU tests establish software contracts; they
do not qualify those four-rank hardware behaviors.
