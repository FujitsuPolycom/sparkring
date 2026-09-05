# Run GLM-5.3 Flash Spark with native MTP3 and hardware-forwarded mesh

Status: **research-only**. The profile's composition, managed host service,
and CPU checks are **implemented**. The
[managed functional record](../performance/records/glm53-flash/spark-mtp3-managed-mesh-functional-20260905.md)
qualifies bounded installer, policy-scoped fault/recovery, post-recovery
readiness, and one persistent-cache recall case for the published managed
image identified below.
Broader cache/failure coverage and unattended serving remain unqualified.

The [public application-install record](../performance/records/glm53-flash/spark-mtp3-public-application-install-20260905.md)
covers fresh public checkouts, extracted image artifacts, empty application
caches, installation, native correctness, and model-restart cache restoration
on four prepared hosts. It does not qualify a factory-reset OS/network setup.

**Starting with four stock Sparks and no image?** Follow
[the managed-mesh prerequisite section](PREREQUISITES.md#four-spark-managed-hardware-forwarded-mesh)
first. It reuses the shared blank-cluster bootstrap and adds the secondary
data interfaces, GID/MTU checks, and driver configuration required below.
Return here to pull the published image and deploy the model.

The profile uses the `GLM-5.3-Flash-NVFP4-Spark` target's built-in multi-token
predictor with three speculative tokens. Graph-native SIRCL handles most
captured target verification, fused SIRCL handles large eager prefill, and
RoCEnante handles selected small all-reduces. Patched NCCL retains the other
collectives. The host fabric supplies hardware-forwarded paths between
opposite ranks without extra diagonal cables.

The [profile contract](../runtime/glm53-spark-mtp3-mesh/README.md) and
[pins](../runtime/glm53-spark-mtp3-mesh/pins.json) are the canonical inputs.
The [throughput record](../performance/records/glm53-flash/spark-mtp3-mesh-20260905.md)
reports observations, not a general performance guarantee.

## Attribution and design origins

RoCEnante originates with Local Inference Lab's contributors, not SparkRing.
Credit goes to Luke (`lukealonso`) and the Local Inference Lab community,
including PR author `original-el8`, for the communication implementation in
[B12X #295](https://github.com/local-inference-lab/b12x/pull/295) and its
[vLLM integration in #597](https://github.com/local-inference-lab/vllm/pull/597).

Those PRs motivated SparkRing's investigation of hardware-forwarded paths
between opposite ranks on a four-node ring. SparkRing adapts the donor
communication package to those paths and combines it with SIRCL routing and
managed deployment. It does not claim to originate RoCEnante or install both
complete PRs unchanged. The [vendored-source provenance](../third_party/b12x_roce/README.md)
identifies the included code and retained license.

## Recorded benchmark observations

See the [consolidated validation report](../performance/records/glm53-flash/spark-mtp3-validation-summary-20260905.md)
for completed checks, repeat counts, and the remaining test plan.

The [profile results table](../runtime/glm53-spark-mtp3-mesh/README.md#operator-benchmark-observations)
shows the full concurrency matrix: aggregate decode reached **231.3 tok/s at
8K/C16**, and concurrency-one prefill scouts measured **2,703–2,787 prompt
tokens/s** across 8K–128K contexts. The linked record provides the measured
configuration, sampling settings, and single-run measurement conditions.

A separate [Estonia long-context accuracy benchmark](../performance/records/glm53-flash/spark-mtp3-country-recall-20260905.md)
completed **30/30 correct answers at C8** on one repeated 133,208-token prompt,
with no output-budget hits. The records include both operator screenshots,
numeric results, and distinct throughput definitions.

## Preparation order and command locations

Use one management shell to prepare the shared site and coordinate all four
hosts. Host installation commands run locally on each Spark. Commands below
assume a Linux Bash shell and a checkout containing this guide:

```bash
set -euo pipefail
git clone --branch codex/glm53-spark-mtp3-mesh https://github.com/FujitsuPolycom/sparkring.git
cd sparkring
git rev-parse HEAD
test -f runtime/glm53-spark-mtp3-mesh/managed_install.py
```

Use the same reviewed Git revision on the management host and all four
Sparks; keep it alongside the image receipt. If the profile is supplied in a
draft PR rather than merged `main`, check out that PR's exact commit before
continuing. A checkout that lacks `managed_install.py` cannot follow this
guide. Do not substitute files from a private experiment directory.
Keep `set -euo pipefail` enabled in each shell running the following command
blocks so a failed identity check prevents subsequent steps.

The preparation order is image/model → extracted artifacts → private
site/topology → rendered launch → distribution → stopped containers →
managed installation → native checks → model startup/readiness. The
[managed operations guide](../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md)
provides the key-generation, install, startup, and recovery commands.

The [deployment recipe](../recipes/glm53-spark-mtp3-managed-mesh-tp4.json)
indexes image, model, topology, settings, and evidence metadata using
`sparkring-recipe/v1`. The profile renderer consumes the separate dedicated
site/topology contracts described below; the recipe is not a substitute for
those private inputs or an executable host installer.

## Hardware and safety prerequisites

- Four NVIDIA DGX Spark GB10 systems, arranged in the physical cable cycle
  `0-1-2-3-0`, with two physical ConnectX-7 ports per system.
- Both Socket Direct PCI functions for each physical port must be available:
  four RDMA functions per system. These are not four additional cables.
- A verified rank, interface, MAC, IP, GID, and cable inventory matching the
  mesh contract. Do not infer physical connections from interface names.
- Working direct-neighbor RoCE, Ethernet MTU 9,000, RoCE MTU 4,096, GID index
  3, and the ability to install hardware-only traffic-control redirect rules.
- The same pinned image, model revision, transport bundle, and marker binary
  on all four ranks. Docker with NVIDIA runtime support is required.
- A systemd Linux host with Python 3.10 or later, Git, OpenSSH, `rsync`, `openssl`,
  `ip`/`tc` from iproute2, `ethtool`, rdma-core, and `ibv_devinfo` from
  ibverbs-utils. The host Python helpers use the standard library. The
  download host additionally needs the Hugging Face `hf` CLI. Building
  locally requires a Linux/ARM64 Docker host; no model weights are embedded
  in the image.
- Space for the complete checkpoint and image on every host, plus the
  configured 40 GiB persistent cache per rank and JIT/build cache headroom.
  Image export can require another tens-of-GiB archive. Confirm free disk
  and memory instead of deleting another workload's artifacts.
- A trusted management network. This profile serves port 8015 without an API
  key; restrict access to intended clients. Adding API authentication requires
  a reviewed profile extension, not an untracked edit to a rendered rank file.
- An operator-approved maintenance window for host networking and model
  startup. Preserve host routes, neighbor entries, traffic-control rules,
  qdiscs, MTUs, and process ownership before making changes.

A small native helper, called the source marker, installs the source NIC's
packet-header rewrite used by the hardware-forwarded paths. It selects
**every RDMA-TX packet with UDP source port 65535
on its selected device**. It is not scoped to a particular IP or QP number.
Reserve that source port for the mesh and exclude unrelated users of the
selected functions during testing.

The source-bound peer map requires clockwise links on physical f0 and
counter-clockwise links on physical f1, using RDMA names `rocep1s0f0`,
`rocep1s0f1`, `roceP2p1s0f0`, and `roceP2p1s0f1`. Netdev names may differ,
but the RDMA-function roles must agree. Other cabling/HCA maps need a
separately validated transport contract; changing the site JSON alone is
insufficient.

For each verified interface/function pair, read its identity before editing
the private topology. Replace the two variables with one actual pair:

```bash
MESH_NETDEV='REPLACE_WITH_ACTUAL_NETDEV'
MESH_RDMA_DEVICE='rocep1s0f0'
ip -j addr show dev "$MESH_NETDEV"
ethtool -k "$MESH_NETDEV"
ibv_devinfo -d "$MESH_RDMA_DEVICE" -i 1
cat "/sys/class/infiniband/$MESH_RDMA_DEVICE/ports/1/gids/3"
cat "/sys/class/infiniband/$MESH_RDMA_DEVICE/ports/1/gid_attrs/ndevs/3"
```

Repeat for all four functions on every host. Require the GID's IPv4-mapped
address and netdev to match the private inventory, Ethernet MTU 9,000, and
active RDMA MTU 4,096. Hardware TC offload must be available; the managed
service requires actual `skip_sw`/`in_hw` rules. It does not configure
cabling, IP addresses, MTUs, firmware, or switch modes. Establish those
direct-link prerequisites first using your host network configuration.
Do not blindly enable switchdev or modify management interfaces. Hardware
support is accepted only when the exact managed TC rules and bidirectional
native RC checks succeed on the deployment.

### ConnectX-7 driver configuration for hardware forwarding

The qualified deployment used the following NIC profile. These are host
driver prerequisites, not settings embedded in the model image:

| Setting | Observed value and scope |
|---|---|
| `hairpin_num_queues` | `4`, `driverinit`, all four PCI functions on every host |
| `hairpin_queue_size` | `1024` packets, `driverinit`, all four PCI functions on every host |
| `flow_steering_mode` | `hmfs`, `runtime`, all four PCI functions on every host |
| eSwitch | `legacy`, inline mode `none`, encapsulation `basic`, all 16 PCI functions across the four hosts |
| Hardware TC offload | Enabled on all 16 verified data functions across the four hosts |

`hmfs` is an observed capability of the tested driver, not a promise that
every upstream kernel, vendor driver, or firmware exposes that value. If the
driver rejects the parameter or reports a different steering mode, stop and
establish a compatible host configuration. Do not substitute `smfs`/`dmfs`,
enable switchdev, or update firmware merely to get past a check.

Hairpin queues implement mlx5 hardware forwarding for TC redirect rules.
Their count and packet capacity are `driverinit` parameters in the
[kernel mlx5 devlink documentation](https://docs.kernel.org/next/networking/device_drivers/ethernet/mellanox/mlx5/devlink.html#hairpin-num-queues-number-of-hairpin-queues).
Discover each function's PCI identity from its verified RDMA device instead
of copying PCI addresses from another host:

```bash
MESH_RDMA_DEVICE='rocep1s0f0'
MESH_BDF=$(basename "$(readlink -f "/sys/class/infiniband/$MESH_RDMA_DEVICE/device")")
[[ "$MESH_BDF" =~ ^[[:xdigit:]]{4}:[[:xdigit:]]{2}:[[:xdigit:]]{2}\.[0-7]$ ]]
MESH_DEVLINK="pci/$MESH_BDF"
sudo devlink dev info "$MESH_DEVLINK"
sudo devlink dev param show "$MESH_DEVLINK" name hairpin_num_queues
sudo devlink dev param show "$MESH_DEVLINK" name hairpin_queue_size
sudo devlink dev param show "$MESH_DEVLINK" name flow_steering_mode
sudo devlink dev eswitch show "$MESH_DEVLINK"
```

Repeat the read-only inventory for `rocep1s0f1`, `roceP2p1s0f0`, and
`roceP2p1s0f1` on every host. Retain driver/firmware information and the
observed values with the private site record. A configured queue count alone
does not prove that a TC rule is hardware resident or that RC works.

### Optional stopped-stack hairpin provisioning

Skip this section when the verified settings already match. These commands
change the NIC driver and are **not safe during serving**. Before running
them, stop the managed stack across all four hosts and stop **every other
RDMA user** of the affected ASIC, not just this model. Use a separate
management connection or physical console. An SSH alias routed over the
RoCE links can disconnect during reload.

Save the host's persistent IP/MTU configuration, routes, neighbor state,
TC rules, and devlink settings. Have an explicit restoration method for
that host's network manager before proceeding. Driver reinitialization
applies `driverinit` values but can remove/recreate driver entities and
cause additional reset or downtime depending on the driver; inspect the
actions actually performed. See
[devlink reload semantics](https://docs.kernel.org/networking/devlink/devlink-reload.html).

For **one reviewed data PCI function**, using the discovered `MESH_DEVLINK`
above and only after the stopped-stack prerequisites:

```bash
sudo devlink dev param set "$MESH_DEVLINK" name hairpin_num_queues value 4 cmode driverinit
sudo devlink dev param set "$MESH_DEVLINK" name hairpin_queue_size value 1024 cmode driverinit
sudo devlink dev reload "$MESH_DEVLINK" action driver_reinit
```

Re-establish the saved host IP/MTU configuration after reload, rediscover
the RDMA/netdev mapping, and repeat the read-only inventory before touching
another function. Do not run a blind reload loop over an entire PCI tree.
Provision the four verified functions deliberately, accounting for their
shared ASIC and disruption scope. This procedure does not change steering
mode, eSwitch mode, or firmware.

If hardware TC offload is disabled on a verified **data** netdev used by the
plan and the driver supports enabling it, enable only that interface while
the stack remains stopped:

```bash
sudo ethtool -K "$MESH_NETDEV" hw-tc-offload on
ethtool -k "$MESH_NETDEV"
```

Recheck addresses, MTUs, GID index 3, steering mode, and direct-neighbor RoCE.
Then install/start the managed mesh and require actual `in_hw` rules plus
the native all-rank RC correctness checks before model startup. The managed
installer intentionally does not reload NIC drivers or perform this bootstrap.

Use the [managed mesh service](../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md)
to configure and monitor the hardware-forwarded paths between opposite
Sparks. Model startup requires authenticated readiness from all four ranks;
unhealthy fabric stops dependent serving, and the operator explicitly
initiates recovery. Changing forwarding helpers beneath live RDMA connections
is unsupported. The bounded diagnostic mode is separate from this serving
procedure.

## Exact serving settings

| Setting | Value |
|---|---|
| Target | `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision `df116c4fb16b1d37ae43d2cfd624de26ffbc832e` |
| Speculation | Built-in MTP, depth 3, draft TP4, probabilistic proposals, standard rejection |
| External draft weights | None |
| Rank layout | TP4 / DCP4 / PP1 |
| Compute / weights | BF16 compute; ModelOpt mixed quantization, including NVFP4 routed experts and MXFP8 attention/shared-expert projections |
| Request length / sequences | 1,048,576 tokens / 16 |
| Batched-token budget | 8,192 |
| Scheduler | Asynchronous, prefill interval 2, chunked prefill, prefix caching |
| KV | FP8, 24 GiB per rank |
| Kernels | B12X attention, KDA prefill, MoE, and linear |
| Graphs | `FULL_AND_PIECEWISE`, rows 4, 8, 12, ..., 64 |
| Fusion / autotuning | All-reduce/RMSNorm fusion disabled; FlashInfer autotuning disabled |
| Loader / CPU | fastsafetensors, queue 1; OMP threads 16 |
| Serving ports | API 8015; scheduler liveness 8016; rendezvous 29775 |
| SIRCL graph control | Ports 9970/9971; submit CPU 10; progress CPU 11; inflight limit 64 |
| SIRCL eager transport | Fused dual rail; two 64 MiB operation arenas; 120-second operation timeout |
| RoCEnante | Six origin QPs per rank, two paths per peer, two operation slots, proxy CPU 13 |

The launcher warms concurrency 1 through 16. Its `DFLASH_WARMUP*` environment
names are compatibility interfaces for the shared speculation warmup entry
point; they do not select DFlash when `SPECULATION_METHOD=mtp`.
The required source-bound child image uses temperature one with thinking
disabled, selected by `SPARKRING_WARMUP_TEMPERATURE=1`. Its verified receipt
must attest both the sampling warmup helper and the managed marker source.
The parent image alone does not provide this managed quickstart contract.
This addresses the greedy-only warmup gap tracked in
[issue #214](https://github.com/FujitsuPolycom/sparkring/issues/214).
Completed warmup establishes that its requests ran, not comprehensive
sampling correctness or thinking-enabled generation coverage.

For a full MTP3 verification batch, target rows are approximately
`Q = 4 × active requests`. Draft execution and partial batches can use
different shapes. A capture list is not proof that every live step replays a
full graph.

The scheduled-token budget is 8,192. Native MTP3 in the pinned runtime
reserves zero extra parallel-drafting slots. This guide does not change
the scheduler's input capacity independently of that budget.

### Collective routing

The custom paths require a contiguous CUDA BF16 TP tensor of shape
`[Q,4096]`. `Q` is the number of rows in that collective, not context length.

| Execution / rows | Transport |
|---|---|
| Captured Q4, Q8, Q12 | RoCEnante over direct and hardware-forwarded mesh paths |
| Captured Q16, Q20, Q24, Q28, Q32 | Graph-native SIRCL |
| Captured Q36, Q40, ..., Q64 | Graph-native SIRCL |
| Eager Q1 through Q32 | RoCEnante |
| Eager Q33 through Q127 | NCCL fallback |
| Eager Q128 through Q8192 | Dual-rail fused SIRCL |
| Other signatures, DCP, and other collectives | Existing fallback dispatch |

RoCEnante all-gather is not enabled. RoCEnante sends both path sets below
Q24 and uses direct-then-diagonal scheduling from Q24. WQE tiling and
hardware rate limiting are not enabled. The source-bound overlay selects
these settings from its bundle configuration; the image's model kernels are
not replaced by the RoCEnante package.

### Persistent cache

SparkCache runs read-write with `tail-cow-v2` publication, a 300-second shared
GPU-prefix lease, 40 GiB capacity per rank, a 32 GiB low watermark, and no TTL
expiration. Publication spans range from 4,096 to 1,048,576 tokens. Restore
uses eight I/O workers, eight load threads, eight pending operations, and two
256 MiB arenas. Async capture uses two 3 GiB slots per rank; these are
separate from SIRCL's two 64 MiB transport arenas.

Native MTP's cache draft identity is the target checkpoint. The profile uses
the dedicated namespace in `pins.json`; external-draft-tagged entries must
not be renamed into it. The `draft_policy=separate` field describes cache
registration layout, not an external draft model. The linked functional record
includes an uncached publication and stopped-container restoration under this
identity. It covers one recall prompt and does not qualify other checkpoints,
all context lengths, or concurrent cache workloads.

## Obtain the image and target

Pull the published Linux/ARM64 managed image on every Spark. No local build
is required. The [registry receipt](../runtime/glm53-spark-mtp3-mesh/public-image.json)
records anonymous access and its match to the tested image. The separate
[content receipt](../runtime/glm53-spark-mtp3-mesh/image-receipt.json) is the
input accepted by the renderer, installer, and native qualification runner.
Keep both with the checkout; do not substitute `public-image.json` for the
content receipt.

```bash
mtp_image='ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:23f00af873ccc784cfb742b7be2a29c6d3c20ebec9741843c025320bb9c04685'
mtp_image_id='sha256:26273b8e358df139ae913610a5d43084ff0fd08aafe282ef633a3bc74afefe47'
docker pull "${mtp_image}"
test "$(docker image inspect "${mtp_image}" --format '{{.Id}}')" = "${mtp_image_id}"

hf download local-inference-lab/GLM-5.3-Flash-NVFP4-Spark \
  --revision df116c4fb16b1d37ae43d2cfd624de26ffbc832e \
  --local-dir /srv/models/GLM-5.3-Flash-NVFP4-Spark/df116c4fb16b1d37ae43d2cfd624de26ffbc832e
```

Distribute that checkpoint directory and the identical verified child image
to all four ranks. Follow the [image and model distribution procedure](GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md)
for transfer mechanics, but do not download or configure its external draft.
Do not substitute a mutable model branch or a different image tag. The
launcher checks the target configuration and index hashes; operators must
also verify every indexed weight shard after transfer.

For example, if all hosts use the checkpoint path above, copy it from the
download host over trusted SSH aliases defined in your topology:

```bash
mtp_target='/srv/models/GLM-5.3-Flash-NVFP4-Spark/df116c4fb16b1d37ae43d2cfd624de26ffbc832e'
for host in spark-r1 spark-r2 spark-r3; do
  ssh "$host" mkdir -p "$mtp_target"
  rsync -aH --partial --info=progress2 "$mtp_target/" "$host:$mtp_target/"
  rsync -aHnc --itemize-changes "$mtp_target/" "$host:$mtp_target/"
done
```

Provision destination directory permissions first. The final checksum dry
run must report no content differences; it verifies transfer equality, not
independent model-publisher authenticity. You may instead download the exact
revision independently on every host. Do not download an external drafter.

## Extract the transport bundle and host marker

On a host with the published image already pulled, extract its packaged
artifacts without starting a container. Provision `/srv/sparkring` for the
operator first. Both artifact destinations and the temporary container name
must be unused:

```bash
mkdir -p /srv/sparkring/artifacts
test ! -e /srv/sparkring/artifacts/mtp3-mesh-bundle
test ! -e /srv/sparkring/artifacts/mlx5-rdma-tx-marker
docker create --name sparkring-mtp3-extract --entrypoint /bin/true "$mtp_image_id"
docker cp sparkring-mtp3-extract:/opt/spark-sircl /srv/sparkring/artifacts/mtp3-mesh-bundle
docker cp sparkring-mtp3-extract:/opt/sparkring/bin/mlx5-rdma-tx-marker /srv/sparkring/artifacts/mlx5-rdma-tx-marker
docker rm sparkring-mtp3-extract
cp runtime/glm53-spark-mtp3-mesh/image-receipt.json /srv/sparkring/verified-image-receipt.json

printf '%s  %s\n' \
  '4204fabc93303226b9a120b094ef3c82ed4aadd1d7f97cfbe291204c027ed45f' \
  '/srv/sparkring/artifacts/mtp3-mesh-bundle/sparkring-overlay-manifest.json' \
  '2828c07e4255c4962c77425be2c88969e7eb7dd4b1bf9e36485bc705bb5d6d64' \
  '/srv/sparkring/artifacts/mlx5-rdma-tx-marker' | sha256sum --check
```

Do not run `profile.py bundle` on this extracted child bundle: that command
composes a child bundle from the **parent** image's SIRCL input and is only
needed for source reproduction. The published child already contains the
canonical bundle. The renderer and installer verify manifest entries.

Keep these artifacts outside `/opt/sparkring/managed-mesh`, which must not
exist before service installation. Check the extracted marker's host-library
linkage before attachment. For an optional local source build, follow
[image reproduction](../runtime/glm53-spark-mtp3-mesh/IMAGE_BUILD.md);
use the resulting verified receipt instead of the published receipt if its
image identity differs.

## Describe and render the site

Copy `runtime/glm53-spark-mtp3-mesh/site.example.json` and
`fabric.example.json` into a private site directory. Replace every synthetic
address, MAC, path, and device mapping with the verified four-rank inventory.
The site document must identify the marker executable by its actual SHA-256.
Extract the managed executable from the verified child image using the
[image packaging guide](../runtime/glm53-spark-mtp3-mesh/IMAGE_BUILD.md).
The installer verifies its source and binary identities against the receipt.

Create a private site directory and copy the templates without overwriting
an existing deployment:

```bash
mkdir -p /srv/sparkring/site
cp -n runtime/glm53-spark-mtp3-mesh/site.example.json /srv/sparkring/site/mtp3-mesh.json
cp -n runtime/glm53-spark-mtp3-mesh/fabric.example.json /srv/sparkring/site/fabric.example.json
```

Edit those private copies. At minimum, review every field below:

| Input | Required value |
|---|---|
| `topology_file` | The sibling private fabric JSON filename |
| `management_addresses` and each rank's SSH alias | Four distinct reachable hosts in rank order; passwordless SSH plus authorized `sudo -n` |
| Fabric ports and reciprocal peers | Actual direction/function, netdev, RDMA device, IPv4, MAC, and cable mapping |
| `model_roots` | Actual immutable target directory on each host |
| `cache_roots` | Dedicated writable persistent/JIT cache root on each host; preserve across model restart |
| `bundle_root` | `/srv/sparkring/artifacts/mtp3-mesh-bundle`, or your verified staged equivalent |
| `marker_binary` | `/srv/sparkring/artifacts/mlx5-rdma-tx-marker`, or your extracted equivalent |
| `marker_binary_sha256` | `inside_image.marker_binary_sha256` from the verified image receipt, not the template's diagnostic binary hash |
| `container_prefix` | An unused prefix reserved for this deployment |
| `state_root` | Retain the schema's planned state path; the managed service separately uses `/run/sparkring-mesh` |

Retain the topology's `bounded_runtime_seconds: 7200` compatibility field.
It describes diagnostic-plan arguments and is validated by the renderer;
it does not impose a runtime limit on the managed service. Do not change it
to zero to request persistent serving. The managed service supplies
`--managed` itself and does not execute the bounded marker argument arrays.

```bash
python3 runtime/glm53-spark-mtp3-mesh/profile.py render \
  --site /srv/sparkring/site/mtp3-mesh.json \
  --bundle /srv/sparkring/artifacts/mtp3-mesh-bundle \
  --image-receipt runtime/glm53-spark-mtp3-mesh/image-receipt.json \
  --output build/mtp3-mesh-launch
```

Review `rank0.env` through `rank3.env` and `fabric-plan.json`. The latter
contains argument arrays, not permission to execute them. Stage the bundle
at the site's `bundle_root` on every rank. Stage the rendered directory at
`/srv/sparkring/mtp3-mesh-launch`, and retain a checkout for the read-only
inspector. Use identical bytes on all ranks.

Distribute the checkout at the same reviewed commit, rendered launch, image
receipt, canonical bundle, and extracted marker to every host. The private
site JSON is identical on all hosts: the installer chooses its local rank.
Do not run `make_example.py` on a configured private site. Create each
`cache_roots` directory with sufficient capacity before container creation.
Example transfers, after provisioning the destination paths and permissions:

```bash
for host in spark-r0 spark-r1 spark-r2 spark-r3; do
  rsync -a --checksum build/mtp3-mesh-launch/ "$host:/srv/sparkring/mtp3-mesh-launch/"
  rsync -a --checksum /srv/sparkring/artifacts/ "$host:/srv/sparkring/artifacts/"
  scp /srv/sparkring/verified-image-receipt.json "$host:/srv/sparkring/verified-image-receipt.json"
done
```

The artifact directory in this example contains only public transport files,
not the health key or cache contents. Do not transfer an entire private work
directory. Protect configuration and code from unprivileged modification.

The SIRCL endpoint slots are ordered by peer rank: slot 0 connects rank `r`
to `r XOR 1`, and slot 1 connects it to `r XOR 3`. On this four-node cycle,
odd ranks therefore use counter-clockwise before clockwise. The renderer
applies that order to both primary and secondary SIRCL functions. Do not
replace it with clockwise-first on every rank. RoCEnante's canonical HCA
inventory is a separate ordering and remains unchanged.

## Install and operate the managed fabric

Follow the [managed mesh installation and lifecycle guide](../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md).
It is the executable host-operations procedure for this profile:

1. Stage identical image, bundle, marker, and rendered site inputs.
2. Pre-create stopped containers using `SPARKRING_CREATE_ONLY=1`.
3. Install each rank with a shared private health key and epoch.
4. Establish authenticated four-rank mesh readiness.
5. Run native RC correctness checks before starting the model.
6. Start the model through the managed model units and verify completed
   temperature-one warmup.

The service installs only exact missing planned routes, neighbors, and
hardware rules. Matching preexisting state is adopted, not owned for removal.
It never flushes unrelated networking. The service monitors the configured
paths and their helper processes for the lifetime of the deployment.
The host installer copies an explicit allowlist of 16 required source files,
not an entire checkout or user work directory. Private inputs are supplied
separately; do not place keys or cached results into source trees for transfer.

The source ASIC changes Ethernet type `0x0800` to `0x88b5`. The intermediate
ASIC restores `0x0800`, rewrites the destination MAC, and redirects to the
opposite physical port. RC transport headers and payload are preserved.
Endpoints still use pinned, GPU-mapped host memory and CPU-side posting.

After managed startup, require the combined four-rank readiness check:

```bash
python3 runtime/glm53-spark-mtp3-mesh/wait_managed_ready.py \
  --launch /srv/sparkring/mtp3-mesh-launch --timeout 900 \
  --output /path/to/private-receipts/model-ready.json
```

Create the receipt's parent directory first and use an absent output path.
The tool requires all four containers to be running and Docker-health healthy,
plus rank-zero API health and scheduler liveness HTTP 200. It is read-only
and does not treat systemd active state as model readiness.

With the example container prefix, inspect serving logs and endpoints:

```bash
ssh spark-r0 'docker logs -f --tail 100 glm53-spark-mtp3-mesh-r0'
curl --fail http://RANK0_MANAGEMENT_ADDRESS:8015/health
curl --fail http://RANK0_MANAGEMENT_ADDRESS:8015/v1/models
curl --fail http://RANK0_MANAGEMENT_ADDRESS:8016/liveness
```

Use model name `glm-5.3-flash-spark`. Confirm native MTP depth three, completed
warmup, all ranks healthy, and the expected image/bundle identities. API
health alone is not scheduler liveness, numerical correctness, or end-to-end
transport qualification.

For shutdown or recovery, use `managed_cluster.py stop-model`, `down`, or
`recover` as documented in the managed guide. Drain requests first. The
coordinator verifies all four pinned model containers are stopped before
planned fabric removal. Hot replacement beneath live QPs and automatic model
restart are unsupported. No unattended-high-availability claim is made.

## Run bounded qualification checks

Status: **implemented** test tooling. These commands produce evidence for
individual checks; they do not qualify failure containment or unattended
serving. Run from the repository root with Python and SSH available. Keep
receipts outside version control: they contain resolved site information.

### Native RC correctness before model startup

After managed `up` passes the four-rank readiness gate, test the packaged
child image before starting any model containers. The following
command prints a plan without contacting hosts:

```bash
python3 runtime/glm53-spark-mtp3-mesh/qualification/run_native.py \
  --launch build/mtp3-mesh-launch \
  --image-receipt runtime/glm53-spark-mtp3-mesh/image-receipt.json \
  --output /path/to/private-receipts/native
```

Review the SSH targets, exact image ID, temporary container names and cache
mounts. Run the same command with `--execute-authorized` to execute it. The
output directory must not exist. This requires a render made with that exact
image receipt; the parent-image-only mode does not contain the embedded mesh
source that the standalone probe selects.

The runner refuses running containers on any rank and never stops them. It
uses the site's management network and the source-bound six-QP map. Four
concurrent ranks check Q4, Q20, Q28 and Q64 BF16 payloads, changed graph inputs,
poisoned outputs, repeated slot reuse, and all 24 origin-QP completion/byte
balances. Require every cell to pass with zero completion errors. Native
RoCEnante checks do not test the serving adapter's SIRCL routing for Q20/Q28.

Each container has a 240-second process timeout. If SSH or a rank fails,
inspect the named test containers on all hosts before proceeding; a timeout
is not proof of cleanup. No route, marker, or traffic-control cleanup is
performed by the runner. Its three timing samples per cell are diagnostics,
not performance claims. Managed markers remain under supervision throughout
model loading and qualification; the test runner does not own or stop the
mesh service.

### Model output and persistent-cache restoration

Start the four-rank model through `managed_cluster.py start-model` and wait
for completed speculation warmup. Set the endpoint to the rank-zero
management address:

```bash
export MTP_ENDPOINT='http://RANK0_MANAGEMENT_ADDRESS:8015'
mkdir -p /path/to/private-receipts
python3 runtime/glm53-spark-mtp3-mesh/qualification/recall_prompt.py \
  --output /path/to/private-receipts/recall.txt
python3 runtime/glm53-spark-mtp3-mesh/qualification/model_cache.py \
  --endpoint "$MTP_ENDPOINT" --model glm-5.3-flash-spark \
  --kind semantic --output /path/to/private-receipts/semantic-before \
  --execute-authorized
python3 runtime/glm53-spark-mtp3-mesh/qualification/model_cache.py \
  --endpoint "$MTP_ENDPOINT" --model glm-5.3-flash-spark \
  --kind persistent --phase before-restart \
  --prompt-file /path/to/private-receipts/recall.txt \
  --expected-text 'cobalt orchard lantern' --max-tokens 512 --temperature 1 \
  --output /path/to/private-receipts/prefix-before --execute-authorized
```

Without `--execute-authorized`, the request tool prints its plan and makes no
API calls. Requests default to temperature one, a fixed seed, and a bounded output.
Require the exact semantic answer and `stop`. The recall generator creates
129,455 UTF-8 bytes, with SHA-256
`3d2bc5228895566b1497e6f35f6c5aa051685f99438f27226134dfcfab15c277`.
It makes no API calls and refuses to overwrite a file. The long-prefix request
records actual token usage; text length is not a token-count guarantee. Retain
successful publication and compatible cache-identity evidence on every rank
before restarting. Preserve the cache directory and namespace.

For this cache check, preserve the pinned container IDs, image, cache mounts,
namespace, and forwarding contract. Drain API traffic and use the managed
all-rank model-stop barrier; do not invoke the launcher again or independently
restart Docker containers:

```bash
python3 runtime/glm53-spark-mtp3-mesh/managed_cluster.py stop-model \
  --site /srv/sparkring/mtp3-mesh-launch/site.json \
  --output /path/to/private-receipts/model-stop.json --execute-authorized

python3 runtime/glm53-spark-mtp3-mesh/managed_cluster.py start-model \
  --site /srv/sparkring/mtp3-mesh-launch/site.json \
  --output /path/to/private-receipts/model-restart.json --execute-authorized
```

Do not proceed after a failed phase. Inspect its receipt and use the managed
recovery procedure if the fabric or a supervisor failed. The same-fabric
model restart does not require marker renewal.

A successful model-unit start is not readiness. Inspect rank logs, completed
speculation warmup, API health and scheduler liveness as described above.
Restarting replaces GPU prefix state; simply repeating a request within one
process cannot prove disk restoration. Wait for compatible inventory discovery
on all ranks, then run:

```bash
python3 runtime/glm53-spark-mtp3-mesh/qualification/model_cache.py \
  --endpoint "$MTP_ENDPOINT" --model glm-5.3-flash-spark \
  --kind persistent --phase after-restart \
  --prompt-file /path/to/private-receipts/recall.txt \
  --expected-text 'cobalt orchard lantern' --max-tokens 512 --temperature 1 \
  --reference /path/to/private-receipts/prefix-before \
  --output /path/to/private-receipts/prefix-after --execute-authorized
python3 runtime/glm53-spark-mtp3-mesh/qualification/model_cache.py \
  --endpoint "$MTP_ENDPOINT" --model glm-5.3-flash-spark \
  --kind semantic --output /path/to/private-receipts/semantic-after \
  --execute-authorized
```

The tool verifies identical request bodies across the restart and records
visible-output agreement, raw responses, usage, and separate prefix/external
cache metric deltas. It deliberately does not set `persistent_restore_proven`
to true. Acceptance additionally requires successful restoration logs on all
four ranks, a nonzero external-prefix-hit delta, the matching native-MTP cache
identity, and no transport/model errors. RAM-prefix hits, shorter latency,
or a successful HTTP response cannot substitute for those checks. A clean
recomputation after late inventory discovery is a cache miss, not a restore.

These canaries are bounded semantic checks, not general model-quality tests.
Retain native correctness, serving routing evidence, restart evidence and
persistent-cache evidence as distinct results.

The generated recall workload places a known phrase before 700 varied neutral
records and asks for that phrase at the end. Alternative private UTF-8 prompt
files are supported through `--prompt-file` and `--expected-text`, but identify
them separately in any evidence. Verify actual prompt-token counts from the
response. The tool
requires identical submitted requests for the restart comparison; a fixed
seed does not guarantee identical generated sequences at temperature one.
Visible-output equality is diagnostic, not a restart pass criterion. A
single failed semantic canary is inconclusive and requires repeated controls;
it must not be erased or replaced by a passing workload.

[Profile validation: performance, accuracy, and restart checks](PROFILE_VALIDATION.md).
