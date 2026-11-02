# SIRCL

SIRCL, the **Switchless Inference RDMA Collective Layer**, moves tensors
between the DGX Sparks of a model while they serve. Every collective of the
tensor-parallel and decode-context-parallel (DCP) groups, such as the
all-reduce that sums each Spark's partial result, runs over RDMA on the
Sparks' ConnectX-7 cables. There is no switch, so SIRCL also reaches Sparks
that share no cable, through the Sparks between them.

`sparkring install` uses SIRCL by default on an image that carries it, such
as release 2026.10.5's. The [architecture overview](overview.md) shows where
SIRCL sits; this page explains how it works and where each part is used.

## How SIRCL reaches every Spark

`sparkring setup` cables two to eight Sparks as a pair, a line or a ring
([topology](overview.md#topology)). A SIRCL **group** is the set of Sparks
that serve one model, or one DCP subgroup of it. Inside a group:

- **Cable neighbours** send directly. Each ConnectX-7 port has two network
  functions, so each neighbour is reached over two **lanes**: one RDMA
  connection (queue pair) on each function. A message is split into stripes
  across the lanes.
- **Other members** are reached through **relays**: the Sparks in between
  forward the packets in their ConnectX-7 hardware. On a ring of eight,
  Spark 0 reaches Spark 3 through Sparks 1 and 2, and Spark 4 through three
  Sparks either way round. A relay's CPU and GPU take no part.
- **Lanes follow the shortest way** over the group's own cables. Several
  groups on one fabric share no cable and no Spark, so their traffic never
  mixes.

A lane crosses at most three relays, the qualified limit
(`SIRCL_MAX_RELAYS` raises it, **research-only**). That covers a whole ring
of up to eight Sparks and a line of up to five. A line of six or more is
**unsupported**.

Relaying needs the relay table and the ConnectX hairpin setting on every
Spark between two cables; `sparkring setup` installs both
([the relay table](../operations/install-reference.md#the-relay-table)). A
marker on the sending Spark tags each relayed RDMA packet with the number of
relays left, and each relay passes it on and lowers the count. A relay holds forwarded bytes in a
hairpin queue of 512 KiB that cannot pause the sender, so SIRCL keeps the
unacknowledged bytes of every relayed lane within a **forward window**: at
most 75 % of each queue the lane crosses, shared among the lanes that cross
it. Relays forward only these tagged RDMA packets, not TCP, so SIRCL sets
up its sessions over the serving engine's CPU process group.

SIRCL has two session generations:

| | Ring sessions | Four-rank native sessions |
|---|---|---|
| Groups | 2 to 8 ranks: pairs, lines, rings, several independent groups on one fabric, DCP subgroups; lanes through up to three relays | Exactly four ranks on a four-Spark ring |
| Code | [`spark_transport/sircl`](../../spark_transport/sircl/README.md): Python package `sparkring_sircl` with a native progress thread built per source digest (`roce_proxy-<digest>.so`) | [`spark_transport/`](../../spark_transport/README.md) C/C++ sources, built as `libspark_transport_capi.so` |
| vLLM integration | Platform and general plugin `sircl` ([adapter](../../spark_transport/sircl/sparkring_sircl/vllm/README.md)) | Startup hooks in [`integrations/vllm`](../../integrations/vllm/README.md), turned off as a whole by `SPARK_TP4_ENABLED=0` |
| Used by | `sparkring install` and the SIRCL serve launcher | The four-Spark catalog profiles under [Profile use](#profile-use) |
| Status | Per component in the [status table](../../spark_transport/sircl/STATUS.md#component-status): one-shot all-reduce and all-gather **qualified** on the eight-Spark ring; serving through vLLM **research-only** | **implemented**; qualification is per profile, under [Profile use](#profile-use) |

## Ring sessions

A **session** is one rank's part of one group's collectives. It owns:

- an **arena**: pinned host memory registered with every RDMA device it
  opens. Peers write into it and GPU kernels read it in place. The GB10 GPU
  cannot register its own memory for RDMA, so the arena is host memory.
- a **command ring** through which GPU kernels hand work to the progress
  thread, and a **native progress thread** (plain C over libibverbs) that
  posts the RDMA writes.

One all-reduce runs like this: the kernel copies the rank's input into the
arena and rings a doorbell; the progress thread writes the data to every
peer, striped over the lanes, and then a 4-byte flag per lane; each kernel
waits for every peer's flags and adds the values in float32 in rank order.
Every rank therefore gets identical bits.

**Setup** is collective. Each rank derives its lanes from the group's
layout and the fabric document (`SIRCL_FABRIC_DOCUMENT`, schema
`sparkring-fabric/v1`), which names each Spark's RDMA devices. The ranks
exchange their settings and connection records, check that every shared
setting matches, connect, and prove every lane with one small write. A
mismatch fails on every rank with one message naming each rank and reason.

**Failure.** A flag wait has a time limit: 600 s during startup, 20 s while
serving. A wait that times out marks the session failed; every later call
raises, and the vLLM adapter checks every session at each worker step, so
the model stops instead of serving wrong output. Each rank writes a receipt
per group that names what carried each collective.

**Collectives**: one-shot and two-shot all-reduce, all-gather,
reduce-scatter and all-to-all, and large messages in pieces, as a chain
between cable neighbours or as a ring. Every schedule gives identical bits
on every rank. A **tuning table** chooses the schedule, piece size and
launch grid per collective and message size. The installer's default table
([`sircl-tuning-defaults.json`](../../runtime/common/sircl-tuning-defaults.json))
has measured rows for a pair, a line of four, a ring of four and a ring of
eight; `sudo sparkring fabric tune` measures one for your fabric
([measure the tuning table](../operations/install-reference.md#measure-the-tuning-table)).
A table chooses only among SIRCL's own settings and never sends a call to
NCCL.

The [package README](../../spark_transport/sircl/README.md) describes the
interface and settings; the [runbook](../../spark_transport/sircl/RUNBOOK.md)
covers the ring harness, which tests and tunes sessions outside vLLM, and the
standalone relay plan installer.

## Implemented boundary

Ring sessions run what the
[status table](../../spark_transport/sircl/STATUS.md#component-status) lists:
layouts whose lanes cross at most three relays, BF16, FP16 and FP32
reductions, and any plain dtype in gathers. Swing all-reduce runs only in the
ring harness (**research-only**); direct mlx5 posting and phase tracing are
**unsupported** and refused at setup. Sessions read host networking and never
configure it.

Four-rank native sessions keep RDMA sessions, registered arenas and
device-published command rings for exactly four ranks on a four-Spark ring.
Captured CUDA graphs submit work through device command descriptors; native
progress threads perform the host protocol. The native interface and the
adapter check tensor geometry, so these sessions do not accept arbitrary
dtypes or shapes. Other layouts use ring sessions. Patched NCCL has its own
[pair and ring configurations](../../spark_transport/nccl/README.md).

## Libraries built on SIRCL

Two libraries carry SIRCL's cabling to programs other than SparkRing's vLLM
plugins:

- **libsircl** implements NCCL's public C API on SIRCL's wire protocol, so a
  program written against NCCL loads it in place of NCCL
  ([libsircl](libsircl.md)). `sparkring install --transport libsircl` serves
  vLLM's PyNccl on it (**research-only**).
- **net-sircl** is an NCCL network plugin: NVIDIA NCCL keeps its own
  algorithms and SIRCL's relays carry its traffic. It lives in the separate
  repository [FujitsuPolycom/net-sircl](https://github.com/FujitsuPolycom/net-sircl)
  (**research-only**); `sparkring install` does not use it.

## Composition with hardware-forwarded mesh

The four-Spark mesh composition combines four-rank SIRCL, an adapted
RoCEnante all-reduce and patched NCCL. RoCEnante's selected opposite-peer
operations cross two cables through an intermediate ConnectX-7; calls
outside its admission rules go to the saved SIRCL or NCCL backend. A
per-deployment mesh service installs its relays. Only the retired
`glm53-spark-mtp3-managed-mesh-tp4` profile uses it. The
[mesh runtime contract](../../runtime/glm53-spark-mtp3-mesh/README.md)
identifies its dispatch configuration, source packages and evidence;
RoCEnante's origins are in its
[source attribution](../../third_party/b12x_roce/README.md).

Ring sessions use the same kind of hardware relay. On a fabric that
`sparkring setup` formed, the relay table serves every transport, and a
four-Spark ring's prepared deployments need no mesh service.

## Profile use

**Ring sessions.** `sparkring install` serves every installer profile on
them when the image carries the SIRCL layer (image lock v3,
[SIRCL layer](../../runtime/images/installer-images.md#sircl-layer)) and
`sparkring setup` installed the fabric's relay table; otherwise it uses the
prepared transport
([transport and receipts](../operations/install-reference.md#transport-and-receipts)).
No profile names a transport. The installer's transport adapter,
[`runtime/common/transport.py`](../../runtime/common/transport.py), sets
the same session settings as the SIRCL serve launcher's plan for the same
group. The six-Spark profile `glm53-nvfp4-csf-tp6` (**research-only**) routes
its lanes over the whole ring (`SIRCL_ROUTE_SCOPE=fabric`) so none crosses
more than three relays. The serve and bundle launchers also run ring
sessions outside the installer
([serve runbook](../../spark_transport/sircl/sparkring_sircl/vllm/RUNBOOK.md#installer-profiles-on-the-ring-of-eight)).

**Four-rank native sessions:**

- The GLM-5.3-Flash four-Spark profiles of shared runtime 2026.09.3, such as
  [`glm53-flash-spark-tp4-dcp1-sparkcache`](../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md),
  set `SIRCL_ENABLED=1` and run SIRCL's tensor-parallel and vocabulary
  collectives with patched NCCL for the rest
  ([`runtime/common/glm_tp4.py`](../../runtime/common/glm_tp4.py)). That
  profile is **qualified** within its release record's scope.
- The retired GLM-5.2 EXL3 3.5-bpw profile, a TP4/DCP4 configuration that
  SparkRing does not support, uses SIRCL for qualified tensor-parallel
  all-reduce and vocabulary collective families; patched NCCL handles the
  rest.
- The DeepSeek-V4-Flash-0731 quickstart uses patched NCCL. Its width-4096
  SIRCL CUDA-graph configuration is research-only; see the [DeepSeek SIRCL
  evidence record](../../performance/records/deepseek-v4-flash/sircl-width4096-nccl-ab-20260822.md).
- The [GLM-5.3 DFlash2 operator image](../../runtime/glm53-flash-jj-r8-gb10/glm53-dcp4-sircl-public-image-receipt.json)
  belongs to a retired profile and embeds a source-bound SIRCL bundle. Its
  receipt records **qualified** four-rank TP4/DCP4 functional checks, from
  startup to injected failure containment; its throughput matrix is
  **research-only**. See the [GLM-5.3 runtime
  guide](../../runtime/glm53-flash-jj-r8-gb10/README.md) and the [vLLM
  adapter contract](../../spark_transport/integrations/vllm/README.md).
- The Qwen3.8-27B EXL3 K5/K6 pair and ring profiles use patched NCCL with
  SIRCL off: their width-5,120 tensor-parallel shape is outside four-rank
  SIRCL.

## Persistent host rail configuration

Every SIRCL session needs its fabric interfaces to keep their IPv4 address,
MTU 9000 and RoCE v2 GID after a reboot. `sparkring setup` makes the fabric
it forms survive a reboot, and boot services restore the rest:
`sparkring-hairpin.service` applies the
hairpin setting, `sparkring-fabric.service` restores routes and the relay
table, and `sparkring-relay-marker.service` runs the markers
([boot and reboot](../operations/install-reference.md#boot-and-reboot)). A
relay plan installed by the standalone `sircl-fabric up` is not restored at
boot; run it again after every reboot.

For the four-rank profiles' dedicated secondary rails,
[`configure_sircl_rail.py`](../../scripts/configure_sircl_rail.py) creates one
NetworkManager profile at a time. Its default mode prints the plan;
`--verify` performs read-only checks; `--execute` requires root and the
confirmation `CONFIGURE_SIRCL_RAIL`. Run it on each rank:

```bash
rail_args=(
  --management-interface REPLACE_MANAGEMENT_NETDEV
  --interface REPLACE_SECONDARY_NETDEV
  --address-cidr REPLACE_LOCAL_SECONDARY_ADDRESS/PREFIX
  --peer-address REPLACE_SECONDARY_PEER_ADDRESS
  --rdma-device REPLACE_SECONDARY_RDMA_DEVICE
  --rdma-port 1
  --gid-index REPLACE_VERIFIED_GID_INDEX
  --mtu 9000
)
python scripts/configure_sircl_rail.py "${rail_args[@]}"            # offline plan
sudo python scripts/configure_sircl_rail.py "${rail_args[@]}" \
  --execute --confirmation CONFIGURE_SIRCL_RAIL                     # host change
python scripts/configure_sircl_rail.py "${rail_args[@]}" --verify   # read-only check
```

The helper disables IPv6 on the rail. Managed GLM-5.3 mesh deployments keep
IPv6 link-local addressing for their fixed GID index 3 and use the [mesh host
setup](../GLM53_SPARK_MESH_HOST_SETUP.md#6-configure-persistent-data-ipv4-and-mtu)
instead. The [secondary-rail persistence
validation](../../runtime/glm53-flash-jj-r8-gb10/sircl-secondary-rail-persistence-live-validation.json)
records eight rails on four GB10 ranks passing every check after a reboot.

## Operational invariants

- All ranks of a session share one layout and one set of shared settings;
  each derives its own lanes and devices from the layout.
- NCCL runs only where the operator opts in (`--nccl auto`, which sets
  `SIRCL_NCCL=auto`; the default is `never`). The group's cabling then bounds
  it: every collective on a pair, NCCL's ring algorithm alone
  (`NCCL_ALGO=Ring`, `NCCL_SKIP_TREE_CONNECT=1`) on a whole ring, nothing on
  a line or any group with relayed lanes.
- A collective a session declines goes to the caller's own path; on a group
  where NCCL may not run, the vLLM adapter refuses it instead of letting NCCL
  connect Sparks that share no cable.
- Four-rank dual-rail prefill uses both RDMA functions of each cabled ring
  edge; it needs neither extra cables nor diagonal links.
- Transport evidence does not establish model correctness or performance
  unless the corresponding profile result states those conditions.

Four-rank deployment commands and limits are in the
[GLM-5.3-Flash four-Spark quickstart](../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md),
[GLM-5.2 quickstart](../../profiles/glm52-exl3-r7-3.5bpw/README.md),
[DeepSeek quickstart](../operations/deepseek-0731.md),
[Qwen3.8-27B pair quickstart](../../profiles/qwen38-27b-exl3-k5k6-pair/README.md) and
[Qwen3.8-27B ring quickstart](../../profiles/qwen38-27b-exl3-k5k6/README.md).
