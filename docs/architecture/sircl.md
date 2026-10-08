# SIRCL

SIRCL is the **Switchless Inference RDMA Collective Layer**, SparkRing's
collective transport between DGX Sparks over RoCE without a switch. It
carries tensor-parallel and decode-context-parallel collectives for groups of
2 to 8 Sparks cabled as a pair, a path or a cycle. SIRCL is the end state of
SparkRing's communication layer; the `prepared` RoCEnante transport and
patched NCCL serve the published installer profiles and retained images. SIRCL
has two session generations:

| | Ring sessions | Four-rank native sessions |
|---|---|---|
| Groups | 2 to 8 ranks: pairs, paths, cycles, several independent groups on one fabric, DCP subgroups; members without a shared cable reach each other through up to three ConnectX-7 relays | exactly four ranks on a four-Spark cycle |
| Code | [`spark_transport/sircl`](../../spark_transport/sircl/README.md): Python package `sparkring_sircl` and a native progress thread built per source digest (`roce_proxy-<digest>.so`) | [`spark_transport/`](../../spark_transport/README.md) C/C++ sources, built as `libspark_transport_capi.so` |
| vLLM integration | platform and general plugin `sircl` ([adapter](../../spark_transport/sircl/sparkring_sircl/vllm/README.md)) | startup hooks in [`integrations/vllm`](../../integrations/vllm/README.md), turned off as a whole by `SPARK_TP4_ENABLED=0` |
| Role | the default generation: every SIRCL deployment outside the retained images uses it | retained images and the retired profiles under [Profile use](#profile-use) |
| Status | per component in the [ring-session status table](../../spark_transport/sircl/STATUS.md#component-status): one-shot all-reduce and all-gather **qualified** on the eight-Spark ring; serving through vLLM **research-only** | **implemented**; qualified only for the retained images named under [Profile use](#profile-use) |

No profile names a transport. `sparkring install` runs an installer profile
on ring sessions, with NCCL off, when its image carries the SIRCL layer (image
lock v3, [SIRCL layer](../../runtime/images/installer-images.md#sircl-layer))
and `sparkring setup` recorded the fabric with its relay table; elsewhere it
uses the `prepared` RoCEnante transport and patched NCCL
([transport and receipts](../operations/install-reference.md#transport-and-receipts)).
The published installer images carry no SIRCL layer. The serve and bundle
launchers also run ring sessions outside the installer ([serve and bundle
runbook](../../spark_transport/sircl/sparkring_sircl/vllm/RUNBOOK.md)).

## Ring sessions

A session spans the ranks of one group. Every rank reaches every peer over one
or two lanes, each a queue pair between one local RDMA function and one
function of the peer. A rank's route map (`SIRCL_PEER_ROUTES`) names the
local device of every lane; lanes follow the shortest paths of the group's
own cables, lane 0 on the primary and lane 1 on the secondary function. The
device names are the DGX OS names unless a fabric document
(`SIRCL_FABRIC_DOCUMENT`, schema `sparkring-fabric/v1`) names others.

Members that share no cable are joined through the ConnectX-7 of every member
between them. The relay plan tags each relayed lane's packets by destination
with the number of relays left, and relay filters forward them out of the
other port in hardware. A relay's hairpin queue holds 512 KiB and cannot pause
its sender, so every relayed lane keeps a forward window of unacknowledged
bytes within 75 % of the queue it shares.

A session owns a pinned host arena registered on every opened device, a
device command ring and a native progress thread. Kernels stage a payload and
ring a doorbell; the progress thread writes each lane's stripe and then its
flag. Setup is collective over the serving engine's CPU process group: ranks
agree on every shared setting, validate every connection record, connect and
prove every lane. A flag wait past its time limit poisons the group, and every
later call raises. Each rank writes a receipt per group that names the
collectives' carriers.

Collectives: one-shot and two-shot all-reduce, all-gather, reduce-scatter and
all-to-all, and large messages in pieces, as a chain between cable neighbours
or as a ring. Every schedule gives identical bits on every rank. The
[package README](../../spark_transport/sircl/README.md) describes the
interface and settings; the [runbook](../../spark_transport/sircl/RUNBOOK.md)
the ring harness and the relay plan installer.

## Implemented boundary

Ring sessions run what the [status
table](../../spark_transport/sircl/STATUS.md#component-status) lists:
layouts with lanes through at most three relays, BF16, FP16 and FP32
reductions, any plain dtype in gathers. Swing all-reduce, direct mlx5 posting
and phase tracing are not offered by sessions. Sessions read host networking
and never configure it.

Four-rank native sessions maintain RDMA sessions, registered arenas and
device-published command rings for exactly four ranks on a four-Spark cycle.
Captured CUDA graphs submit work through device command descriptors; native
progress threads perform the host protocol. The pairwise-exchange schedule
uses two perfect matchings of the cycle; bidirectional and fused variants use
their own ring schedules. The native interface and the adapter validate
tensor geometry, so a model-independent transport does not imply arbitrary
dtype or shape support. This generation stays for the retained images and
profiles under [Profile use](#profile-use); new layouts use ring sessions.
Patched NCCL has its own [pair and cycle
configurations](../../spark_transport/nccl/README.md).

## Composition with hardware-forwarded mesh

The `prepared` transport's four-Spark mesh composition combines four-rank
SIRCL, an adapted RoCEnante all-reduce and patched NCCL. RoCEnante's selected
opposite-peer operations cross two physical links through an intermediate
ConnectX-7; calls outside its admission rules go to the saved SIRCL or NCCL
backend. A per-deployment mesh service installs its relays. The [mesh runtime
contract](../../runtime/glm53-spark-mtp3-mesh/README.md) identifies its
dispatch configuration, source packages and evidence; RoCEnante's origins are
in its [source attribution](../../third_party/b12x_roce/README.md).

Ring sessions use the same kind of hardware relay. The relay plan installer
(`sircl-fabric`, [runbook](../../spark_transport/sircl/RUNBOOK.md#relay-plan-installer))
derives a layout's plan from the sessions' own route module; for a whole
eight-Spark ring it installs the universal relay table, which reaches every
address of every Spark that shares no cable with the sender.

## Profile use

Ring sessions: `sparkring install` serves every installer profile on them
on an image with the SIRCL layer; its transport adapter
([transport.py](../../runtime/common/transport.py)) sets what the serve
launcher's plan sets for the same group. Its tuning table is the release's
default or one that `sudo sparkring fabric tune` measured on the cluster's
fabric with the ring harness
([measure the tuning table](../operations/install-reference.md#measure-the-tuning-table)). The serve launcher plans the
catalog's installer profiles on SIRCL groups; which profiles plan on
Sparks 0-1 or 0-3 and what blocks the others is in the [serve
runbook](../../spark_transport/sircl/sparkring_sircl/vllm/RUNBOOK.md#installer-profiles-on-the-ring-of-eight).

Four-rank native sessions:

- The retired GLM-5.2 EXL3 3.5-bpw profile, a TP4/DCP4 configuration that
  SparkRing does not support, uses SIRCL for qualified tensor-parallel
  all-reduce and vocabulary collective families; patched NCCL handles the
  rest.
- The DeepSeek-V4-Flash-0731 quickstart uses patched NCCL. Its width-4096
  SIRCL CUDA-graph configuration is research-only; see the [DeepSeek SIRCL
  evidence record](../../performance/records/deepseek-v4-flash/sircl-width4096-nccl-ab-20260822.md).
- The [GLM-5.3 DFlash2 operator image](../../runtime/glm53-flash-jj-r8-gb10/glm53-dcp4-sircl-public-image-receipt.json)
  belongs to a retired profile and embeds a source-bound SIRCL bundle. Its
  receipt records **qualified** four-rank TP4/DCP4 functional checks:
  capability agreement, startup, semantic inference, persistent SparkCache
  restore, store-ownership drain and injected failure containment. Its
  throughput matrix is **research-only**. See the [GLM-5.3 runtime
  guide](../../runtime/glm53-flash-jj-r8-gb10/README.md) and the [vLLM
  adapter contract](../../spark_transport/integrations/vllm/README.md).
  Before native construction the adapter exchanges a capability record over
  the CPU process group; an unset GID index is resolved per device from the
  host's GID table ([GID resolution](../../integrations/vllm/README.md#roce-gid-resolution)).
- The Qwen3.8-27B EXL3 K5/K6 pair and cycle profiles use patched NCCL: their
  width-5,120 tensor-parallel shape is outside four-rank SIRCL.

## Persistent host rail configuration

Every SIRCL session needs its fabric interfaces to keep their IPv4 address,
MTU 9000 and RoCE v2 GID after a reboot. `sparkring setup` configures the
fabric addresses of pairs and four-Spark cycles persistently; on a four-Spark
cycle it also applies the ConnectX hairpin setting, which
`sparkring-hairpin.service` repeats at every boot. The relay plan of a
ring-session layout is not persistent: install it with `sircl-fabric up`
after every boot.

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

- All N ranks of a session share one layout and one set of shared settings;
  the setup agreement makes ranks with different values fail together. Each
  rank derives its own lanes and devices from the shared layout.
- NCCL runs only where the operator opts in (`SIRCL_NCCL=auto`, launchers'
  `--nccl auto`; the default is `never`). With the opt-in, the group's
  cabling bounds it: every collective on a pair, NCCL's ring algorithm alone
  (`NCCL_ALGO=Ring`, `NCCL_SKIP_TREE_CONNECT=1`) on a whole cycle, nothing on
  a path or any group with relayed lanes.
- A collective a session declines goes to the caller's own path; on a group
  NCCL may not run, the vLLM adapter refuses it instead of letting NCCL
  connect Sparks that share no cable.
- The management network is not a fabric edge. Relays forward only tagged
  RDMA traffic, so setup exchanges use the CPU process group.
- Four-rank dual-rail prefill uses both RDMA functions of each cabled cycle
  edge; it needs neither extra cables nor diagonal links.
- Transport evidence does not establish model correctness or performance
  unless the corresponding profile result states those conditions.

Four-rank deployment commands and limits are in the
[GLM-5.2 quickstart](../../profiles/glm52-exl3-r7-3.5bpw/README.md),
[GLM-5.3 four-Spark quickstart](../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md),
[DeepSeek quickstart](../operations/deepseek-0731.md),
[Qwen3.8-27B pair quickstart](../../profiles/qwen38-27b-exl3-k5k6-pair/README.md) and
[Qwen3.8-27B cycle quickstart](../../profiles/qwen38-27b-exl3-k5k6/README.md).
