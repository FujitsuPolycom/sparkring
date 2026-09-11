# SIRCL

SIRCL is the **Switchless Inference RDMA Collective Layer**, SparkRing's native
collective transport for four participating ranks. It operates on tensor
buffers and rank groups; runtime adapters select admitted signatures for each
model profile. The native interfaces and implementations live in
[`spark_transport/`](../../spark_transport/README.md).

Status: **implemented**. Qualification applies to the exact artifacts and
profile conditions linked below. SIRCL is one component of SparkRing's
communication stack alongside adapted RoCEnante communication and patched
NCCL.

## Implemented boundary

SIRCL maintains RDMA sessions, registered arenas, and device-published command
rings. Captured CUDA graphs submit work to established sessions through device
command descriptors. Native progress threads perform the host protocol work
required by the selected transport.

The pairwise-exchange schedule uses two perfect matchings of the physical
cycle; bidirectional and fused variants use their own ring schedules. These
native APIs require four ranks. Tensor geometry is validated by
the native interface and runtime adapter; a model-independent transport does
not imply arbitrary rank-count, dtype, or shape support. Patched NCCL has its
own [pair/cycle configurations](../../spark_transport/nccl/README.md).

## Composition with hardware-forwarded mesh

The four-rank mesh profile combines SIRCL, an adapted RoCEnante all-reduce,
and patched NCCL. RoCEnante's selected opposite-peer operations cross two
physical links through an intermediate ConnectX-7 ASIC. The overlay delegates
calls outside its admission rules to the saved SIRCL/NCCL backend. It does not
extend the SIRCL C API to other rank counts.

The [mesh runtime contract](../../runtime/glm53-spark-mtp3-mesh/README.md) identifies
its dispatch configuration, source packages, and evidence. The
[MTP3 cache/checkpoint quickstart](../history/glm53-cache-checkpoints.md)
provides one serving composition, while the
[transport overview](../../spark_transport/README.md) describes the shared components.
RoCEnante's origins and local adaptations are recorded in its
[source attribution](../../third_party/b12x_roce/README.md).

## Profile use

The GLM-5.2 EXL3 3.5-bpw profile uses SIRCL for qualified tensor-parallel
all-reduce and vocabulary collective families. Patched NCCL handles operations
outside those families; DCP and indexer collectives use stock paths.

The DeepSeek-V4-Flash-0731 quickstart uses patched NCCL. Its width-4096 SIRCL
CUDA-graph configuration is research-only and excluded from functional profile
qualification. A four-rank matched comparison established native replay,
API health, and zero overflow for the target and DSpark capture path; see the
[DeepSeek SIRCL evidence record](../../performance/records/deepseek-v4-flash/sircl-width4096-nccl-ab-20260822.md).

The [GLM-5.3 DFlash2 operator image](../../runtime/glm53-flash-jj-r8-gb10/glm53-dcp4-sircl-public-image-receipt.json)
embeds a source-bound SIRCL bundle. Its receipt records **qualified** four-rank TP4/DCP4
functional checks: capability agreement, startup, semantic inference,
persistent SparkCache restore, concurrent store-ownership drain, and injected
failure containment. Its artifact-bound throughput matrix is
**research-only**; no broad SIRCL-versus-NCCL performance comparison has been
established. A developer can replace the embedded bundle with a read-only host
mount. Its captured width-4096 path and eager fused-prefill path use separate
signature checks. The fused path accepts contiguous TP4 BF16 `[Q, 4096]`
tensors from Q128 through Q8192 and uses two operation slots. Unsupported
signatures remain on NCCL. The GLM-5.3 profile captures every eight-row DFlash
request-batch shape from Q8 through Q128; those captured collectives use
graph-native SIRCL with direct doorbells. The
fused session uses four persistent QPs and two 67,109,888-byte operation
arenas. See the
[GLM-5.3 runtime guide](../../runtime/glm53-flash-jj-r8-gb10/README.md) and the
[vLLM adapter contract](../../spark_transport/integrations/vllm/README.md). The
[public SIRCL build receipt](../../runtime/glm53-flash-jj-r8-gb10/sircl-public-build-receipt.json)
binds the native build and single-node test identity; it does not establish a
four-rank serving result. The
[operator-image receipt](../../runtime/glm53-flash-jj-r8-gb10/glm53-dcp4-sircl-public-image-receipt.json)
records the four-rank functional result and its limits.

Before native construction, the GLM-5.3 adapter exchanges a capability record
over the CPU process group. Shared protocol and artifact identities must match,
while each rank proves its own RDMA device and GID availability. Model output
is checked against every process-local native session after vLLM's existing
output synchronization. Fused kernels publish poison into mapped host control
state so this check can reject their output without adding CUDA synchronization.

## Persistent host rail configuration

SIRCL reads host networking but does not configure it. Every Ethernet interface
named by a SIRCL profile must retain its IPv4 address and MTU after a reboot.
The corresponding RoCEv2 GID must encode that IPv4 address. A transient
`ip address add` or `ip link set` command can satisfy a same-boot check but does
not meet this requirement.

[`configure_sircl_rail.py`](../../scripts/configure_sircl_rail.py) creates one
dedicated NetworkManager profile at a time. Its default mode only validates the
arguments and prints the complete plan. `--verify` performs read-only checks.
`--execute` requires root plus the exact confirmation
`CONFIGURE_SIRCL_RAIL`; it creates or updates the named profile, activates it,
then verifies:

- profile autoconnect, manual IPv4, no default route, disabled IPv6, and MTU;
- the active connection, live address, link state, and live MTU;
- both interfaces exist, the declared management interface owns the active
  IPv4 default route, and the rail interface does not, before any mutation;
- the configured RDMA port is active in Ethernet mode and exposes the expected
  GID value, RoCEv2 type, and Ethernet device; and
- a don't-fragment peer ping whose payload exercises the configured MTU.

Run the helper locally on each rank. Replace every value below before running
the plan. Repeat the procedure for every dedicated secondary-rail interface:

```bash
management_netdev='REPLACE_MANAGEMENT_NETDEV'
rail_netdev='REPLACE_SECONDARY_NETDEV'
rail_cidr='REPLACE_LOCAL_SECONDARY_ADDRESS/PREFIX'
rail_peer='REPLACE_SECONDARY_PEER_ADDRESS'
rail_rdma_device='REPLACE_SECONDARY_RDMA_DEVICE'

rail_args=(
  --management-interface "${management_netdev}"
  --interface "${rail_netdev}"
  --address-cidr "${rail_cidr}"
  --peer-address "${rail_peer}"
  --rdma-device "${rail_rdma_device}"
  --rdma-port 1
  --gid-index 3
  --mtu 9000
)

# Offline plan: validates values and prints the exact profile contract.
python scripts/configure_sircl_rail.py "${rail_args[@]}"

# Host mutation: inspect the plan before supplying the confirmation.
sudo python scripts/configure_sircl_rail.py "${rail_args[@]}" \
  --execute --confirmation CONFIGURE_SIRCL_RAIL

# Read-only validation, including the peer path.
python scripts/configure_sircl_rail.py "${rail_args[@]}" --verify
```

The default connection name is `sparkring-sircl-<interface>`. The helper will
not change a profile with that name when it belongs to another interface. It
also rejects a rail interface that is identical to the declared management
interface. Verify both ends of every direct link and rerun `--verify` after a
host reboot before starting a four-rank service.

The
[`secondary-rail persistence validation`](../../runtime/glm53-flash-jj-r8-gb10/sircl-secondary-rail-persistence-live-validation.json)
records eight successful live rail verifications on four GB10 ranks after a
reboot exposed non-persistent addresses. Each rail passed 23 profile, route,
link, RDMA, GID, and peer-path checks while the public GLM-5.3 image remained
healthy. The helper's confirmed `--execute` path has CPU-only regression
coverage; the recorded profiles were created with equivalent NetworkManager
commands before the helper was available.

The Qwen3.8-27B EXL3 K5/K6 pair and cycle profiles use patched NCCL. Their
width-5,120 tensor-parallel shape is unsupported by SIRCL, so neither loads a
custom SparkRing collective adapter.

## Operational invariants

- All four ranks require the same topology, peer ordering, RDMA device mapping,
  and transport configuration.
- A collective shape not admitted to the native path must use the NCCL fallback.
- The management network is not an RDMA cycle edge.
- Dual-rail prefill uses both RDMA device functions associated with each
  existing cabled cycle edge. It requires neither additional cables nor
  diagonal rank-to-rank links.
- Transport evidence does not establish model correctness or performance unless
  the corresponding profile result states those conditions.

Deployment commands and profile limits are in the
[GLM-5.2 quickstart](../../profiles/glm52-exl3-r7-3.5bpw/README.md),
[GLM-5.3 quickstart](../history/glm53-dflash-operator.md),
[DeepSeek quickstart](../operations/deepseek-0731.md),
[Qwen3.8-27B pair quickstart](../../profiles/qwen38-27b-exl3-k5k6-pair/README.md), and
[Qwen3.8-27B cycle quickstart](../../profiles/qwen38-27b-exl3-k5k6/README.md).
