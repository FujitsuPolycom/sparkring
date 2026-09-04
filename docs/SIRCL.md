# SIRCL

SIRCL is the **Switchless Inference RDMA Collective Layer**. It is SparkRing's
native collective transport for the directly cabled four-DGX-Spark cycle; it is
not a separate service or an NCCL fork.

## Implemented boundary

SIRCL maintains RDMA sessions, registered arenas, and device-published command
rings. CUDA graph replay submits pre-established work without Python or host
control work in the replay path.

A four-rank collective is decomposed into two perfect matchings of the physical
cycle. This scheduling is specific to the four-Spark topology documented in
[architecture](ARCHITECTURE.md). SIRCL does not claim a generic multi-node
collective interface or support beyond that topology.

## Profile use

The GLM-5.2 EXL3 3.5-bpw profile uses SIRCL for qualified tensor-parallel
all-reduce and vocabulary collective families. Patched NCCL handles operations
outside those families; DCP and indexer collectives use stock paths.

The DeepSeek-V4-Flash-0731 quickstart uses patched NCCL. Its width-4096 SIRCL
CUDA-graph configuration is research-only and excluded from functional profile
qualification. A four-rank matched comparison established native replay,
API health, and zero overflow for the target and DSpark capture path; see the
[DeepSeek SIRCL evidence record](../performance/records/deepseek-v4-flash/sircl-width4096-nccl-ab-20260822.md).

The GLM-5.3 Flash GB10 operator image embeds a source-bound SIRCL bundle. The
exact public image is **qualified** for the recorded four-rank TP4/DCP4
functional checks: capability agreement, startup, semantic inference,
persistent SparkCache restore, concurrent store-ownership drain, and injected
failure containment. Its artifact-bound throughput matrix is
**research-only**; no broad SIRCL-versus-NCCL performance comparison has been
established. A developer can replace the embedded bundle with a read-only host
mount. Its captured width-4096 path and eager fused-prefill path use separate
signature checks. The fused path accepts contiguous TP4 BF16 `[Q, 4096]`
tensors from Q128 through Q8192 and uses two operation slots. Unsupported
signatures remain on NCCL. The GLM-5.3 profile captures Q8/Q16/Q32/Q64/Q128;
those captured collectives use graph-native SIRCL with direct doorbells. The
fused session uses four persistent QPs and two 67,109,888-byte operation
arenas. See the
[GLM-5.3 runtime guide](../runtime/glm53-flash-jj-r8-gb10/README.md) and the
[vLLM adapter contract](../spark_transport/integrations/vllm/README.md). The
[public SIRCL build receipt](../runtime/glm53-flash-jj-r8-gb10/sircl-public-build-receipt.json)
binds the native build and single-node test identity; it does not establish a
four-rank serving result. The
[operator-image receipt](../runtime/glm53-flash-jj-r8-gb10/glm53-dcp4-sircl-public-image-receipt.json)
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

[`configure_sircl_rail.py`](../scripts/configure_sircl_rail.py) creates one
dedicated NetworkManager profile at a time. Its default mode only validates the
arguments and prints the complete plan. `--verify` performs read-only checks.
`--execute` requires root plus the exact confirmation
`CONFIGURE_SIRCL_RAIL`; it creates or updates the named profile, activates it,
then verifies:

- profile autoconnect, manual IPv4, no default route, disabled IPv6, and MTU;
- the active connection, live address, link state, and live MTU;
- both the rail and declared management interfaces exist before any mutation;
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
[GLM-5.2 quickstart](GLM52_35BPW_QUICKSTART.md),
[GLM-5.3 quickstart](GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md),
[DeepSeek quickstart](DEEPSEEK_V4_FLASH_QUICKSTART.md),
[Qwen3.8-27B pair quickstart](QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md), and
[Qwen3.8-27B cycle quickstart](QWEN38_27B_EXL3_K5K6_QUICKSTART.md).
