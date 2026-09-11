# Patched NCCL fallback

Status: implemented patched-NCCL fallback for direct pairs, four-rank and six-rank
direct-cable cycles.

## Status and scope

Patched NCCL is the supported fallback for collectives that
`spark_transport` does not implement: DCP and sparse-indexer collectives, and
every non-admitted tensor-parallel collective. It is not a custom transport
path.

On a four-rank cycle, the patch set constrains NCCL to the direct-cable RoCE ring. It
prevents Tree and PAT connection setup, which would require non-adjacent
peers, and advertises eligible listener GIDs so subnet-aware connection
selection reaches the directly attached peer. No collective payload is routed
through an intermediate rank. A two-rank pair uses the same verified library
but a separate single-HCA environment: subnet-aware routing is off and Tree,
algorithm, and channel overrides remain unset because both ranks are directly
adjacent.

The [Qwen builder's pinned patch](../../runtime/qwen38/pins.json) publishes at
most two listener GIDs, matching its two selected cycle devices. Despite the
`advertise-all-listener-gids` filename, it is not an unlimited inventory.
Profiles selecting four functions across two PCIe domains instead use the
[four-GID routing contract](DUAL_PCI_DOMAIN.md).

## Runtime contract

Use the patched NCCL library selected by the profile, either embedded in its
image or supplied through its documented host mount. The following variables
describe the common fallback; retain the selected profile's exact environment:

```text
LD_PRELOAD=<patched-nccl-library>
VLLM_NCCL_SO_PATH=<patched-nccl-library>
NCCL_NET=IB
NCCL_IB_DISABLE=0
NCCL_CUMEM_ENABLE=0
```

GID selection is profile-specific. A pinned policy sets
`NCCL_IB_GID_INDEX` to a verified local index. The DeepSeek automatic policy
validates eligible GIDs per HCA and leaves that variable unset in the
container; it must not be replaced by an arbitrary fixed index.

### Two-rank pair

Both ranks are directly adjacent. Single-HCA profiles, including the DeepSeek
and Qwen pair recipes, name one function attached to the cable and use its
fabric interface for bootstrap. GLM-5.3 pair profiles select both host-domain
functions through their own runtime contract:

```text
NCCL_SOCKET_IFNAME=<direct-fabric-interface>
GLOO_SOCKET_IFNAME=<direct-fabric-interface>
NCCL_IB_HCA=<one-direct-roce-device>
NCCL_IB_SUBNET_AWARE_ROUTING=0
NCCL_IB_MERGE_NICS=0
NCCL_CROSS_NIC=1
```

`NCCL_ALGO`, `NCCL_MIN_NCHANNELS`, `NCCL_MAX_NCHANNELS`,
`NCCL_SKIP_TREE_CONNECT`, and `NCCL_IB_SUBNET_PREFIX_LEN` remain unset in those
single-HCA pair recipes. Tree connectivity is valid because there is no
non-adjacent rank; other pair profiles retain their own channel settings.

### Four-rank cycle

The cycle names both neighbor-facing HCAs. Its bootstrap interface is selected
by the model profile: GLM mesh and Qwen cycle profiles use management, while
the DeepSeek cycle template specifies fabric interfaces.

```text
NCCL_IB_HCA=<two-direct-roce-devices>
NCCL_IB_MERGE_NICS=0
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_IB_SUBNET_PREFIX_LEN=24
NCCL_CROSS_NIC=1
NCCL_ALGO=Ring
NCCL_SKIP_TREE_CONNECT=1
NCCL_SOCKET_IFNAME=<profile-bootstrap-interface>
```

`NCCL_PROTO` remains unset for the generic fallback so NCCL can select a
protocol per communicator. A model profile may override it only when the
profile's environment, recipe, and live evidence bind the same value. The
DeepSeek and Qwen pair/cycle profiles bind `LL,LL128,Simple`; that setting is
not a default for other serving objects. Collective payloads use the selected
direct RoCE interfaces regardless of the bootstrap interface.

## Fail-closed requirements

Before serving, validate the patched library identity and its image or mount,
selected pair/cycle topology, direct-peer subnet mapping, and complete runtime
environment on every rank. Any failed identity, topology, environment, or
collective-correctness check is a hard stop. Do not substitute Socket
transport or route RoCE traffic through another rank.
