# Patched NCCL fallback

Status: implemented patched-NCCL fallback for direct pairs and four-rank
direct-cable cycles.

## Status and scope

Patched NCCL is the supported fallback for collectives that
`spark_transport` does not implement: DCP and sparse-indexer collectives, and
every non-admitted tensor-parallel collective. It is not a custom transport
path.

On a four-rank cycle, the patch set constrains NCCL to the direct-cable RoCE ring. It
prevents Tree and PAT connection setup, which would require non-adjacent
peers, and advertises both eligible listener GIDs so subnet-aware connection
selection reaches the directly attached peer. No collective payload is routed
through an intermediate rank. A two-rank pair uses the same verified library
but a separate single-HCA environment: subnet-aware routing is off and Tree,
algorithm, and channel overrides remain unset because both ranks are directly
adjacent.

## Runtime contract

Every serving image must use the site-provided patched NCCL library and set:

```text
LD_PRELOAD=<patched-nccl-library>
VLLM_NCCL_SO_PATH=<patched-nccl-library>
NCCL_NET=IB
NCCL_IB_DISABLE=0
NCCL_IB_GID_INDEX=<site-gid-index>
NCCL_CUMEM_ENABLE=0
```

### Two-rank pair

Both ranks are directly adjacent. The pair names the one HCA attached to the
cable and uses that fabric interface for bootstrap:

```text
NCCL_SOCKET_IFNAME=<direct-fabric-interface>
GLOO_SOCKET_IFNAME=<direct-fabric-interface>
NCCL_IB_HCA=<one-direct-roce-device>
NCCL_IB_SUBNET_AWARE_ROUTING=0
NCCL_IB_MERGE_NICS=0
NCCL_CROSS_NIC=1
```

`NCCL_ALGO`, `NCCL_MIN_NCHANNELS`, `NCCL_MAX_NCHANNELS`,
`NCCL_SKIP_TREE_CONNECT`, and `NCCL_IB_SUBNET_PREFIX_LEN` remain unset on a
pair. Tree connectivity is valid because there is no non-adjacent rank.

### Four-rank cycle

The cycle names both neighbor-facing HCAs and uses the management interface
for bootstrap:

```text
NCCL_IB_HCA=<two-direct-roce-devices>
NCCL_IB_MERGE_NICS=0
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_IB_SUBNET_PREFIX_LEN=24
NCCL_CROSS_NIC=1
NCCL_ALGO=Ring
NCCL_SKIP_TREE_CONNECT=1
NCCL_SOCKET_IFNAME=<management-interface>
```

`NCCL_PROTO` remains unset for the generic fallback so NCCL can select a
protocol per communicator. A profile may override it only when its
environment, recipe, and live evidence bind the same value. On a cycle, the
management interface is bootstrap-only; collective payloads use the two direct
RoCE interfaces.

## Validation requirements

Before serving, validate the patched library identity, read-only mount,
selected pair/cycle topology, direct-peer subnet mapping, and complete runtime
environment on every rank. Any failed identity, topology, environment, or
collective-correctness check is a hard stop. Do not substitute Socket
transport or route RoCE traffic through another rank.
