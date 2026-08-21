# Patched NCCL fallback

Status: implemented patched-NCCL fallback for the four-rank direct-cable
topology.

## Status and scope

Patched NCCL is the supported fallback for collectives that
`spark_transport` does not implement: DCP and sparse-indexer collectives, and
every non-admitted tensor-parallel collective. It is not a custom transport
path.

The patch set constrains NCCL to the four-rank direct-cable RoCE ring. It
prevents Tree and PAT connection setup, which would require non-adjacent
peers, and advertises both eligible listener GIDs so subnet-aware connection
selection reaches the directly attached peer. No collective payload is routed
through an intermediate rank.

## Runtime contract

The serving image must use the site-provided patched NCCL library and set:

```text
LD_PRELOAD=<patched-nccl-library>
VLLM_NCCL_SO_PATH=<patched-nccl-library>
NCCL_NET=IB
NCCL_IB_DISABLE=0
NCCL_IB_HCA=<two-direct-roce-devices>
NCCL_IB_GID_INDEX=<site-gid-index>
NCCL_IB_MERGE_NICS=0
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_IB_SUBNET_PREFIX_LEN=24
NCCL_CROSS_NIC=1
NCCL_ALGO=Ring
NCCL_SKIP_TREE_CONNECT=1
NCCL_CUMEM_ENABLE=0
NCCL_SOCKET_IFNAME=<management-interface>
```

`NCCL_PROTO` remains unset so NCCL can select a protocol per communicator.
The management interface is bootstrap-only; collective payloads use the two
direct RoCE interfaces.

## Fail-closed requirements

Before serving, validate the patched library identity, read-only mount,
four-rank topology, direct-peer subnet mapping, and complete runtime
environment on every rank. Any failed identity, topology, environment, or
collective-correctness check is a hard stop. Do not substitute Socket
transport or route RoCE traffic through another rank.
