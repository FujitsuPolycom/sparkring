# Switchless NCCL-IB bridge for TP4/DCP2

## Purpose

Keep SparkRing's custom transport on the hot TP4 paths while giving the
remaining stock vLLM collectives a correct direct-cable RDMA fallback.

The admitted topology is deliberately narrow:

- four DGX Sparks in the physical 200 Gbit/s ring;
- TP4 across ranks `0,1,2,3`;
- DCP2 groups `[0,1]` and `[2,3]`;
- ring-only NCCL algorithms; and
- no routed or non-adjacent RoCE queue pair.

This is not a general routed-RDMA solution. It makes NCCL choose only links
that physically exist.

## Why two NCCL changes are required

Joseph Rose's switchless patch supplies the first requirement: do not create
Tree or PAT connections, because those algorithms request non-adjacent rank
pairs on a four-node ring. That change is reproduced by
`nccl-2.30.7-skip-tree-pat.patch`.

NCCL 2.30.7 already contains subnet-aware RoCE device selection, but its
listener normally advertises only the topology-selected device when NIC
merging is disabled. In this topology, both peers can initially select the
wrong cable and then have no peer GID with which to recover.

`nccl-2.30.7-advertise-all-listener-gids.patch` makes a listener advertise the
GIDs of both eligible local RoCE devices. The existing subnet-aware connector
then selects the one whose `/24` contains the peer. No packet is routed through
an intermediate Spark.

## Pinned build

The validated source base is NVIDIA NCCL tag `v2.30.7-1`. Apply both 2.30.7
patches and build for GB10/SM121:

```bash
git clone --branch v2.30.7-1 --depth 1 \
  https://github.com/NVIDIA/nccl.git nccl-switchless-2.30.7
cd nccl-switchless-2.30.7
git apply /path/to/nccl-2.30.7-skip-tree-pat.patch
git apply /path/to/nccl-2.30.7-advertise-all-listener-gids.patch

docker run --rm --gpus all \
  -v "$PWD:/src" -w /src \
  nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 \
  bash -lc \
  'make -j"$(nproc)" src.build \
    NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"'
```

The exact admitted binary is:

```text
file:   libnccl-spark-switchless-2.30.7-gidfix.so
sha256: 106150aebf7ef9d997f4dcab5edea13082d4bc72fd55b1160a030dbc05c60202
```

The graph-window controller refuses switchless mode unless that checksum is
present on every rank. It bind-mounts the library read-only and attests the
container mount, command line, environment, start time, and image identity.

## Required runtime contract

```text
LD_PRELOAD=/opt/sparkring/libnccl-switchless.so.2
VLLM_NCCL_SO_PATH=/opt/sparkring/libnccl-switchless.so.2
NCCL_NET=IB
NCCL_IB_DISABLE=0
NCCL_IB_HCA=rocep1s0f0,rocep1s0f1
NCCL_IB_GID_INDEX=3
NCCL_IB_MERGE_NICS=0
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_IB_SUBNET_PREFIX_LEN=24
NCCL_CROSS_NIC=1
NCCL_ALGO=Ring
NCCL_SKIP_TREE_CONNECT=1
NCCL_CUMEM_ENABLE=0
NCCL_SOCKET_IFNAME=wlP9s9
```

`NCCL_PROTO` is intentionally unset so NCCL can tune the two-rank DCP and
four-rank TP communicators separately. The Wi-Fi interface is bootstrap and
management only; collective payloads use the direct RoCE interfaces.

## Model-down proof

`probe_dcp2_collectives.py` creates the same communicator scopes as vLLM:
one four-rank TP group and adjacent DCP groups `[0,1]` and `[2,3]`. It checks
the exact values and rank-major layout, not only completion or latency.

The four-rank proof passed on all ranks with actual runtime log evidence for
NCCL `2.30.7+cuda13.2`, `NET/IB`, the ring-only skips, and cable-local subnet
overrides.

| Scope and payload | Observed p50 |
|---|---:|
| TP4 all-reduce, `[6144]` BF16 | about 51.2 us |
| DCP2 owner top-k, `[1,2,2048]` INT32 | 26.1--28.5 us |
| DCP2 query, `[1,16,576]` BF16 | 25.1--27.4 us |
| DCP2 LSE, `[1,32]` FP32 | 18.5--20.0 us |
| DCP2 query Q40, 737,280 bytes | 89.8--90.8 us |
| DCP2 query Q4096, 75,497,472 bytes | 5.62--5.69 ms |

CUDA graph capture plus 2,000 replays also passed exact correctness for the
owner-top-k, query, and LSE collectives on both DCP pairs. The aggregate
back-to-back replay mean is a proxy-queue saturation test, not a
single-collective critical-path latency measurement.

## Live launch

The fail-closed launch path is:

```powershell
powershell -ExecutionPolicy Bypass `
  -File scripts/run-glm52-graph-window.ps1 `
  -Mode Execute `
  -Confirmation STOP-GLM52-TRACE-ON-ALL-FOUR `
  -AllowUnavailableApiForExecute `
  -DcpSize 2 `
  -NcclTransportMode switchless_ib `
  -MtpTokens 4 `
  -MtpMode adaptive-2-4 `
  -MaxNumSeqs 8 `
  -EnablePrefillQ512 `
  -KvScaleMode per-token `
  -B12xPrefillBlockK auto
```

The live gate still records every stock collective. In switchless mode those
calls are admitted only after the exact patched NCCL artifact and runtime
contract above have been attested. Any unsupported topology, checksum
mismatch, writable/wrong mount, Socket configuration, or failed correctness
gate remains a hard stop.

## Live result

The first TP4/DCP2 production candidate passed on 2026-07-27. It exposed
187,520 KV tokens, completed all 26 PIECEWISE and 16 FULL graph captures, and
passed the live transition with 720 stock collective calls carried by this
attested NCCL-IB bridge.

Measured headline results were:

- C1 decode: 18.9 / 20.2 / 17.5 tok/s at 8K / 16K / 32K;
- C8/8K aggregate decode: 59.0 tok/s;
- prefill: 729 / 699 / 658 tok/s at 8K / 16K / 32K; and
- coding peak: 23.6 median tok/s.

See
`deliverables/glm52-dcp2-switchless-result-20260727.md` (private archive)
for the exact DCP1 comparison, capture census, configuration, and remaining
work.
