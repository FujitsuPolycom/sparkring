# Prepared RoCEnante: two/four-rank qualification

Status: **qualified for bounded collective correctness and graph replay**.
The [sanitized receipts](rocenante-prepared-a75bd02f-20260918.json) identify image
`sha256:a75bd02ffc1dd29713f97f599ccb10a88ba76bb5cd9d3554a98c875e96c0b942`,
manifest `7d8beed57e541c756995b73b0909a826642cef7c00f8147316cc8cd97a2f905c`
and [executed probe](../../../runtime/releases/shared-2026.09.0-rc.1/qualification/transport-probe-15e522b0.tar.gz)
SHA256 `15e522b076afb4885c0d6eda1d80eeacc2a1a79dba6164ccc24f5d3c5ba4ed4f`.
The image ID is not a registry pull digest.

| Configuration | Selected transport | Result |
|---|---|---|
| Two GB10 nodes, one rank/GPU each | Two HCA functions (`rocep1s0f0`, `roceP2p1s0f0`), peer map `0/1` | Both ranks passed 15 cases; exit 0 |
| Four GB10 nodes, one rank/GPU each | Four HCA functions; two paths per peer with rank-specific maps in receipts | All four ranks passed 15 cases; exit 0 |

Every rank passed nine BF16/FP16/FP32 reductions at 16 B, 4 KiB and 1 MiB;
three direct/padded gathers; independent misaligned-gather output ownership;
four mixed-grid CUDA graph replays with frozen kernel resolution, stable
addresses and no replay allocation; and positive traffic on every selected
HCA path. Collective error sequences remained zero. No container was OOM-killed.

The reference uses PyTorch ProcessGroupNCCL, logged as **2.31.2+cuda13.3**.
This does not establish use of the separately patched NCCL 2.30.7 library.
NCCL emitted `ibv_query_port_speed` errno 93 warnings; the bounded checks passed.
The probe archive SHA256 is
`5aefa700c021547c24cbc1f6ee97ab0e68d0a6f94d23c69c58cab054bd338e08`.

No model was loaded. Throughput, SparkCache restore, model serving, four-path
mode, failover and soak remain outside this qualification. The
[35cf12b2 transport record](rocenante-prepared-35cf12b2-20260918.md) uses different
image, manifest and probe identities; its results have not been relabeled.
