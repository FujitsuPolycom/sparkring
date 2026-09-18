# Prepared RoCEnante collective qualification

Status: **qualified for the bounded collective checks below**. No model was
loaded. This record does not qualify throughput, model serving, SparkCache
restore, long-running stability or a profile default.

The [sanitized rank receipts](rocenante-prepared-35cf12b2-20260918.json) bind image
configuration ID `sha256:35cf12b2d644661a9e1443d8afc388742e9d516305e5014f21be1c20f7e50340`,
the prepared transport manifest and the exact
[executed probe source](../../../runtime/releases/shared-2026.09.0-rc.1/qualification/transport-probe-be8d5f7d.tar.gz)
(`probe.py` in the archive).
The image ID is not a registry pull digest.

| Configuration | HCA inventory / selected paths | Result |
|---|---|---|
| Two GB10 nodes, one rank/GPU per node | Two functions: `rocep1s0f0`, `roceP2p1s0f0`; local peer map `0/1`, two paths per peer | 15 cases passed on both ranks; both exited 0 |
| Four GB10 nodes, one rank/GPU per node | Four functions: `rocep1s0f0`, `rocep1s0f1`, `roceP2p1s0f0`, `roceP2p1s0f1`; two paths per peer, per-rank maps in the receipts | 15 cases passed on all four ranks; all exited 0 |

The two-node selection uses the two PCI-domain functions of cage p0. Four local
HCA functions on the four-node deployment do not mean four paths to each peer.
The tested configuration retained two paths per peer; four-path mode was not
qualified. Every selected path recorded positive payload traffic, with no path
completion errors or collective error sequence. No container was OOM-killed.

## Workload and checks

- Nine reduction cases: BF16/FP16/FP32 at 16 bytes, 4 KiB and 1 MiB, compared
  with PyTorch ProcessGroupNCCL and checked for identical outputs across ranks.
- Three gathers: `[4,32]` along dimension 0 and the last dimension, plus padded
  `[5,3]` last-dimension gather.
- Consecutive misaligned, pack-sized gathers retained independently owned output.
- Four CUDA graph replays mixed small/large reductions and direct/padded gathers
  with kernel resolution frozen. Output addresses remained stable and replay
  allocated no GPU storage.
- The final case checked actual per-peer/path counters and proxy health.

The reference logs identify **NCCL 2.31.2+cuda13.3**. This is not evidence that
the separately patched NCCL 2.30.7 serving library executed these reference
operations. NCCL logged `ibv_query_port_speed` errno 93 warnings; the recorded
collectives nevertheless completed. No bandwidth or latency conclusion is
drawn from this short correctness probe.

Manifest SHA256: `e8577c447a69ac75253758a0964791e862ccca1ad13168e7addefa6ec96369c9`.
Harness SHA256: `be8d5f7d12bef39c453953dade256b36904200700da2c68df443fef909826558`.
The formatted [maintained probe](../../../integrations/vllm/rocenante_prepared/probe.py)
has a separate artifact hash; these GPU results still identify the archived bytes.
The receipts retain the original successful per-rank JSON hashes while omitting
site addresses, container IDs and ephemeral QP identifiers. Follow the
[coordinated probe procedure](../../../integrations/vllm/rocenante_prepared/INSTALLATION.md#bounded-real-hardware-probe)
to reproduce the workload; changing the image, manifest, topology or harness
requires separate evidence.
