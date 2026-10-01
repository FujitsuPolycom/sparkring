# RoCEnante GID index per port after a neighbor restart, 2026-10-01

Status: **research-only**. One pair, one probe run.

## Question

When a cabled neighbor restarts, a Spark's fabric addresses return at RoCE
GID index 4 instead of 3 on the ports facing it, and `sparkring install`
repairs them back to index 3 before a model starts. Does the prepared RoCEnante
transport with a GID index per port
([GID index per port](../../../integrations/vllm/rocenante_prepared/README.md#gid-index-per-port),
proxy ABI 6) start and pass its collective checks on the moved index without
that repair?

## Conditions

- Two directly cabled DGX Sparks: spark-3286 (rank 0) and spark-0a0f
  (rank 1), one p0-to-p0 cable using both PCIe functions of port 0.
- Image: `runtime/images/derive_transport_port_gid.py` at SparkRing revision
  `81d99674` over `dev-20260930-spinwait-cuda1342-nccl2323-status033`,
  built on spark-3286 (image `sha256:07baba8539f2`, transport manifest
  `d5e790c5173c`). The published release
  `dev-20261001-portgid-cuda1342-nccl2323-status033` carries the same layer
  and manifest; its image was built separately.
- Procedure:
  1. With a `qwen38-flash-next-tp2` deployment on that image serving, both
     ranks' HCAs used index 3 (`spark_roce_gid.py` and the transport's
     startup lines agreed).
  2. spark-0a0f was restarted with `systemctl reboot`.
  3. Afterwards `integrations/vllm/spark_roce_gid.py` reported index 4 for
     both of spark-3286's devices (`rocep1s0f0`, `roceP2p1s0f0`) and index 3
     on spark-0a0f. No `sparkring install` or `up`, and therefore no GID
     repair, ran before the probe.
  4. Rank 0's model container was stopped.
- Probe: [probe.py](../../../integrations/vllm/rocenante_prepared/probe.py)
  (SHA-256 prefix `4ac6c0d02013`) under `torch.distributed.run` on both ranks,
  in containers made from each rank's deployment Compose file with the model
  command replaced. `NCCL_IB_GID_INDEX` and `B12X_ROCE_GID_INDEX` were removed
  from the environment, so both NCCL, which supplies the reference sums, and
  the transport selected each port's index themselves. Rendezvous used the
  deployment's master address and port.

## Results

| Rank | Status | Cases passed | `gid_indices` per selected HCA |
|---|---|---|---|
| 0 (spark-3286) | passed | 15 of 15 | `rocep1s0f0`: 4, `roceP2p1s0f0`: 4 |
| 1 (spark-0a0f) | passed | 15 of 15 | 3, 3 |

The cases were nine FP16/BF16/FP32 reductions against NCCL, three gathers,
gather output ownership, alternating-grid frozen-graph replays and selected
HCA traffic, completed in 8.6 s. Rank 0's standard error named both devices
with "uses RoCE GID index 4, the RoCE v2 GID of 198.18.0.1 on enp1s0f0np0"
(and `198.18.1.1` on `enP2p1s0f0np0`). NCCL, with its GID index unset,
connected over both HCAs (`NET/IB : Using [0]rocep1s0f0:1/RoCE
[1]roceP2p1s0f0:1/RoCE`).

On the same image before the restart, a `qwen38-flash-next-tp2` deployment
decoded at 23.9–24.0 steps/s with one request and 84.9–86.5 with eight
(llm-inference-bench 0.6.2, temperature 1.0, two runs), within the range of the
parent image's runs (24.0–24.2 and 86.0–88.5).

## Conclusion

After a neighbor restart moved one Spark's fabric addresses to GID index 4,
the per-port transport selected index 4 on that Spark and index 3 on the
other, and every probe case passed without the installer's index-3 repair.
NCCL also started with its index unset. The installer profiles still set
`NCCL_IB_GID_INDEX=3`, so a model start still depends on the repair until the
profiles leave NCCL to select per port; a model start without the repair was
not tested. Four-Spark rings were not probed this way.
