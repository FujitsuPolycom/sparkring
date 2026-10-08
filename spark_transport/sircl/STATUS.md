# SIRCL ring sessions: status

SIRCL ring sessions (`spark_transport/sircl`, Python package
`sparkring_sircl`) carry tensor-parallel and decode-context-parallel
collectives between 2 to 8 DGX Sparks over RoCE without a switch: pairs,
paths, cycles, several independent groups on one fabric, and subgroups whose
members share no cable. Members that share no cable reach each other through
ConnectX-7 hardware relays. This page holds the package's one status table;
the [package README](README.md), the [runbook](RUNBOOK.md) and the
[vLLM adapter documents](sparkring_sircl/vllm/README.md) refer to it.

Status labels: **implemented** (built and tested on CPU or in GPU
emulation), **qualified** (correct on the eight-Spark ring under the stated
conditions), **research-only** (runs, but its evidence is light or partial),
**unsupported** (refused or not offered).

## Component status

| Component | Status | Evidence |
|---|---|---|
| Route maps and layouts (`routes.py`): lane derivation, validation, cross-rank pairing, group isolation, relay load, forward windows; device names from a fabric document (`SIRCL_FABRIC_DOCUMENT`) | implemented | CPU tests: all 23 reference layouts of `tests/data/routes.json` match rank for rank and lane for lane; invalid maps fail naming the rank, entry and rule; fabric documents with other device names (`tests/test_fabric_document.py`) |
| Wire protocol arithmetic (`protocol.py`, `pieces.py`, `posting.py`): stripes, chunks, flags, posting orders, op words, launch grids, counters | implemented | every reference vector of `tests/data/numeric.json`, plus exhaustive small cases |
| Native progress thread (`oneshot/_roce_proxy.c`, ABI 9): arena, validated connection records, lanes, posting orders, CPU pinning, forward windows with delivery proofs, chain and link streams, event trace | implemented | CPU proxy simulator (74 cases: groups of 2 to 8 on cycles and paths, one and two lanes, sequence wrap, missed doorbells, injected failures, relayed lanes under simulated delays) and binding tests on the verbs stand-in; ring harness transport-only runs on pairs, a path of four, two paths of four at once and the cycle of eight |
| Setup agreement (`agreement.py`) | implemented | CPU tests |
| One-shot all-reduce and all-gather (`AllReduce.all_reduce`, `all_gather`) | qualified | ring harness on the eight-Spark ring (pairs, a path of four, two paths of four at once, the cycle of eight; eager and CUDA graph replay; all-reduce 16 B to 128 KiB, all-gather shards to 155,648 B and padded row shapes): every output bit-exact against the host reference; no RDMA, retransmission, `rx_out_of_buffer` or hairpin counter moved |
| Eager launch path (`oneshot/_fast_launch.py`, `SIRCL_FAST_LAUNCH`) | qualified | ring harness on a pair and the cycle of eight, every output exact; CPU tests of the argument block against the CuTe DSL's own conversion |
| Two-shot all-reduce; `all_reduce_large` in pieces; `all_gather_large` in tiles | implemented | GPU emulation (one RTX 5090, ranks as threads over the verbs stand-in) on `path:0-3`, `ring:8`, `path:0-1`, `ring:3`: every output bit-exact, eager and in graph replay; ring harness large-message cases on a path of four and the cycle of eight, every case exact |
| Chain schedule (`chain`): all-reduce, all-gather and reduce-scatter as pipelined ops between cable neighbours | implemented | proxy simulator; GPU emulation as above; ring harness on a path of four and the cycle of eight, every case exact |
| Ring schedule (`ring`): all-reduce, reduce-scatter and all-gather over the chain closed by its end ranks, with staggered relays (`SIRCL_RING_STAGGER`, `SIRCL_RING_GATHER_STAGGER`) | implemented | CPU link simulator (`tests/test_ring_links.py`); GPU emulation; ring harness on a path of four (closing lanes through relays) and the cycle of eight, every case exact |
| Scatter collectives: `reduce_scatter` in scatter ops, `all_to_all` | implemented | proxy simulator; ring harness on a path of four and on both DCP 4 groups of the cycle of eight at once (configuration `dcp4`), every case exact |
| Swing all-reduce (`oneshot/_swing_ops.py`, `oneshot/_swing_cute.py`) | research-only | runs from the ring harness configuration `ring-swing` only; sessions report it unavailable and refuse `SIRCL_ALLREDUCE_ALGORITHM=swing` |
| Flag waits: limits in seconds of the GPU clock, a startup regime (600 s) and a serving regime (20 s, after `enter_serving()`); `SIRCL_SPIN_LIMIT` bounds only waits without a time limit | implemented | GPU emulation (a late peer waited out at startup, the group poisoned under the serving limit); proxy simulator case `wait-regimes` |
| Tuning tables (`tuning.py`, `SIRCL_TUNING_TABLE`; ring harness `tune` and `tune-table`): a measured choice of algorithm, schedule, piece, stagger and grid per collective, size and mode | implemented | CPU tests of the tables and the harness; GPU emulation on `path:0-3`, `ring:8` and `path:0-1` (every op under the table's choice bit-exact). No table ships with the package; without one, the rules of [Dispatch settings](README.md#dispatch-settings) choose every op |
| Point-to-point channels (`p2p/`, native ABI 1): send, receive, batched send and receive between any two ranks of a group | implemented | point-to-point simulator (30 cases), binding tests (12), GPU emulation on `path:0-1`, `path:0-3` and `ring:8` (19 of 19 checks each); not run on a ring |
| Event trace (`SIRCL_EVENT_TRACE`; `python -m sparkring_sircl.ring trace`) | implemented | GPU emulation; ring harness on the cycle of eight (64 MiB ring all-reduce, no events lost) |
| CPU placement of launching and progress threads (`cpus.py`) | implemented | CPU tests |
| NCCL baseline of the ring harness (`--baseline nccl`) | implemented | CPU tests; ring harness on a pair and the cycle of eight |
| Relay plan installer (`fabric/`, `sircl-fabric`): origin routes, marker rules and relay filters per layout, ownership marks, group isolation | implemented | CPU tests against a host-command simulator; the `ring8` plan reproduces the universal relay table object for object (`tests/fabric_reference.py`); the installer has not run on a ring. Restoring a plan after a reboot: unsupported |
| Fused all-reduce + residual add + RMSNorm (`fused_norm/`; adapter `SIRCL_FUSED_NORM=1`) | research-only | CPU tests of geometry and reference arithmetic; bit-identity with the unfused path in GPU emulation on an RTX 5090, not on GB10; GLM-5.3 at TP8 with DCP 4 served with it on the cycle of eight (light test, one run) |
| Direct mlx5 posting (`SIRCL_POST_MODE=direct`), phase tracing (`SIRCL_TRACE`) | unsupported | refused at setup naming the setting |
| vLLM adapter: fabric placement, NCCL policy, route maps, planner and executor, NCCL guard, plugins, pins, hook table, shim catalog (`sparkring_sircl/vllm/`) | implemented | CPU tests against a stand-in vLLM package; every pin and hook anchor matched in the vLLM builds `pins.py` names |
| vLLM adapter: serving with SIRCL in front of every collective (group adapter and communicator, DCP collectives, mHC and Qwen3.8 hyper-connection prefill row ownership, flag-wait regimes) | research-only | served on the eight-Spark ring, one run per configuration: GLM-5.3-Flash at TP4 on Sparks 0-3 and 4-7 (alone and both at once) and at TP2 on a pair; Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash, MiMo-V2.6-Flash and Swift-1.5 at TP4 on four consecutive Sparks; GLM-5.3 at TP8 with decode context parallelism 1 and 4; every functional check passed |
| vLLM adapter: point-to-point channels for pipeline-parallel groups | research-only | CPU tests on emulated ranks; not run on a ring |
| vLLM adapter: column gathers on the session's ring or chain (`executor.ColumnGather`, `SIRCL_COLUMN_GATHER`, default on): an all-gather along a dimension with rows in front of it, carried as a dimension-0 gather into staging plus one local copy | implemented | CPU tests (`tests/test_vllm_column_gather.py`); GPU emulation on `ring:8` with two lanes, 17 of 17 checks (shards of 4 to 8 MiB along the last, a middle and dimension 1; odd rows, FP32 and FP16, transposed and misaligned inputs; ring, chain and pieces schedules; one CUDA graph per rank replayed twice): staged and tiled outputs bit-identical to each other and to the concatenation on every rank; not run on a ring |
| Serving without NCCL (`--nccl never`, the default; `--require-no-nccl`, `--nccl-debug`, receipt and log checks in `check` and `bundle-check`) | implemented | CPU tests; GLM-5.3 at TP8 with DCP 4 on the cycle of eight: every rank's receipts `nccl=none pynccl=skipped`, no NCCL line in any rank's log with `NCCL_DEBUG=INFO` |
| Serve launcher and bundle (`sparkring_sircl/vllm/serve/`) | implemented | CPU tests on a synthetic SparkRing checkout and simulated Sparks, and against the repository's profile catalog; the serving runs above |

## Supported layouts

| Layout | Lanes per peer | Longest lane | Notes |
|---|---|---|---|
| pair, one cable | 2 | direct | both functions of the cable |
| pair, two cables | 2 | direct | one lane per cable |
| triangle, cycle of 4 to 8 | 2 | half the cycle | up to 3 relays on a cycle of eight |
| path of consecutive Sparks | 2 | `N - 1` cables | lanes through more than 3 relays are refused (`SIRCL_MAX_RELAYS` raises the limit: research-only) |
| independent groups on one fabric | 2 | inside each group | no lane crosses another group's cable or Spark |
| subgroups (DCP) | 2 | over the parent's fabric | non-adjacent members relay through parent members |
| one lane per peer | 1 | as above | research-only |

Measured on the ring: pairs, a path of four, two paths of four at once and
the cycle of eight. Triangles, cycles of 4 and 6 and paths of 5 to 7 Sparks
are covered by CPU tests and GPU emulation only.

## Algorithms and exactness

| Collective | Algorithm | Sizes | Exactness |
|---|---|---|---|
| all-reduce | one-shot | BF16, FP16, FP32; multiples of 16 bytes up to the one-shot limit | float32 sum in rank order 0..W-1, rounded once; identical bits on every rank; not bit-equal to NCCL |
| all-reduce | two-shot | above the one-shot limit, up to the capacity | the one-shot bits |
| all-reduce | `all_reduce_large`, pieces | any size, in two-shot (or one-shot) ops | the one-shot bits, for any piece size |
| all-reduce | `all_reduce_large`, chain | any size on a chain of cable neighbours | identical on every rank; half the message summed in chain order and half in reverse, rounded per hop; deterministic per size (`references.large_all_reduce`) |
| all-reduce, reduce-scatter, all-gather | ring | any size on a chain closed by its end ranks | identical on every rank; ring order, rounded per hop (`references`) |
| all-gather | one-shot | any plain dtype; dimension 0 or the last; unaligned shapes padded | byte-exact concatenation |
| all-gather | `all_gather_large`, tiles or chain | any dense dtype, any dimension, any size | byte-exact concatenation |
| reduce-scatter, all-to-all | scatter ops | `W` equal chunks of whole 16-byte packs | the one-shot bits per chunk; all-to-all moves bytes unchanged |

Model constraints (heads divisible by the tensor-parallel or DCP size,
hidden sizes in whole 16-byte packs) belong to vLLM and the model; the
sessions do not check them. A collective a session declines goes to the
caller's own path.

## Measured performance

Conditions for every row: the eight-Spark ring (ConnectX-7, MTU 9000, RoCE
v2, hairpin queues of 8,192 entries), serving image `aba309e4610c` (torch
2.13 with CUDA 13.0, nvidia-cutlass-dsl 4.7.0), ring harness, BF16, one
process per rank on GPU 0, median of the slowest rank.

One-shot all-reduce, CUDA graph replay, microseconds:

| Group | 8 KiB | 32 KiB | 64 KiB | 128 KiB |
|---|---|---|---|---|
| pairs (four at once) | 13.6-15.5 | 19.6-21.4 | 25.6-25.8 | 33.5-34.1 |
| path of four (Sparks 0-3, end ranks two relays apart) | 17.6 | 22.0 | 30.3 | 50.1 |
| cycle of eight | 23.9 | 35.9 | 56.3 | 92.2 |

Eager calls: one 8 KiB all-reduce on a pair takes 10.3 µs of host time
through the eager launch path.

Large messages, eager unless noted:

| Collective | Group | Schedule and link geometry | Time |
|---|---|---|---|
| all-reduce, 64 MiB | path of four | ring, 1 MiB pieces and slots | 5.21 ms |
| all-gather, 16 MiB shards | path of four | ring, 1 MiB pieces and slots | 2.58 ms |
| reduce-scatter, 64 MiB input | path of four | ring, 1 MiB pieces and slots, stagger 1; graph replay | 2.55 ms |
| all-reduce, 64 MiB | cycle of eight | ring, 2 MiB pieces and slots, 12 slots | 5.10 ms |
| all-reduce, 128 MiB | cycle of eight | ring, 2 MiB pieces and slots, 12 slots | 9.86 ms |
| all-gather, 16 MiB shards | cycle of eight | ring, 1 MiB pieces, 2 MiB slots, 12 slots | 5.05 ms |

Each Spark's NIC host interface bounds large collectives: about 24 GB/s sent
and 26.8 GB/s received, summed over its four RoCE functions; relayed traffic
does not count against it (`sparkring_sircl.bounds`).

Serving through the vLLM adapter (research-only; light tests, one run each):

| Model and layout | Settings | Prefill | Decode |
|---|---|---|---|
| GLM-5.3-Flash NVFP4, TP4 on a path of four | ring schedules from 2 MiB, mHC prefill sharding on the session reduce-scatter | 3,641 / 3,542 tok/s at 16K / 64K tokens | 25.3 / 53.1 / 70.4 / 100.7 steps/s at 1 / 4 / 8 / 16 users |
| GLM-5.3, TP8 with DCP 4 on the cycle of eight, no NCCL | fused all-reduce + RMSNorm, 1 MiB capacity and dispatch ceiling, 28 KiB one-shot limit | 1,327 tok/s at 16K tokens | 20.4 / 30.0 / 40.8 / 56.7 steps/s at 1 / 2 / 4 / 8 streams |

## Limitations

- Serving through the vLLM adapter is research-only: each configuration has
  one light test run.
- Members without a shared cable need the relay plan installed
  (`sircl-fabric up`, [runbook](RUNBOOK.md#relay-plan-installer)) and the
  ConnectX hairpin setting in effect. Nothing restores a relay plan after a
  reboot.
- A relay's hairpin queue holds 512 KiB and cannot pause its sender; forward
  windows keep every relayed lane within 75 % of it. Lanes through more than
  three relays are refused.
- Every Spark of a fabric must name its four RDMA functions alike; the names
  come from the DGX OS defaults or from `SIRCL_FABRIC_DOCUMENT`.
- GB10 offers no GPU-memory registration: session arenas are pinned host
  memory registered with `ibv_reg_mr`.
- Relays forward only tagged RDMA traffic, so TCP between Sparks without a
  shared cable cannot use fabric addresses; setup exchanges run over the
  serving engine's CPU process group or, in the ring harness, gloo over the
  LAN.
- No tuning table ships with the package.
