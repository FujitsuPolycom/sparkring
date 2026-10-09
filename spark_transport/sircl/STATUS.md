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
| Native progress thread (`oneshot/_roce_proxy.c`, ABI 9): arena, validated connection records, lanes, posting orders, CPU pinning, forward windows with delivery proofs, chain and link streams, event trace | implemented | CPU proxy simulator (78 cases: groups of 2 to 8 on cycles and paths, one and two lanes, sequence wrap, missed doorbells, injected failures, relayed lanes under simulated delays) and binding tests on the verbs stand-in; ring harness transport-only runs on pairs, a path of four, two paths of four at once and the cycle of eight |
| Setup agreement (`agreement.py`) | implemented | CPU tests |
| One-shot all-reduce and all-gather (`AllReduce.all_reduce`, `all_gather`) | qualified | ring harness on the eight-Spark ring (pairs, a path of four, two paths of four at once, the cycle of eight; eager and CUDA graph replay; all-reduce 16 B to 128 KiB, all-gather shards to 155,648 B and padded row shapes): every output bit-exact against the host reference; no RDMA, retransmission, `rx_out_of_buffer` or hairpin counter moved |
| Eager launch path (`oneshot/_fast_launch.py`, `SIRCL_FAST_LAUNCH`) | qualified | ring harness on a pair and the cycle of eight, every output exact; CPU tests of the argument block against the CuTe DSL's own conversion |
| Two-shot all-reduce; `all_reduce_large` in pieces; `all_gather_large` in tiles | implemented | GPU emulation (one RTX 5090, ranks as threads over the verbs stand-in) on `path:0-3`, `ring:8`, `path:0-1`, `ring:3`: every output bit-exact, eager and in graph replay; ring harness large-message cases on a path of four and the cycle of eight, every case exact |
| Collective ops independent of rank pointer alignment (`oneshot/_aligned.py`; `large_reduce_plan(nbytes)`, `large_reduce_staging(nbytes, input_aligned, output_aligned)`; [README.md](README.md#pointer-alignment)): `all_reduce_large`, `all_gather_large`, `reduce_scatter` and `all_to_all` choose their ring, chain, scatter or transport ops from the shared size, shape, dtype and agreed settings only; a rank whose input or output pointer is not 16-byte aligned runs the same ops on aligned working buffers (fresh allocations, from the graph's pool inside a capture) | implemented | CPU tests (`tests/test_alignment.py`: every rank of the cycle of eight and of a path of four, configured by the session's own steps, plans the same ops for every input and output alignment at 3 MiB, 2 MiB + 16 B and 64 MiB, and at 3 MiB, 8 MiB and 8 MiB + 48 B, and stages exactly its unaligned buffers when an op is a ring or chain op; the package before this behavior planned two-shot pieces on a rank with an unaligned input or output where its peers planned one ring op at 3 MiB on the cycle of eight and one chain op at 8 MiB on the path of four); GPU emulation on a DGX Spark's GB10 in serving image aba309e4610c (ranks as threads of one process over the verbs stand-in; 2026-10-09): `--alignment-only`, rank 0's input or output one element past 16-byte alignment in `all_reduce_large` of 3 MiB (eager, and two CUDA graph replays), `all_gather_large` of 1 MiB shards and `reduce_scatter` of 512 KiB chunks, 0 of 8 checks failed on `ring:8`, `path:0-3` and `path:0-1` with two lanes (every output exact, the ring, chain and scatter ops the same on every rank), where the package before this behavior failed 6 of 8 on `ring:8` (each op's flag wait on rank 0 timed out at the 20 s limit); the suite on `path:0-1` (151 checks), `path:0-3` (149) and `ring:8` (149) with two lanes and on `ring:3` with one lane (149), 0 failed; the scatter-collective checks on `ring:8` and `path:0-3`, 41 of 41 each; the DCP checks of one DCP group without tensor parallelism and of the TP8 and DCP4 world, 18 of 18 each; not run on a ring |
| Chain schedule (`chain`): all-reduce, all-gather and reduce-scatter as pipelined ops between cable neighbours | implemented | proxy simulator; GPU emulation as above; ring harness on a path of four and the cycle of eight, every case exact |
| Ring schedule (`ring`): all-reduce, reduce-scatter and all-gather over the chain closed by its end ranks, with staggered relays (`SIRCL_RING_STAGGER`, `SIRCL_RING_GATHER_STAGGER`) | implemented | CPU link simulator (`tests/test_ring_links.py`); GPU emulation; ring harness on a path of four (closing lanes through relays) and the cycle of eight, every case exact |
| Ring kernels' own finished pieces in one pass: a ring rank writes its own finished piece (the all-reduce's own result, the all-gather's own piece) to the link's own slot and to the output in one kernel pass and publishes it after that pass; every stored byte, the link-op words, the native ABI (9) and the launch arguments are unchanged | implemented | GPU emulation on a DGX Spark's GB10 (ranks as threads of one process over the verbs stand-in): 138 checks on `path:0-1` with two lanes, 136 on each of `path:0-3` with two lanes and `ring:3` with one lane, every case exact; in the event trace on `path:0-1`, the kernel pass per own result took 19.7-20.5 us against 28.5-41.2 us with two passes (8 MiB ring all-reduce, 512 KiB pieces). Ring harness on a cabled pair, eager periods at 1 block per role: the ring all-gather of 16 MiB shards in 512 KiB pieces 757.7 us against 814.1 us with two passes |
| Link blocks per role by group shape and link kernel (`protocol.link_blocks`, `stats()["link_blocks"]`): 1 for the ring all-reduce, all-gather and reduce-scatter on a cabled pair and on the cycle of eight and for the ring all-reduce and all-gather on a path of four, 4 for every other kernel and shape; `SIRCL_GATHER_LINK_BLOCKS`, `SIRCL_SCATTER_LINK_BLOCKS` and `SIRCL_REDUCE_LINK_BLOCKS` set one collective's under both its schedules, `SIRCL_LINK_BLOCKS` every other collective's | implemented | CPU tests (`tests/test_link_blocks.py`); ring harness eager periods at 4 / 2 / 1 blocks per role: on a cabled pair the ring all-reduce in 256 KiB pieces 245.8 / 221.3 / 212.0 us at 4 MiB and 2,857.5 / 2,823.9 / 2,812.5 us at 64 MiB, the ring reduce-scatter within 2-4 %; on a path of four the ring all-reduce in 512 KiB pieces 395.1 / 375.9 / 366.8 us at 4 MiB and 5,163.5 / 5,185.1 / 5,184.0 us at 64 MiB, the ring all-gather of 16 MiB shards 2,622.5 / 2,597.3 / 2,586.9 us; on the cycle of eight (16 link slots) the ring all-reduce in 512 KiB pieces 1,306.5 / 1,286.2 / 1,269.4 us at 16 MiB and 4,915.8 / 4,888.2 / 4,875.8 us at 64 MiB, the ring all-gather of 16 MiB shards 4,922.5 / 4,884.2 / 4,868.9 us and the ring reduce-scatter of 64 MiB 2,502.9 / 2,481.8 / 2,475.1 us, 1 block the fastest in 32 of 33 ring cases; not measured on other cycles |
| Built-in pair plan (`tuning.BUILTIN_PLANS`, `SIRCL_BUILTIN_PLAN`): without a matching tuning table a cabled pair runs the ring all-reduce from 3 MiB messages, the ring all-gather from 2 MiB shards and the ring reduce-scatter from 8 MiB inputs, in 256 KiB pieces (512 KiB from 16 MiB shards and from 64 MiB inputs), for every collective whose schedule variable is unset; no result bit changes on two ranks | implemented | CPU tests (`tests/test_builtin_plan.py`); GPU emulation on a GB10, the plan's checks among the 138 on `path:0-1` with two lanes; ring harness on cabled pairs ([Large messages on a cabled pair](#large-messages-on-a-cabled-pair)) |
| Built-in plan of the cycle of eight (`tuning.BUILTIN_PLANS["cycle:8"]`): without a matching tuning table a session of all eight Sparks of a cycle runs the ring all-reduce from the first message above the two-shot capacity (2 MiB) in 128 KiB pieces, in 256 KiB pieces from 4 MiB and in 512 KiB pieces from 16 MiB, while `SIRCL_LARGE_SCHEDULE` is unset; below, and for the all-gather and reduce-scatter, the rules apply. On eight ranks the ring adds each element's values in ring order, rounding at every hop, where the two-shot op adds them in rank order and the chain in chain order: every rank's result is the same, but its bits differ from those of the rules' schedules, which a tuning table or `SIRCL_LARGE_SCHEDULE` keeps | implemented | CPU tests (`tests/test_builtin_plan.py`: the decisions at and around each interval, a 3 MiB all-reduce as one ring op, a 2 MiB one as one two-shot op); ring harness on the cycle of eight, eager periods of the slowest rank at 1 block per role and 16 link slots, one run per arm: at 2 MiB the two-shot op 287.4 us and the ring in 128 KiB pieces 290.0 us, at 3 MiB the ring 349.4 us against 412.8 us for two-shot pieces; the ring at 4 / 8 / 16 / 32 / 64 MiB 423.6 / 713.3 / 1,269.4 / 2,473.1 / 4,875.8 us in its plan's pieces |
| Blocks per role per op: a tuning table's chain or ring choice names its blocks (`blocks`, schema `sircl-tuning-table/v2`), which the session applies to that op unless `SIRCL_GATHER_LINK_BLOCKS`, `SIRCL_SCATTER_LINK_BLOCKS`, `SIRCL_REDUCE_LINK_BLOCKS`, `SIRCL_LINK_BLOCKS` or `SIRCL_CHAIN_BLOCKS` sets the collective's; every link kernel and the chain kernel find a launch's last block as the arrival that completes its grid and return their tail word to 0, so consecutive launches of one kernel may change their grid; `tune` searches blocks per role with the pieces (`--tune-link-blocks`, `--tune-chain-blocks`, default 1, 2 and 4); a launcher is compiled for one count, and an op in a CUDA graph capture at a count none compiled raises, so `prepare` compiles the session's counts, every count of a tuning table's choices and every count `op_blocks` names (the counts a caller sets with `set_op_blocks` before ops it captures), and `tune` compiles every candidate's counts on every rank before its first case | implemented | CPU tests (`tests/test_op_blocks.py`: the tail rule against counting modulo the grid, schema v2, the tune candidates, every candidate's counts prepared before the tune's first case; `tests/test_prepare_blocks.py`: on a cycle-of-eight session, capture-time launcher lookups of every chain and link kernel at 1, 2 and 4 blocks per role refuse each count `prepare` was not given, and none after `op_blocks` names them); GPU emulation `_tune_capture_checks` (ranks as threads of one process on an RTX 5090 (WSL, 170 multiprocessors) and on a DGX Spark's GB10 (48 multiprocessors, image aba309e4610c), 2026-10-09: before this change, on sessions prepared as the ring harness's setup prepares them, a captured 262144 B chain all-reduce at the tune's chain candidates' blocks per role other than the session's own was refused ("SIRCL chain all-reduce for torch.bfloat16 was not prepared before CUDA graph capture") on `ring:8` and `path:0-3` with two lanes on both GPUs; with it, `--tune-capture-only` passed on `ring:8`, `path:0-3` and `path:0-1` with two lanes (RTX 5090: 22 checks each, every chain and link kernel at 1, 2 and 4 blocks per role captured and replayed exactly; GB10: 10, 16 and 22 checks, where the resident cap keeps `ring:8` at 1 and `path:0-3` at 2 blocks per role), and the `gpu_emulation` suite passed on GB10 on `path:0-1` (170 checks), `path:0-3` (162) and `ring:8` (156) with two lanes and `ring:3` with one lane (159), on a build that differed from this one only in the emulated tune's harness. The emulated tune (`--tune`, `path:0-3`, two lanes) passed 5 of 5 runs on GB10 (110 cases each) and 5 of 5 on the RTX 5090 (158 cases each) with its harnesses on rank-local synchronizes (`_RankTorch`); with the device-wide synchronize it passed 1 of 2 GB10 runs, and the other poisoned its sessions at the 120 s wait limit: after the barrier before the timed calls, a rank whose thread ran late waited in a device synchronization for a peer's first timed collective, which waited for that rank. A probe on an emulated pair reproduces that cycle on both GPUs (rank-local synchronize: exact in 1.0 s; device-wide: poisoned at the 10 s wait limit). The coupling `dcp_gpu_checks` documents (a rank thread's copy from pageable host memory holding a driver lock while another rank's CUDA call holds the interpreter lock, so the verbs stand-in's delivery thread stops) needs a host copy queued behind a collective in flight; the tune's host copies follow a synchronize of the rank's own stream, so they do not wait, and the tune needs no input staging from the main thread. Both couplings need every rank in one process on one GPU and cannot occur with one rank per GPU); GPU emulation of the per-op blocks on a DGX Spark's GB10 (ranks as threads of one process): the `gpu_emulation` suite on `path:0-1` with two lanes (145 checks), `path:0-3` with two lanes and `ring:3` with one lane (143 checks each) and `ring:8` with two lanes at one link and chain block per role (143 checks); not measured on hardware |
| Rotated buffers in the ring harness (`--rotate-buffers N`, default 1, `tune` 8): every timed case, NCCL's included, cycles through N input and output windows call by call (at most 4 GiB of windows per case), so a call reads and writes buffers the previous N-1 calls did not touch; tuning tables record N (`conditions.rotate_buffers`) and the summary's first line names it; `tune` also measures 1.5, 3 and 6 MiB | implemented | CPU tests (`tests/test_op_blocks.py`); not run on hardware |
| Scatter collectives: `reduce_scatter` in scatter ops, `all_to_all` | implemented | proxy simulator; ring harness on a path of four and on both DCP 4 groups of the cycle of eight at once (configuration `dcp4`), every case exact |
| Swing all-reduce (`oneshot/_swing_ops.py`, `oneshot/_swing_cute.py`) | research-only | runs from the ring harness configuration `ring-swing` only; sessions report it unavailable and refuse `SIRCL_ALLREDUCE_ALGORITHM=swing` |
| Flag waits: limits in seconds of the GPU clock, a startup regime (600 s) and a serving regime (20 s, after `enter_serving()`); `SIRCL_SPIN_LIMIT` bounds only waits without a time limit. A wait that times out is recorded by the kernel (control words in pinned host memory, then the poison word) after its op returned, so that op's outputs are not results, every later kernel of the session returns without work, and the session raises the failure at its next eager call's health check (`check_health`, `poisoned`); the vLLM adapter checks every session when each worker step starts and once the step's output is on the host ([vLLM adapter status](sparkring_sircl/vllm/STATUS.md)) | implemented | GPU emulation (a late peer waited out at startup, the group poisoned under the serving limit); proxy simulator case `wait-regimes` |
| Tuning tables (`tuning.py`, `SIRCL_TUNING_TABLE`; ring harness `tune` and `tune-table`): a measured choice of algorithm, schedule, piece, stagger, grid and (schema v2) blocks per role per collective, size and mode, the fastest by the group's period of back-to-back calls, with the session settings those choices need (link slots, link slot, chain slot, large-message piece), applied where the environment leaves them unset; a table chooses only among SIRCL's settings and its NCCL marks route no call | implemented | CPU tests of the tables, their settings and the harness; GPU emulation on `path:0-3`, `ring:8` and `path:0-1` (every op under the table's choice bit-exact) and the tune round trip on `ring:8` (the harness's `tune` on the emulated group, a table that records its settings, sessions that apply them, every output exact). No table ships with the package; without one, a cabled pair and the cycle of eight take their built-in plans and the rules of [Dispatch settings](README.md#dispatch-settings) choose every other op |
| Ordered close of sessions and point-to-point channel sets (`sparkring_sircl.teardown`; `close(*, abort=False)`, `close_result`; [README.md](README.md#close)): a close refuses further work, finishes the rank's own work (device synchronization; a channel set's streams) and holds two rounds over the exchange group, each one all-gather of a 512-byte note per rank waited for at most the flag-wait limit plus 5 s. Round 1 carries every rank's own failure while every progress thread runs; the rank stops its progress thread; round 2 runs only after round 1 arrived and waits for every rank's stop; then the native context is destroyed. The result is None after a healthy close, else this rank's failure, the first failed peer's or the round that did not complete, and a later close returns it. `roce_destroy` and `p2p_destroy` return the number of verbs calls that failed (native ABI 9 and point-to-point ABI 1 unchanged), and the bindings refuse a library whose local feature identity (`roce_local_features`, `p2p_local_features`) does not promise it, such as an earlier build named by `SIRCL_NATIVE_LIBRARY`; when it is not zero, or after a failed device or stream synchronization, the registered arena stays allocated for the rest of the process. `abort=True` (setup failures, garbage collection) holds no round. A note names its close (the object's kind, its ordinal among the rank's objects of the group, the round), and a round whose notes name another close does not complete, so every rank of a group must run one package. Residual risk: an abandoned round stays pending on the CPU group and pairs with the next collective any user issues there; it arises only after a failure, and a group whose round did not complete must not carry further sessions, channel sets or collectives | implemented | CPU tests (`tests/test_teardown.py`, 40: the fixed-size note, bounded rounds, the health vote, round 2 only after round 1, the unusable group, the retained arena; the session's and the channel set's `close` on stand-in native contexts for a healthy group and for each failure mode, each a terminal failure with cleanup exactly once; notes through a two-rank gloo group; `roce_destroy` and `p2p_destroy` counting a failed queue-pair destroy on the verbs stand-in); proxy simulator cases `teardown-race`, `teardown-ordered` and `teardown-failed-destroy`; GPU emulation on a DGX Spark's GB10: `--teardown` on `ring:8` and `path:0-3` with two lanes, 4 of 4 checks each (a healthy group; a poisoned rank named on every rank; one failed queue-pair destroy retaining exactly that rank's arena; an aborted rank, whose peers gave up round 1 after 6.01 s at a 6 s limit), and the suite on `ring:8`, `path:0-3` (144 checks each) and `path:0-1` (146) ending with every rank's close; the local feature check: CPU tests (`tests/test_native_binding.py` and `tests/test_p2p_binding.py` with `tests/native_stub.py`: a library without the required features or of another ABI is refused before any context exists, and the source builds report their features), and GPU emulation on a DGX Spark's GB10 in serving image aba309e4610c (2026-10-09), whose simulator builds of both native libraries passed the check at load and then served: `--teardown` on `ring:8` with two lanes, 4 of 4; the suite on `path:0-1` (146 checks) and `ring:8` (144) with two lanes, 0 failed; the point-to-point emulation on `path:0-3` and `ring:8` with two lanes, 20 of 20 each, every channel set up; not run on a ring |
| Flags-only own items: bit 24 of a link op word (`protocol.RING_OWN_FLAGS`) sends every own item of the op as its flags only, for a rank whose peer discards them (the empty direction of libsircl's pair exchange); native ABI 9, connection record and wire unchanged; SIRCL's own kernels do not set the bit | implemented | CPU tests (`tests/test_link_own_flags.py`: a pair on the verbs stand-in, 12 pieces through 8 slots, the flags-only rank's payload never reaches its peer and its link bytes posted are 0, every flag arrives and the other direction is exact; without the bit both directions exact; `tests/test_ring_links.py`: bits 25-31 refused); in libsircl on ConnectX-7 (two Sparks, nccl-tests v2.21.1, bfloat16 broadcast, reduce, gather, scatter and all-to-all, 512 KiB to 256 MiB): 133.1 GB per direction against 232.7 GB without it (NVIDIA NCCL 2.32.3: 133.2 GB) |
| Point-to-point channels (`p2p/`, native ABI 1): send, receive, batched send and receive between any two ranks of a group | implemented | point-to-point simulator (30 cases), binding tests (12), GPU emulation on `path:0-1`, `path:0-3` and `ring:8` (19 of 19 checks each); not run on a ring |
| Event trace (`SIRCL_EVENT_TRACE`; `python -m sparkring_sircl.ring trace`) | implemented | GPU emulation; ring harness on the cycle of eight (64 MiB ring all-reduce, no events lost) |
| CPU placement of launching and progress threads (`cpus.py`) | implemented | CPU tests |
| NCCL baseline of the ring harness (`--baseline nccl`) | implemented | CPU tests; ring harness on a pair and the cycle of eight |
| Period of back-to-back calls in the ring harness summary (`period_us`: per rank the median over consecutive calls of their mean time, the group's the slowest rank's; `period_busbw_gbps`; with the NCCL baseline `nccl_period_us` and `vs_nccl_period`), by which `tune` ranks candidates; NCCL rows take the bus factor of the collective they measure and the same untimed calls after the barrier as SIRCL's | implemented | CPU tests (`tests/test_ring_period.py`, among them a SIRCL and an NCCL row of the same size and time giving the same bus bandwidth); recomputed from the rank results of six pair groups: the 8 MiB ring all-reduce's slowest-rank median 519-535 us against periods of 434-446 us, the ranks alternating which one starts a call late |
| Ring harness on sites cabled as a path or by a list of cables (`sircl-ring-site/v1` `cabling`): a pair cabled port 0 to port 0, the whole of a path (configuration `path`), groups on the site's own cables, a preflight that requires active RDMA functions only on the ports the site cables, the rank watchdog (`--worker-timeout`) and the chain slot (`--chain-slot-bytes`) on the command line; the relay plan installer, the serve launcher and the point-to-point harness refuse a site not cabled as a ring | implemented | CPU tests (the cablings and their refusals, a pair's lanes over port 0, a path's groups equal to the same groups inside a ring, preflight of a path's end Sparks and of a pair, the watchdog and the run's wait derived from it, the chain slot); not run on Sparks |
| Relay plan installer (`fabric/`, `sircl-fabric`): origin routes, marker rules and relay filters per layout, ownership marks, group isolation | implemented | CPU tests against a host-command simulator; the `ring8` plan reproduces the universal relay table object for object (`tests/fabric_reference.py`); the installer has not run on a ring. Restoring a plan after a reboot: unsupported |
| Fused all-reduce + residual add + RMSNorm (`fused_norm/`; adapter `SIRCL_FUSED_NORM=1`) | research-only | CPU tests of geometry and reference arithmetic; bit-identity with the unfused path in GPU emulation on an RTX 5090, not on GB10; GLM-5.3 at TP8 with DCP 4 served with it on the cycle of eight (light test, one run) |
| Direct mlx5 posting (`SIRCL_POST_MODE=direct`), phase tracing (`SIRCL_TRACE`) | unsupported | refused at setup naming the setting |
| vLLM adapter: fabric placement, NCCL policy, route maps, planner and executor, NCCL guard, plugins, pins, hook table, shim catalog (`sparkring_sircl/vllm/`) | implemented | CPU tests against a stand-in vLLM package; every pin and hook anchor matched in the vLLM builds `pins.py` names |
| vLLM adapter: serving with SIRCL in front of every collective (group adapter and communicator, DCP collectives, mHC and Qwen3.8 hyper-connection prefill row ownership, flag-wait regimes) | research-only | served on the eight-Spark ring, one run per configuration: GLM-5.3-Flash at TP4 on Sparks 0-3 and 4-7 (alone and both at once) and at TP2 on a pair; Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash, MiMo-V2.6-Flash and Swift-1.5 at TP4 on four consecutive Sparks; GLM-5.3 at TP8 with decode context parallelism 1 and 4; every functional check passed |
| vLLM adapter: start-of-step failure check (`adapter.check_all_failures`, shim `step_health`): each of the worker's step methods (`execute_model`, `sample_tokens`, `execute_dummy_batch`) first checks every SIRCL session and point-to-point channel set of the process for a recorded failure, reading host memory only | implemented | CPU tests against the stand-in vLLM (`tests/test_vllm_step_health.py`): every step method runs while the sessions are healthy, and with one rank's session failed each raises before its body runs; a failed channel set raises at the step start; the check makes no device synchronization, statistics read, receipt write or regime change; an unpinned worker logs a warning and the failure raises at the post-step check. The shim's anchors and file pins match the vLLM builds `lil-image-aba309e4610c` and `sparkring-kraken-beta-20261007-bc9ea774`. The shim installed on every rank of GLM-5.3 at TP8 with DCP 4 on the cycle of eight (one serving run) |
| vLLM adapter: ordered shutdown (`adapter.py` `close`, `dcp_collectives.py`, `tp_slot.py`): each group adapter closes its DCP session, then its point-to-point channels, and the tensor-parallel slot its session, every rank in the same order, each close holding the two teardown rounds over the group's CPU group ([README.md](README.md#close)) and logging its result as a warning when a rank failed; a failure while the adapter builds a group closes what it built with `abort` (no rounds) | implemented | CPU tests of the session's and channel set's close (`tests/test_teardown.py`); not run in serving |
| vLLM adapter: point-to-point channels for pipeline-parallel groups | research-only | CPU tests on emulated ranks; not run on a ring |
| vLLM adapter: column gathers on the session's ring or chain (`executor.ColumnGather`, `SIRCL_COLUMN_GATHER`, default on): an all-gather along a dimension with rows in front of it, carried as a dimension-0 gather into staging plus one local copy | implemented | CPU tests (`tests/test_vllm_column_gather.py`); GPU emulation on `ring:8` with two lanes, 17 of 17 checks (shards of 4 to 8 MiB along the last, a middle and dimension 1; odd rows, FP32 and FP16, transposed and misaligned inputs; ring, chain and pieces schedules; one CUDA graph per rank replayed twice): staged and tiled outputs bit-identical to each other and to the concatenation on every rank; not run on a ring |
| Serving without NCCL (`--nccl never`, the default; `--require-no-nccl`, `--nccl-debug`, receipt and log checks in `check` and `bundle-check`) | implemented | CPU tests; GLM-5.3 at TP8 with DCP 4 on the cycle of eight: every rank's receipts `nccl=none pynccl=skipped`, no NCCL line in any rank's log with `NCCL_DEBUG=INFO` |
| Serve launcher and bundle (`sparkring_sircl/vllm/serve/`) | implemented | CPU tests on a synthetic SparkRing checkout and simulated Sparks, and against the repository's profile catalog; the serving runs above |
| Decode-context parallelism in profile serving (`serve --dcp-size N`) for the GLM-5.3-Flash checkpoints, with their startup conditions (KV-cache interleave of 4, mHC sizes admitted by `pins.MHC_ADMITS`), and for GLM-5.3 | implemented | CPU tests on the synthetic checkout (GLM-5.3-Flash at TP2 with DCP 2 on a pair and at TP4 with DCP 2 on a path, GLM-5.3 at TP8 with DCP 4 on eight Sparks, `--nccl never`; the refusals) and against the repository's `glm53-flash-nvfp4-spark-tp2` profile; not run on a ring |

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

### Large messages on a cabled pair

Conditions: DGX Spark pairs cabled port to port (both lanes on the one
cable), the serving image above, BF16, the ring harness's large cases, eager,
timed as periods of back-to-back calls (`period_us`) with buffers reused
between calls; NVIDIA NCCL 2.32.3 timed the same way in the same runs
(`--baseline nccl`); 1 block per role for the ring kernels.

| All-reduce | Two-shot (one op to 4 MiB, pieces above) | Ring | NCCL |
|---|---|---|---|
| 1 MiB | 77-78 us | | 101-102 us |
| 2 MiB | 129-130 us | 129.2-129.3 us (128 KiB pieces) | 122-125 us |
| 3 MiB | 183-184 us | 171.3 us (256 KiB) | 169-170 us |
| 4 MiB | | 211.0-212.1 us (256 KiB) | 218-221 us |
| 8 MiB | | 385.6-387 us (256 KiB) | 408-413 us |
| 64 MiB | 4,935 us (pieces of 4 MiB) | 2,806 us (256 KiB; 23.9 GB/s) | 3,105 us |

| All-gather, shard | Tiles | Ring | NCCL |
|---|---|---|---|
| 1 MiB | 80-81 us | 81-82 us (128 KiB) | 82-83 us |
| 2 MiB | | 132.2 us (256 KiB) | 138-143 us |
| 4 MiB | | 223.2-227.0 us (256 KiB) | 249-252 us |
| 16 MiB | | 757.7 us (512 KiB) | 938.1 us |

| Reduce-scatter, input | Scatter ops | Ring, 256 KiB | Ring, 512 KiB |
|---|---|---|---|
| 4 MiB | 157 us | 161 us | |
| 8 MiB | 266 us | 248 us | 268 us |
| 32 MiB | 1,321 us | 802 us | 811 us |
| 64 MiB | 2,637 us | 1,561 us | 1,497 us |

Results: the two-shot op leads to 2 MiB and ties the ring there; the ring
leads from 3 MiB for the all-reduce, from 2 MiB shards for the all-gather and
from 8 MiB inputs for the reduce-scatter, and its period is below NCCL's in
the same runs from 4 MiB (the all-gather's from 2 MiB shards). Two blocks per
role are 1-6 % slower than one at every all-reduce size. Buffers reused
between calls favour the two-shot op, so the 1-4 MiB crossovers under freshly
written buffers are not established. These are harness periods, which rank
settings within a run; a comparison with NCCL rests on nccl-tests with
repeated, counterbalanced runs.

Conclusion: a cabled pair takes the built-in pair plan (the ring all-reduce
from 3 MiB, all-gather from 2 MiB shards, reduce-scatter from 8 MiB inputs, 1
block per role); a pair's tuning table, measured with `tune`, replaces it. A
pair of Sparks inside a larger cycle, such as a DCP group of the ring of
eight, has no ring of its own and keeps the rules.

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
- No tuning table ships with the package; a cabled pair without one takes
  the built-in pair plan, every other group shape SIRCL's rules.
