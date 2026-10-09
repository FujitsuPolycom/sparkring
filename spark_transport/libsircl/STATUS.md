# libsircl status

Status of the library as a whole: **implemented; verified in GPU emulation; on a cabled pair of NVIDIA
DGX Spark systems, bit-identical to the SIRCL Python session's outputs and timed.** The NCCL-compatible
API layer, bootstrap, session engine, the four kernel packs, both transports and the point-to-point channels
are implemented. The pair at
ring positions 0-1 ran `RUNBOOK.md` sections 3.1 to 3.5 on snapshot fb63329a, nccl-tests v2.21.1 included,
and the pair at positions 6-7 ran sections 3.1 to 3.5, teardown rounds and a fail-stop late-rank case on
snapshot 0f94c72b, and the path of four at positions 4-7 and the cycle of eight ran nccl-tests, the
bit-exact check, timing sweeps and (path) teardown and fail-stop cases on snapshot a3477af2, point-to-point
channels included (hardware evidence below); no serving or training claim follows from this document.

Workstation conditions for every GPU-emulation result: one RTX 5090 (driver 595.79, CUDA 13.2 driver
API), WSL2 Ubuntu 24.04, x86_64, GCC 13.3, nvcc 13.3.73 for the kernel packs, Python 3.12 with torch
2.10.0+cu128 and nvidia-cutlass-dsl 4.5.0.dev0 for SIRCL's sessions, every GPU run under the
workstation's GPU lock, 2026-10-08 America/Chicago. Its clocks (WSL2, measured as the fail-stop row
describes): CLOCK_MONOTONIC runs about 5.6% fast against the GPU's `%globaltimer` (a 2.000 s kernel wait
measures 2.11 s), and CLOCK_REALTIME is stepped back by 1.2 to 1.7 s about every 30 s. The library binary of
every library, PyTorch,
setup-failure and timing run is `libsircl.so` SHA-256 `7ccc632e...6c42b2f61`
(`verification/binary.json`; a second build from the same sources in another directory has the same
hash).

Identifiers and logs. A snapshot named by eight hexadecimal digits (`fb63329a`, `0f94c72b`, `a3477af2`,
`e31abc5c` and others) is a copy of the library's files that its development workspace wrote, named by
the SHA-256 of its file list; this directory's first commit in the repository holds snapshot
`e31abc5c`'s files. Paths under `verification/`, `../sircl-ccl-snapshots/` and `ccl-pair/` name run
logs, snapshot builds and the ring operator's gate outputs that the machines which made them keep; they
are not in this repository.

## Sources

The SIRCL reference of the evidence below is SIRCL's implementation tree as captured on 2026-10-08 at
06:00 UTC: `SOURCE_SNAPSHOT.json` records the SHA-256 of each of its 200 files (build outputs, caches
and the outputs of SIRCL's ring harness runs excluded) and the 39 files that changed or were added since
the capture before it; at 03:00 CDT (08:00 UTC) every source file of that tree still matched the capture and no
source file had been added. SIRCL's own CPU tests passed on it (503 passed, 16 skipped;
`verification/sircl-cpu-tests.log`). This repository's SIRCL package is `spark_transport/sircl`
(`sparkring_sircl` 0.3.1). The library
carries byte-identical copies of two native libraries of SIRCL 0.3.1 (with change LF), in the LF-ending
bytes SIRCL's release builds its natives from, each SHA-256 recorded beside it and checked by the build
(which also refuses carriage returns; `tests/test_vendored_sources.py`): the native proxy
`oneshot/_roce_proxy.c` (`src/transport/sircl_roce_proxy.c`, `09db8715...`, the source of the serving
image's `roce_proxy-09db871538819191.so`), with the ordered session close (request TD: `roce_destroy`
returns the number of verbs calls that failed), flags-only own items (request FO) and the local feature
word `roce_local_features` (bits 0 and 1 for those two); and the point-to-point library `p2p/_p2p_proxy.c`
(`src/transport/sircl_p2p_proxy.c`, `c8ccfcb9...`, the source of `p2p_proxy-c8ccfcb93ffa2d6a.so`), whose
`p2p_local_features` bit 0 reports its `p2p_destroy` count. This repository's SIRCL package (`spark_transport/sircl`) holds the same
bytes. The reference copy above predates TD and FO (`_roce_proxy.c` `7deb5b1a...f2ae83`).
Also SIRCL's verbs-subset header (`src/transport/fake_verbs/infiniband/verbs.h`).
The interface follows NVIDIA's public `nccl.h.in` v2.32.3-1 and the user guide only.

## Components

| Component | Status | Evidence |
|---|---|---|
| Symbol surface: 73 `ncclX` host functions and 73 `pnccl` twins of the 2.32.3 header, and the 16 host entry points of NCCL's device API (`nccl_device/core.h` and its barrier and LL all-to-all headers, NCCL 2.28 and later) with their twins, which programs built against current headers import (`src/device_api.c`: `ncclCommQueryProperties` and the team queries implemented, the device communicator, device pointers and requirement builders refused); SONAME `libnccl.so.2`, version 22705; only `nccl*`, `pnccl*` and the three extension functions `sirclGetInfo`, `sirclSetWaitRegime` and `sirclGetReceipt` exported, each by exact name; `tests/api_manifest.json` marks each function of the header implemented or not; `tools/check_exports.py` lists any function a header set declares or a program imports that the library lacks | implemented | ABI suite (5 tests); engine suite's device-API test; generator check; `tools/check_exports.py`: 0 missing against the NCCL 2.29.7 headers and the nccl-tests v2.21.1 binaries built on the workstation; PyTorch and vLLM bind to it (`verification/framework-loading.json`) |
| Kernel-entry check: before the library links, its pack loader resolves every entry it names in the three embedded packs through a stand-in CUDA driver that reads every architecture's cubin offline (`tests/check_entries.c`, `tests/fake_cuda.c`; make and CMake) | implemented | kernel-entry suite (3 tests: every entry present; hiding one entry of each pack fails and names it; refused without the stand-in driver); a loader that names the two-pass ring all-reduce entries, built with link pack `5935b067...` (which has none), fails the build at `sircl_ring_reduce_two_pass_f32_u1` |
| Loading: the library links only libc and the dynamic loader, has no constructor that touches CUDA, and opens `libcuda.so.1` only when a communicator is created | implemented | CPU test in an isolated interpreter, with a negative control that initializes CUDA in a constructor; `tools/probe_cuda_on_load.py` (`verification/cuda-on-load.json`) |
| Bootstrap: unique id with root address, port, nonce and site hash; loopback, or a LAN address from `SIRCL_BOOTSTRAP_ADDR`, `SIRCL_BOOTSTRAP_IFNAME` or `NCCL_SOCKET_IFNAME`; all-gather rounds through the root; a rank leaving a complete group fails every later round at once | implemented | bootstrap suite (21 tests); between the two Sparks of the pair (hardware evidence) |
| Communicator lifecycle: blocking and non-blocking init, groups (a collective or point-to-point call refused with `ncclInvalidArgument` leaves its group as it was; a queued point-to-point call holds no reference on its communicator, so `ncclCommAbort` from another thread returns at once and that group's `ncclGroupEnd` fails the calls with `ncclInvalidUsage`), finalize and destroy (wait for every rank; see the teardown row), abort, revoke, a communicator's error and its text published together (one lock), `ncclRedOpCreatePreMulSum` (not implemented) returning `ncclNumOps`, which every collective refuses, async errors from the command ring and the progress thread; `ncclCommSplit` (members ordered by key, then parent rank; `NCCL_SPLIT_NOCOLOR` gets NULL; children keep the parent's route-map position; a NULL config, or a config whose `blocking` is undefined, takes the parent's blocking mode) | implemented | lifecycle (14) and API (17) suites, also under AddressSanitizer and UndefinedBehaviorSanitizer; GPU emulation (below): `library_rank.py`'s checks of an invalid-peer send inside a group (the group's valid send and receive complete), of `ncclCommAbort` with a send queued in another thread's open group (returns within 0.1 s; that group's end returns `ncclInvalidUsage`) and of `ncclRedOpCreatePreMulSum`'s op in an all-reduce of unequal inputs (refused with `ncclInvalidArgument`, the buffer unchanged). Abort cannot stop a kernel waiting for a dead peer before its wait limit (request AW) |
| Session engine: settings and both packs' hashes agreed across ranks; route maps keyed by process position (`LIBSIRCL_POSITION`, exchanged at setup), so one map serves every communicator a process joins; pinned arena (cuMemHostAlloc on hardware, shared memory registered with CUDA in emulation), device counters, SIRCL's native proxy, queue-pair connection, lane check, progress thread per communicator | implemented | GPU emulation (below); setup-failure cases; verbs transport on the pair (hardware evidence) |
| Transport kernel pack: one-shot and two-shot all-reduce, all-gather, reduce-scatter and all-to-all scatter ops in CUDA C++, sm_120 and sm_121, compiled by nvcc when the library is built; its hash, the fold pack's and the link pack's join the setup agreement | implemented | `KERNEL_ROUTE.md`: 905 of 905 mixed-group checks against SIRCL's DSL kernels; on the pair (hardware evidence) |
| Fold kernel pack: rank-ordered local reduction of gathered rows, 12 datatypes x 5 built-in ops, sm_120 and sm_121, compiled when the library is built | implemented | library emulation and the pair: every datatype and op equal to the host model |
| `ncclAllReduce`, every datatype, `ncclSum`/`ncclProd`/`ncclMax`/`ncclMin`/`ncclAvg`: float16, bfloat16 and float32 sums in the transport kernels (one-shot to `SIRCL_ONESHOT_MAX_BYTES`, two-shot above, pieces of `SIRCL_LARGE_PIECE_BYTES`, zero-padded tails below 16 bytes, unaligned buffers through scratch, in place); every other datatype and op as an all-gather of slot-sized tiles and the fold | implemented | library emulation, PyTorch emulation, the pair |
| `ncclAllGather` (every datatype): tiles of at most one slot sized from agreed settings only, so ranks whose buffers differ in alignment split a call into the same ops; tiles that are not whole 16-byte packs or not aligned travel through scratch; in place | implemented | library emulation, the pair |
| `ncclReduceScatter`, every datatype and built-in op: float16, bfloat16 and float32 sums in the scatter ops; every other datatype and op as an all-to-all of column tiles and the fold; padded and unaligned chunks through scratch; in place | implemented | library emulation, the pair |
| `ncclBroadcast`, `ncclBcast` (an all-gather of the root's tiles; other ranks may pass no send buffer) and `ncclReduce` (the all-reduce, kept on the root; other ranks may pass no receive buffer) | implemented | library emulation, the pair |
| `ncclAlltoAll` (every datatype; overlapping buffers through scratch), `ncclGather` (the all-gather's tiles, kept on the root) and `ncclScatter` (the all-to-all's tiles from the root); ranks that do not use a buffer may pass NULL | implemented | library emulation, the pair |
| `ncclSend`, `ncclRecv` between the two ranks of a two-rank communicator: outside a group each call is one exchange; inside a group the k-th send to the peer and the k-th receive from it form exchange k, carried at the outermost `ncclGroupEnd`; a rank's sends to itself are local copies on any communicator; CUDA graph capture | implemented | library emulation, PyTorch emulation, the pair |
| Stream order: a call on another stream than the communicator's previous call waits for that stream (an event); inside a CUDA graph capture every collective of one capture uses one stream; calls enqueue and return | implemented | alternating-stream and graph checks (below) |
| Receipts: per-communicator counts of calls, ops by algorithm and dtype, fold ops by datatype and op, link ops by kind and by blocks per role, pair exchanges, point-to-point sends, receives and bytes, captured calls, padded and unaligned staging, staging buffers and graph staging allocations, refusals and native counters; the pair plan, the fail-stop mode, the progress thread's CPUs and the ring all-reduce's relay form; in one file per communicator written at creation, refreshed after calls every `LIBSIRCL_RECEIPT_INTERVAL_S` and at destroy (`LIBSIRCL_RECEIPT`), and in process (`sirclGetReceipt`); `"forwarded": 0` always | implemented | library emulation and the pair check every rank's receipt |
| One-way pair exchanges post the empty direction's items as flags only: the exchange kernel sets bit 24 of the link op word when its input is 0, and the native proxy then posts those items without payload; the peer discards them as before | implemented | Conditions: snapshot ca09d28c (SIRCL's landed proxy with this change, link pack `dc9dd167...`) against snapshot 7d1c6f19 and NVIDIA NCCL 2.32.3 on ConnectX-7 between two Sparks (pair 4-5), nccl-tests v2.21.1, bfloat16, root 0, 512 KiB to 256 MiB, out of place, two runs of this tree and of NVIDIA. Measurement (us at 256 MiB and 4 MiB): broadcast 11,008 and 11,000 / 177 and 173 against 7d1c6f19's 11,976 / 216 and NVIDIA's 11,033 and 11,029 / 208 and 190; reduce 10,993 and 10,993 / 177 and 180 against 12,447 and NVIDIA's 11,025 and 10,990 / 205 and 220; gather 5,505 and 5,495 / 90 and 90 against NVIDIA's 5,531 and 5,543 / 109 and 115; scatter 5,483 and 5,494 / 88 and 86 against NVIDIA's 5,545 and 5,548 / 99 and 104; every job exact; NIC bytes per direction over the five rows 133.1 GB against 232.7 GB without the change (NVIDIA 133.2 GB), as in research snapshot bcff23bc. The all-to-all at API level 22705 (grouped point-to-point) does not use this path: 7,229 / 157 us against NVIDIA's 6,228 and 6,215 / 166 and 176. GPU emulation on the workstation: every one-way exchange of the two-rank library run bit-exact, the same link items with fewer link bytes on both ranks; `library_rank.py`'s flags-only check on a fresh split child, three rounds of ten one-way exchanges of 1.1 to 2.8 MiB (10 to 13 pieces each against 8 link slots, so every exchange wraps the slots at a shifting place) whose empty direction alternates between the ranks (broadcasts, gathers, scatters and ncclReduce sums of roots 0 and 1, sends each way), each round followed by a two-way all-to-all and all-gather: every output exact and 33 pair exchanges for 33 calls under the pair plan on one and two lanes, and on a relayed pair with forward windows and a ring window of 393,216 bytes, which carried them in 1,151 window chunks. Conclusion: the four rooted one-way collectives match NVIDIA within 1% at 256 MiB and are faster from 4 MiB |
| The all-to-all of two ranks as grouped point-to-point calls (torch's pattern; nccl-tests' below NCCL API level 22800): a send to and a receive from the peer and a send to and a receive from the rank itself of one size, the receive blocks adjacent in rank order and neither send block overlapping them, carried as one pair exchange whose kernel copies the own block in parallel with the transfer (otherwise a copy ahead of the exchange) | implemented | GPU emulation on the workstation: `library_rank.py` on two ranks, torch's pattern of 1 MiB blocks under the pair plan exact with one pair exchange and no separate local copy, and exact under the pieces schedule (no pair plan, the copy ahead); the emulation suite's nccl-tests lines (11, `alltoall_perf` included) exit 0 with `#wrong` 0. On ConnectX-7 between two Sparks (pair 4-5; snapshot b75d4b26 = this change on ca09d28c, against ca09d28c and NVIDIA NCCL 2.32.3; nccl-tests v2.21.1 `alltoall_perf` at API level 22705, bfloat16, 512 KiB to 256 MiB, out of place, two runs each; us at 256 / 16 / 4 MiB): 6,147 and 6,139 / 428 and 423 / 141 and 142 against ca09d28c's 7,217 and 7,266 / 493 and 493 / 159 and 159 and NVIDIA's 6,216 and 6,220 / 455 and 447 / 163 and 177; every job exact; broadcast, reduce, gather and scatter unchanged within the runs' spread. Conclusion: the all-to-all through torch's grouped point-to-point path is 1% faster than NVIDIA at 256 MiB, 5-7% at 16 MiB and 13-20% at 4 MiB |
| Shared-memory verbs stand-in (`src/transport/shm_verbs.c`): the verbs subset of SIRCL's proxy across processes, with seeded interleaving across queue pairs; segments named by process id and process start time, so containers that share the host's `/dev/shm` with reused process ids never share a name, and a name that exists already is skipped (never removed); a report of every queue pair and failed write for link dumps; the emulation runners remove the segments of their ended rank processes | research-only (test infrastructure) | CPU suite across processes (5 tests, one with stale segments under the names the ranks try first: the group runs exact and leaves them untouched; with the skip disabled the same test fails with `shm_open: File exists`); every library emulation run |
| MPI shim for nccl-tests (`tools/mpi-shim`: every MPI call and constant of nccl-tests v2.21.1, including `MPI_Allgatherv`, `MPI_Reduce`, `MPI_Error_string`, splits of any color and `MPI_UNDEFINED`, splits of splits, collectives among a communicator's members over a full TCP mesh; a job tag, `SIRCL_MPI_JOB`, in every hello, so processes of different jobs refuse each other; a watchdog that ends the process when a peer's process ends outside `MPI_Finalize`; with `SIRCL_MPI_DISTINCT_HOSTS=1`, a host name per rank, so nccl-tests on one host in emulation counts one rank per host and every rank uses device 0), nccl-tests driver and evaluator (`tools/nccl_tests_pair.sh`: one shim job and a time limit per test line, the ring harness's CPU placement, a job manifest `jobs.tsv` per rank and exit status 1 when any line failed; `tools/check_nccl_tests.py`: the manifests of every rank, the same jobs on every rank and, given the lines file, every expected job, exit 0 per job, every job's sweep complete, `#wrong` 0 with an all-to-all's in-place `N/A` counted as not covered, receipts whose all-reduce ops (transport, fold, chain, ring) cover their calls), all-reduce timing sweep (`tests/emulation/perf_rank.py`) | research-only (test infrastructure) | MPI shim CPU suite (6 tests, 1 to 4 processes: a peer that exits after `MPI_Init` ends the other within 15 s, ranks of different jobs never join, each rank's host name under `SIRCL_MPI_DISTINCT_HOSTS`); the emulation suite's nccl-tests section on the workstation (two processes, one GPU, 11 lines, every line exit 0 on both ranks and `Out of bounds values : 0 OK`); nccl-tests v2.21.1 (commit `afd59ab`) compiles and links against the shim on the workstation with every test binary, `comm_ops_perf` included (nvcc 13.3, NCCL 2.29.7 headers, sm_121; its GIN device-API tests need NCCL 2.30.7 headers and were not built, and their MPI calls compile and run as C++ against the shim on two processes); evaluator on synthetic outputs (`tests/test_check_nccl_tests.py`, 6 cases: a complete pair, a missing manifest, a partial log with a failed exit, a job missing from the lines file, ranks with different jobs, an in-place `N/A` outside all-to-all); timing sweep in emulation (below) |
| Link kernel pack (`kernels/sircl_links.cu`, `dc9dd167...`): SIRCL's chain all-reduce and link collectives in CUDA C++ (chain all-gather and reduce-scatter over links 0 and 1; ring all-gather, reduce-scatter and all-reduce over links 2 and 3 with both staggers), chain position and rank order as launch parameters, unroll 1-8, at most 512 threads per block (up to 128 registers per thread, no spills), sm_120 and sm_121, compiled when the library is built; the ring all-reduce's relay in the two-pass form (default: the slot, then the output) and in one pass (`LIBSIRCL_RING_REDUCE_PASSES=1`); the pair exchange entries; a launch's last block is the one whose arrival completes its own grid, so link ops of one kernel type may differ in blocks per role | implemented; the one-pass relay: research-only (on a GB10, one run under forward windows on four ranks returned differing bits on one rank of a split communicator; not yet isolated) | mixed groups against SIRCL's DSL kernels on `path:0-1`, `path:0-3` and `ring:8` of the two-pass pack `5935b067...`: chain and ring schedules and SIRCL's own link checks, every check passed (below); the one-pass pack's mixed groups are queued |
| Point-to-point kernel pack (`kernels/sircl_p2p.cu`, `0f39a3b9...`): the send and receive kernels of SIRCL's point-to-point channels (`p2p/_kernels.py`) in CUDA C++, one entry per direction and unroll 1-8, threads, lanes, slots, slot bytes and the native layer's block offsets as launch parameters (`sccl_p2p_args`), at most 512 threads per block (44 to 96 registers, no spills), sm_120 and sm_121, compiled when the library is built; its hash joins the setup agreement | implemented | GPU emulation against SIRCL's native point-to-point library (below, "Point-to-point channels"); not run in one group with SIRCL's DSL point-to-point kernels |
| The cycle plan (README.md, "Cycle default"): a communicator of three or more ranks without `SIRCL_LARGE_SCHEDULE`, given a ring plan, whose ring closes over cables only (every rank's relayed positions exchanged with the ranks' positions, the decision compared with the settings), runs all-reduces, all-gathers and reduce-scatters from 8 MiB as one ring op each and the pieces below; the receipt's `cycle_plan` | implemented | GPU emulation (`tools/emulation_suite.sh`): four and eight ranks with a ring plan and no relays take the plan and every check of `library_rank.py` passes against the pieces digests below 8 MiB and the ring digests from it; four ranks whose ring closes through relays, and four with `SIRCL_LARGE_SCHEDULE=pieces`, do not take it; with every rank's own planner settings the channel test's ring:8 layouts take it and path:0-3 does not. The measurement behind it: the ring schedules overtake the pieces between 4 and 8 MiB on the cycle of eight Sparks (snapshot a3477af2, hardware evidence). On the cycle of eight Sparks with the library of image 1a8c10354eb0 (tree dbf36074, 2026-10-09; [record](../../performance/records/transport/libsircl-ring8-image-1a8c10354eb0-20261009.md)): every eight-rank communicator without a schedule setting took the plan (96), the bit-exact check expecting it passed on every rank (16 runs), and from 8 MiB the default all-reduce ran within 1.15 times the ring schedules' time (256 MiB: 19,288 against 19,290 us eager; nccl-tests 19,314 us, 24.32 GB/s bus bandwidth) |
| Schedules of large messages and relayed groups, SIRCL's plan: `SIRCL_LARGE_SCHEDULE` (all-reduce: one ring op for the largest prefix of W equal chunks, or one chain op for the 16-byte-aligned body), `SIRCL_GATHER_SCHEDULE` (one chain or ring all-gather for 16-byte-multiple shards), `SIRCL_SCATTER_SCHEDULE` (chain or ring reduce-scatter of 16-byte-multiple chunks, in column tiles of `LIBSIRCL_LINK_TILE_BYTES`); each `pieces`, `chain`, `auto` (from SIRCL's per-collective chain minimums) or `ring` (from its ring minimums, else as `auto`); unset, `pieces` on groups of three or more and the pair plan on two ranks (a cabled pair, one block per link role: ring all-reduces from 2 MiB in 256 KiB pieces, ring all-gathers from 1 MiB shards with pieces by shard size, ring reduce-scatters from 4 MiB of input, and pair exchanges for all-to-alls, point-to-point exchanges of equal or one-way sizes, broadcasts, scatters, gathers and ncclReduce sums from 1 MiB; a relayed pair: the chain from 8 MiB; README.md, "Pair default"); blocks per link role per collective and per call (`LIBSIRCL_*_LINK_BLOCKS`, `SIRCL_LINK_BLOCKS`, the plan), agreed at setup; decided from agreed settings and sizes only, with unaligned and in-place buffers staged, so every rank runs the same ops; chain order from `LIBSIRCL_CHAIN_ORDER` or the ranks by position | implemented | CPU suites; GPU emulation (below); the pair default on the cabled pair at positions 6-7 (snapshot 0f94c72b, hardware evidence): every nccl-tests row of `RUNBOOK.md` sections 3.4 and 3.5 correct. Its sizes follow SIRCL's ring harness on two cabled pairs (section 3.6); libsircl's large-message times under the pair default have not been compared on Sparks with its other schedules, the ring harness or NVIDIA NCCL |
| Op selection from shared state only: a call's schedule, op kinds and count, pieces, blocks per role and link op words follow from its shape, datatype, root and the settings agreed at setup; a rank's buffer alignment, in-place or overlapping buffers, a discarded output (off the root) and its staging state choose only local copies through scratch or staging around the same ops. The staging buffer grows outside capture with every earlier buffer kept until destroy (no limit on the number of sizes); under CUDA graph capture a call that needs more staging than the communicator holds takes a graph allocation on the capturing stream (`cuMemAllocAsync`), freed on that stream when the call returns, so the graph owns it (README.md, "Op selection and staging") | implemented | Conditions: GPU emulation on the workstation; `library_rank.py`'s rank-local staging check on a fresh split child: the collectives of every selection point (one-shot, pieces and padded tiles; chain, ring and link ops; pair exchanges of roots 0 and 1; `ncclAlltoAll`; on two ranks torch's all-to-all pattern and one-way sends each way), with every rank's buffers aligned, then rank 1's unaligned, then rank 1's in place, then captured cold into one graph on a second fresh child with rank 1's unaligned and replayed twice; and the staging growth check, 19 all-gathers of doubling shards from 16 B to 4 MiB with rank 1 unaligned. Measurement (the runs under "Op selection under rank-local staging, and alternating flags-only directions" below): in every run each rank's receipt counts of launched ops (pair exchanges and their bytes, link ops by kind and by blocks per role, transport, chain and point-to-point ops) and of carried transfers (native link ops and items, one-shot ops and phases) equal its aligned run's, and the launched counts are the same on every rank; every output exact; the cold capture is taken on every rank, each of rank 1's staged calls as a graph allocation (16 on the pair default, 20 under the ring schedule on two ranks), and both replays are exact; rank 1 keeps 19 staging buffers under the chain and ring schedules. The same checks against snapshot ba5a337b's library, with a 15 s wait limit: under the pair default, rank 1 refused its cold capture of the staged calls with `ncclInvalidUsage` while rank 0 captured them, rank 0's replays returned wrong outputs at the wait limit and the child's destroy failed; under the ring schedule, rank 1 also refused the 18th and 19th all-gathers of the growth check (`ncclInternalError`, too many staging buffer sizes) while rank 0 launched them, and rank 0's session was poisoned at the wait limit. Conclusion: in these runs a rank's alignment, in-place buffers, role and staging state change no rank's ops |
| Progress thread placement: `SIRCL_PROGRESS_CPU` pins it; otherwise `LIBSIRCL_CPU_POLICY=performance` (default) starts it on the fastest CPU class the creating thread may use (GB10: the Cortex-X925 cores), `none` leaves it to the scheduler; the application's threads keep their affinity; receipts name the thread's CPUs (`progress_cpus`) | implemented | build and CPU suites (the workstation's CPUs are of one class); on GB10s (Sparks 6-7, snapshot 0f94c72b, hardware evidence) the default policy started the thread on CPUs 5-9 and 15-19 and `SIRCL_PROGRESS_CPU=19` pinned it to CPU 19, as each receipt's `progress_cpus` records |
| The chain and link areas in the arena after the control line, the link counters, piece counters and decision words, SIRCL's link settings (`SIRCL_LINK_*`, staggers) agreed at setup; the ring plan from `LIBSIRCL_RING_WINDOW`; forward windows of relayed lanes from `LIBSIRCL_FORWARD_WINDOWS`; `tools/site_routes.py` prints every rank's route map, chain order, forward windows and ring plan from SIRCL's route planner | implemented | CPU suites; setup-failure cases; `tools/site_routes.py` reproduces the pair's route maps and gives `path:0-3` a ring window of 393,216 bytes on position 3 |
| Teardown: `ncclCommDestroy` and `ncclCommFinalize` of a communicator of two or more ranks wait for this rank's work, then a bootstrap round of every rank that also carries each rank's health, then stop the progress thread, then a second round, before any queue pair, registration or arena is freed; each round bounded by the wait limit; the close is terminal (new work refused, its result kept); a failed release of the transport keeps the memory allocated; `ncclCommAbort` does not wait | implemented | Conditions: GPU emulation on the workstation and on GB10s (snapshot 7d1c6f19, built in the serving image); `tests/emulation/teardown_race.py`: a reversed `ncclCommSplit` child, one ring all-reduce, `ncclCommDestroy` at once, repeated, one rank's writes delayed 5 ms so its predecessor finishes first; with `--expect close-error`, a rank 4 s late under a 2 s wait limit, or the stand-in's deregistrations failed (`SIRCL_EMU_FAIL_DEREG=1`). Measurement: four ranks and two ranks at 8 MiB per rank, 20 rounds each, every round exact with no async error and the parent healthy (GB10: four ranks, 5 runs of 20 rounds, 0 failed); without the rounds (snapshot 6fe2882f) the same check failed 7 of 10 rounds on four ranks on the workstation and 98 of 100 on a GB10 (each failure poisons the parent): a rank's kernel had completed while its progress thread still held an outbound item gated by a peer's credit, destroy stopped that thread and freed its queue pairs, the peer waited out the wait limit, and the peer's late credit write failed (transport retry counter exceeded; the destination queue pair was gone). In close-error mode (workstation and GB10) every round's first finalize or destroy returned the error, a second finalize the same, and a collective after the close was refused. The full emulation suite on a GB10 (snapshot 7d1c6f19) passed every section, the teardown check on four ranks and on two at 8 MiB included. On the fabric (ConnectX-7, four Sparks on a path and a cabled pair, 8 MiB per rank, no delayed rank): 100 rounds each without the rounds and 60 with them (snapshot 7d1c6f19), all exact and healthy, and 10 runs of the relayed-group bit-exact check under the ring schedules (RUNBOOK 4.1) on the path, 0 failed (Sparks 0-3 and 0-1, `ccl-pair/done-hw-teardown-7d1-20261008T194629Z`); the same 60 rounds per layout and 10 checks at snapshot ba5a337b on Sparks 4-7 and 4-5, 0 failed (`ccl-pair/hw-teardown-ba5-47-20261008T225027Z`); the stranded item did not occur at natural timing there; on the cabled pair at positions 6-7 (snapshot 0f94c72b, the rounds of this tree, 2026-10-09): 6 runs of 20 rounds at 8 MiB, 3 with fail-stop off and 3 on, 0 failed, the parent exact and healthy. Result: round 1 proves that every item and flag a kernel waits for has landed; round 2 that no rank posts toward queue pairs or regions about to be freed. Writes no kernel waits for (credits) may still be in flight; each rank destroys its queue pairs before its registrations and memory, so they land in registered memory or are dropped. Limits: the rounds are bounded, but this rank's own wait for its work, the progress thread's join and the verbs destroys are not; in emulation the stand-in resolves a write under its registry lock and applies it after, so a write resolved before a deregistration still lands in the old segment, and emulation does not test a NIC's revocation of a deregistered region. Callers: torch 2.10's `destroy_process_group` shuts its groups down in one order on every rank (finalize, then destroy); vLLM 0.19.1's pynccl `destroy` calls `ncclCommDestroy`; a caller that tears down with `ncclCommAbort` runs no rounds |
| Destroying or finalizing communicators in a different order on different ranks | unsupported | each teardown round waits for every rank of its communicator, so crossed orders wait out the wait limit on each round (20 s serving, 600 s startup) and return `ncclRemoteError`; they do not wait longer |
| Process exit without `ncclCommDestroy`, `ncclCommFinalize` or `ncclCommAbort` while a peer's collective on the communicator still runs | unsupported | at library unload only the native threads stop; a peer may wait out its wait limit |
| `ncclCommDestroy` while replays of captured CUDA graphs of the communicator's collectives still run | unsupported (caller contract) | teardown waits for the engine's last eager launch (a launch under capture records no stream), not for graph replays; a communicator used only under capture has nothing to wait for |
| CUDA graph capture of a call that needs more staging than the communicator holds, on a driver or device without stream-ordered allocation (a driver before CUDA 11.2, or a device without memory pools; the GB10 and the workstation's RTX 5090 have them) | unsupported | the call is refused with `ncclInvalidUsage` on the rank that needs the staging while its peers capture theirs; it captures after one eager call of that size has run on that rank. Whether every rank has stream-ordered allocation is agreed at setup |
| Fail-stop (`LIBSIRCL_FAIL_STOP=1` or `abort`, per process): at the first asynchronous error (a flag wait that timed out, a failed progress thread, a failed point-to-point channel) of a watched communicator of two or more ranks, found by the watcher thread (every `LIBSIRCL_FAIL_STOP_POLL_MS`, default 5) or by the first library call that would report it (`ncclCommGetAsyncError`, an enqueue's check, `ncclCommFinalize`, `ncclCommDestroy`, `ncclCommAbort`), the library writes the error, the time and what found it to stderr and the communicator's receipt, then ends the process, with `1` at once (exit status 70, no atexit handlers, no core dump), with `abort` by `abort()`; destroy and abort check once more, then take the communicator off the watch before releasing it, and library unload stops the watcher first | implemented | Conditions: the workstation, `tests/emulation/fail_stop.py`, two ranks, SIRCL_STARTUP_WAIT_S 2, rank 1 starting its second all-reduce 12 s late, rank 0 enqueuing its own at once, waiting for its stream, keeping the output and calling the library no more. Measurement (library `638cca1b...`, every time CLOCK_MONOTONIC): 15 runs of the exit and control cases and one of all three: with `1`, the fail-stop line 2.11 to 2.13 s and the process's end 2.43 to 2.53 s after the enqueue, exit status 70; with `abort`, the line 2.11 s after the enqueue and SIGABRT 20.8 s after it (WSL's crash capture ran first); without fail-stop, rank 0's stream wait returned 2.10 to 2.13 s after the enqueue with a wrong output, every call it made returned 0, and the process ran on past 7 s. With `LIBSIRCL_FAIL_STOP=1` the pair default's 533 library checks passed and no process ended; on a build of the same fail-stop code, so did the ring schedules' 989 checks on four ranks and 10 teardown rounds on four ranks. The wait limit itself (research build of this tree whose timed-out flag wait records its `%globaltimer` start and end and the limit it read): 34 timed-out 2 s waits lasted 2.00002 to 2.00095 s of `%globaltimer` with the limit read as 2,000,000 us; a 300 s kernel and five 10 s kernels that sampled `%globaltimer` every 1,024 polls showed no step beyond 11 ms (host-stamping jitter) and a rate of 0.943 to 0.948 of the workstation's CLOCK_MONOTONIC; the workstation's CLOCK_REALTIME stepped back by 1.2 to 1.7 s about every 30 s (WSL2's time synchronization), so two of 28 waits timed by the wall clock looked 0.40 s long while the kernel measured 2.0003 s; earlier runs of this check that used the wall clock showed the same. On the fabric (cabled pair at positions 6-7, snapshot 0f94c72b, 2026-10-09; `teardown_race.py`, rank 1 starting its child all-reduce 8 s late, SIRCL_STARTUP_WAIT_S 2): with `1`, rank 0 exited with status 70 3.9 s after it started, setup included, its fail-stop line naming the timed-out flag wait for the late rank; without fail-stop it exited 1 after 10.4 s with the round reported failed. Library calls (this tree, library `9776657f...`, GPU emulation, 2026-10-09, `verification/emulation/suite-s1/`): `fail_stop.py`'s cases in which rank 0 calls `ncclCommGetAsyncError`, `ncclCommDestroy` or `ncclCommAbort` right after its stream wait, with `LIBSIRCL_FAIL_STOP_POLL_MS=600000` so the watcher does not poll, end rank 0 with status 70 inside the call (2.31 to 2.38 s after the enqueue, the line 2.00 s after it, "found by a library call" or "found by ncclCommDestroy or ncclCommAbort"), the call never returning; against the library before this path the same cases had `ncclCommGetAsyncError` and `ncclCommAbort` return the error to rank 0, which ran on and exited 0. `teardown_race.py --expect fail-stop`, four ranks, rank 3's child all-reduce 8 s late, 2 s wait limit: every rank exited 70 in 6 of 6 runs; in 9 of the 15 non-late exits of 5 runs a library call found the error before the watcher (the gate's path of four lost one rank that way before this path). Conclusion: with fail-stop, a caller that checks only enqueue codes keeps a failed collective's output at most for the wait limit plus one poll; a read within that poll after its stream wait returns is not prevented |
| `ncclCommSuspend` and `ncclCommResume` | unsupported | refused with `ncclInvalidUsage` (logged once); a communicator's arena and device memory stay allocated while it exists |
| Point-to-point between two ranks of a communicator of three or more ranks on SIRCL's point-to-point channels (`LIBSIRCL_P2P_CHANNELS=on`, off by default; README.md, "Point-to-point channels"): SIRCL's native point-to-point library vendored byte-identical (`src/transport/sircl_p2p_proxy.c`, `c8ccfcb9...`, SIRCL 0.3.1's copy with change LF, checked by the build, its symbols prefixed per transport and not exported), created inside the communicator's setup exchanges (each rank's lanes and `LIBSIRCL_P2P_WINDOWS` give a channel table every rank derives alike; connection records validated together; lane check; windows of relayed lanes), first-in first-out matching per ordered pair, one stream per channel direction joined to the caller's stream, sends queued before receives, staging of messages not 16-byte aligned or not whole packs (stream-ordered allocation, or each channel's own buffer), refusal on both ranks of a pair without a channel, asynchronous errors and fail-stop on every rank after a kernel's timeout or size mismatch (abort notices), the channels' progress thread stopped in the teardown rounds and their verbs objects counted at destroy, receipts (`channels`) | implemented | GPU emulation on the workstation (below, "Point-to-point channels"): four and eight ranks, ring:8 under the route planner's settings, torch initialized eagerly, nccl-tests' sendrecv, hypercube and alltoallv on four processes; CPU: the native library across processes over the verbs stand-in. Not run on Sparks |
| Point-to-point channel items under CUDA graph capture | unsupported | refused with `ncclInvalidUsage`: a replay would reuse the item numbers fixed at enqueue |
| Point-to-point channels created at the first `ncclSend` or `ncclRecv` | unsupported | connecting the channels is collective over every rank, and a first send involves two; the channels are created with the communicator when `LIBSIRCL_P2P_CHANNELS=on`, and a communicator without them holds none of their memory, queue pairs or thread |
| Point-to-point windows of the route planner (`tools/site_routes.py`, SIRCL's `p2p/budget.py`): `--p2p-reserve session` reserves the collective session's forward windows first, and its ring windows only with `--ring-schedules` (libsircl's ring lanes carry traffic only under the ring schedules); `--session-share F` (default 1) sizes the session's forward and ring windows against F of every relay queue's share and gives the channels the rest; a relayed pair without a window on every lane in both directions has no channel, named in the JSON output | implemented | CPU (`tests/test_site_routes.py`, 8 tests, with `SIRCL_PACKAGE` naming the SIRCL reference tree's `spark_transport/sircl`): path:4-7 gives all 6 relayed ordered pairs a window without `--ring-schedules` and leaves 3 without one with it (rank 3's 393,216-byte ring window reserved); ring:8 leaves all 40 without one at the default plan and gives all 40 one with `--max-window 32768` and with `--session-share` 0.95, 0.5 and 0.25 (at 0.95 every relayed session lane one 32 KiB chunk smaller, the channels' windows 32 KiB); a share of 1 prints the default's output; shares outside (0, 1] are refused; in every case no relay queue holds more than its 393,216-byte share (session plus channels), and `--p2p-reserve none` on ring:8, as the check's negative control, overfills one; the ring:8 files of `tests/data` equal the tool's output |
| The setup check of the native proxy's local feature word (`roce_local_features`, SIRCL change LF): communicator setup fails on every rank, naming both bits, unless bit 0 (`roce_destroy` counts failed verbs calls) and bit 1 (flags-only own items, which the pair exchange without input uses) are set; the build refuses a vendored copy that differs from its recorded SHA-256 or does not define the word | implemented | CPU: the native probe reads bits 0 and 1 from the vendored copy (`tests/test_shm_verbs.py`); the library carries the check (`tests/test_engine_cpu.py`); a build against a modified copy, a copy with CRLF endings, or the copy before LF with its own hash, stops with the reason. GPU emulation: every communicator of the suite passed the check. Not run on Sparks |
| The setup check of the point-to-point library's local feature word (`p2p_local_features` bit 0: `p2p_destroy` counts failed verbs calls; SIRCL change LF): channel setup fails with a named reason when the bit is absent; compiled by default (`P2P_FEATURES=1`, CMake `LIBSIRCL_P2P_FEATURES`), and the build refuses a vendored copy without the word | implemented | CPU: the native probe reads bit 0 from the vendored copy (`tests/test_p2p_native.py`); the default build carries the check (`tests/test_engine_cpu.py`); a default build against the copy before LF stops with the reason, and `P2P_FEATURES=0` builds it. GPU emulation: every channel setup of the suite below passed the check. Not run on Sparks |
| Pipeline parallelism on two-rank communicators: a pipeline group of two ranks (TP 4 x PP 2 on a ring of eight: positions i and i + 4, every lane through three relays), and torch's lazily initialized ProcessGroupNCCL, which gives each stage pair of a larger pipeline group a two-rank communicator of its own (vLLM 0.19.1's V1 pipeline exchange: `isend_tensor_dict` and `irecv_tensor_dict`, one `torch.distributed.isend` or `irecv` per tensor) | implemented | Conditions: GPU emulation on the workstation (library `02573b7c...`), every rank with the settings `tools/site_routes.py --layout ring:8 --lanes 2` emits for its position (`tests/data/site_routes_ring8_l2.json`). Measurement: `tests/emulation/pp_pairs.py` (eight ranks; tensor-parallel splits of positions 0-3 and 4-7, pair splits of positions i and i + 4; six microbatches per pair at its own pace of a tensor dictionary of 2 MiB, 4 MiB, 6 KiB and 1,000,003 B sent as torch's batch_isend_irecv groups them, one tensor back, and every third microbatch both ways in one group): 64 of 64 checks, every received byte exact; each pair ran 20 pair exchanges through its ring lanes' window of 65,536 B (640 to 1,408 window chunks) and its small transfers through the forward windows; without the ring plan 64 of 64, every transfer through the forward windows. `tests/emulation/torch_pp.py`: vLLM's exchange through torch.distributed, initialized lazily: TP 4 x PP 2 under the same ring:8 settings (an all-reduce on the default group, `new_group` tensor-parallel and pipeline groups, four microbatches) 56 of 56 checks exact over four two-rank point-to-point communicators, each pair's ring lanes windowed; a chain of four stages 12 of 12 over three two-rank communicators. Not run on Sparks |
| Communicators of some of a layout's positions (split children; two-rank communicators such as torch's per-pair ones) under the route planner's per-process settings, one `LIBSIRCL_CHAIN_ORDER`, `LIBSIRCL_FORWARD_WINDOWS` and `LIBSIRCL_RING_WINDOW` for every communicator a process joins: the chain order's listed members in its order, the other positions skipped; a ring next reached through relays keeps the smallest forward window toward it when the layout's ring plan gives 0 | implemented | Libraries up to and including snapshot db529218 (the serving image's) refuse every such communicator at setup ("LIBSIRCL_CHAIN_ORDER=0,1,2,3,4,5,6,7 must list the positions of the communicator's ranks once"): `torch_pp.py --shape tp4pp2` failed there with 17 problems (the tensor-parallel groups did not form) and `pp_pairs.py` 8 of 8 (the splits); this tree's library forms them (the pipeline row's runs). With the ring lanes' window not derived (a research build of this tree), every pipeline pair ran its 20 pair exchanges through relays unwindowed (ring window 0 B, 0 window chunks; emulation, which has no relay queues, still exact). Unchanged where the ring next is a cable or the plan gives a window: cabled pairs, `path:0-3` (393,216 B on position 3) and ring:8; the pair default, windowed pair, relayed pair, ring schedules on two, four and eight ranks, the four-rank ring with windows and pieces on four and eight ranks passed every check of `library_rank.py` (533, 533, 531, 533, 989, 1,977, 989, 985 and 1,969) |
| `ncclMemAlloc` and `ncclMemFree` (device memory of the current CUDA context, ordinary buffers for every collective, eager and in graphs); buffer and window registration (`ncclCommRegister`, `ncclCommWindowRegister` and their deregistrations) accepted as hints, the handle and the window being the buffer's address, and `ncclWinGetUserPtr` | implemented | API suite (registration and windows on a CPU communicator); nccl-tests v2.21.1 allocates its buffers with `ncclMemAlloc` |
| `ncclRedOpCreatePreMulSum` and other ops beyond the five built in, the per-call `nccl*Config` collectives, shrink, grow, the scalable initialization, host RMA signals, the device communicator and device pointers | unsupported | refused with `ncclInvalidUsage` (an op number beyond the built-in five: `ncclInvalidArgument`, counted in the receipt) and logged once |
| One progress thread per process; an abort word watched by every kernel wait | unsupported | each needs a change to SIRCL's package (`spark_transport/sircl`) that it does not have: an externally driven progress loop with which one thread serves every session of a process (request PO), and an abort word in the command ring that every timed flag wait watches (request AW) |
| All-reduce timing on a cabled pair (`tests/emulation/perf_rank.py`: bf16, fp16, fp32, 8 B to 256 MiB, eager and graph) | qualified | the pair (hardware evidence): fp32 8 KiB 9.73 us eager, 9.61 us in graphs |
| nccl-tests v2.21.1, unmodified, through `LD_PRELOAD` on a cabled pair: every collective test and point-to-point | qualified | snapshot fb63329a on Sparks 0-1 (hardware evidence): `#wrong 0` on every line of `RUNBOOK.md` sections 3.4 and 3.5; snapshot 0f94c72b, this tree's library and pair plan, on Sparks 6-7, 2026-10-09 (hardware evidence): `#wrong 0` on every row of sections 3.4 and 3.5 and of `sendrecv_perf` to 256 MiB (in-place `N/A` on all-to-all and sendrecv, which have no in-place result) |
| Frameworks on the ring | unsupported | PyTorch in emulation only |

## Evidence

The files behind each result are in `verification/` (CPU, build and loading records),
`verification/emulation/` (GPU emulation logs and per-rank results) and `verification/hardware/` (the
cabled pairs).

### Hardware: the cabled pair at ring positions 0-1

Conditions common to both runs: two DGX Sparks (GB10, aarch64) at positions 0 and 1 of the eight-Spark
ring, cabled directly; the serving image `aba309e4610c...`; containers with `--privileged --gpus
device=0 --network host --ipc host --ulimit memlock=-1`; the library built from source in each container
with the kernel packs transport `c2e6e5a1...` and fold `66da585d...`; `LIBSIRCL_TRANSPORT=verbs`,
2 lanes, route maps `1=rocep1s0f0/roceP2p1s0f0` and `0=rocep1s0f1/roceP2p1s0f1`; run by the ring
operator on 2026-10-08. The rank scripts load the library with ctypes, so the NVIDIA NCCL the image
preloads (below) stays out of their calls; every receipt names the verbs transport and libsircl's packs.

Run at 08:43 UTC, `RUNBOOK.md` sections 3.1 and 3.2, source archive SHA-256 `70727348...0f17d5e85` (a
tree without point-to-point, split and route-map positions;
`verification/hardware/pair-20261008T0843Z/`):

- `library_rank.py`: 217 of 217 checks per rank against the rank-order reference, the fold model and
  the bytes sent; communicator setup 0.22 s, the whole check 9 s; receipts `"forwarded":0`, refusals
  only the deliberate op-7 case, `"healthy":true`, all-reduce 130 calls (9 captured), the native layer
  294 ops posted and 646 RDMA writes completed;
- `make check` in the image: every suite but the CUDA-on-load test, whose check (any mapped file named
  `libcuda`) the image met before the library loaded.

Run at 10:06 UTC, sections 3.1 to 3.3, source archive SHA-256 `89380984...ae7027e4a`, library SHA-256
`c8c659c9...` on both ranks (`verification/hardware/pair-20261008T1006Z/`):

- `make check` in the image: every suite passed (ABI 5, API 14, bootstrap 21, lifecycle 13, engine 4,
  shared-memory verbs 4, MPI shim 3).
- `tools/probe_cuda_on_load.py`: the library mapped and initialized nothing of CUDA, with and without
  site packages. The image's environment sets `LD_PRELOAD` to NVIDIA NCCL
  (`/opt/sparkring/toolchain/nccl/lib/libnccl.so.2`), `libcudart.so.13` and eight further CUDA
  libraries, so `libcuda` and `libcudart` are mapped before any program starts (`cuDeviceGetCount` 3:
  not initialized). `RUNBOOK.md` section 3 puts libsircl first in that list and drops the other
  `libnccl.so.2` for programs that bind NCCL through the dynamic linker.
- `library_rank.py` with SIRCL-session digests: 228 checks per rank. Every check against the rank-order
  reference, the fold model and the bytes sent passed, including the 11 point-to-point and split checks
  (the split reverses the ranks, so each process joins the child at another rank with its own route-map
  position). 178 and 180 checks passed: the other 50 and 48 compared outputs with digests made on the
  workstation from inputs that `torch.randn` generated: the image's torch on aarch64 and the
  workstation's torch 2.10 on x86_64 generate different floating inputs from the same seed for all but
  the smallest counts. The all-gathers among them move bytes unchanged, and every uint8 all-gather,
  whose inputs came from `torch.randint`, matched. The inputs are
  now SplitMix64 bits and the digests record each case's input digest too (Library evidence below).
- `perf_rank.py`, all-reduce sum, out of place, 50 timed calls after 10 warm-up calls, graphs of 20
  calls, 72 sizes (bf16, fp16, fp32 from 8 B to 256 MiB), every size up to 64 MiB checked bit for bit
  (`perf-rank0.out`):

  | Measurement | bf16 | fp16 | fp32 |
  |---|---|---|---|
  | 8 KiB eager, per call | 9.83 us | 9.74 us | 9.73 us |
  | 8 KiB in a CUDA graph, per call | 9.55 us | 9.64 us | 9.61 us |
  | 8 B to 8 KiB eager | 8.29-12.35 us | 8.07-12.08 us | 8.07-11.68 us |
  | 128 KiB eager | 18.35 us | 18.35 us | 18.32 us |
  | peak bus bandwidth (4 MiB) | 17.63 GB/s | 17.72 GB/s | 17.85 GB/s |
  | bus bandwidth, 16 MiB to 256 MiB | 14.88-15.11 GB/s | 14.89-14.93 GB/s | 14.86-15.04 GB/s |

  72 of 72 sizes correct; the receipt reports 6,618 all-reduce calls (1,560 captured), `"forwarded":0`,
  `"healthy":true`.

Run at 10:44 UTC, sections 3.1 to 3.3, source archive SHA-256 `e9f947f5...49b4d171` (the SplitMix64
inputs and digests; library sources as in the 10:06 run, library SHA-256 `c8c659c9...` on both ranks;
`verification/hardware/pair-20261008T1044Z/`):

- `make check` in the image: every suite passed; the probe and the nccl-tests preflight ran clean; the
  preflight finds the NCCL 2.32.3 header at `/opt/sparkring/toolchain/nccl` and the rewritten preload
  list with libsircl first.
- `library_rank.py` with SIRCL-session digests: 228 of 228 checks passed on each rank; 87 per rank equal
  the SIRCL session's bytes, 72 the fold model, none report different inputs.
- `perf_rank.py` with `SIRCL_LARGE_PIECE_BYTES=16777216` (16 MiB pieces; the receipt confirms the
  setting): bus bandwidth 15.30-15.34 GB/s at 16 MiB and 15.06-15.14 GB/s at 64 MiB and 256 MiB, against
  14.86-15.04 GB/s with 4 MiB pieces; 4 MiB itself still reaches 17.78-17.89 GB/s; 8 KiB fp32 9.73 us
  eager and 9.60 us in graphs; no size wrong.

Conclusion: on the cabled pair, through its NCCL API and the real RDMA fabric, every collective and the
point-to-point calls produce the bits of the rank-order reference, of the fold model and, for the
float16, bfloat16 and float32 sums, the all-gathers and the reduce-scatters, of the SIRCL Python
session, eager and in CUDA graph replay, with nothing forwarded; a split communicator with other rank
numbers finds its routes by position. The eager 8 KiB all-reduce takes 9.7-9.9 us per call, within the
milestone's 20 us; the ring operator's earlier measurement of NVIDIA NCCL 2.32.3 on the same pair, 18.8
us eager, is not in this tree's evidence. Messages above 4 MiB stay near 15 GB/s of bus bandwidth whether
they travel in 4 MiB or 16 MiB pieces, so the piece size does not set that limit; the 4 MiB single op
reaches 17.8 GB/s.

nccl-tests on the pair, snapshot fb63329a (the pair default at that snapshot: the ring all-reduce from 6 MiB
at 4 blocks per link role, ring all-gathers from 8 MiB of output, pieces below; the tree, `MANIFEST` and
build logs in `../sircl-ccl-snapshots/fb63329a/`), built in the serving image on Sparks 0-1 and run by
the ring operator with `tools/nccl_tests_pair.sh` (each line its own MPI-shim job under a time limit, the
binaries on the Cortex-X925 cores but one and the progress thread on the remaining one):

- Bit-exact check of section 3.2: exit 0 on both ranks.
- Section 3.4, `all_reduce_perf -b 8 -e 256M -f 2`, bf16, fp16 and fp32, eager and `-G 20`: `#wrong 0` on
  every line. Float, eager, out of place, us per call: 8 KiB 11.4, 1 MiB 86.6, 2 MiB 152.8, 4 MiB 285.7
  (two-shot), 8 MiB 425.6, 16 MiB 772.9, 64 MiB 2,842 (23.6 GB/s bus bandwidth), 256 MiB 11,101 (24.18
  GB/s); with `SIRCL_LINK_BLOCKS=1`: 8 MiB 399.1, 16 MiB 742.5, 64 MiB 2,812.7, 256 MiB 11,060 (24.27
  GB/s). NVIDIA NCCL 2.32.3 in the same window: 8 MiB 427-441, 16 MiB 811-828, 64 MiB 3,119-3,134, 256 MiB
  12,129-12,192; libsircl at one block per role is 7-10% faster from 8 MiB.
- Section 3.5, every further test (all-gather, reduce-scatter, broadcast, reduce, all-to-all, sendrecv,
  gather, scatter, and the integer, double, average and min all-reduces): `#wrong 0`, no errors, the MPI
  shim's watchdog never ended a process. bf16 times at 8 MiB, libsircl against NVIDIA NCCL, us:
  all-gather 279 / 315, reduce-scatter 325 / 350, broadcast 596 / 346, reduce 531 / 371, all-to-all 403 /
  264, sendrecv 680 / 406, gather 298 / 198, scatter 324 / 199 (at fb63329a these moved store-and-forward
  through scratch; the pair plan's link paths replace them in this tree).
- nccl-tests moves its buffers to a new window of a large allocation at every call (`common.cu`), so
  each call reads cold input; the same library's own sweep (`perf_rank.py`, buffers reused) took 126-129
  us at 2 MiB and 235 us at 4 MiB where nccl-tests' two-shot took 152.8 and 285.7; with `-b` equal to `-e`
  (no movement) nccl-tests took 131.8 and 239.4.

Conclusion: libsircl runs nccl-tests unmodified through `LD_PRELOAD` on the pair, every test correct; at
one block per link role it is faster than NVIDIA NCCL 2.32.3 from 8 MiB of all-reduce. The pair plan of
this tree replaces the store-and-forward paths of the slower collectives with link paths, which ran on
the pair at positions 6-7 at snapshot 0f94c72b (next section).

### Hardware: the cabled pair at ring positions 6-7, snapshot 0f94c72b

Conditions: snapshot 0f94c72b (`../sircl-ccl-snapshots/0f94c72b/`; this tree adds the fail-stop
observation path and SIRCL 0.3.1's LF native copies with their setup checks); two DGX Sparks (GB10, aarch64) at positions 6 and 7 of
the eight-Spark ring, cabled directly; the serving image `aba309e4610c...`; containers as for positions
0-1; the library built from the snapshot in each container (`build/libsircl.so` SHA-256 `4f83a78c584cef26...`
on both, make check passed); `LIBSIRCL_TRANSPORT=verbs`, 2 lanes, the route maps of positions 0-1
(`1=rocep1s0f0/roceP2p1s0f0`, `0=rocep1s0f1/roceP2p1s0f1`); the pair default (no schedule settings);
nccl-tests v2.21.1 through `tools/nccl_tests_pair.sh`. Run by the ring operator on 2026-10-09 from 07:00 to
07:11 UTC under the ring's measurement lock for the two positions; output directory
`gate-0f94c72b-pair67-20261009T065331Z`, copied to `verification/hardware/pair67-0f94c72b-20261009T065331Z/`
without the transferred archives and with host addresses and host names replaced by position labels. The
gate's own evaluator (the snapshot's `tools/check_nccl_tests.py`) exempted the in-place `N/A` of
`alltoall_perf` only and so failed every `sendrecv_perf` row; the verdicts below are this tree's evaluator
on the same output (`recheck-a3477af2/`). Measurement and result:

- Bit-exact check (`RUNBOOK.md` 3.2, `library_rank.py` against SIRCL's digests): 266 checks over 75 ops
  and 8 communicators per rank, exit 0 on both ranks, with `LIBSIRCL_FAIL_STOP=0`, with `=1` and with
  `LIBSIRCL_P2P_CHANNELS=on`; under the channel setting every receipt says `"channels":{"on":false`
  (the channels need three or more ranks). The default placement started the progress thread on CPUs
  5-9 and 15-19 (`progress_cpus`).
- Section 3.4, `all_reduce_perf -b 8 -e 256M -f 2`, bf16, fp16 and fp32, eager and `-G 20`, with fail-stop
  off and on: `#wrong 0` on all 156 rows of each run. 8 KiB, out of place, us per call: eager 11.17 to
  17.50 with fail-stop off (fp32 17.50) and 10.25 to 11.33 with it on; graph 9.51 to 9.73. 256 MiB: 11,065
  to 11,148 us, 24.08 to 24.26 GB/s bus bandwidth. The graph criterion (within 1 us of the SIRCL ring
  harness's graph p50 on the same pair) was not evaluated: no harness measurement of positions 6-7 was given.
- Section 3.5 (all-gather, reduce-scatter, broadcast, reduce, all-to-all, sendrecv, gather and scatter in
  bf16 to 64 MiB; all-reduces of int32 max, int64 sum, double prod, float avg and uint8 min to 16 MiB):
  `#wrong 0` on all 151 rows out of place and in place, except the 24 in-place rows of all-to-all and
  sendrecv, which nccl-tests reports as `N/A`. bf16 at 8 MiB, us: all-gather 249.8, reduce-scatter 227.1,
  broadcast 342.5, reduce 353.2, all-to-all 247.8, sendrecv 419.9, gather 183.5, scatter 178.1 (at 32 MiB
  20.0 to 24.5 GB/s bus bandwidth); the other all-reduces 609.8 to 633.5 us at 8 MiB. Snapshot fb63329a's
  times on positions 0-1 (previous section) came from another pair.
- `sendrecv_perf` 8 B to 256 MiB, bf16, with `LIBSIRCL_P2P_CHANNELS=on`: `#wrong 0` on all 26 rows out of
  place, in place `N/A`; 8 KiB 15.97 us, 8 MiB 416.3 us, 256 MiB 11,187 us (24.00 GB/s); every receipt
  says channels off.
- Every nccl-tests receipt (52): nothing forwarded, no refusals, healthy, all-reduce ops covering the
  calls; every log "Out of bounds values : 0 OK". The runner pinned the progress thread to CPU 19.
- Teardown (`teardown_race.py`: 20 rounds of a reversed split child, one 8 MiB bf16 all-reduce, an
  immediate destroy), 3 runs with fail-stop off and 3 with it on: 20 of 20 rounds on both ranks in every
  run, the parent exact and healthy.
- Late rank (rank 1 starts its child all-reduce 8 s late, `SIRCL_STARTUP_WAIT_S=2`): with
  `LIBSIRCL_FAIL_STOP=1` rank 0 exited with status 70 3.9 s after it started (setup included), its
  fail-stop line naming the child communicator's timed-out flag wait for the late rank; without fail-stop
  rank 0 exited 1 after 10.4 s, reporting the round failed, and no fail-stop line.

Conclusion: on a second cabled pair, snapshot 0f94c72b's library reproduces the reference bits, runs every
nccl-tests line of sections 3.4 and 3.5 and sendrecv to 256 MiB with `#wrong 0`, survives 120 teardown
rounds, and under fail-stop ends a rank whose peer is late within setup plus the wait limit; the
point-to-point channel setting changes nothing on two ranks. The pair plan's link paths carry the further
collectives at 178 to 420 us for 8 MiB of bf16.

### Hardware: the path of four at positions 4-7 and the cycle of eight, snapshot a3477af2

Conditions: snapshot a3477af2 (`../sircl-ccl-snapshots/a3477af2/`; library sources equal to 0f94c72b's;
this tree adds the fail-stop observation path, SIRCL 0.3.1's LF native copies and their setup checks), built
in the serving image `aba309e4610c...` on every Spark (`build/libsircl.so` `4f83a78c584cef26...`, make check
passed); the route planner's settings per rank (`tools/site_routes.py`, two lanes): `path:4-7` on Sparks 4-7
and `ring:8` on Sparks 0-7, every rank at its own position; nccl-tests v2.21.1 through
`tools/nccl_tests_pair.sh` placement, bf16, 8 B to 256 MiB doubling, `-n 20 -w 5`; NVIDIA NCCL 2.32.3 from
the image as the baseline on the cycle. Run by the ring operator with the gate script `gate_multi_0f94c72b.sh`
(`SNAP=a3477af2`) on 2026-10-09, 08:46-09:03 UTC (path) and 09:03-10:07 UTC (cycle, `CAP32=1
NCCL_BASELINE=1`); outputs in `verification/hardware/path4-a3477af2-20261009T084624Z/` and
`ring8-a3477af2-20261009T090306Z/` (host identities replaced by position labels; `recheck/`: this tree's
nccl-tests evaluator on the same output, whose verdicts are the ones below). Measurement and result:

- Bit-exact check (`library_rank.py`, every rank), channels off with SIRCL's budget and channels on with
  every relayed lane windowed: exit 0 on 4 of 4 and 8 of 8 ranks.
- nccl-tests, `#wrong 0` on every row, every job exit 0 on every rank, every receipt forwarded 0, no
  refusals, healthy: all-reduce, all-gather, reduce-scatter and broadcast (104 rows per arm) with channels
  off and on, and on the cycle with the forward windows capped at 32 KiB and under the ring schedules
  (all-reduce, all-gather, reduce-scatter, 78 rows); alltoall, sendrecv and hypercube with channels on
  (78 rows, in-place `N/A` of alltoall and sendrecv not covered), on the cycle also capped. Without channels
  the point-to-point lines are refused on every rank (`ncclInvalidUsage`). Under SIRCL's budget hypercube
  passes with channels on the path (every relayed pair has a window) and is refused on the cycle (40
  relayed ordered pairs without one). The gate's own evaluator failed the hypercube rows: their reduction
  op column is blank and it parsed none of them; this tree's evaluator parses them.
- Teardown on the path (`teardown_race.py`, 20 rounds, 4 ranks), 3 runs with fail-stop off and 3 on: 20 of
  20 rounds on every rank, the parent exact and healthy.
- Late rank on the path (rank 3's child all-reduce 8 s late, `SIRCL_STARTUP_WAIT_S=2`): with fail-stop,
  ranks 0 and 1 exited with status 70 naming the timed-out wait; rank 2's wait timed out the same way
  ("a flag wait for rank 0 lane 0 timed out at sequence 1", child rank numbers) but it exited 1: it read
  the error with `ncclCommGetAsyncError` and destroyed the communicator within one 5 ms watcher poll, and
  destroy took it off the watch. This tree ends the process at that call (the fail-stop row); the late-rank
  step has not run on hardware with it. Without fail-stop no rank was ended by it.
- NVIDIA NCCL 2.32.3 cannot run on the path (probe exit 3 or 1 on every rank). On the cycle it runs the
  collectives; its alltoall and hypercube fail (exit 3 or 1), its sendrecv passes.
- Rank 0's nccl-tests time on the cycle, us (bus bandwidth GB/s at 256 MiB), NCCL against libsircl's default
  and libsircl under the ring schedules:

  | test | 4 KiB | 32 KiB | 128 KiB | 1 MiB | 4 MiB | 8 MiB | 256 MiB |
  |---|---|---|---|---|---|---|---|
  | all-reduce, NCCL | 120.3 | 106.3 | 185.1 | 291.3 | 751.6 | 782.6 | 19,940 (23.56) |
  | all-reduce, default | 22.5 | 34.6 | 78.8 | 162.5 | 579.8 | 1,165.2 | 36,785 (12.77) |
  | all-reduce, ring | 30.9 | 37.2 | 86.1 | 169.7 | 666.2 | 786.5 | 19,305 (24.33) |
  | all-gather, NCCL | 92.0 | 117.5 | 130.4 | 199.8 | 301.5 | 428.2 | 10,127 (23.19) |
  | all-gather, default | 25.0 | 27.9 | 40.6 | 94.8 | 296.4 | 574.6 | 17,326 (13.56) |
  | all-gather, ring | 29.2 | 32.4 | 40.5 | 91.2 | 299.7 | 431.2 | 9,714 (24.18) |
  | reduce-scatter, NCCL | 98.2 | 100.8 | 134.9 | 153.7 | 260.1 | 436.9 | 10,320 (22.76) |
  | reduce-scatter, default | 28.5 | 27.8 | 39.7 | 105.8 | 324.3 | 631.1 | 19,970 (11.76) |
  | reduce-scatter, ring | 25.5 | 29.3 | 41.7 | 108.6 | 377.3 | 444.1 | 9,822 (23.91) |
  | broadcast, NCCL | 57.6 | 52.7 | 57.3 | 104.8 | | 579.9 | 11,082 (24.22) |
  | broadcast, default | 25.6 | 41.6 | 82.8 | 551.4 | | 4,371.5 | 138,623 (1.94) |

  (Sizes are nccl-tests' sizes: the all-gather's output, the reduce-scatter's input.) `perf_rank.py` on
  the cycle (bf16 all-reduce, eager, bus bandwidth GB/s): default 13.67 at 4 MiB, 13.27 at 8 MiB, 12.77 at
  256 MiB; ring schedules 12.02, 18.97 and 24.35; forward windows capped at 32 KiB 11.60, 11.45, 11.04.
- `perf_rank.py` under the ring schedules on the cycle reported 40 wrong of 120 size checks: on every rank
  the sizes 4 to 64 MiB, the ring op's sizes. Its check compared with the rank-order float32 sum rounded
  once; the ring op rounds to bf16 at every hop in ring order (`library_rank.py`'s `ring_reduce_scatter`).
  In GPU emulation on eight ranks with the same settings, the old check reports the same sizes wrong and
  every rank's output equals the ring reference bit for bit from 4 to 16 MiB
  (`verification/emulation/perf-ring8/`); `perf_rank.py` now checks against the library's result for its
  schedule.

Conclusion: on a path of four and a cycle of eight Sparks, every collective and point-to-point line is
correct and every receipt healthy. On the cycle libsircl's all-reduce, all-gather and reduce-scatter take
0.19 to 0.69 of NVIDIA NCCL's time from 4 KiB to 1 MiB; under the default schedule its bandwidth from 8 MiB stays near 12.8 GB/s while the ring schedules
reach 24.3 GB/s, NCCL's level, and overtake the default from 8 MiB (all-reduce message, all-gather output,
reduce-scatter input); broadcast on the cycle reaches 1.9 GB/s. The fail-stop gap the late-rank step found
is closed in this tree and not yet run on hardware.

### Link pack against SIRCL's DSL kernels

Chain all-reduce. Conditions: `tests/emulation/mixed_group.py` with `SIRCL_LARGE_SCHEDULE=chain`, so
every `all_reduce_large` runs as one chain op (plus the zero-padded tail); each session's chain
launchers taken from SIRCL's DSL or from the link pack, DSL-only, mixed (even ranks C++) and C++-only on
the same sessions, eager and in CUDA graph replay; chain chunks of 512 KiB in 4 slots of 1 MiB, so
larger cases reuse slots within one op. References: SIRCL's `references.large_all_reduce` for the
session's plan (per-hop rounding in chain order) and the DSL outputs. Measurement and result, with the
pack built from the chain all-reduce alone (`6c29474d...`, before the link collectives joined the file;
`verification/emulation/chain-*.log`): `path:0-1` (1 lane) 208 of 208; `path:0-3` (2 lanes) 208 of
208; `ring:8` (2 lanes) 208 of 208. Conclusion: the CUDA C++ chain kernel speaks SIRCL's chain protocol
with SIRCL's progress thread and reproduces the DSL kernel's bits at two, four and eight ranks, where
every hop rounds. The pack that adds the link collectives (`5935b067...`) changes the chain entries'
launch bound from 1,024 to 512 threads; with that pack the chain all-reduce passed again in every run below (the chain schedules and SIRCL's chain checks).

Link collectives. Conditions: the same harness with every compiled link launcher swapped
(`("link-gather",)`, `("link-scatter", dtype)`, `("link-ring", mode, dtype)`), pack `5935b067...`;
the chain schedules (`SIRCL_GATHER_SCHEDULE=chain`, `SIRCL_SCATTER_SCHEDULE=chain`), the ring schedules
(`ring`, `SIRCL_RING_MIN_BYTES=0`) and SIRCL's own link checks of its emulation harness
(`--suite sircl-links`: chain and ring collectives across pieces and slot reuse, link ops taken late by
one rank's progress thread, pieces of their own, ring minimums and every stagger the slots hold).
Reduce-scatters are compared with `references.chain_reduce_scatter` or `ring_reduce_scatter` for the
path the session takes. Measurement and result (`verification/emulation/links-*.log`; launches from C
counted over the mixed and C++-only modes):

| Layout | Chain schedules | Ring schedules | SIRCL's link checks |
|---|---|---|---|
| `path:0-1`, 1 lane | 209 of 209 | 209 of 209 | 266 of 266 |
| `path:0-3`, 2 lanes (the ring closes through two relays) | 209 of 209; 6 chain all-gathers, 90 chain reduce-scatters | 209 of 209; 6 ring all-gathers, 90 ring reduce-scatters, 90 ring all-reduces | 266 of 266; 192 chain all-gathers, 258 chain reduce-scatters, 198 ring all-gathers, 192 ring reduce-scatters, 324 ring all-reduces |
| `ring:8`, 2 lanes | 209 of 209; 12 chain all-gathers, 180 chain reduce-scatters | 209 of 209; 12 ring all-gathers, 180 ring reduce-scatters, 180 ring all-reduces | 266 of 266; 384 chain all-gathers, 516 chain reduce-scatters, 396 ring all-gathers, 384 ring reduce-scatters, 648 ring all-reduces |

In SIRCL's link checks the C++ kernels also ran with one rank's progress thread taking every link op
2 ms late, under every ring stagger the slots hold, and with pieces of their own per collective; every
session stayed healthy. Conclusion: the CUDA C++ link collectives speak SIRCL's link protocol
with SIRCL's progress thread, alone and mixed with DSL ranks, and reproduce SIRCL's per-hop rounding on a
pair, on a path of four whose ring crosses relays and on the cycle of eight.

### Transport kernel pack against SIRCL's DSL kernels

Conditions: SIRCL's emulation harness, unmodified, with each session's launchers taken from SIRCL's
DSL or from the pack (`tests/emulation/mixed_group.py`). Measurement, result and conclusion:
`KERNEL_ROUTE.md`. In short: 905 of 905 checks on `path:0-1`, `ring:3`, `path:0-3` (one-block and
every-block polling) and `ring:8`, every output bit-exact against the host reference and the DSL's
output, eager and in graph replay, mixed groups included; pack `c2e6e5a1...be4eaa25`. The test shim,
rebuilt to load both packs, repeats the `path:0-1` run with 181 of 181 checks passed
(`verification/emulation/mixed-pair-rebuilt-shim.log`).

### Library end to end: one process per rank on one GPU

Conditions: `tests/emulation/run_library.py`, one process per rank, the library loaded with ctypes and
called through its NCCL C API, `LIBSIRCL_TRANSPORT=emulation`, torch only for tensors, streams and
CUDA graphs. Two references: the SIRCL Python session's bytes for the same inputs
(`tests/emulation/sircl_golden.py`, DSL kernels, large-message schedule fixed to pieces; as files for
two ranks with one lane, as SHA-256 digests in `tests/emulation/golden/w*.json` for the other groups,
with the digest of each case's inputs) for the float16, bfloat16 and float32 sums, the all-gathers and
the reduce-scatters; and a host model of
the fold arithmetic (`tests/emulation/fold_model.py`, NumPy and torch CPU conversions) for every other
datatype and op, which SIRCL's session does not reduce.

Measurement per rank, 228 checks with two ranks and 221 with more:

- transport reductions: all-reduce of float16, bfloat16 and float32 from one element to 10 MiB (tails,
  two pieces and more, unaligned, in place, four calls alternating between two streams), reduce-scatter
  of 2 B to 4.2 MiB chunks (padded, unaligned, in place, several ops per call), reduce to rotating roots
  with the other ranks' buffers untouched;
- byte movement: all-gather of bf16, uint8 and fp32 shards of 1 B to 8.4 MiB (padded, unaligned, in
  place, tiles larger than a slot) and of 3 MiB shards with rank 0's output and rank 1's input
  unaligned; broadcast from every root with no send buffer on the others; in-place `ncclBcast`;
  `ncclAlltoAll` of 16 B to 4.2 MiB chunks (padded, unaligned, in place); `ncclGather` and
  `ncclScatter` to and from rotating roots (padded, several tiles, in place, NULL buffers on the ranks
  that do not use them);
- fold reductions: all-reduce of all 12 datatypes with all 5 built-in ops at 1,027 elements, and six
  further cases (one element to 8 MiB in three ops, unaligned, in place); reduce of int64 and float16
  to a root; reduce-scatter of int32, float64, bfloat16, float8 e5m2 and uint64 (padded, two ops,
  unaligned, in place);
- point-to-point with two ranks: sends of 1 B, 4,097 B and 4 MiB in three ops outside a group, a group
  exchanging 1,000 B and 70,000 B in opposite directions, three sends matched in issue order, torch's
  all-to-all pattern with sends to the rank itself, a send and a receive captured in a CUDA graph and
  replayed twice; with more ranks, a send to another rank refused and a send to the rank itself carried;
- `ncclCommSplit` with keys that reverse the rank order, then an all-reduce on the child summed in the
  child's rank order, and the child destroyed;
- CUDA graphs: one of three all-reduces (one-shot, two-shot, pieces with a padded tail), one of an
  all-gather and a reduce-scatter, one of a fold all-reduce, an all-to-all, a gather and a scatter, each
  captured once and replayed twice with new inputs;
- an op number beyond the built-in five refused with `ncclInvalidArgument`; the receipt (calls, ops,
  57 distinct fold pairs, point-to-point counts, captured calls, nothing forwarded);
  `ncclCommGetAsyncError`, finalize, destroy.

Result:

| Group | Lanes | Checks | Failed | Equal to the SIRCL session's bytes | Equal to the fold model | Wall time |
|---|---|---|---|---|---|---|
| 2 ranks | 1 | 456 | 0 | 174 | 144 | 9.7 s |
| 2 ranks | 2 | 456 | 0 | 174 | 144 | 10.4 s |
| 3 ranks | 1 | 663 | 0 | 261 | 215 | 17.7 s |
| 4 ranks | 2 | 884 | 0 | 348 | 286 | 27.1 s |
| 8 ranks | 2 | 1,768 | 0 | 552 | 570 | 121.3 s |

Every check not compared with a reference is compared with the rank-order host reference or with the
bytes the ranks sent. For eight ranks the SIRCL session's reduce-scatter bytes are absent: on its
emulated `ring:8` (2 lanes, `max_gather_bytes` 64 KiB), SIRCL's `reduce_scatter` completed the 16 B and
4,096 B chunk cases and did not complete the 65,552 B chunks, the first case above 64 KiB, within its
harness's 120 s stream wait (`verification/emulation/sircl-golden-w8-full.log`). The golden run for eight
ranks therefore skips reduce-scatter (`sircl_golden.py --skip-reduce-scatter`), and the library's
eight-rank reduce-scatters are checked against the rank-order reference only. The library's emulation
gives every pair of ranks a direct lane, so it does not exercise SIRCL's relayed lanes.

Every case's inputs are SplitMix64 bits generated with NumPy integer arithmetic, the same on every
platform (`library_rank.inputs_for`); the SIRCL-session outputs of the table were generated from them
(`verification/emulation/sircl-golden-w*.log`). Five eight-rank runs passed in full since the checks
order their buffer fills: the table's and four with the earlier `torch.randn` inputs
(`verification/emulation/runs.out`).
One earlier eight-rank run, made with a version of the checks that filled buffers on torch's default
stream without ordering the fills before the library's stream, had five of the eight child ranks of the
split check reach the wait limit; unordered fills change the values a rank sends, not whether they
arrive, so that run's timeout has no identified cause. That version also failed the two-rank, two-lane
4 MiB send in five of six runs, as an unordered zero-fill of the receive buffer racing the library's copy
into it would; with ordered fills, eight of eight two-rank, two-lane runs passed.

Conclusion: through its NCCL API, the library's float16, bfloat16 and float32 all-reduce,
reduce-scatter and reduce reproduce the SIRCL Python session's outputs bit for bit; every other
datatype and op reproduces the fold model bit for bit on every rank; the byte-moving collectives and
point-to-point move exact bytes; all of it eager and under CUDA graph capture, across processes. The
processes share the GPU by time slicing, so these runs measure correctness only.

### Op selection under rank-local staging, and alternating flags-only directions

Conditions: the workstation (above), library SHA-256 `995b8c1c...45e24cc4` (except where marked),
`tests/emulation/run_library.py` with the checks of `library_rank.py` described in the op-selection and
flags-only rows (Components), the SIRCL session's digests as golden references, 2026-10-08.

Measurement, checks over all ranks and failures per run:

| Group and settings | Lanes | Checks | Failed |
|---|---|---|---|
| 2 ranks, pair default | 2 | 533 | 0 |
| 2 ranks, pair default | 1 | 533 | 0 |
| 2 ranks, relayed pair given a ring plan (forward windows of 65,536 bytes, `LIBSIRCL_RING_WINDOW=393216`) | 2 | 533 | 0 |
| 2 ranks, relayed pair (forward windows of 65,536 bytes, no ring plan) | 2 | 531 | 0 |
| 2 ranks, `SIRCL_LARGE_SCHEDULE=pieces` | 2 | 531 | 0 |
| 2 ranks, ring schedules from 0 bytes | 1 | 533 | 0 |
| 2 ranks, chain schedules | 1 | 533 | 0 |
| 4 ranks, ring schedules from 0 bytes | 2 | 989 | 0 |
| 4 ranks, chain schedules (*) | 2 | 989 | 0 |
| 4 ranks, ring schedules with forward windows and a ring window of 393,216 bytes (*) | 2 | 989 | 0 |
| 4 ranks, pieces (*) | 2 | 985 | 0 |
| 8 ranks, ring schedules from 0 bytes (*) | 2 | 1,977 | 0 |
| 8 ranks, pieces (*) | 2 | 1,969 | 0 |

(*) A build whose sources differ from this tree's only in comments, and checks whose file differs only in
one docstring.

Result: every check of every run passed. The same checks against snapshot ba5a337b's library (built from
that snapshot's tree, 15 s wait limit) failed 13 of 531 on the pair default and 15 of 531 under the ring
schedules on two ranks (the op-selection row names the failures). Conclusion: the op-selection and
flags-only rows above rest on these runs.

### PyTorch ProcessGroupNCCL, unmodified, through LD_PRELOAD

Conditions: `tests/emulation/torch_pg.py`, torch 2.10.0+cu128 built against NCCL 2.27.5, one process
per rank on one GPU, `LD_PRELOAD` of the library, emulation transport, `init_process_group("nccl",
device_id=...)`. Measurement per rank: the process maps libsircl and `ncclGetVersion` in its global
scope reports 22705; all-reduce of bf16, fp16 and fp32 sums (7 elements to 5 MiB), of int64 (sum),
fp32 (`MAX`) and bf16 (`AVG`); `all_gather_object`, `broadcast_object_list`; an all-reduce on
`new_group([0, 1])`; with two ranks `send`/`recv`, `batch_isend_irecv` and `all_to_all_single`;
`barrier`, `all_gather_into_tensor`, `reduce_scatter_tensor`, `broadcast`, `reduce`; an all-reduce
captured in a CUDA graph and replayed twice; `destroy_process_group`; every tensor compared with the host
reference bit for bit. Result: every check passed with 2 ranks (46 checks) and with 4 ranks (79 checks; point-to-point
runs with two ranks only, since torch then issues it on the four-rank communicator); torch reports
`comm_split_count` 1, so `new_group` split the group's communicator through `ncclCommSplit`; every
receipt reports both packs' hashes, `"forwarded":0` and no refusals
(`verification/emulation/torch-pg-w2.log`, `torch-pg-w4.log`). Conclusion: torch's NCCL backend runs unmodified on
libsircl in emulation, including object collectives, integer and averaging reductions, subgroups and,
on two ranks, point-to-point.

### Point-to-point channels

Conditions: GPU emulation on the workstation, library `bb536638...` (this tree built with `make` in its own
directory; a build of the same sources in another directory differs only in its GNU build ID, which covers
the debug information's paths: `.text`, `.rodata`, `.data` and the dynamic symbols are byte-identical), one
process per rank on one GPU, emulation transport, `LIBSIRCL_P2P_CHANNELS=on`, SIRCL's channel defaults (8 slots of 512 KiB, 4
blocks of 512 threads, unroll 4), every run under the GPU lock, 2026-10-09. Measurement and result
(`verification/emulation/p2p/`):

- `tests/emulation/p2p_channels.py`: four ranks over two lanes (60 checks), over one lane (56), with
  `LIBSIRCL_STREAM_ORDERED_ALLOC=off` (56; the channels' own staging buffers), eight ranks (112): every
  ordered pair at once in one group per round in a shuffled issue order (0 B to 8 MiB + 16, up to 17 items
  against 8 slots, buffers offset by 3 to 9 bytes and sizes not whole packs), pairs (0, 1) and (4, 5)
  exchanging at their own pace while the other ranks idle, three sends outside groups received later in issue
  order, an all-reduce after them, a pipeline chain forwarding along every stage, nccl-tests' sendrecv ring
  at 8 MiB, and the receipt: every received byte exact, per-peer messages, bytes and items as issued, every
  sent item posted and released, 8 to 38 messages staged per rank, healthy, `ncclCommDestroy` 0. Eight ranks with the
  route planner's ring:8 settings (120): SIRCL's budget leaves the layout's relayed lanes no window, so only
  cable neighbors have channels; each rank's sends to its five other ranks are refused with
  `ncclInvalidUsage` naming the lane and `LIBSIRCL_P2P_WINDOWS` (counted 5 in the receipt) and the rest
  passes. With `--p2p-reserve none` windows (120): every pair has a channel, the relayed peers' lanes keep
  their windows (65,536 to 131,072 B), and the receipts show window waits (7 on rank 0) with at most one
  window's bytes unacknowledged.
- Failures: rank 0 sends 4,096 B to rank 1, which receives 8,192 B: every rank reports the channels' failure
  within 0.05 s (rank 1 the two sizes, the others rank 1 as the origin), and `ncclCommAbort` returns. Under
  `LIBSIRCL_FAIL_STOP=1` and a 3 s wait limit, rank 3 ends after setup and rank 0 receives from it: rank 0's
  receive times out and ranks 0, 1 and 2 end with status 70 within 1 ms of each other, their stderr lines naming
  the channels. Setup refuses `LIBSIRCL_P2P_CHANNELS` on rank 0 only, `SIRCL_P2P_THREADS=1024` and a
  malformed `LIBSIRCL_P2P_WINDOWS` on every rank, naming the rank and the setting.
- `tests/emulation/torch_pp.py --shape eager`: torch 2.10.0+cu128 with the default group initialized eagerly
  (`device_id`), four ranks: an all-reduce, `batch_isend_irecv` of a ring exchange and of one batch with every
  rank, and unbatched `isend` and `irecv` along the chain; 48 of 48 checks exact, all of it carried on each
  rank's one four-rank communicator as channel items. Against snapshot cddad8cd's library every rank fails at
  its first send with `ncclInvalidUsage` ("carried on communicators of two ranks; this one has 4").
- nccl-tests v2.21.1 through the MPI shim on four processes: `sendrecv_perf`, `hypercube_perf` and
  `alltoallv_perf`, bfloat16, 8 B to 16 MiB, each exit 0 on every rank with `Out of bounds values : 0 OK`,
  their sends and receives on the four-rank communicators' channels (receipts).
- The library-level run of four ranks with channels on (981 checks), and the rest of the emulation suite
  without them (every library run of two to eight ranks, teardown, pipeline pairs of ring:8, fail-stop,
  torch's chain and TP 4 x PP 2 shapes, setup failures, timing sweeps, nccl-tests on two processes): every
  run passed (`suite-summary.txt` and `suite-nccl-tests-summary.txt`; the first run's nccl-tests lines did not
  start, their binaries' `libcudart.so.13` off the loader path, and passed when rerun with it).
- This tree's library (`9776657f...`: SIRCL 0.3.1's native libraries with change LF, both feature words
  checked at setup, the fail-stop observation path), 2026-10-09, the whole emulation suite, every one of its
  60 runs passed (`verification/emulation/suite-s1/`), the channel runs included. Before the fail-stop path,
  the library with the LF point-to-point library and its check (`6eb5e19a...`) repeated the channel runs
  above (`verification/emulation/p2p-lf/`):
  the library run of four ranks with channels on (981 checks), `p2p_channels.py` four ranks over two lanes
  (60), over one lane (56), own staging (56), eight ranks (112), ring:8 with SIRCL's budget (120) and with
  every relayed lane windowed (120), the size mismatch (12), the vanished peer and the setup refusals (3 of
  3), torch's eager shape (48 of 48), and nccl-tests' `sendrecv_perf`, `hypercube_perf`, `alltoallv_perf`
  and `all_reduce_perf` on four processes with channels on (`Out of bounds values : 0 OK`): every run
  passed, so every channel setup passed the feature check.
- CPU (`tests/test_p2p_native.py`, 5 tests): the vendored library's emulation build across processes, host
  threads playing the kernels: four ranks over two lanes and eight over one with every ordered pair at once
  (messages of up to five items through 4 slots of 8,192 B), channels between ring neighbors with two ranks
  exchanging while the others idle, windowed lanes (window waits), a size mismatch after which every rank's
  progress thread stops naming its origin; `p2p_layout` equals SIRCL's protocol layout and
  `p2p_local_features` has bit 0.

Conclusion: on one GPU, point-to-point between any two ranks of communicators of four and eight ranks matches
issue order, moves exact bytes at independent paces and with idle ranks, keeps relayed lanes within their
windows, refuses pairs without a channel on both ranks, and turns a peer's timeout, a size mismatch or a
vanished peer into an error (or, under fail-stop, an exit) on every rank. Relays, the fabric's timing and the
hardware verbs path are not exercised.

### Setup failures

Conditions: `tests/emulation/setup_failures.py`, two processes on one GPU per case. Measurement: the
result code and `ncclGetLastError` text of `ncclCommInitRank` on both ranks. Result: 5 of 5 cases passed: the verbs transport without a route map, a route map naming a device
the host lacks, RoCE v2 GID resolution over a sysfs tree with two candidate entries,
`SIRCL_ONESHOT_MAX_BYTES` differing between the ranks, and an invalid `SIRCL_THREADS` on one rank;
both ranks returned an error within 3.7 to 4.3 s of starting (process start and CUDA initialization
included), naming every failing rank and its reason (`verification/emulation/setup-failures.log`).
Conclusion: a communicator that cannot form fails on every rank promptly, and the error names each
failing rank and its reason.

### All-reduce timing sweep in emulation

Conditions: `tests/emulation/perf_rank.py`, two ranks, emulation transport, bf16, fp16 and fp32 from 8 B
to 2 MiB (factor 8), 10 calls eager and in graphs of 5. Measurement: correctness of each size and the
tool's run to completion; the times measure GPU time slicing between the two processes, not a fabric.
Result: 21 of 21 sizes correct; the sweep ran to completion, eager and in graphs (`verification/emulation/timing-sweep-w2.log`). Conclusion: the sweep that
`RUNBOOK.md` section 3.3 runs on the pair works end to end.

### Launch cost from C

Conditions: RTX 5090, WSL2; pack kernels launched on a poisoned session so each returns at once
(`tests/emulation/launch_bench.c`); 20,000 launches in batches of 100. Result: one-shot 4.25 us mean
(3.48 us best batch), two-shot 4.19 us (3.67 us) per `cuLaunchKernel`. Conclusion: a C launch of the
pack costs about 4 us on the workstation; GB10 measured 2.75 us for a DSL kernel through a prebuilt
argument block (handoff section 4), so the eager 8 KiB target of 20 us on a pair keeps its margin. Not
measured on GB10.

### CPU suites, builds and loading

Conditions: WSL2, GCC 13.3, Python 3.12, no GPU or RDMA device. Results:

- `make check` with `SIRCL_PACKAGE` naming the SIRCL reference's `spark_transport/sircl`: ABI 5,
  API 17, bootstrap 22, lifecycle 14, engine 8, shared-memory verbs 5, vendored native sources 4,
  point-to-point native library 5, route planner 8, MPI shim 6, nccl-tests evaluator 12 and kernel entries
  3 tests passed; 10,057 fabric
  vector checks passed (`verification/library-tests.log`). Without `SIRCL_PACKAGE` the route planner's 8
  tests are skipped.
- The CUDA-on-load test passes against the library and fails against a control library that initializes
  CUDA in a constructor; `tools/probe_cuda_on_load.py` on the workstation finds that the library maps and
  initializes nothing of CUDA, with and without site packages (`verification/cuda-on-load.json`).
- CMake 4.4.4 build of `CMakeLists.txt` without a build type (unoptimized): 13 of 13 tests passed, with
  `SIRCL_PACKAGE` set (`verification/cmake-tests.log`). Libraries up to snapshot cddad8cd fail 3 of them there: rdma-core's inline
  `ibv_reg_mr` keeps its call to `ibv_reg_mr_iova2` unoptimized, which their verbs loader did not define, so
  the library did not load.
- AddressSanitizer and UndefinedBehaviorSanitizer build: lifecycle 13 and API 14 tests passed, engine 3
  of 4 and ABI 4 of 5; the other two replace the preload list, which the sanitizer runtime refuses, and
  pass in the plain build (`verification/sanitizer-cpu.log`).
- The kernel packs built from the sources by nvcc 13.3.73 (the workstation's CUDA pip packages,
  2026-10-09) have the SHA-256 values the rows above name: transport `c2e6e5a1...`, fold `66da585d...`,
  link `dc9dd167...` and point-to-point `0f39a3b9...`. The library and its four packs built in 63 s
  (`make -j8` under `nice -n 19`); without nvcc the build stops at the first pack, naming `NVCC`.
- Host loading without a CUDA context: torch's c10d and vLLM's pynccl bind to libsircl and report
  22705, or the value of `LIBSIRCL_NCCL_API_VERSION` (`verification/framework-loading*.json`).
- `tools/check_nccl_tests.py` evaluates synthetic nccl-tests outputs and receipts as intended
  (`tests/test_check_nccl_tests.py`, 12 tests), including the in-place `N/A` that nccl-tests v2.21.1 prints
  for `alltoall_perf`, `alltoallv_perf` and `sendrecv_perf`, counted as not covered, and the blank reduction
  op column of `hypercube_perf`; an in-place `N/A` of any other test, a nonzero count of these three, and a
  line that starts like a data row but does not parse, fail.
- The generated API matches the header; ruff E, F and W are clean on the library tree and on the SIRCL
  copy.

## Limits

- Groups larger than a cabled pair run on hardware only with the routing settings `tools/site_routes.py`
  prints (route maps, chain order, forward windows, ring plan; `RUNBOOK.md` section 4). Groups of four on
  a path have run on Sparks with the planner's `path:0-3` settings: teardown rounds (`teardown_race.py`, 3
  runs of 20 rounds on 4 ranks, every rank 20 of 20, no failed round, the parent exact and healthy) and 10
  runs of the relayed-group bit-exact check under the ring schedules (`RUNBOOK.md` 4.1, every rank exit
  0), at snapshot 7d1c6f19 on Sparks 0-3 (the ring operator's gate output
  `ccl-pair/done-hw-teardown-7d1-20261008T194629Z`) and at snapshot ba5a337b on Sparks 4-7
  (`ccl-pair/hw-teardown-ba5-47-20261008T225027Z`); at snapshot a3477af2 the path of four at positions
  4-7 and the cycle of eight ran nccl-tests, the bit-exact check and timing sweeps (Evidence, "the path of
  four ... and the cycle of eight"). The library emulation gives every pair of ranks a direct lane, so
  relayed lanes are exercised only in SIRCL's emulation harness (mixed groups) and in those Sparks runs.
- Broadcast on the cycle of eight Sparks: 1.9 GB/s of bus bandwidth from 1 MiB against NVIDIA NCCL
  2.32.3's 24.2 GB/s at 256 MiB (snapshot a3477af2, `ring8-a3477af2-20261009T090306Z`); libsircl has no
  ring broadcast. Status: unsupported (known gap), deferred.
- A CUDA graph that captures a call needing more staging than the communicator holds contains memory
  allocation and free nodes, and CUDA's rules for such graphs apply (for example, one executable
  instance of the graph at a time). Without stream-ordered allocation such a capture is refused on
  the rank that needs the staging (Components, unsupported).
- Point-to-point between ranks of a communicator of three or more ranks needs `LIBSIRCL_P2P_CHANNELS=on`
  at its creation (otherwise it is refused). With it, every such communicator holds an arena of 32 MiB (four
  ranks) or 64 MiB (eight) at the default geometry, a queue pair per lane of every channel and one more
  progress thread, which spins 20,000,000 idle passes before it naps. Channel items are not captured in CUDA
  graphs. `LIBSIRCL_P2P_WINDOWS` is one plan per process, as the forward windows are: the windows of one
  communicator's channels are not budgeted against other communicators' relayed traffic through the same relay
  queues. On ring:8 the collective session's forward windows of the default plan (64 to 128 KiB per relayed
  lane) leave the channels no window on any relayed pair. Windows are whole 32 KiB chunks, so any
  `tools/site_routes.py --session-share` below 1 (0.95 to 0.05 checked) takes at least one chunk from every
  relayed session lane (at 0.95: 64-128 KiB become 32-96 KiB) and gives all 40 relayed ordered pairs a
  32 KiB channel window, as `--max-window 32768` does (every session lane one chunk); the session's cost is not
  yet measured on Sparks. The channels have run in emulation only, where no lane crosses
  a relay.
- The library orders eager calls across streams, not CUDA graph replays: a graph that holds a
  communicator's collectives must not run while that communicator's other work runs.
- The fold path sends every rank's whole message to every rank (an all-gather or all-to-all, then a
  local fold): (W-1) times the message per rank, against about 2(W-1)/W for the two-shot all-reduce.
  It suits the integer, control and averaging reductions frameworks issue; it is not tuned for
  bandwidth. Two-rank point-to-point stages every exchange through scratch.
- `ncclGather`, `ncclScatter` and `ncclBroadcast` move every rank's tiles to every rank; only the root's
  (or the root-bound) bytes are kept.
- Fold semantics: integers wrap; integer `ncclAvg` truncates toward zero; float8 results saturate to the
  largest finite value; max and min keep the earlier rank's value on ties and do not order NaNs.
- `ncclCommSplit` creates the child communicators before it returns, whatever the config's blocking
  field says.
- One progress thread per communicator, spinning 20,000,000 idle passes before it naps (request PO).
- A kernel waiting for a dead peer returns only at its wait limit; `LIBSIRCL_WAIT_REGIME=serving` or
  `sirclSetWaitRegime(comm, "serving")` shortens it to `SIRCL_SERVING_WAIT_S` (request AW). Its output is
  then wrong while the enqueue that started it returned success; the next call reports the error, or, with
  `LIBSIRCL_FAIL_STOP`, the process ends within one poll of the wait limit or at that next call, whichever
  comes first.
- Collectives inside `ncclGroupStart`/`ncclGroupEnd` launch at the call, in issue order, and
  point-to-point calls at the group's end; a group that issues collectives of several communicators in
  different orders on different ranks can deadlock.
- Within one CUDA graph capture a communicator's collectives use one stream; another stream is refused.
- A rank that leaves after setup breaks later setup rounds; shrink and grow are unsupported.

## Next items

1. On the pair (operator): `RUNBOOK.md` section 3.6, libsircl's large-message times against the ring
   harness's and NVIDIA NCCL 2.32.3's on the same pair, and section 3.4's graph criterion against the ring
   harness's graph p50 on that pair.
2. Relayed groups on Sparks (operator): the late-rank fail-stop step on the path of four with this tree;
   `RUNBOOK.md` 4.1's bit-exact check under the chain and ring schedules on the cycle of eight.
3. Land requests PO and AW in SIRCL, then adopt them: one progress thread per process, abort through
   the command ring.
4. A ring broadcast for cycles (deferred). The per-call `nccl*Config` collectives.
