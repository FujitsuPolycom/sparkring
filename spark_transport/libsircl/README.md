# libsircl

libsircl is SIRCL's NCCL-compatible C API: a Linux shared library that implements the public C
interface of NVIDIA NCCL 2.32 (the 73 functions of the NCCL 2.32.3 header and their `pnccl`
profiling twins) on SIRCL, SparkRing's Switchless Inference RDMA Collective Layer for clusters of
NVIDIA DGX Spark systems. A program written against NCCL's C API, such as PyTorch, vLLM, SGLang or
nccl-tests, loads libsircl in place of NCCL's `libnccl.so.2` without modification.

Compatibility is at the C API and ABI. libsircl implements a subset of the API (`STATUS.md` lists
each function) and does not exchange data with NVIDIA NCCL processes. libsircl is an independent
implementation: it contains no NCCL implementation source, it is not NVIDIA NCCL, and it is not
sponsored or endorsed by NVIDIA. `STATUS.md` states what is implemented and the evidence; `RUNBOOK.md`
gives the build, emulation and hardware commands.

## Names and versions

- File: `libsircl.so.<version>` (the version is in `VERSION`; the build also makes the development
  link `libsircl.so`). ELF SONAME: `libnccl.so.2`, the interface name the dynamic linker and
  `ctypes.CDLL("libnccl.so.2")` resolve. Load it with `LD_PRELOAD=<path>/libsircl.so.<version>`. A
  `libnccl.so.2` link belongs only in a private image directory, never in a directory that `ldconfig`
  scans or in a system package.
- `ncclGetVersion` reports 22705, an NCCL API level chosen to pass framework feature gates (at least
  2.22 and 2.27.3, below 2.30.4); it does not mean NCCL 2.27.5 is present.
  `LIBSIRCL_NCCL_API_VERSION` overrides it. `sirclGetInfo` reports libsircl's own version. With
  `NCCL_DEBUG` set to `VERSION`, `WARN`, `INFO` or `TRACE`, the first communicator creation of a
  process writes one stderr line, `libsircl <version> (SIRCL's NCCL-compatible C API; not NVIDIA NCCL),
  NCCL API level <level>`, so a framework log that prints an NCCL version can be traced to libsircl.
- `include/nccl.h` is NVIDIA's 2.32.3 header with marked changes (`NCCL_VERSION_CODE` 23203), used
  only to build libsircl.
- The name `libsircl` without a suffix means this library; SIRCL's own test libraries are named
  `libsircl_sim-<digest>.so` and `libsircl_p2p_sim-<digest>.so`.
- Settings named `LIBSIRCL_*` are libsircl's own. Until release 0.7.0 each is also read under its
  earlier name `SIRCL_CCL_*` (and `SIRCL_NCCLAPI_VERSION_CODE` for `LIBSIRCL_NCCL_API_VERSION`) when the
  `LIBSIRCL_*` name is unset. `SIRCL_*` settings shared with SIRCL's package keep SIRCL's names.

NVIDIA, NCCL, CUDA, DGX and DGX Spark are trademarks and/or registered trademarks of NVIDIA
Corporation in the United States and other countries. Other names may be trademarks of their
respective owners.

## What carries a call

A communicator of W ranks (1 to 8) owns one SIRCL ring session:

- **Bootstrap**: a 128-byte unique id carrying the root's address and port, a nonce and an optional site
  hash; TCP rendezvous over loopback or the wired LAN (`SIRCL_BOOTSTRAP_ADDR`, `SIRCL_BOOTSTRAP_IFNAME`
  or `NCCL_SOCKET_IFNAME`); all-gather rounds through the root for setup.
- **Session engine** (`src/engine.c`): settings agreed across ranks, the pinned arena, device counters,
  SIRCL's native progress thread (`src/transport/sircl_roce_proxy.c`, a byte-identical copy of SIRCL's
  `_roce_proxy.c`), queue-pair connection and the lane check, stream ordering, CUDA graph capture rules
  and receipts.
- **Kernel packs** (`kernels/`): the transport pack, SIRCL's one-shot and two-shot all-reduce, all-gather
  and scatter kernels ported to CUDA C++ (`sircl_kernels.cu`); the fold pack, a local rank-ordered
  reduction of gathered rows for every NCCL datatype and built-in op (`sircl_fold.cu`); and the link
  pack, SIRCL's chain all-reduce and link collectives (chain and ring all-gather and reduce-scatter, ring
  all-reduce) for groups whose ranks form a chain of cable neighbors (`sircl_links.cu`). All three are
  compiled ahead of time for `sm_120` and `sm_121`, embedded as fatbins (`kernels/prebuilt/`, each
  checked by SHA-256) and launched from C through the CUDA driver API. `KERNEL_ROUTE.md` records why the
  kernels are CUDA C++ rather than extracted DSL output.
- **Transport**: libibverbs (`LIBSIRCL_TRANSPORT=verbs`, the default), loaded at run time; or the
  shared-memory verbs stand-in (`LIBSIRCL_TRANSPORT=emulation`), with which several processes act as
  the ranks of a group on one GPU.

Collectives carried, for every NCCL datatype: `ncclAllReduce`, `ncclReduceScatter` and `ncclReduce`
(the all-reduce, kept on the root) with `ncclSum`, `ncclProd`, `ncclMax`, `ncclMin` and `ncclAvg`;
`ncclAllGather`, `ncclBroadcast`, `ncclBcast`, `ncclAlltoAll`, `ncclGather` and `ncclScatter`;
`ncclSend` and `ncclRecv` between the two ranks of a two-rank communicator (and from a rank to itself on
any communicator), inside and outside groups; `ncclCommSplit`.

- float16, bfloat16 and float32 sums run in the transport kernels: the float32 sum in rank order rounded
  once, equal to the SIRCL Python session's one-shot and two-shot bits. Under the chain and ring schedules
  (below) large all-reduces and reduce-scatters round at every hop in chain or ring order instead, as
  SIRCL's chain and ring ops do; the schedule, not the buffers' alignment, decides which ops run, so
  every rank of a call runs the same ops (op selection, below).
- Every other datatype and op is an all-gather (all-reduce, reduce) or all-to-all (reduce-scatter) of the
  ranks' bytes, then the fold pack: the rank-ordered fold, integers in their own type with wrap-around,
  16-bit and 8-bit floats in float32 rounded once, float32 and float64 in their own type.
- Every reduction gives identical bits on every rank, deterministically, independent of the message's
  split into ops.

Op selection and staging: the ops of a call (schedule, op kinds and count, pieces, blocks per role and
the link op words) follow from its shape, datatype, root and the settings agreed at setup, never from one
rank's buffers, so every rank of a call launches ops that match its peers'. A rank whose buffers are not
16-byte aligned, overlap where an op needs them apart, or whose output is discarded (`ncclReduce` and
`ncclGather` off the root under the chain and ring schedules) copies through the transport kernels'
scratch or a staging buffer around those same ops. Outside CUDA graph capture the staging buffer grows to
the largest such call and keeps every earlier buffer until destroy, since queued work and captured graphs
may still use them. Under capture, a call that needs more staging than the communicator holds takes a
graph allocation on the capturing stream (`cuMemAllocAsync`), freed on that stream when the call returns,
so the graph owns its staging; such a graph contains memory allocation and free nodes, and CUDA's rules
for those apply (for example, one executable instance of the graph at a time). Whether the driver and
device have stream-ordered allocation is agreed at setup; without it, such a capture is refused with
`ncclInvalidUsage` on the rank that needs the staging until one eager call of that size has run there.

Teardown: `ncclCommDestroy` and `ncclCommFinalize` of a ready communicator of two or more ranks wait
for every rank. This rank's enqueued work completes; a first bootstrap round proves that every rank's work
completed, so every item and flag any kernel waits for has landed (a kernel completes once it has its
inbound items, while its progress thread may still owe a forward or a last item to a peer); the progress
thread then stops, and a second round proves that every rank's progress thread stopped before any queue
pair, registration or arena is freed. Each round is bounded by the session's wait limit; a rank that does
not arrive in time makes the call return `ncclRemoteError` after the local teardown. The first round
also carries each rank's health: a rank whose own work failed (a flag wait that timed out, a failed progress
thread, an earlier asynchronous error) returns that error, and every other rank returns `ncclRemoteError`
naming it. The close is terminal: from its start the communicator refuses new work, a failed close becomes
its error, and a second `ncclCommFinalize` or the destroy that follows returns the same result. When the
transport's queue pairs or registrations cannot be released, destroy keeps the communicator's memory
allocated and returns `ncclSystemError`. Ranks destroy their communicators in the same order, as they issue
collectives. Finalize runs the rounds once and destroy then only frees; `ncclCommAbort` does not wait.

Errors of single calls: a collective or point-to-point call refused with `ncclInvalidArgument` did nothing
and leaves the calling thread's group as it was (the group's other calls still run at `ncclGroupEnd`); other
errors inside a group become the group's result. A point-to-point call queued in a group holds no reference
on its communicator, so `ncclCommAbort` from another thread returns at once, and that group's
`ncclGroupEnd` then fails its calls with `ncclInvalidUsage`. `ncclCommSplit` with a NULL config (or a config
whose `blocking` is undefined) takes the parent's blocking mode.

`ncclMemAlloc` and `ncclMemFree` allocate and free device memory of the current CUDA context; buffer and
window registration are accepted as hints (the collectives move data through the session's arena
whatever the buffer). The header's functions this library does not implement (among them
`ncclRedOpCreatePreMulSum`, `ncclCommShrink` and the per-call `nccl*Config` collectives) and the device
API's device communicator and device pointers return `ncclInvalidUsage` (`ncclRedOpCreatePreMulSum` also
sets the op it returns to `ncclNumOps`, which every collective refuses, so a caller that ignores the refusal
gets an error, never a plain sum); a reduction op that is not built in returns `ncclInvalidArgument` and is counted in
the receipt. Nothing is forwarded to another NCCL. `tests/api_manifest.json` marks every function of the
header implemented or not.

## Build and test

```sh
make -j                 # build/libsircl.so; needs a C11 compiler, make, Python 3, rdma-core headers
make check              # CPU suites; no GPU, RDMA device or CUDA toolkit
```

The library loads without CUDA or libibverbs; it resolves both at communicator creation.
`make kernels NVCC=<nvcc>` regenerates the three kernel packs and `make kernels-check` proves the
prebuilt fatbins match the sources. GPU emulation and the hardware runs are in `RUNBOOK.md`.

## Settings

| Variable | Meaning |
|---|---|
| `LIBSIRCL_TRANSPORT` | `verbs` (default) or `emulation` |
| `SIRCL_PEER_ROUTES` | verbs: `<position>=<device>[/<device>],...`, the devices toward each other process (SIRCL's route map); a communicator uses the entries of its members' positions |
| `LIBSIRCL_POSITION` | this process's position (0-63) in the route map, the same in every communicator it joins; unset, the rank in each communicator created by `ncclCommInitRank` (split communicators keep their parent's) |
| `SIRCL_GID_INDEX`, `NCCL_IB_GID_INDEX` | RoCE GID index for every device; unset, each device's single RoCE v2 IPv4 GID |
| `SIRCL_TRAFFIC_CLASS`, `NCCL_IB_TC` | traffic class of the queue pairs |
| `LIBSIRCL_MAX_SIZE` | all-reduce capacity of one op (default 2 MiB) |
| `SIRCL_LARGE_PIECE_BYTES` | op size of large messages (default the larger of 4 MiB and the capacity); the arena's slot holds it |
| `SIRCL_ONESHOT_MAX_BYTES` | largest one-shot all-reduce (default 131072); larger ops are two-shot |
| `SIRCL_ALLREDUCE_ALGORITHM` | `auto`, `oneshot` or `twoshot` |
| `SIRCL_THREADS`, `SIRCL_BLOCKS`, `SIRCL_LARGE_BLOCKS`, `SIRCL_PACKS_PER_THREAD`, `SIRCL_FLAG_POLLERS`, `SIRCL_SPIN_LIMIT` | kernel geometry and polling, as in SIRCL; the link pack's kernels run at most 512 threads per block |
| `SIRCL_LARGE_SCHEDULE` | large float16, bfloat16 and float32 all-reduces: `pieces` (ops of `SIRCL_LARGE_PIECE_BYTES`), `chain` (one chain op for the 16-byte-aligned body), `auto` (chain from `SIRCL_CHAIN_MIN_BYTES`) or `ring` (one ring op for the largest prefix of W equal chunks, from `SIRCL_RING_MIN_BYTES`; below it as `auto`). Unset: `pieces` on groups of three or more ranks, the pair default (below) on two |
| `SIRCL_GATHER_SCHEDULE`, `SIRCL_SCATTER_SCHEDULE` | all-gathers of 16-byte-multiple shards and float16, bfloat16 and float32 reduce-scatters of 16-byte-multiple chunks: `pieces` (default: tiles and scatter ops), `chain`, `auto` or `ring`, as above |
| `SIRCL_CHAIN_MIN_BYTES`, `SIRCL_RING_MIN_BYTES` | one minimum for all three collectives (all-reduce message, all-gather output, reduce-scatter input); unset, SIRCL's: chain 8, 8 and 4 MiB, ring 4, 8 and 4 MiB |
| `LIBSIRCL_CHAIN_ORDER` | positions in chain order (cable neighbors); unset, the ranks by position |
| `SIRCL_CHAIN_SLOTS`, `SIRCL_CHAIN_SLOT_BYTES`, `SIRCL_CHAIN_CHUNK_BYTES`, `SIRCL_CHAIN_BLOCKS`, `SIRCL_CHAIN_UNROLL` | the chain all-reduce's area and kernel geometry, SIRCL's names and defaults (4 slots of 1 MiB, chunks of 512 KiB, 4 blocks per role, unroll 4) |
| `SIRCL_LINK_SLOTS`, `SIRCL_LINK_SLOT_BYTES`, `SIRCL_LINK_CHUNK_BYTES`, `SIRCL_GATHER_LINK_CHUNK_BYTES`, `SIRCL_SCATTER_LINK_CHUNK_BYTES`, `SIRCL_REDUCE_LINK_CHUNK_BYTES`, `SIRCL_LINK_BLOCKS`, `SIRCL_LINK_UNROLL`, `SIRCL_RING_STAGGER`, `SIRCL_RING_GATHER_STAGGER` | the link area and link kernels, SIRCL's names and defaults (2 W slots, 8 to 32; slots of 512 KiB growing to the largest configured piece up to 1 MiB; pieces of 512 KiB; 4 blocks per role; unroll 4; staggers of one round when the slots hold them) |
| `LIBSIRCL_REDUCE_LINK_BLOCKS`, `LIBSIRCL_GATHER_LINK_BLOCKS`, `LIBSIRCL_SCATTER_LINK_BLOCKS` (or SIRCL's `SIRCL_REDUCE_LINK_BLOCKS`, `SIRCL_GATHER_LINK_BLOCKS`, `SIRCL_SCATTER_LINK_BLOCKS`) | blocks per link role (1-64) of every chain and ring link op of that collective (pair exchanges count as all-gathers); unset, `SIRCL_LINK_BLOCKS` when set, else the pair plan's on a pair, else 4. The chain all-reduce's blocks are `SIRCL_CHAIN_BLOCKS`. Ops of one kernel type may use different blocks: a launch's last block is the one whose arrival completes its own grid |
| `LIBSIRCL_LINK_DUMP` | a diagnostic: a file prefix; one JSON line in `<prefix>.rank<r>.<pid>.c<n>.links.json` when a wait of a communicator first times out or its native progress thread fails, and at destroy: the time (CLOCK_REALTIME nanoseconds, the clock of the native event trace), the link area's words (credit, sent, ready, consumed and inbound flag words of every link), the native counters, the device link counters, the native event trace since the previous dump, whether the progress thread failed and its message, and on the emulation transport the stand-in's report (every queue pair's posted, executed and completed writes; every failed write with the resolution check that failed); after destroy, a line with the times the progress thread's stop was requested and completed and the transport was destroyed |
| `LIBSIRCL_LINK_BLOCKS_CYCLE` | a test hook: `<n>[,<n>...]` gives successive link ops these blocks per role in turn, to exercise launches of one kernel type with different grids |
| `LIBSIRCL_RING_REDUCE_PASSES` | the ring all-reduce's relay: `2` (default) stores each pack of its finished piece to link 3's slot, then copies the slot to the output; `1` stores both in one pass (the same bytes and protocol, one read of the slot fewer; research-only, see `STATUS.md`); per rank |
| `LIBSIRCL_LINK_TILE_BYTES` | bytes of every chunk one link reduce-scatter op carries (default 16 MiB); a call's chunks split into column tiles, which leave every element's arithmetic unchanged |
| `LIBSIRCL_RING_WINDOW` | the ring plan: set on every rank when the ring that closes the chain can run, to the bytes this rank's ring lanes keep unacknowledged through relays (0 for cables); the ring schedules need it. |
| `LIBSIRCL_FORWARD_WINDOWS`, `SIRCL_FORWARD_CHUNK_BYTES` | forward windows of relayed lanes: `<position>=<bytes>[/<bytes>],...` per lane, 0 for a direct lane, and the chunk a windowed stripe posts in (default 32,768) |
| `SIRCL_STARTUP_WAIT_S`, `SIRCL_SERVING_WAIT_S`, `LIBSIRCL_WAIT_REGIME` | flag-wait limits and the regime a communicator starts in (`startup`, default, or `serving`); `sirclSetWaitRegime` switches it |
| `LIBSIRCL_FAIL_STOP` | `0` (default), `1` or `abort`, per process: with `1` or `abort` a watcher thread checks every communicator of two or more ranks every 5 ms for an asynchronous error (a flag wait that timed out, a failed progress thread) and at the first one writes a line to stderr (`libsircl: LIBSIRCL_FAIL_STOP: ending the process at <Unix time> ...` with the error) and the communicator's receipt, then ends the process: `1` at once with exit status 70 (no atexit handlers, no core dump), `abort` by `abort()` (SIGABRT, after the system's core-dump handling). A caller that only checks the codes of enqueue calls then keeps a failed collective's output at most for the wait limit plus one poll; a read of the output within that poll after its stream wait returns is not prevented. Destroy and abort take a communicator off the watch before releasing it |
| `SIRCL_POST_ORDER`, `SIRCL_PROGRESS_CPU`, `SIRCL_FORWARD_PROOF` | read by the native progress thread, as in SIRCL; `SIRCL_PROGRESS_CPU` pins it to a CPU list |
| `LIBSIRCL_CPU_POLICY` | where the progress thread runs without `SIRCL_PROGRESS_CPU`: `performance` (default: on the fastest CPU class the creating thread may use, the Cortex-X925 cores of a GB10, found by `/proc/cpuinfo` part numbers or sysfs `cpu_capacity`) or `none` (where the scheduler puts it). The library never changes the affinity of the application's own threads; receipts name the progress thread's CPUs (`progress_cpus`) |
| `SIRCL_BOOTSTRAP_ADDR`, `SIRCL_BOOTSTRAP_IFNAME`, `NCCL_SOCKET_IFNAME` | the root's LAN address; a rank contacts a non-loopback root only when one is set |
| `SIRCL_BOOTSTRAP_TIMEOUT_MS`, `LIBSIRCL_SETUP_TIMEOUT_MS`, `LIBSIRCL_LANE_CHECK_MS` | rendezvous, setup-exchange and lane-check deadlines |
| `LIBSIRCL_RECEIPT`, `LIBSIRCL_RECEIPT_INTERVAL_S` | prefix of the receipt files, one per communicator (`<prefix>.rank<r>.<pid>.c<n>.json`, n the communicator's number in the process), written when the communicator forms, refreshed after calls at most every `LIBSIRCL_RECEIPT_INTERVAL_S` seconds (default 10; 0: only at creation and destroy) and at destroy, each replaced whole; `sirclGetReceipt` returns the same JSON in process |
| `LIBSIRCL_EMU_LANES`, `SIRCL_EMU_FABRIC`, `SIRCL_EMU_SEED`, `SIRCL_EMU_LATENCY_NS` | emulation transport |
| `LIBSIRCL_BOOTSTRAP_ONLY=1` | CPU-only test communicators (bootstrap and lifecycle; no collectives) |
| `LIBSIRCL_NCCL_API_VERSION` | the NCCL API level `ncclGetVersion` reports (default 22705) |
| `LIBSIRCL_SYSFS_INFINIBAND` | the sysfs directory of RDMA devices read for RoCE v2 GID resolution (default `/sys/class/infiniband`; a test hook) |

Pair default (the pair plan): a two-rank communicator without `SIRCL_LARGE_SCHEDULE` that is joined by
cables (no forward window toward the peer) or given a ring plan runs, with one block per link role:

- every all-reduce from 2 MiB as one ring op in pieces of 256 KiB;
- every all-gather from 1 MiB shards as one ring op, in pieces of 128 KiB below 2 MiB shards, 256 KiB from
  2 MiB and 512 KiB from 16 MiB;
- every float16, bfloat16 and float32 reduce-scatter sum from 4 MiB of input as ring ops (column tiles of
  `LIBSIRCL_LINK_TILE_BYTES`) in pieces of 256 KiB, 512 KiB from 64 MiB of input;
- every block that moves between the two ranks in one direction or both, from 1 MiB of whole packs, as one
  pair exchange: an all-to-all (the peer's block sent, the own block copied), a point-to-point exchange
  whose two directions carry equal bytes or one carries none (with a send to and a receive from the
rank itself of the same size beside it, the receive blocks adjacent in rank order, as torch's all-to-all
issues them, the own block is copied in the same kernel), a broadcast and a scatter (the root's own
  block copied in the same kernel), a gather (the other rank's shard sent to the root, the root's own
  copied), and a float16, bfloat16 or float32 ncclReduce sum (the other rank's values sent to the root,
  whose kernel adds them to its own as each piece lands). A pair exchange is the ring all-gather on the
  wire (the same items, op word and native proxy work) with the all-gather's pieces for a shard of that
  size; a direction that carries nothing posts its items as flags only (bit 24 of SIRCL's link op word)
  and the receiver discards them.

Smaller messages keep their earlier paths. A pair joined through relays
without a ring plan runs all-reduces from 8 MiB as one chain op. On two ranks the ring and the chain add
the same two values once, so every result's bits equal the pieces'. A variable set in the environment
keeps its value and takes precedence over the plan: `SIRCL_GATHER_SCHEDULE`, `SIRCL_RING_MIN_BYTES`,
`SIRCL_CHAIN_MIN_BYTES`, the link chunk variables (the piece of every op of that collective), the link
block variables (below) and `LIBSIRCL_RING_WINDOW`. RUNBOOK.md section 3.6 gives the measurement behind the
plan.

What the plan's settings reach: setting `SIRCL_LARGE_SCHEDULE` to any value, `ring` included, turns the
pair plan off, so its ring minima, pieces and blocks no longer apply and the schedule runs with the general
defaults (`SIRCL_RING_MIN_BYTES`, `SIRCL_LARGE_PIECE_BYTES`, 4 blocks per role); a measurement of the pair
plan leaves it unset. Every pair exchange, whatever its collective, takes its eligibility and its pieces from
the all-gather (the gather plan step, `LIBSIRCL_GATHER_LINK_BLOCKS` and the gather link chunk);
`LIBSIRCL_RING_REDUCE_PASSES` selects the ring all-reduce's relay, not the pair `ncclReduce` (its exchange
has a reduce kernel of its own). Link kernels run at most 512 threads per block, whatever `SIRCL_THREADS`
says. Reductions other than float16, bfloat16 and float32 sums (for example float32 max or avg) run as an
all-gather and the fold pack, so the link and pair settings do not reach them.

`tools/site_routes.py --layout <layout> --lanes <n>` prints every rank's route map, chain order, forward
windows and ring plan from SIRCL's route planner for a layout such as `path:0-3` or `ring:8`.

Extension API (`include/sircl.h`), exported by exact name: `sirclGetInfo`, `sirclSetWaitRegime`,
`sirclGetReceipt`.

## Sources, licences and boundaries

libsircl is licensed under the Apache License, Version 2.0 (`LICENSE`); `NOTICE` lists its components
and their terms, and every source or binary copy carries `LICENSE`, `NOTICE`, `vendor/NCCL-LICENSE.txt`,
`vendor/SIRCL-NOTICE` and `LICENSES/`.

- The interface follows NVIDIA NCCL's public header `src/nccl.h.in` at tag v2.32.3-1 (Apache-2.0;
  unmodified in `vendor/`, adapted with marked changes in `include/nccl.h`) and the NCCL user guide;
  no NCCL implementation source is used.
- The SIRCL reference this library follows is the clean-room implementation tree copied to
  `../sircl-current`; `SOURCE_SNAPSHOT.json` records its files' SHA-256 hashes. `requests/` holds
  changes the library needs inside SIRCL's package, prepared for its lead.
- The RDMA transport contains code from rdma-core's `<infiniband/verbs.h>` inline functions, under the
  OpenIB.org BSD option (`LICENSES/rdma-core-verbs.txt`).
- The prebuilt kernel packs (`kernels/prebuilt/*.fatbin`) contain object code that nvcc generated from
  NVIDIA CUDA Toolkit headers; that object code is under NVIDIA's CUDA Toolkit End User License
  Agreement, not Apache-2.0 (`LICENSES/CUDA-NOTICE.txt`).
- The MPI shim for nccl-tests (`tools/mpi-shim`) was written from nccl-tests v2.21.1's sources, read to
  list the MPI calls they make; nccl-tests is NVIDIA's BSD-3-Clause test program, not NCCL's
  implementation.

This working copy is private: no repository, commit or publication.
