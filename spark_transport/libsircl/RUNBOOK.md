# libsircl runbook

Commands to build libsircl, check it in GPU emulation on the workstation, run the pair milestone on a
cabled pair of NVIDIA DGX Spark systems, and run relayed groups (a path of four, the cycle of eight). Status: sections
3.1 to 3.3 ran on the pair at positions 0-1 (`STATUS.md`, hardware evidence); sections 3.4, 3.5 and 4
have not run. Only the operator who holds the ring measurement lock runs sections 3 and 4.

Safety classes, as in SIRCL's ring runbook:

| Class | Meaning |
|---|---|
| OFFLINE | reads and writes only the build machine |
| EMULATION | uses the workstation GPU under its lock (`gpu-lock.sh`) |
| MUTATES HOST | writes files or starts containers on the Sparks |

## 1. Build (OFFLINE)

Requirements: Linux (x86_64 or aarch64), a C11 compiler, make, Python 3, the rdma-core development
headers (`infiniband/verbs.h`; the serving image has them, since SIRCL builds its native layer there)
and nvcc of CUDA 13.3 or later (`NVCC=<nvcc>`, by default `nvcc` on the path), which compiles the four
kernel packs into `build/packs/` before the library embeds them. The build also checks the vendored
copies of SIRCL's native libraries (`src/transport/sircl_roce_proxy.c`, `src/transport/sircl_p2p_proxy.c`)
against their recorded SHA-256.

```sh
make -j BUILD=build              # build/libsircl.so, SONAME libnccl.so.2, and build/packs/*.fatbin
make check BUILD=build           # CPU suites: ABI, API, bootstrap, lifecycle, engine refusals,
                                 # shared-memory verbs across processes, SIRCL's point-to-point
                                 # library across processes, MPI shim, fabric vectors
make emulation-tools mpi-shim BUILD=build
```

`SIRCL_PACKAGE=<directory holding sparkring_sircl> make check` also runs the route planner's tests
(`tests/test_site_routes.py`, skipped without it).

`make kernels` builds the transport, fold, link and point-to-point packs alone. With nvcc 13.3.73 (the
CUDA pip packages that CI's `libsircl` job installs) the packs have the SHA-256 values `STATUS.md` names
(transport `c2e6e5a1...`, fold `66da585d...`, link `dc9dd167...`, point-to-point `0f39a3b9...`); another
nvcc version may give other bytes, and the setup agreement refuses ranks whose packs differ. The default build (`P2P_FEATURES=1`) compiles the channels' setup check of SIRCL's
`p2p_local_features` word (SIRCL change LF) and refuses a vendored point-to-point library without it;
`P2P_FEATURES=0` leaves the check out. Before it links the library, the build runs the kernel-entry check
(`tests/check_entries.c`): the library's pack loader resolves every entry it names in the embedded packs
through a stand-in CUDA driver that reads their cubins offline (`tests/fake_cuda.c`), so a loader that
names an entry some architecture's cubin lacks stops the build with that entry's name.

## 2. GPU emulation on the workstation (EMULATION)

Every command runs inside WSL under the workstation's GPU lock. `REF` is this repository's
`spark_transport/sircl` directory (SIRCL's package, which the emulation imports). `LOCK` is the
workstation's GPU-lock wrapper: a script that takes an owner name and a command and runs the command while
it holds the one lock that every GPU job on the workstation takes (here, the owner `sircl-ccl`); a
workstation with one GPU user can run the commands without it.
`CUDA_DEVICE_MAX_CONNECTIONS=32` keeps ranks that share one process from serializing on
the GPU's default eight hardware queues.

```sh
export PYTHONPATH=$REF CUTE_DSL_ARCH=sm_120a CUDA_DEVICE_MAX_CONNECTIONS=32
# Mixed group: SIRCL's sessions with the C++ transport pack on even ranks (and on all ranks) against the DSL.
bash $LOCK sircl-ccl python tests/emulation/mixed_group.py --kp-lib build/libsirclkp_test.so \
    --layout path:0-1 --lanes 1
# The link pack against SIRCL's DSL link kernels: chain schedules, ring schedules, SIRCL's own link checks.
SIRCL_LARGE_SCHEDULE=chain SIRCL_GATHER_SCHEDULE=chain SIRCL_SCATTER_SCHEDULE=chain bash $LOCK sircl-ccl \
    python tests/emulation/mixed_group.py --kp-lib build/libsirclkp_test.so --layout ring:8 --lanes 2
SIRCL_LARGE_SCHEDULE=ring SIRCL_GATHER_SCHEDULE=ring SIRCL_SCATTER_SCHEDULE=ring SIRCL_RING_MIN_BYTES=0 \
    bash $LOCK sircl-ccl python tests/emulation/mixed_group.py --kp-lib build/libsirclkp_test.so --layout ring:8 --lanes 2
bash $LOCK sircl-ccl python tests/emulation/mixed_group.py --kp-lib build/libsirclkp_test.so \
    --layout path:0-3 --lanes 2 --suite sircl-links
# The SIRCL Python session's outputs for the library cases (one directory per group size; for ring:8
# add --skip-reduce-scatter, since SIRCL's harness does not complete scatter ops on that layout).
bash $LOCK sircl-ccl python tests/emulation/sircl_golden.py --out /tmp/golden/w2 --layout path:0-1 --lanes 1
# The library end to end: one process per rank on one GPU, NCCL API, shared-memory verbs stand-in.
bash $LOCK sircl-ccl python tests/emulation/run_library.py --library build/libsircl.so --world 2 \
    --lanes 1 --golden /tmp/golden/w2
# A relayed pair given a ring plan: the pair plan's ring ops and pair exchanges (flags-only directions
# included) through the ring window.
bash $LOCK sircl-ccl python tests/emulation/run_library.py --library build/libsircl.so --world 2 --lanes 2 \
    --golden tests/emulation/golden/w2.json --env LIBSIRCL_FORWARD_WINDOWS=0=65536/65536,1=65536/65536 \
    --env LIBSIRCL_RING_WINDOW=393216
# The same under the chain or ring schedules, against SIRCL sessions under the same schedules
# (SIRCL_GOLDEN_{LARGE,GATHER,SCATTER}_SCHEDULE; tests/emulation/golden/w<W>-chain.json and -ring.json).
SIRCL_GOLDEN_LARGE_SCHEDULE=ring SIRCL_GOLDEN_GATHER_SCHEDULE=ring SIRCL_GOLDEN_SCATTER_SCHEDULE=ring \
    SIRCL_RING_MIN_BYTES=0 bash $LOCK sircl-ccl python tests/emulation/sircl_golden.py --out /tmp/golden/w4-ring \
    --layout path:0-3 --lanes 2 --digests tests/emulation/golden/w4-ring.json
bash $LOCK sircl-ccl python tests/emulation/run_library.py --library build/libsircl.so --world 4 --lanes 2 \
    --golden tests/emulation/golden/w4-ring.json --env SIRCL_LARGE_SCHEDULE=ring --env SIRCL_GATHER_SCHEDULE=ring \
    --env SIRCL_SCATTER_SCHEDULE=ring --env SIRCL_RING_MIN_BYTES=0 --env LIBSIRCL_RING_WINDOW=0
# Teardown right after a ring collective: a reversed split child, one ring all-reduce and an immediate
# ncclCommDestroy, 20 rounds, rank 0's writes delayed 5 ms (with LIBSIRCL_LINK_DUMP, the dumps of a failure);
# then two ranks at 8 MiB per rank. Expected: "0 rounds failed, 0 problems".
bash $LOCK sircl-ccl python tests/emulation/teardown_race.py --library build/libsircl.so --world 4 --lanes 2 \
    --rounds 20 --slow-rank 0 --slow-ns 5000000
bash $LOCK sircl-ccl python tests/emulation/teardown_race.py --library build/libsircl.so --world 2 --lanes 2 \
    --rounds 20 --count 4194304 --slow-rank 0 --slow-ns 5000000
# The close's error paths (expected: "0 rounds failed, 0 problems"): a rank 4 s late under a 2 s wait limit,
# so every flag wait times out; and every deregistration failed, so destroy keeps the memory and fails.
bash $LOCK sircl-ccl python tests/emulation/teardown_race.py --library build/libsircl.so --world 4 --rounds 2 \
    --slow-ns 0 --expect close-error --late-rank 1 --late-s 4 --env SIRCL_STARTUP_WAIT_S=2
bash $LOCK sircl-ccl python tests/emulation/teardown_race.py --library build/libsircl.so --world 4 --rounds 2 \
    --slow-ns 0 --expect close-error --env SIRCL_EMU_FAIL_DEREG=1
# Fail-stop: with LIBSIRCL_FAIL_STOP=1 (and abort) a late rank's peer ends its process within the 2 s wait limit
# plus 3 s; without it the peer keeps a wrong output; with the watcher's poll set past the case, the peer's
# ncclCommGetAsyncError, ncclCommDestroy or ncclCommAbort right after its stream wait ends it. Expected:
# "fail-stop: 0 problems". Then four ranks with rank 3 late, as the fabric gates run it: every other rank
# ends with status 70.
bash $LOCK sircl-ccl python tests/emulation/fail_stop.py --library build/libsircl.so
bash $LOCK sircl-ccl python tests/emulation/teardown_race.py --library build/libsircl.so --world 4 --rounds 1 \
    --expect fail-stop --late-rank 3 --late-s 8 --env LIBSIRCL_FAIL_STOP=1 --env SIRCL_STARTUP_WAIT_S=2
# Point-to-point channels (LIBSIRCL_P2P_CHANNELS=on) on four and eight ranks: every ordered pair at once, a
# subset of pairs while the other ranks idle, a pipeline chain, the sendrecv ring; ring:8 under the route
# planner's settings (relayed pairs refused under SIRCL's budget; with --layout ring8-alone, windowed); a
# size mismatch, a peer that is gone under fail-stop, setup refusals. Expected: "0 failed" on every line.
bash $LOCK sircl-ccl python tests/emulation/p2p_channels.py --library build/libsircl.so --world 4 --rounds 3
bash $LOCK sircl-ccl python tests/emulation/p2p_channels.py --library build/libsircl.so --world 8 --rounds 2
bash $LOCK sircl-ccl python tests/emulation/p2p_channels.py --library build/libsircl.so --layout ring8
bash $LOCK sircl-ccl python tests/emulation/p2p_channels.py --library build/libsircl.so --layout ring8-alone
for case in size gone setup; do
  bash $LOCK sircl-ccl python tests/emulation/p2p_channels.py --library build/libsircl.so --case $case
done
# PyTorch ProcessGroupNCCL through LD_PRELOAD (with --shape eager, the default group initialized eagerly,
# its point-to-point on the four-rank communicator's channels), and communicator setup failures.
bash $LOCK sircl-ccl python tests/emulation/torch_pg.py --launch --library build/libsircl.so --world 2
bash $LOCK sircl-ccl python tests/emulation/torch_pp.py --launch --library build/libsircl.so --shape eager
bash $LOCK sircl-ccl python tests/emulation/setup_failures.py --library build/libsircl.so
```

### 2.1 The library-level suite on one GPU of any host

`tools/emulation_suite.sh` runs, from the tree root, the build and CPU checks and then every
library-level emulation run of section 2 that needs no SIRCL tree: two to eight ranks against the digests
in `tests/emulation/golden/` under the pair plan, pieces, chain and ring schedules, a relayed pair with
and without a ring plan and forward windows, fail-stop, pipeline stage pairs of ring:8, point-to-point
channels on four and eight ranks, PyTorch's `ProcessGroupNCCL` and its pipeline exchange (lazily and
eagerly initialized), the setup failures and the timing sweeps; with `NCCL_TESTS_BUILD` (nccl-tests v2.21.1
built against `build/mpi-shim`, their CUDA runtime `libcudart` on `LD_LIBRARY_PATH`) also nccl-tests on two
processes, and its sendrecv, hypercube and alltoallv tests on four processes with point-to-point channels.
It needs Python 3 with torch, make, a C compiler and one CUDA GPU (sm_120 or sm_121), so one Spark in the
serving image runs it as well as the workstation. It writes only `build/` and its output directory and
prints one PASS or FAIL line per run (`<out>/SUMMARY`).

```sh
cd <tree> && bash tools/emulation_suite.sh /tmp/libsircl-emulation
NCCL_TESTS_BUILD=/opt/nccl-tests/build bash tools/emulation_suite.sh /tmp/libsircl-emulation
```

## 3. Pair milestone on Sparks at positions 0 and 1 (MUTATES HOST; operator only)

Conditions: the serving image (the image whose ID starts `aba309e4610c`, or its successor), host
networking, GPU 0, the RDMA devices of the cabled pair, containers started the way SIRCL's ring harness
starts them. `LAN0` is the wired-LAN address of the Spark at position 0 and `IFACE` the wired-LAN
interface name (the site file's `lan_interface`). The route maps are SIRCL's for `path:0-1` with two
lanes (`sparkring_sircl.routes.derive_routes`):

| Rank | Position | `SIRCL_PEER_ROUTES` |
|---|---|---|
| 0 | 0 | `1=rocep1s0f0/roceP2p1s0f0` |
| 1 | 1 | `0=rocep1s0f1/roceP2p1s0f1` |

Every rank's environment:

```sh
export LIB=/opt/sircl-ccl/build/libsircl.so
export LIBSIRCL_TRANSPORT=verbs
export SIRCL_BOOTSTRAP_IFNAME="=$IFACE"    # "=" selects the interface by exact name
export LIBSIRCL_RECEIPT=/tmp/sircl-ccl/receipt LIBSIRCL_SETUP_TIMEOUT_MS=120000
export LD_LIBRARY_PATH=/opt/sircl-ccl/build/mpi-shim/lib:$LD_LIBRARY_PATH
# optional: SIRCL_GID_INDEX=<n> to bypass per-device RoCE v2 GID resolution
```

The route map's keys are positions. Unset, a process's position is its rank in each communicator it
creates, which for the pair equals the ring position in the table above; a process that joins
communicators with other rank numberings (torch subgroups created without splitting) sets
`LIBSIRCL_POSITION` to its ring position, and split communicators keep their parent's.

The serving image's environment preloads NVIDIA NCCL (`/opt/sparkring/toolchain/nccl/lib/libnccl.so.2`)
and CUDA libraries through `LD_PRELOAD`. The rank scripts of 3.2 and 3.3 load libsircl with ctypes and
call it directly, so they need nothing more. A program that binds NCCL through the dynamic linker
(torch, vLLM, nccl-tests) needs libsircl first in `LD_PRELOAD` and no other `libnccl.so.2` in it, so
that exactly one library with that SONAME is mapped; `tools/nccl_tests_pair.sh run` does this itself:

```sh
export LD_PRELOAD="$LIB$(printf '%s' "$LD_PRELOAD" | tr ':' '\n' | grep -v '/libnccl\.so' | sed 's/^/:/' | tr -d '\n')"
# The NCCL in the process's global scope must report libsircl's 22705 (NVIDIA NCCL 2.32.3 reports 23203):
python3 -c 'import ctypes; v = ctypes.c_int(); ctypes.CDLL(None).ncclGetVersion(ctypes.byref(v)); print(v.value)'
```

### 3.1 Build inside the serving image on each Spark

```sh
cd /opt/sircl-ccl && make -j BUILD=build && make check BUILD=build && make mpi-shim BUILD=build
# Which libraries the image maps or initializes at Python start, against what the library adds:
python3 tools/probe_cuda_on_load.py --library build/libsircl.so > /tmp/sircl-ccl/cuda-on-load.json
```

Exit: every CPU suite passes. The probe's `verdicts` must say that the library mapped and initialized
nothing of CUDA; its other fields name what the image itself maps (`LD_PRELOAD`, `/etc/ld.so.preload`,
`.pth` files, modules imported at startup).

### 3.2 Bit-exact check over the fabric (both ranks, same moment)

The library emulation's rank script runs unchanged over the verbs transport; rank 0 serves the unique
id on the LAN. `tests/emulation/golden/w2.json` holds the SHA-256 of the SIRCL Python session's output
for every case that has one, and of that case's inputs, so each output is also compared with SIRCL's own
bytes. The inputs are SplitMix64 bits, the same on every platform; a check that reports `the inputs
differ from those of the SIRCL session's bytes` means the digests and the tree do not belong together.
Rank 0 (position 0):

```sh
SIRCL_PEER_ROUTES=1=rocep1s0f0/roceP2p1s0f0 python3 tests/emulation/library_rank.py --library $LIB \
    --world 2 --rank 0 --id-server $LAN0:29711 --golden tests/emulation/golden/w2.json \
    --out /tmp/sircl-ccl/check-rank0.json
```

Rank 1 (position 1):

```sh
SIRCL_PEER_ROUTES=0=rocep1s0f1/roceP2p1s0f1 python3 tests/emulation/library_rank.py --library $LIB \
    --world 2 --rank 1 --id-server $LAN0:29711 --golden tests/emulation/golden/w2.json \
    --out /tmp/sircl-ccl/check-rank1.json
```

Exit: every check of both result files passes (all-reduce, all-gather and reduce-scatter of every size,
padded, unaligned and in place, two streams, CUDA graph replay; every datatype and op through the fold
pack; all-to-all, gather and scatter; send and receive; receipts with nothing refused beyond the
deliberate op-7 case and nothing forwarded), and 87 checks per rank report `equals the SIRCL session's
bytes`.

### 3.3 All-reduce timing without nccl-tests (both ranks, same moment)

`tests/emulation/perf_rank.py` sweeps `ncclAllReduce` the way `all_reduce_perf` does (bf16, fp16 and
fp32, 8 B to 256 MiB doubling, out of place, 50 timed calls after 10 warm-up calls, and the same in
CUDA graphs of 20 calls), checks each size up to 64 MiB bit for bit against the library's result for
its schedule (4.2), and prints time per call and bus bandwidth. It needs nothing beyond the image. Rank `<r>` with its route map as in 3.2:

```sh
python3 tests/emulation/perf_rank.py --library $LIB --world 2 --rank <r> --id-server $LAN0:29713 \
    --out /tmp/sircl-ccl/perf-rank<r>.json
```

Exit: every size checked is correct; the eager 8192-byte rows give the milestone's eager latency
measurement (criterion: at or below 20 us) and the graph rows its graph measurement, pending 3.4.

### 3.4 nccl-tests, unmodified, through `LD_PRELOAD`

The NCCL header comes from the image: the serving image holds NVIDIA NCCL 2.32.3 at
`/opt/sparkring/toolchain/nccl` (`include/nccl.h`, `lib/libnccl.so.2.32.3`), and nvcc at
`/usr/local/cuda-13.4/bin/nvcc`, so the nccl-tests v2.21.1 source tree is the only input from outside
the image. `tools/nccl_tests_pair.sh` builds it against that header (or `NCCL_HEADER_HOME`, or an
`nvidia-nccl` wheel) and libsircl's MPI shim (in place of an MPI installation), runs the lines with
libsircl first in `LD_PRELOAD`, and keeps every log and receipt; `tools/check_nccl_tests.py` evaluates
the exit criteria from both ranks' outputs.

Each test line is one MPI-shim job: the script sets `SIRCL_MPI_JOB` to the line, and the shim's ranks
refuse a peer of another job, so the two ranks never pair different lines. A binary that ends outside
`MPI_Finalize` (a failed check, an abort, a crash) ends the other rank's binary within seconds through the
shim's watchdog, and every binary runs under `timeout` (`LINE_TIMEOUT_S`, default 1200 s); a failed or
stuck line costs that line on both ranks and the ranks start the next line together. The script places
CPUs as SIRCL's ring harness does (`sparkring_sircl.cpus`, policy `performance`): each binary on the
Cortex-X925 cores but one (`taskset`), the progress thread on the remaining one (`SIRCL_PROGRESS_CPU`);
`PLACEMENT=none` leaves both to the scheduler. In GPU emulation, where both ranks share one host and one GPU, `tools/emulation_suite.sh` sets `SIRCL_MPI_DISTINCT_HOSTS=1`: each rank reports a host name of its own, so nccl-tests (which uses device <local rank>) counts one rank per host and both use device 0. The printed `placement:` line and each receipt's
`progress_cpus` record what ran. `run <r> <root> <bin> <out> lines <file>` runs the rows of a file instead,
one `<binary> <arguments>` per row, under the same jobs, time limit and placement.

```sh
bash tools/nccl_tests_pair.sh preflight                    # nvcc, the NCCL header, binaries, LD_PRELOAD
bash tools/nccl_tests_pair.sh build /opt/nccl-tests        # the nccl-tests source tree, once per Spark
# both Sparks at once, rank <r> with its route map as in 3.2:
bash tools/nccl_tests_pair.sh run <r> $LAN0:29712 /opt/nccl-tests/build /tmp/sircl-ccl/nccl-tests
# after both ranks finish, with both output directories in one place:
python3 tools/check_nccl_tests.py nccl-tests-rank0 nccl-tests-rank1 --graph-reference-us <SIRCL graph p50>
```

The binaries carry `DT_NEEDED libnccl.so.2`; at run time `LD_PRELOAD` puts libsircl (SONAME
`libnccl.so.2`) in its place, and the script puts the MPI shim on `LD_LIBRARY_PATH`. The `run` step
executes, for bf16, half and float, eager and `-G 20`:

```sh
LD_PRELOAD=$LIB all_reduce_perf -b 8 -e 256M -f 2 -d <dtype> -o sum -c 1 -n 50 -w 10 -G <0|20>
```

Exit criteria (the handoff's pair milestone):

- `#wrong 0` on every line, bf16, fp16 and fp32, 8 B to 256 MiB, eager and `-G 20`;
- the bit-exact check of 3.2 passes on the same build;
- every receipt (`receipt.rank<r>.<pid>.c<n>.json`, one per communicator) has `"forwarded":0`, refusals 0 and `"healthy":true`,
  and its all-reduce op counts cover the calls;
- eager 8 KiB time at or below 20 us (the `time` column of the 8192-byte row, out of place);
- graph replay at 8 KiB within 1 us of the SIRCL ring harness's graph p50 on the same pair.

### 3.5 Further collectives on the same build (after 3.4 passes)

`bash tools/nccl_tests_pair.sh run <r> $LAN0:29712 /opt/nccl-tests/build /tmp/sircl-ccl/further further`
runs all-gather, reduce-scatter, broadcast and reduce in bf16, and all-reduce of int32 max, int64 sum,
double prod, float avg and uint8 min. The exit is `#wrong 0` on every line and receipts as in 3.4. On a
two-rank communicator nccl-tests' `alltoall_perf`, `sendrecv_perf`, `gather_perf` and `scatter_perf`
(which use `ncclSend` and `ncclRecv`) can run the same way. nccl-tests v2.21.1 prints the in-place
`#wrong` of `alltoall_perf`, `alltoallv_perf` and `sendrecv_perf` as `N/A` (these tests have no in-place
result); `tools/check_nccl_tests.py` counts those rows as not covered and requires their out-of-place
`#wrong` 0, and treats an in-place `N/A` of any other test as wrong.

### 3.6 Large messages on the pair: pieces, chain and ring schedules

The pieces schedule runs a message as two-shot ops of `SIRCL_LARGE_PIECE_BYTES`; within one op the GPU
stages the whole piece into the arena, the NIC sends it, and the peer reduces it, one after another, so the
fabric idles while the GPUs stage and reduce. The chain and ring schedules move a message as one op in
pieces that cycle through the link slots, so staging, sending and reducing overlap. Their sums round to
the dtype at every hop, in chain or ring order: on two ranks that equals the rank-order sum bit for bit (a
chain or ring of two adds two values once), on more ranks it does not. `perf_rank.py` checks every size
against the library's own result for the schedule the environment sets (`library_rank.py`'s
`allreduce_reference`), so its checks hold under each schedule and on any number of ranks placed at their
own positions.

The pair plan (README.md, "Pair default") rests on two measurements of SIRCL's ring harness, eager calls,
each rank's period per call (the nccl-tests time metric), NVIDIA NCCL 2.32.3 on the same pairs in the same
runs.

Schedules, at SIRCL's 4 blocks per link role:

- Conditions: run `20261008-102616-68064`, Sparks 0-1 and 2-3 as two cabled pairs (both pairs agree).
- Measurement, all-reduce, us per call:

  | Size | pieces (4 MiB) | chain, 512 KiB chunks | ring, 256 KiB pieces | NVIDIA NCCL |
  |---|---|---|---|---|
  | 4 MiB | 238-243 | 266-267 | 247-248 | 222-224 |
  | 8 MiB | 499-510 | 444-445 | 422-423 | 431-433 |
  | 16 MiB | 1,128-1,129 | 803-807 | 772-773 | 805-819 |
  | 64 MiB | 4,521-4,540 | 2,900-2,915 | 2,851-2,867 | 3,098-3,126 |

- Result: from 8 MiB the ring in 256 KiB pieces is fastest, 2-5% ahead of the best chain.

Blocks per link role (`SIRCL_LINK_BLOCKS`), ring:

- Conditions: runs `20261008-110124-12964` (Sparks 0-1, 2 blocks), `20261008-110144-82436` (2-3, 1
  block) and `20261008-110204-80648` (6-7, 4 blocks), the three pairs at the same time; every case passed.
- Measurement, all-reduce in 256 KiB pieces, us per call:

  | Size | 4 blocks | 2 blocks | 1 block | NVIDIA NCCL |
  |---|---|---|---|---|
  | 4 MiB | 245.8 | 221.3 | 212.0 | 219-222 |
  | 8 MiB | 419.4 | 399.4 | 387.8 | 411-413 |
  | 16 MiB | 772.9 | 748.2 | 737.4 | 803-816 |
  | 64 MiB | 2,857.5 | 2,823.9 | 2,812.5 (23.9 GB/s) | 3,105-3,127 |

  All-gather, us per call:

  | Shard | Pieces | 1 block | 2 blocks | NVIDIA NCCL |
  |---|---|---|---|---|
  | 1 MiB | 256 KiB | 83.2 | 88.1 | 82-85 |
  | 2 MiB | 256 KiB | 132.3 | 133.2 | 137-143 |
  | 4 MiB | 256 KiB | 226.4 | 223.4 | 248-252 |
  | 16 MiB | 512 KiB | 814.1 | 771.8 | 939-950 |

  The chain all-reduce does not change with `SIRCL_LINK_BLOCKS` (445 us at 8 MiB, 2,900 us at 64 MiB). Ring
  pieces of 128 KiB were slower at every size; 512 KiB pieces at one block took 393.8 us at 8 MiB and
  2,811.5 us at 64 MiB.
- Result: one block per role is fastest for the all-reduce at every size and ties NVIDIA NCCL at 4 MiB;
  for the all-gather one block leads up to 2 MiB shards and two blocks from 4 MiB shards.

Sizes from 1 to 8 MiB at one block per role:

- Conditions: SIRCL's ring harness on pairs, counterbalanced arms, buffers reused between calls, NVIDIA
  NCCL 2.32.3 in the same runs.
- Measurement, all-reduce, us per call: the two-shot op 77 at 1 MiB and 103 at 1.5 MiB (NVIDIA NCCL 101
  and 135); at 2 MiB the two-shot op and the ring in 128 KiB pieces 129 (NVIDIA NCCL 122-125); the ring in
  256 KiB pieces 171 at 3 MiB, 212 at 4 MiB, 299 at 6 MiB and 386 at 8 MiB (NVIDIA NCCL 170, 218-221,
  312-317 and 408-413). All-gather: 1 MiB shards, the ring in 128 KiB pieces 81-82 (NVIDIA NCCL a tie);
  2 MiB shards, 256 KiB pieces, 132 (138-143); 4 MiB shards 227 at one block and 223 at two (249-252).
- Result: one block per role is within 2% of two for every all-gather; the ring wins from 3 MiB for the
  all-reduce and ties at 2 MiB.
- Conclusion: one block per role throughout, and the ring all-gather from 1 MiB shards with 128 KiB pieces
  below 2 MiB shards.

The all-reduce boundary under nccl-tests' buffers, which move to a new window of a large buffer at every
call (`common.cu`: the send and receive pointers shift by the message size per iteration), so each call
reads cold input:

- Conditions: libsircl snapshot fb63329a on Sparks 0-1 under nccl-tests v2.21.1 with the placement of
  3.4, float, out of place, eager; arms `SIRCL_LARGE_SCHEDULE=pieces` (two-shot), and the ring at one
  block in 256 KiB and 128 KiB pieces; NVIDIA NCCL 2.32.3 on the same lines.
- Measurement, us per call, rotating buffers (two-shot, ring 256 KiB, ring 128 KiB, NVIDIA NCCL): 1 MiB
  89.9, 104.0, 95.2, 133.7; 1.5 MiB 118.5, 118.5, 116.7, 190.1; 2 MiB 154.9, 137.7, 141.5, 138.4; 3 MiB
  217.3, 181.8, 190.4, 186.2; 4 MiB 287.6, 223.9, 239.5, 231.3; 8 MiB 570.5, 395.9, 432.9, 436.1. Fixed
  buffers (`-b` equal to `-e`): 2 MiB 131.8, 132.7, 131.7, 157.8; 4 MiB 239.4, 218.3, 231.4, 234.6.
- Result: the two-shot op leads up to 1.5 MiB; from 2 MiB the ring in 256 KiB pieces leads by 11-31% on
  rotating buffers and ties or leads on fixed ones; with it libsircl ties or beats NVIDIA NCCL at every size.
- Conclusion: the pair plan's ring all-reduce starts at 2 MiB.

Reduce-scatter:

- Conditions: SIRCL's ring harness on Sparks 2-3, runs `20261008-114011-84856` (1 block per role),
  `20261008-114257-71240` (2) and `20261008-114544-56856` (4); bf16 `[rows, 4096]`, eager, the slowest
  rank's median per call, buffers reused.
- Measurement, us per call by input bytes per rank (pieces, ring in 256 KiB pieces, ring in 512 KiB pieces,
  one block): 2 MiB 114.7, 109.8, 132.3; 4 MiB 157.1, 160.8, 179.3; 8 MiB 266.3, 247.9, 267.5; 16 MiB
  612.4, 435.2, 437.0; 32 MiB 1,321, 802, 811; 64 MiB 2,637, 1,561, 1,497. Blocks 1, 2 and 4 lie within
  2-4% of each other at every size. NVIDIA NCCL's reduce_scatter under nccl-tests on the same pair took 350
  us at 8 MiB and 951 us at 32 MiB.
- Conclusion: the pair plan runs the ring at one block from 4 MiB of input, in 256 KiB pieces below 64 MiB
  and 512 KiB from 64 MiB, and keeps the pieces below 4 MiB. At 4 MiB the two were within 3% on reused
  buffers, and under nccl-tests' moving buffers the pieces took 167 us against NVIDIA NCCL's 146 (snapshot
  1e77143f on Sparks 2-3).

The same comparison under libsircl, each rank's line with its route map as in 3.2:

```sh
# the pair plan, then the pieces, the chain, and the ring at 4 blocks per role
for setting in "" "SIRCL_LARGE_SCHEDULE=pieces" "SIRCL_LARGE_SCHEDULE=chain" "SIRCL_LINK_BLOCKS=4"; do
  env $setting python3 tests/emulation/perf_rank.py --library $LIB --world 2 --rank <r> --id-server $LAN0:29714 \
      --dtypes float32 --min 1048576 --out /tmp/sircl-ccl/perf-large-$(echo "$setting" | tr -c 'A-Za-z0-9' _)-rank<r>.json
done
```

`perf_rank.py` loads the library through ctypes and does not place CPUs; run it under the placement of 3.4
(`taskset -c <X925 cores but one>` and `SIRCL_PROGRESS_CPU=<the remaining one>`) for times comparable with
the harness's. The same settings apply under nccl-tests (`env <setting> bash tools/nccl_tests_pair.sh run
...`); its float check tolerates per-hop rounding on larger groups. Exit: every size correct; from 4 MiB the
pair plan's time at or below the pieces', the chain's and NVIDIA NCCL's on the same pair.

Diagnostics: a setup failure names every failing rank and reason (route map, GID, connection, lane
check); `LIBSIRCL_SETUP_TIMEOUT_MS` bounds each setup exchange. A peer that stops makes the survivors
poison their sessions after the wait limit (600 s at startup; `LIBSIRCL_WAIT_REGIME=serving` selects
`SIRCL_SERVING_WAIT_S`, default 20 s), and `ncclCommGetAsyncError` reports `ncclRemoteError` naming the
peer, lane and sequence.

## 4. Relayed groups: a path of four and the cycle of eight (MUTATES HOST; operator only)

Conditions as in section 3, on the Sparks at positions 0-3 (`path:0-3`, the path's ends reach each
other through relays) or 0-7 (`ring:8`), one rank per Spark. Every rank's routing settings come from
SIRCL's route planner, one line per rank (rank r runs at position r):

```sh
PYTHONPATH=<SIRCL's package> python3 tools/site_routes.py --layout path:0-3 --lanes 2
```

The line sets `LIBSIRCL_POSITION`, `SIRCL_PEER_ROUTES`, `LIBSIRCL_CHAIN_ORDER`, the forward windows of
the relayed lanes (`LIBSIRCL_FORWARD_WINDOWS`, `SIRCL_FORWARD_CHUNK_BYTES`) and, where the ring that
closes the chain can run, `LIBSIRCL_RING_WINDOW`. For `path:0-3` the planner gives windows of 131,072
bytes on every relayed lane and a ring window of 393,216 bytes on position 3, whose ring lanes toward
position 0 cross two relays; for `ring:8` every ring edge is a cable (ring window 0). With point-to-point
channels (`LIBSIRCL_P2P_CHANNELS=on`) the line also sets `LIBSIRCL_P2P_WINDOWS`: pass `--ring-schedules`
when the site runs the ring schedules, so the channels leave the ring windows their room, and on `ring:8`
give the session less than the whole share (`--session-share` below 1, or `--max-window 32768`) for the
relayed pairs to have channels; the JSON output (`--json`) names every ordered pair left without one.

### 4.1 Bit-exact check under each schedule (every rank, same moment)

Rank `<r>` of `<W>`, with `<line r>` from `site_routes.py`, first with the default schedules, then with
the chain schedules, then with the ring schedules; each run compares with SIRCL sessions under the same
schedules (`tests/emulation/golden/w<W>.json`, `w<W>-chain.json`, `w<W>-ring.json`):

```sh
env <line r> python3 tests/emulation/library_rank.py --library $LIB --world <W> --rank <r> \
    --id-server $LAN0:29721 --golden tests/emulation/golden/w<W>.json --out /tmp/sircl-ccl/relay-pieces-rank<r>.json
env <line r> SIRCL_LARGE_SCHEDULE=chain SIRCL_GATHER_SCHEDULE=chain SIRCL_SCATTER_SCHEDULE=chain \
    python3 tests/emulation/library_rank.py --library $LIB --world <W> --rank <r> --id-server $LAN0:29722 \
    --golden tests/emulation/golden/w<W>-chain.json --out /tmp/sircl-ccl/relay-chain-rank<r>.json
env <line r> SIRCL_LARGE_SCHEDULE=ring SIRCL_GATHER_SCHEDULE=ring SIRCL_SCATTER_SCHEDULE=ring SIRCL_RING_MIN_BYTES=0 \
    python3 tests/emulation/library_rank.py --library $LIB --world <W> --rank <r> --id-server $LAN0:29723 \
    --golden tests/emulation/golden/w<W>-ring.json --out /tmp/sircl-ccl/relay-ring-rank<r>.json
```

Exit: every check passes on every rank, including `link collectives ran` under the chain and ring
schedules; the receipts show `"links"` ops of the schedule's kinds, `"forwarded":0` and
`"healthy":true`; the `equals the SIRCL session's bytes` counts match the emulation's for the group.

### 4.2 Large-message timing under each schedule

`perf_rank.py` with `--world <W>` on every rank, with the line of 4.1 and each schedule's variables in
turn (pieces, `SIRCL_LARGE_SCHEDULE=chain`, `SIRCL_LARGE_SCHEDULE=ring`), compares the all-reduce bus
bandwidth of the three schedules from 4 MiB to 256 MiB:

```sh
env <line r> [schedule variables] python3 tests/emulation/perf_rank.py --library $LIB --world <W> --rank <r> \
    --id-server $LAN0:29724 --out /tmp/sircl-ccl/perf-<schedule>-rank<r>.json
```

The library's default large-message schedule stays `pieces` until these measurements choose another.
