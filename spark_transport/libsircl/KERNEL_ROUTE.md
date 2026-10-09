# Native kernel route

Decision: **libsircl carries its kernels as ahead-of-time CUDA C++, ported from SIRCL's CuTe DSL
kernels, speaking SIRCL's wire protocol unchanged.** Cubins extracted from the DSL's output are not
used. Status: the one-shot and two-shot all-reduce, the all-gather (plain and tiled), the scatter
ops (reduce-scatter and all-to-all), the chain all-reduce and the link collectives (chain and ring
all-gather and reduce-scatter, ring all-reduce) are **implemented** and verified in GPU emulation
against SIRCL's DSL kernels (below and `STATUS.md`). The send and receive kernels of SIRCL's
point-to-point channels are **implemented** and verified in GPU emulation against SIRCL's native
point-to-point library; they have not run in one group with SIRCL's DSL point-to-point kernels.

## What the pack is

- Source: `kernels/sircl_kernels.cu` and `kernels/sircl_common.cuh`, ports of SIRCL's
  `oneshot/_oneshot_cute.py`, `_twoshot_cute.py`, `_allgather_cute.py`, `_scatter_cute.py`,
  `_cute_intrinsics.py` and `_timed_wait.py`, templated on dtype (float16, bfloat16, float32) and group
  size (2 to 8). Rank, lane count and the flag-polling mode are launch arguments. The arena layout,
  command-ring words, op words, flag lines, device counters, arrival words, timed waits and error words
  are SIRCL's, so these kernels run against SIRCL's native progress thread and interoperate with ranks
  that run SIRCL's DSL kernels.
- Build: the library's build (or `make kernels`, the packs alone) runs nvcc for `sm_120` (RTX 5090) and
  `sm_121` (GB10) SASS into `build/packs/sircl_kernels.fatbin`; the library embeds that file with its
  SHA-256 and loads it with `cuModuleLoadData` once per CUDA context at communicator creation. The
  fatbin holds GPU code only, so a pack built on the x86_64 workstation or on an aarch64 Spark runs on
  both GPUs.
- Launch: `src/kernelpack.c` resolves 77 entry points per architecture (`sircl_oneshot_<dtype>_w<W>`,
  `sircl_twoshot_<dtype>_w<W>`, `sircl_allgather_w<W>`, `sircl_scatter_<dtype>_w<W>`,
  `sircl_alltoall_w<W>`) and launches them with `cuLaunchKernel` from a parameter array of plain
  integers. The library uses only the CUDA driver API, loaded at run time, so it shares the process's
  driver and contexts whatever CUDA runtime the application carries.
- Fold pack: `kernels/sircl_fold.cu` (libsircl's own code, no SIRCL counterpart) folds the W rows an
  all-gather or all-to-all of the transport pack delivered, element by element in rank order, for the
  12 NCCL datatypes and 5 built-in ops (60 entries `sircl_fold_d<datatype>_o<op>`). It touches no arena
  and moves no data between ranks. It is built like the transport pack, into
  `build/packs/sircl_fold.fatbin` with its own SHA-256, so the transport pack and its evidence
  below stay unchanged when the fold pack changes.
- Link pack: `kernels/sircl_links.cu`, ports of SIRCL's `oneshot/_chain_cute.py` (chain all-reduce) and
  `_links_cute.py` (`LinkGather`, `LinkScatter`, `LinkRing`), into `build/packs/sircl_links.fatbin`
  with its own SHA-256. SIRCL compiles one specialization per chain position, neighbors, rank order,
  threads, lanes and link geometry; the port takes all of them as launch parameters (the link
  collectives as one 176-byte parameter struct, `sccl_link_args`), so 168 entries per architecture
  serve every rank: `sircl_chain_<dtype>_u<U>`, `sircl_link_gather_u<U>`, `sircl_link_scatter_<dtype>_u<U>`,
  `sircl_ring_gather_u<U>`, `sircl_ring_exchange_u<U>`, `sircl_ring_exchange_reduce_<dtype>_u<U>`,
  `sircl_ring_scatter_<dtype>_u<U>`,
  `sircl_ring_reduce_<dtype>_u<U>` and `sircl_ring_reduce_two_pass_<dtype>_u<U>` for unroll U of 1 to 8 (SIRCL's `SIRCL_CHAIN_UNROLL` and
  `SIRCL_LINK_UNROLL`). The ring all-reduce's relay in `sircl_ring_reduce_<dtype>_u<U>` stores each pack of
  this rank's finished piece to link 3's own slot and to the output from one load and one sum; the
  two-pass entries, which the library launches by default, store the slot, then copy it to the output
  (`LIBSIRCL_RING_REDUCE_PASSES=1` selects the one-pass entries). Both write the same bytes and speak the same protocol,
  and the ring all-gather's own role stores each pack to the slot and the output in one pass as well. The
  pair exchange entries run the ring all-gather's items and op word on a ring of two, carrying one block
  each way (or one way) with three roles: sending, receiving (or discarding), and copying the own block in
  parallel; the reduce-to-root form's receiving role adds the rank's own values to each arriving piece. Every entry picks its launch's last block, which advances the item bases and the
  sequence, as the block whose tail arrival brings the kernel type's tail word to the launch's grid, and
  that block returns the word to 0; launches of one kernel type with different grids therefore never
  mistake an earlier block for the last one. The entries are built for at
  most 512 threads per block, so ptxas may give each thread up to 128 registers and keep every load of a
  pass (up to three sources times the unroll) in flight without spills; the library launches them with
  at most 512 threads whatever `SIRCL_THREADS` says. Loads of a pass are issued together through the
  same zero-gate SIRCL's `_cute_batch.py` uses, and the three-source sum of the chain reduce-scatter's
  owner is `round((L + x) + R)` with float32 additions in that order.
- Point-to-point pack: `kernels/sircl_p2p.cu`, a port of SIRCL's `p2p/_kernels.py` (`P2PSend`, `P2PRecv`,
  `wait_eq_or_poison`, `wait_ge_or_poison`) into `build/packs/sircl_p2p.fatbin` with its own SHA-256.
  SIRCL compiles one specialization per threads, lanes, slots and slot bytes; the port takes them, and the
  block offsets SIRCL's native `p2p_layout` reports, as one 120-byte parameter struct (`sccl_p2p_args`), so
  16 entries per architecture serve every channel: `sircl_p2p_send_u<U>` and `sircl_p2p_recv_u<U>` for unroll
  U of 1 to 8 (`SIRCL_P2P_UNROLL`), built for at most 512 threads per block (44 to 96 registers, no spills).
  One launch carries one message: block b of the grid takes the message's items b, b + gridDim.x, ...; a
  send waits for its slot's sent word, copies the item into the send slot and writes the header and the
  ready tag; a receive waits for every lane flag of the item, checks its header, copies the slot out and
  writes the consumed tag. The waits, the failure record (peer, lane, kind, expected and received header,
  then the tag, then the poison word) and the 64-poll poison and 1,024-poll clock checks are SIRCL's. The
  loads of a copy pass are issued before its stores as volatile loads (system scope for the inbound slot)
  rather than through the zero gate.

Licensing of the packs: the CUDA C++ sources are SparkRing's (Apache-2.0); a built fatbin, and so the
library that embeds it, also contains object code nvcc generated from NVIDIA CUDA Toolkit headers (the
floating-point type headers the fold pack includes and nvcc's implicit device headers), which is under
NVIDIA's CUDA Toolkit End User License Agreement, not Apache-2.0 (`NOTICE`, `LICENSES/CUDA-NOTICE.txt`).
The source holds no compiled pack.

## Evidence

| Conditions | Measurement | Result | Conclusion |
|---|---|---|---|
| nvcc 13.3.73, `-O3 -std=c++17`, sm_120 + sm_121 SASS, WSL2 x86_64 | Build time, registers, spills; rebuilds from two source paths | 6.1 s for 154 cubins; 34-64 registers, 0 spill bytes; fatbin 2,714,984 bytes, SHA-256 `c2e6e5a1...be4eaa25`, identical on every rebuild | The pack is reproducible and path independent: one nvcc and one source give one pack, whatever the build directory. |
| Same compiler and flags, fold pack | Build time, registers, spills; rebuilds from two source paths | 4.0 s for 120 cubins; 46-64 registers, 0 spill bytes, 0-byte stack frames; fatbin 1,388,776 bytes, SHA-256 `66da585d...9f11eee9`, identical on every rebuild | The fold pack is reproducible like the transport pack. |
| Same compiler and flags, link pack | Build time, registers, spills; rebuilds from two source paths | 55 s for 336 cubins; 56-128 registers, 0 spill bytes, 0-byte stack frames (the pair reduce exchange at unroll 4: 96 registers); fatbin 16,734,040 bytes, SHA-256 `2fb2fb0b...f9b340b7`, identical from both paths | The link pack is reproducible like the others. With a 1,024-thread launch bound (64 registers) the unroll-4 chain reduce-scatter spilled 20-44 bytes and unrolls 6-8 more; the 512-thread bound removes every spill. |
| SIRCL's DSL (nvidia-cutlass-dsl 4.5.0.dev0, NVVM 12.9) with `CUTE_DSL_KEEP_PTX`, sm_120a | Arithmetic of the dumped one-shot PTX | `add.f32` (round to nearest, no flush to zero), conversions by SIRCL's own inline PTX | The port uses `add.rn.f32` and the identical conversion PTX, so each reduction is the same IEEE computation. |
| SIRCL's emulation harness unmodified (`EmulatedGroup`: ranks as threads of one process, RTX 5090, SIRCL's in-process verbs stand-in and native progress thread); per rank either SIRCL's DSL launchers or the pack launched from C with the session's own arena, counters and stream (`tests/emulation/mixed_group.py`) | 181 checks per run: all-reduce one-shot and two-shot, auto and forced, 16 B to 256 KiB, bf16, fp16, fp32; `all_reduce_large` with padded tails; reduce-scatter (scatter ops) with chunks of 16 B to 256 KiB; all-gather along dimension 0 and the last dimension, direct, padded and tiled; all-to-all; one CUDA graph per rank holding all of them, replayed for two seeds; all-DSL, mixed (even ranks C++, odd ranks DSL) and all-C++ back to back on the same sessions; a C++ rank alone under a 0.5 s serving limit | 905 of 905 passed: `path:0-1` (1 lane), `ring:3` (1 lane), `path:0-3` (2 lanes; one-block and every-block polling) and `ring:8` (2 lanes). Every output equals the host reference and the DSL output of the same inputs bit for bit, eager and in graph replay; the lone rank poisoned its session after 0.51 s and named peer 1, lane 0 and the sequence | The C++ kernels are wire-compatible with SIRCL's and reproduce the SIRCL Python session's outputs bit for bit. |
| Same harness, DSL launchers | Compile time of `prepare()` | 104 s (pair) to 512 s (`ring:8`, eight ranks in one process) | The DSL compiles per (dtype, world, rank, lanes, polling) at run time; the pack compiles nothing at run time. |
| DSL specializations for 2-8 ranks | Count of one-shot and two-shot cubins per architecture (rank, world, lanes and polling are compile-time constants there) | 35 (world, rank) pairs x 3 dtypes x 2 lanes x 2 polling modes x 2 algorithms = 840, before all-gather, scatter and link kernels | Cubin extraction multiplies files and ties each to one DSL version's undocumented parameter packing. |
| RTX 5090, WSL2, driver 595.79; pack kernel on a poisoned session so each launch returns at once (`tests/emulation/launch_bench.c`) | Host time per `cuLaunchKernel` from C, 20,000 launches in batches of 100 | one-shot 4.25 us mean (3.48 us best batch), two-shot 4.19 us (3.67 us) | A C launch costs what `research/nccl-compat/REPORT.md` section 6.4 measured for an empty kernel with the same parameter count on this machine (3.9 us), against 49.7 us through the DSL executor called from Python. |
| `nvidia-cutlass-dsl` licence as installed (section 1.1) | Distribution grant | Python files of the package in source form; no grant names compiled output; developer tools are for internal use unless identified as distributable | Distributing DSL-generated cubins has no stated permission; the CUDA C++ pack carries only SIRCL's Apache-2.0 terms and the b12x RoCEnante attribution (NOTICE). |

## Consequences

- A build given to others is source plus a checked fatbin; no DSL, Python or JIT runs in the process.
- Bit-exactness of every further kernel is proven the same way: the mixed-group check runs the ported
  kernel against SIRCL's DSL kernel in one group, on the same sessions, eager and captured.
- Kernel changes inside the SIRCL package (an abort word in the timed waits, new collectives) are
  ported here after they land there; both packs' hashes join every communicator's setup agreement, so
  ranks with different packs refuse to form a session.
- Every kernel of SIRCL's session that a collective of the NCCL API reaches is ported; kernels SIRCL
  adds later (an abort word in the timed waits, new collectives) are ported after they land there.
