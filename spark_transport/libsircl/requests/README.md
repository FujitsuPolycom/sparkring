# Requests for the SIRCL package

libsircl vendors SIRCL's native proxy (`src/transport/sircl_roce_proxy.c`, a byte-identical copy of
`oneshot/_roce_proxy.c`) and ports SIRCL's kernels to CUDA C++ with the same wire protocol. Changes the
library needs inside SIRCL's package go to the SIRCL lead as requests; the library adopts them after
they land, by re-vendoring the proxy and porting the kernel change.

| Request | What | Status | Files |
|---|---|---|---|
| PO | An externally driven progress loop: `roce_start_external` and `roce_progress` let one thread serve every session of a process (native ABI 10) | edit script, test and landing script ready; checked on a copy of the reference package (below) | `PO/edit_PO.py`, `PO/land_PO.py`, `PO/base.json` |
| FO | Flags-only own items: bit 24 of a link op word sends every own item of the op as its flags only, for a rank whose peer discards them (native ABI 9 unchanged) | landed in SIRCL's implementation tree after TD (`_roce_proxy.c` SHA-256 `93c65f37...cfa3b3`); libsircl vendors that file and its exchange kernel sets the bit | `FO/edit_FO.py`, `FO/land_FO.py`, `FO/base.json` |
| AW | An abort word in the command ring that every timed flag wait watches | specification (below); no edit script yet | none |

## PO: one progress thread per process

Why: each SIRCL session runs its own progress thread, which spins for 20,000,000 idle passes before it
naps (`ROCE_IDLE_SPINS`). A serving process holds several communicators (PyNccl, ProcessGroupNCCL per
group, two-rank communicators for lazy point-to-point), so it holds several spinning threads on the
GB10's ten performance cores. With PO the library runs one thread per process that calls
`roce_progress` on every session in turn.

What changes: the progress loop's body becomes `progress_pass`, which the thread of `roce_start` calls
exactly as before; `roce_start_external` starts a connected session without a thread (the same start-up
of the posted sequences), `roce_progress` runs one pass from the caller's thread, and `roce_stop` joins
a thread only when the session has one. The wire protocol, records, arena and command ring do not
change. The Python binding gains `Proxy.start_external()` and `Proxy.progress()`.

Evidence. Conditions: `edit_PO.py` applied to a copy of the reference package recorded in
`SOURCE_SNAPSHOT.json` (`_roce_proxy.c` SHA-256 `7deb5b1a...f2ae83`); WSL2 Ubuntu 24.04, GCC 13.3,
Python 3.12.3, pytest 9.1.1. Measurement: `tests/test_progress_external.py` (new),
`tests/test_native_binding.py`, `tests/test_proxy_simulator.py` and `tests/test_ring_links.py`. Result:
32 passed. Conclusion: sessions started either way post exact bytes on the verbs stand-in; the full
SIRCL CPU suite, ruff and the GPU emulation run inside `land_PO.py --build` and the lead's emulation
before landing.

Landing: copy `land_PO.py` to the lead scratchpad and `edit_PO.py` and `base.json` to its `PO/`. The
proxy and binding are also files of the bidirectional all-gather change, so PO lands after it with
`--build --rebase`. The tuning key's native hash covers `_roce_proxy.c`, so a tuning table matches only
sessions built from the proxy source it was measured with; tables are re-measured on the edited proxy.

## FO: flags-only own items

Why: libsircl carries one-way collectives on a pair (broadcast, scatter, gather, an ncclReduce to the root,
a one-way send) as one pair exchange, which is SIRCL's ring all-gather on the wire: each rank sends its own
pieces on link 3 and receives the peer's. In a one-way collective one direction carries nothing the peer
uses; its receiver discards those slots, but the sender's proxy still posts each item at full piece size
from stale slot bytes, so that direction competes with the useful one for the queue pairs, the window and
the completion budget. With bit 24 set by the sending rank's kernel, the proxy posts those items as their
flags only.

What changes: in `_roce_proxy.c`, `link_take_ops` accepts bit 24 of the op word (bits 25-31 are still
refused) and keeps it as the op's `own_flags`; `link_source` sizes every own item of such an op 0 bytes,
which `link_post_item` already posts as flags only (a staggered link's empty items). `protocol.py` gains
`RING_OWN_FLAGS` and `ring_op_word(..., own_flags=)`; the README describes the bit;
`tests/test_ring_links.py`'s high-bit refusal moves to bit 25; `tests/test_link_own_flags.py` is new.
SIRCL's own kernels do not set the bit.

ABI: `ROCE_ABI_VERSION` stays 9.
- Bit 24 is a contract between a process's kernels and its own proxy: the kernel writes the op word into
  its local control line, and only its own progress thread reads it. It never travels.
- The wire is unchanged. A flags-only item is a form every proxy already posts and receives, and the
  receiving rank's kernel waits only for that item's flags and discards its data.
- Ranks with and without FO therefore connect (the connection record's `abi_version` is still 9) and work
  together in both directions. An FO rank's empty direction sends flags only, whose data the peer's kernel
  would not read. A peer without FO sends stale payload, which the FO rank discards.
- The one case in which a proxy without FO meets bit 24 is a kernel pack that sets it, running against
  that proxy in one process. That does not happen: libsircl compiles its kernels and its vendored proxy
  together, and SIRCL's kernels never set the bit. Were it to happen, the old proxy refuses the op word
  loudly ("sets bits 24-31, which hold nothing") rather than posting anything wrong.
- A version bump would buy nothing and cost two things. Every rank without FO would refuse to connect to
  an FO rank (the record compare), and every tuning-table key (`0.2.0/abi9`) would stop matching.

Evidence. Conditions: `edit_FO.py` applied to a copy of the reference package recorded in
`SOURCE_SNAPSHOT.json` (`_roce_proxy.c` SHA-256 `7deb5b1a...f2ae83`); WSL2 Ubuntu 24.04, GCC 13.3,
Python 3.12.3, pytest 9.1.1. Measurement: `tests/test_link_own_flags.py` (new, 2 tests),
`tests/test_ring_links.py`, `tests/test_native_binding.py` and `tests/test_proxy_simulator.py`: 32 passed;
the full CPU suite on that copy: 766 passed, 6 skipped; the new tests with the unedited proxy: the
flags-only test fails, the test without the bit passes. On the
fabric, in libsircl's research snapshot bcff23bc (this proxy change and its exchange kernel setting the
bit when its input is 0), on ConnectX-7 between two Sparks (nccl-tests v2.21.1, bfloat16 broadcast, reduce,
gather, scatter and all-to-all, 512 KiB to 256 MiB, out of place): every job exact; 133.1 GB per direction
over the five rows against 232.7 GB without FO and 133.2 GB for NVIDIA NCCL 2.32.3; broadcast at 256 MiB
11.00 ms against 11.96 ms (NVIDIA 11.02 ms) and at 4 MiB 178 us against 218 us (NVIDIA 191-193 us); reduce,
gather and scatter alike (within 1% of NVIDIA at 256 MiB, 2-24% faster from 4 to 64 MiB). Conclusion: the
proxy change is safe for every existing caller and removes the empty direction's payload; `land_FO.py
--build` runs the full CPU suite again on impl's package.

Landing: copy `land_FO.py` to the lead scratchpad and `edit_FO.py` and `base.json` to its `FO/`. The
session-close change (TD) also edits `_roce_proxy.c` (`roce_destroy` returns the number of failed verbs
calls); FO lands after it with `--build --rebase` (its replacements do not touch TD's code). libsircl then
re-vendors the proxy (byte-identical again, TD included: its `roce_destroy` declaration becomes `int`, and
a nonzero count keeps the arena allocated, as libsircl's own release check does) and adopts the exchange
kernel's bit.

## AW: an abort word in the command ring

Why: `ncclCommAbort` must end operations that wait for a dead peer. A kernel reads the wait limit once,
when a wait starts (`_timed_wait.py`), so a kernel waiting on a dead peer returns only at the limit
(600 s in the startup regime). With PyTorch's `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` the watchdog's abort
then blocks behind that kernel.

Specification:

- Command-ring word 30 (`Ctrl.ABORT`; unused today: words 0-8, 9-16, 17-24, 28, 29 and 31 are taken) is
  written only by the session's host and read only by the session's own kernels; it never travels.
- Every timed wait (`spin_until_eq_timed_sys`, `spin_until_ge_timed_sys`) takes the word's address and
  loads it, system-scope relaxed, on the cadence of its clock check (every 1,024 polls). A nonzero word
  ends the wait as a timeout: the kernel writes the missing peer and lane, the sequence or tag, and
  `ErrorKind.ABORTED` (3) into words 2, 3, 6 and 8, poisons the session, and later launches do nothing.
- The host writes the word, then stops the progress thread; `check_health` names the abort. Zero is the
  only value that lets kernels wait, so a session never clears it.
- Every kernel family that waits passes `ctrl_base + 4 * 30`: one-shot, two-shot, all-gather, scatter,
  chain, link and fused-norm kernels. Point-to-point channels have their own arena and abort notice and
  are outside this word.
- Evidence to collect in GPU emulation: a rank whose peer never launches, with the startup limit of
  600 s, returns within a millisecond of the host writing the word, names `ABORTED`, and every later
  launch on that session returns at once.

libsircl's kernel pack adopts the same word and error kind when AW lands; `ncclCommAbort` and
`ncclCommRevoke` then write it before stopping the progress thread.
