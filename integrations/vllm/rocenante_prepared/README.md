# Prepared adaptive RoCEnante transport

Status: **implemented**. The installer image serves every installer profile
with this transport. The bounded two/four-rank probe qualification below
applies to the unpaced manifest `7d8beed57e54`.
The profile identity `tp2-rocenante-adaptive-prepared` bridges SparkRing's
adaptive peer-path transport to B12X's prepared execution API. It is separate
from the immutable `tp2-rocenante-adaptive` source bundle.

The installer image `dev-20260927-mimovision-cuda1342-nccl2323-status032` pins
transport manifest `2eef276d5403`, whose kernels end a peer wait after a fixed
number of polls. This directory adds the [supervised peer wait](#peer-wait):
the manifest that `package.py` writes here, `31481e43b9a5`, differs from the
image's in six files and in the adaptation, B12X file paths and qualification
fields it records. The supervised peer wait is **implemented** here and in the
image layer builder
[derive_transport_peer_wait.py](../../../runtime/images/derive_transport_peer_wait.py);
no published image carries it and it is not qualified on hardware.

The native proxy paces each stripe on a hardware-forwarded (two-hop) path: it
posts signaled 32 KiB chunks and keeps at most 128 KiB of unacknowledged bytes
per queue pair. `B12X_ROCE_FORWARD_WINDOW_BYTES` and
`B12X_ROCE_FORWARD_CHUNK_BYTES` override the window and chunk; a zero window
restores single-write posting. The window assumes the ConnectX hairpin queue at
8,192 entries (`hairpin_queue_size 8192`, which `sparkring install` applies on
four-Spark rings); with the driver default of 1,024 it must be 32 KiB or less.

The [manifest](manifest.json) records source origins and every installed file.
The native proxy, peer-path policy and collective kernel math retain the
SparkRing source from `60d8d68486540ce9ddb2702dd545fda6b347c087`, derived from
Luke and Local Inference Lab's RoCEnante work. The prepared API derives from
B12X `a83336581a3a907076e60797df69ab66df5a2ff1`. Their Apache-2.0
[license](LICENSE) and source notices are retained.

## Execution contract

- The caller establishes the communicator and declares `query_from_runtime`
  and `plan`. Preparation compiles retained reduce/gather callables and allocates
  shared staging buffers. `all_reduce` and `all_gather` require the matching plan;
  runtime execution performs no kernel lookup.
- Two paths per peer remain the default. Peer HCA indices `0/2` select the two
  PCI domains of physical cage p0 from the four-function inventory. Inherited
  four-path configurations remain research-only, not enabled by this profile.
- Message-size-dependent grids and separate arrival counters for each grid are
  retained. The prepared query binds peer mapping, path count and capacities;
  the kernel's `opposite_paths` argument is not the HCA inventory count.
- Shared staging writes wait for preceding stream work. Completion events follow
  output copies. Padded gathers return independent output storage even when a
  pack-sized, unaligned input would otherwise expose shared scratch.
  Capture-stream admission and health checks retain the transport's source
  contract. A stopped peer wait is fatal to the runtime, never a fallback.

## Peer wait

A collective's kernel spins on each peer's path flags in pinned host memory;
the peer's proxy writes each flag after its stripe on the same reliable queue
pair. Two bounds end a wait whose flags do not arrive:

- **Host verdict.** Every 1,024 polls the waiting thread also reads the abort
  word of the control record. The proxy thread supervises the newest sequence
  it posted. Once that wait reaches 5 s (half of a shorter timeout) it logs the
  flags it is missing and, every second, writes a 4-byte notice into each late
  peer's control record. The write's acknowledgement shows that the peer's
  queue pairs still exist, and the peer logs its own state once per notice. The
  proxy sets the abort word, which stops every wait on its rank, only when a
  notice write is not acknowledged (the peer process or its link is gone), when
  a peer's stop notice arrives, when a flag holds a value that no in-order
  sender can produce for this rank's sequence, when the wait reaches
  `B12X_ROCE_PEER_TIMEOUT_S` (300 s by default), or when the kernel stopped on
  its device poll bound. It then writes a stop notice to every peer on every
  path, so the other ranks stop as well, and `check_health` raises with the
  cause. A wait that ends after a stall is logged with its duration; serving
  continues.
- **Device poll bound.** `B12X_ROCE_SPIN_LIMIT` polls of one flag (default
  4,294,967,295) end a wait without the host, for a host that can no longer
  write the abort word. A poll takes no fixed time, so this bound is not a
  timeout; a smaller value ends waits on live peers, and rank 0 logs a warning
  when it is set.

A stopped wait records its sequence, the missing peer and a sticky stopped
flag, poisons the runtime (later launches do nothing) and stores no output
derived from missing data, also when the sequence has wrapped to 0. Ranks must
agree on both settings. The proxy ABI is version 5; a rank refuses a peer with
another version. `stats()` adds `peer_wait_stalls`,
`peer_wait_stalls_resolved`, `longest_resolved_stall_ms`,
`peer_notices_posted`, `peer_notices_received`, `longest_proxy_loop_gap_ms` and
`peer_wait_timeout_ms`.

Each rank's worker writes these lines to standard error, prefixed with
`RoCEnante rank N:`:

| Line | Meaning |
|---|---|
| `waited 5.0 s at sequence S for rank P path 0 (flag F), ...` | Rank P had not posted S (F is S - 2); the wait continues. |
| `rank R reports waiting for this rank at sequence S; this rank's doorbell is D, posted L, completed C` (on rank P) | D < S: rank P's kernel for S had not started, so its host launched it late. D ≥ S together with a `proxy loop made no progress` line: rank P's proxy thread did not run and posted S late. D ≥ S without one: rank P had posted S, so compare rank R's log. |
| `the proxy loop made no progress for T s` | This rank's proxy thread was not scheduled, or one post waited that long for completions. |
| `waited T s at sequence S although every peer flag for it is in host memory; the kernel has not observed them` | Rank P delivered S; the waiting GPU does not see host memory. |
| `the wait at sequence S ended within T s; serving continues` | The stall resolved. |

## Package and select

```bash
python3 integrations/vllm/rocenante_prepared/package.py \
  --destination /tmp/tp2-rocenante-adaptive-prepared
```

The destination must not exist. The command verifies source hashes and copies
only the manifest-bound runtime files. Install that directory under
`/opt/sparkring/transports/tp2-rocenante-adaptive-prepared` and explicitly admit
the profile in the image's transport selector before selecting its exact
manifest hash. The [image integration procedure](INSTALLATION.md) binds the
replacement selector's preimage and B12X API dependencies. This source addition
does not install a hook or select a running model. Do not substitute this
manifest hash for the legacy bundle's identity.

## Checks

```bash
python -m pytest integrations/vllm/rocenante_prepared/test_prepared_transport.py \
  integrations/vllm/rocenante_prepared/test_probe.py \
  integrations/vllm/rocenante_prepared/test_peer_wait.py -q
```

One hundred sixteen CPU checks pass. They cover retained wire/kernel source,
prepared API and program identities, peer-path compile arguments,
staging/output ownership, selector compatibility, packaging and counter
interpretation. For the peer wait they interpret the wait loop's PTX, run both
kernels' control flow on CPU memory, and compile the proxy with GCC against a
stand-in `infiniband/verbs.h` to run its supervision through a late peer, an
unreachable peer, contradictory flags, the timeout, peer notices and the device
bound. They do not establish network connectivity, GPU numerical results,
memory ordering, graph replay, performance or compatibility of the separate TP4
weighted-mesh bundle.

Before serving, run the [two/four-rank probe](INSTALLATION.md#bounded-real-hardware-probe)
with the selected HCAs:
BF16/FP16/FP32 reduction, dim-0/last-dim gather, small/large messages, interleaved
grid sizes and CUDA graph replay with frozen kernel resolution. Verify native
proxy health and the selected manifest on every rank. Its opt-in
`--peer-delay-seconds` and `--peer-exit` cases check the supervised peer wait.
Qwen's model smoke follows those transport checks; no image/profile promotion
is implied by CPU success.

The [hardware record](../../../performance/records/transport/rocenante-prepared-a75bd02f-20260918.md)
qualifies the unpaced manifest `7d8beed57e54` on image `a75bd02ffc1d`: the two-
and four-rank runs each passed 15 cases per rank, including four frozen-kernel
graph replays with stable addresses and no replay allocation.
For the paced proxy, four GB10 ranks with `hairpin_queue_size 8192` ran 200
back-to-back operations in three trials in the serving image. A 2 MiB
all-reduce took 656 µs with 42,867 hairpin drops and 10 retransmission timeouts
without the window, and 348–356 µs with none with it. A 4 MiB-per-rank
all-gather took 1,491 µs with 125,468 drops and 219 timeouts without the window,
and 769–788 µs with none with it. Results matched NCCL in every case.
The preparation session repeats multi-rank selection races instead of trusting
unequal rank-local selection caches. The archived `e8577c447a69` record retains
its own evidence; neither record establishes model performance or compatibility
of other selections.
