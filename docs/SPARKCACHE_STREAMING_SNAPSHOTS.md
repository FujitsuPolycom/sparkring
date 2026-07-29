# SparkCache streaming snapshots

Status: v50 live pipeline characterized; v51 idle-progress fix implemented
and **v51 LIVE NO-NUDGE VALIDATION PASSED**
Target: remove the multi-second end-of-prefill snapshot without weakening
SparkCache's immutable, manifest-last, all-rank contract.

The transactional journal, bounded planner, lease/preemption model,
checksum-pinned native ring ABI, connector lifecycle, and credit-bounded
10 GbE buddy protocol are implemented. The CUDA 13/SM121 native ring,
throwaway-buffer byte comparison, live model record comparison, and v50
four-rank streaming publication all passed their earlier gates. v50 then
exposed a final lifecycle defect when inference became completely idle.
v51 addresses that defect and has passed both offline tests and the decisive
four-Spark no-nudge test.

## v50 no-nudge discovery

The decisive v50 probe used 32,769 input tokens and `max_tokens=1`, producing
a 32,768-token cache artifact. Each DCP rank:

- admitted one context;
- submitted and wrote eight 16-chunk batches (128 chunks total);
- released its source-block leases; and
- remained without a visible manifest after the request and worker callbacks
  stopped.

The system sat idle for more than three minutes. A later four-token request
caused the next worker callback, at which point all four ranks immediately
published and advertised the same context digest. Their reported
`final_tail_ms` values were 278,753-278,769 ms.

This isolates the defect: chunk durability was not the long pole. The adapter
advanced CUDA fences, writer completions, manifest publication, connector
digest advertisement, and terminal logging only when vLLM called `poll()`
from an inference callback. An idle model could therefore leave the final
ticket and manifest pending indefinitely.

The v50 timing logger also used a module logger that was not enabled by the
worker's configured logging path, so the run produced no timing records even
though timing was enabled. v51 routes the trace sink through the configured
worker logger.

Evidence:
`deliverables/evidence/sparkcache-streaming-v50-timing-20260729T073230Z`
in the private integration tree. Its `timing-probe.json` intentionally has
`passed=false`: manifest quorum and one terminal event per rank passed, but
the timing-record gate failed.

## v51 event-armed progress worker

v51 keeps the existing nonblocking runtime state machine and adds one
worker-owned progress thread per rank:

1. Binding captures the registered CUDA device ordinal and starts the thread.
2. The thread establishes that CUDA device once because CUDA device selection
   is thread-local.
3. An offer wakes the thread only when `runtime.needs_progress` is true.
4. `needs_progress` remains true while a ring/writer batch is inflight or a
   committed/aborted terminal handoff has not been consumed.
5. The worker executes the complete adapter poll path, not only the native
   ring poll. This advances leases, fences, writer completion, manifest
   commit, connector `_held`, terminal status, and timing.
6. Foreground and background paths share one reentrant state lock. Event
   clear and the final `needs_progress` recheck occur under that same lock,
   preventing a lost wake.
7. While active, the thread sleeps 5 ms between nonblocking polls. With no
   work it waits indefinitely on the event and performs zero polls.
8. Shutdown marks the adapter closed and signals the thread under the lock,
   joins it after releasing the lock, and only then destroys the runtime and
   writer.

Expected capacity, writer, and manifest failures still abort only the
optional cache transaction. Escaped CUDA/native ownership errors become
sticky fatal state: continuing after an ambiguous fence could permit vLLM to
reuse a physical KV block still being read.

Offline gates completed on 2026-07-29:

- 132 streaming tests passed;
- 76 connector tests passed;
- the complete SparkCache suite passed 407 tests with 1 skipped;
- Ruff passed on the changed runtime, factory, and tests;
- idle no-spin, foreground/background serialization, exact-once terminal
  handling, writer-failure fail-open behavior, CUDA-init sticky failure, and
  two-phase shutdown are explicitly covered.

## v51 live no-nudge result

The decisive live gate passed on four Sparks on 2026-07-29:

- exactly one 32,769-token completion with `max_tokens=1`;
- request ID `cmpl-a691915a48161692-0-a89689cb`;
- no follow-up inference request or synthetic inference nudge;
- shared digest
  `e298d60abc4d5f01f317a14e095c39231298618f0f573e3aa2da741423bc7dee`;
- 128 chunks plus one manifest on every rank;
- one committed terminal record and one timing record on every rank;
- clean payload SHA verification;
- final tails of 1514.582, 1508.082, 1517.160, and 1526.855 ms on ranks
  0-3;
- dominant fence time of 1283.9-1287.9 ms, writer time of about 208-223 ms,
  and manifest publication time of about 12-18 ms; and
- zero active contexts, leases, and tickets afterward, with background
  progress alive and `error=null`.

The first workstation gate report falsely timed out. The workstation clock
was about 0.44 seconds ahead of rank 0, so a subsecond Docker `--since`
boundary excluded the request POST and worker records. Corrected observer
evidence is sealed in the private validation tree as
`deliverables/evidence/sparkcache-v51-no-nudge-verified.json` (SHA-256
`0f8433ccae345956a0ee109448b6c6d79c172d798c6453be95088bca4e612005`).
This was an observer-window error, not a failed runtime gate.

## Decision

SparkCache will stop treating publication as one end-of-request snapshot.
It will journal completed 256-token regions while chunked prefill is still
running.

For the measured 392,960-token DCP4 artifact, each rank stores approximately
3.14 GB. At 800 prompt tokens/s, prefill takes about 491 seconds and produces
only about 6.4 MB/s of rank-local cache state. The existing 256-token storage
geometry produces roughly one 2 MB immutable chunk per rank every 320 ms.
This is small enough to gather, hash, persist, and optionally replicate
without placing bulk work on request completion.

Hashing is not the foreground bottleneck. The native restore gate SHA-256
verified the complete rank-local artifact in about 190 ms. The current pause
comes from gathering every registered layer and repeatedly executing
device-to-CPU conversion, contiguity, NumPy conversion, and `bytes`
allocation after prefill has already completed.

## Safety invariant

The cache is always optional. Inference must never wait for a cache ring
slot, disk write, replica acknowledgement, or manifest publication.

When any bounded resource is unavailable:

1. abort that cache transaction;
2. release every outstanding block lease and arena slot;
3. leave no visible manifest; and
4. continue serving normally.

Immutable chunks written before an abort remain unreferenced and are later
eligible for orphan collection.

## Pipeline

1. The scheduler knows the complete prompt tokens and calculates the salted
   context digest before prefill.
2. Full-quorum and per-rank-held checks suppress already-published work.
3. A per-digest single-flight transaction begins.
4. Each scheduler step reports the largest completely computed
   256-token-aligned prefix.
5. Completed chunks are grouped into bounded macrobatches. With the current
   4,096-token prefill budget, 16 storage chunks form one macro batch.
6. The KV allocator leases only the physical blocks referenced by that
   macro batch.
7. A native fused gather writes target CKV, sparse-indexer, and MTP draft-KV
   records into a preallocated mapped or managed arena slot.
8. A CUDA completion event transfers ownership of the immutable arena bytes
   to the background journal and releases the source-block lease.
9. The journal encodes, hashes, and durably publishes each content-addressed
   chunk.
10. After every expected chunk is durable, the manifest is fsynced and
    atomically published.
11. The digest is offered to the scheduler only after every DCP rank reports
    a compatible committed manifest.

## Bounded resources

Current v51 candidate:

- 256 logical tokens per immutable storage chunk;
- 16 chunks per gather macro batch;
- 64 MiB per arena slot;
- two mapped-host arena slots;
- one active publication per rank;
- no unbounded Python queue or multi-gigabyte CPU snapshot.

The GPU gather uses short bounded kernels on a low-priority stream. A
serving-activity epoch prevents submission of another batch while foreground
work is runnable. A kernel already submitted is allowed to finish; kernels
must therefore remain small enough to bound non-preemptible interference.

## Required interfaces

### Manifest journal

```text
transaction = begin_context(identity, context_digest, span_tokens)
transaction.append_chunk(context_chunk)
transaction.commit_manifest()
transaction.abort()
```

`append_chunk` publishes immutable content-addressed bytes but does not make
the context discoverable. `commit_manifest` validates exact contiguous
coverage and is the only visibility edge. Repeating an identical append or
commit is idempotent; conflicting immutable content fails closed.

### Block lease

```text
lease = lease_blocks(request_id, physical_block_ids)
event = native_gather(lease, arena_slot, logical_range)
release_after(event, lease)
```

Cancellation, gather failure, timeout, shutdown, and transaction abort must
all converge on the same idempotent release edge. Leases must never survive
worker teardown.

### Native gather ring

```text
slot = ring.try_acquire()
ticket = slot.submit(destinations, slots, logical_range, cuda_stream)
ticket.poll_or_wait()
immutable_view = ticket.complete()
slot.release_after_journal_append()
```

Acquisition is nonblocking. The ring validates generation numbers so a late
completion cannot publish into a reused slot.

## Diagonal 10GbE sideband

The direct S0-S2 and S1-S3 links are optional cache-buddy links:

```text
S0 <========== 10GbE ==========> S2
S1 <========== 10GbE ==========> S3
```

At approximately 6.4 MB/s per rank, a mirrored streaming publication uses
about 0.5% of a 10GbE link. The first implementation should use persistent
bounded TCP sessions with `BEGIN`, `PUT_CHUNK`, `COMMIT`, `ABORT`, and
credit/ack frames. Replication is never on the local commit critical path.
Link failure degrades to a local-only cache entry.

The diagonal replica prevents one missing or corrupt rank-local shard from
forcing a complete DCP4 re-prefill after a reinstall or cache loss. Raw
Ethernet/VFIO can later point TX descriptors at the same mapped arena, but
is not required for throughput and must remain behind the reliable protocol
contract.

## Implementation ladder

### SS-0: admission and duplicate suppression (implemented)

- Suppress scheduler store plans at full DCP quorum.
- Suppress worker snapshots when the digest is already held.
- Add per-digest single-flight and explicit counters.

Gate: repeated identical prompts take no snapshot and cannot produce an
immutable-object collision.

### SS-1: transactional journal (implemented)

- Implement incremental append, abort, and manifest-last commit.
- Do not retain all context chunks in memory.
- Prove crash-before-manifest invisibility and idempotent retries.

Gate: exhaustive GPU-free durability, duplicate, ordering, cancellation, and
corruption tests.

### SS-2: bounded planner and block leases (implemented)

- Convert each completed chunked-prefill range into macro batches.
- Add allocator leases and cancellation-safe release.
- Abort caching on ring backpressure.

Gate: allocator reuse cannot occur before gather completion; every failure
path releases all leases.

### SS-3: native gather ring (live v50 gate passed)

- Register the required destinations once.
- Gather all record families into mapped arena slots without Python
  `cpu/numpy/tobytes` assembly.
- Use generation-checked tickets and bounded CUDA events.

Gate: byte-exact comparison against the current snapshot path for all record
families, DCP ranks, tail sizes, and noncontiguous physical block tables.

### SS-4: chunked-prefill integration (v51 idle-progress gate passed)

- Start the journal when a cache-worthy miss is scheduled.
- Submit completed macrobatches after their final record family is written.
- Publish the manifest after all batches complete.

Gate: cache-off versus streaming-cache prefill and decode differ by no more
than 2% at C1 and C8; completion pause is at most 500 ms p95 initially and
250 ms p95 for promotion.

### SS-5: diagonal buddy replication

- Add bounded reliable chunk replication over the isolated 10GbE pairs.
- Restore from the buddy only when the local entry is missing or corrupt.
- Keep local-only publication valid when the buddy is unavailable.

Gate: cable removal, receiver restart, duplicate frame, corruption, and
partial-transaction drills never delay inference or expose an incomplete
entry.

## Live promotion gates

- **Passed in v51:** a fresh 32K `max_tokens=1` request published all four
  manifests and terminal records without a second inference request or
  synthetic inference nudge.
- **Passed in v51:** every rank emitted exactly one complete final-batch
  timing record through the configured worker logger.
- Byte-identical restored records and continued-generation equivalence on all
  four ranks.
- Zero leaked leases, arena tickets, or background threads after cancellation
  and shutdown.
- Zero manifest visibility before complete local durability.
- Zero inference waits caused by snapshot-ring capacity.
- Foreground completion pause below 500 ms p95, then below 250 ms p95.
- Cache-active TTFT and decode within 2% of cache-off at C1 and C8.
- 393K cold and warm restart restore remains within the accepted native
  restore envelope.
- Cache miss, cache hit, corrupted local shard, missing diagonal buddy, and
  interrupted commit all fail closed.
