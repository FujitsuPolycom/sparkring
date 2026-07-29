# Snapshot-ring production feature gate

`spark_cache_streaming_snapshots` (or
`SPARK_CONTEXT_CACHE_STREAMING_SNAPSHOTS`) is default-off. The connector
accepts the explicit `0/1` and `false/true` forms. `0` does not construct a
ring, import the snapshot CUDA binding, reserve a KV block, alter scheduler
metadata, or change the existing end-of-prefill store path.

An explicit `1` lazy-installs the builtin scheduler or worker factory. It
fails connector construction unless all of the following attest:

- `spark_cache_streaming_native_library` /
  `SPARK_CONTEXT_CACHE_STREAMING_NATIVE_LIBRARY` is an absolute path;
- `spark_cache_streaming_native_library_sha256` /
  `SPARK_CONTEXT_CACHE_STREAMING_NATIVE_LIBRARY_SHA256` is the pinned
  lowercase SHA-256;
- the installed vLLM sources match
  `runtime_patches/vllm-kv-block-lease-contract.json`;
- worker cache registration is exactly the proven GLM-5.2 DCP4 inventory:
  79 target sources at 368 bytes/token and 22 indexer sources at
  132 bytes/token, with colocated MTP state.

The scheduler adapter remains CUDA-free. The worker does not hash/load the
native library, allocate the arena, retain tensor views, or start its writer
until `register_kv_caches()` has produced the final canonical inventory.

## Exact live call seams

1. Scheduler admission sends a stable request ID, digest, span, completed
   watermark, and full block table to every worker. Each worker begins its
   rank-local `ManifestStore.begin_context(...)` transaction and allocates a
   nonzero local context sequence.
2. After each worker prefill step, the model runner must supply the largest
   completed 256-token-aligned prefix, the producer CUDA stream, and the
   request's current physical block table. `build_connector_meta()` carries
   only a promise; `wait_for_save()` converts it to a completed watermark
   after the corresponding target/MTP forward on that stream.
3. The worker maps each offered logical batch with `BlockTableRangeMapper`,
   calls `BlockLeaseRegistry.try_reserve(...)`, then calls native
   `try_submit(...)` *inside* `LeaseHandle.submit(...)`. The submission
   callback must return a completion fence before it releases the registry
   lock; otherwise preemption can recycle source blocks in the submit/event
   gap.
4. Once native `poll(...)` returns READY, the worker releases the block lease
   and claims the generation-checked ring ticket. It lends the immutable view
   to `ManifestSnapshotJournalWriter`; after the returned completion proves
   the writer no longer reads the view, the worker releases the ticket.
5. Only after all expected chunks are durable may it call
   `commit_manifest()`. Backpressure, cancellation, preemption, writer error,
   or shutdown must call `abort()`, abandon the native context, and drain or
   retain leases according to `BlockLeaseRegistry`'s fail-closed rules.

## Fixed production profile

- `NativeSnapshotRing` owns the attested handle lifecycle, a single source
  inventory, generation-checked tickets, zero-copy read-only READY views,
  abandonment, and checked shutdown. The explicit production factory
  constructs it only from worker `register_kv_caches()`.
- The standalone winner is frozen as mapped-host memory, ring depth 2,
  64-MiB slots, and a 1,024-row GLM macro payload of exactly 32,743,424 bytes.
  `Glm52ReadyViewTranslator` fixes the 79 target-CKV then 22 sparse-indexer
  source order; MTP is colocated in target CKV and is not a separate record.
- `ManifestSnapshotJournalWriter` accepts runtime-owned READY views, performs
  the bounded per-256-token canonical copy, appends through
  `ManifestTransaction`, and completes only after durable append. It commits
  only exact full-span coverage. Backpressure and writer errors abort
  invisibly; it never claims, releases, or abandons a ring ticket.
- The connector transports scheduler promises and converts them into completed
  watermarks only from post-forward `wait_for_save()`, with the current CUDA
  producer stream.
- The production factory retains every exact contiguous row alias for the
  entire ring lifetime and rejects copy-producing/noncontiguous layouts.
- Scheduler delayed-free, worker lease completion, preemption/resume,
  fail-open transaction abort, manifest-last publication, and committed
  digest advertisement are covered by the connector/runtime/publisher tests.

This code is ready for the single cache-off versus streaming-cache live model
gate. The standalone native matrix and GPU-free integration tests do not by
themselves claim live GLM performance or zero regression.

## READY-view to `ContextChunk` ownership edge

The offline translator and writer now implement this edge:

- A native macro-batch record is layer-major across every submitted row.
  `_SnapshotChunks` requires layer-major bytes for each individual 256-token
  chunk. Translation takes one 64-row DCP4 slice from each layer and joins the
  slices in the frozen native source-ordinal order.
- `ContextChunk` currently accepts owned `bytes` only. A ring-backed
  `memoryview` cannot satisfy that contract, so the writer makes one bounded
  canonical chunk copy while it borrows the claimed view. Its completion
  becomes observable only after `append_chunk()` has durably returned; the
  runtime then releases the arena.
- `LOGICAL_POSITIONS` is not present in the native payload. Translation must
  derive and pack it; the translator now generates the exact interleave-1
  DCP-rank positions from logical start and shard rank.
- Numeric native record kinds and source ordinals are proven identical to
  the connector's `StateRecord`, `LayerPlan`, draft-policy, and boundary-policy
  ordering. The journal writer enforces DCP4, `live_forward`, and
  `colocated_target`; live connector registration constructs and attests that
  exact inventory and retains its aliasing row views for ring lifetime.

GPU-free tests feed a 1,024-row synthetic GLM READY view through the frozen
layout and require byte-identical canonical encoded chunks, including DCP
positions and the absence of a separate MTP record. A connector-to-runtime-to-
publisher boundary test also proves that `_held` and worker stats advertise a
digest only after the full manifest commit, never on partial append or abort.
