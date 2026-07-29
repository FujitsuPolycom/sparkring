# SparkCache streaming-snapshot runtime contract

The first streaming-snapshot implementation does **not** require a new vLLM
allocator patch.  Official vLLM at the runtime-pinned commit already provides
the ownership contract SparkCache needs:

1. scheduler-side `KVConnectorBase_V1.request_finished()` may return `True`;
2. the scheduler then retains the finished request and its KV block table;
3. worker-side `get_finished(finished_req_ids)` reports the request only after
   the last asynchronous CUDA gather event completes;
4. `Scheduler._update_from_kv_xfer_finished()` then calls `_free_blocks()`.

`vllm-kv-block-lease-contract.json` pins the complete official source files
that implement this behavior, including `KVOutputAggregator`, whose
world-size completion countdown requires every worker to report an owned
request exactly once. Runtime installation must verify those hashes or
explicitly port and re-pin the same semantics. A fuzzy patch or symbol-only
match is not sufficient.

Verify an unpacked vLLM source tree before installing the streaming connector:

```bash
python -m sparkcache.runtime_patches.verify_lease_contract \
  --vllm-root /path/to/vllm/source
```

## Narrow integration

Use `sparkcache.streaming.BlockLeaseRegistry` in the worker connector:

```python
handle = leases.try_reserve(request_id, completed_chunk_block_ids)
if handle is None:
    abandon_cache_publication(request_id)  # never wait for a staging slot
else:
    try:
        handle.submit(lambda: submit_gather_and_record_event(...))
    except LeaseFenceError:
        # A callback failure after possible submission is fatal and retains
        # the lease. Only a failure proven to occur before handle.submit()
        # may use cancel_before_submit().
        raise
```

Never enqueue CUDA work and then call a separate `arm(event)` method. That
creates a reserve/submit/preemption race in which vLLM can recycle the blocks
after GPU work begins but before the event becomes visible. `handle.submit()`
serializes the submission callback and fence publication against preemption.
If the callback raises, submission status is unknown, the lease remains held,
and the worker fails closed.

Map every logical macro batch to only its relevant physical block-table slice:

```python
block_map = BlockTableRangeMapper(
    block_ids=tuple(request_block_ids),
    logical_tokens_per_block=block_size * dcp_degree,
)
offer = planner.offer_completed(request_id, completed_tokens, block_map)
```

The planner calls `blocks_for_range(logical_start, logical_end)` separately for
each emitted batch. Mapping failure aborts the optional cache transaction.
Adjacent aligned batches therefore lease disjoint physical blocks instead of
all contending on the request's complete block table.

Poll `leases.poll()` once per worker step.  Implement worker-side
`get_finished()` as:

```python
owned_finished = finished_req_ids & streaming_seen_request_ids
ready = leases.take_finished(owned_finished)
return ready or None, None
```

vLLM's `finished_req_ids` contains every request newly finished by that
scheduler output, including ordinary requests that never emitted a streaming
offer.  Never pass that unfiltered set to the lease registry: its intentional
"unknown means ready" rule would echo an ordinary ID as a completed async send
after the scheduler had already removed it.  Scheduler-side
`request_finished()` returns `True` only for a request admitted to a streaming
publication transaction.  Returning `True` when an admitted worker transaction
abandoned before submission is safe: after the ownership intersection,
`take_finished()` immediately reports that seen request without an active
lease, adding at most one scheduler round.

During normal prefill, vLLM already owns all blocks for the live request.  Most
chunk gathers finish while that ownership is still active.  The delayed-free
contract is needed only for the final tail that remains in flight when the
request finishes or is aborted.

## Preemption and eviction boundary

Normal request completion is not the only reclamation path. The pinned
`GPUModelRunner.execute_model()` calls:

```python
get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
```

before persistent-batch updates and before the next forward can overwrite
reallocated blocks. Extend `SparkCacheConnectorMetadata` with:

```python
preempted_request_ids: tuple[str, ...] = ()
```

and populate it scheduler-side with:

```python
tuple(sorted(scheduler_output.preempted_req_ids or ()))
```

Worker-side `handle_preemptions()` delegates synchronously:

```python
preemption_adapter.handle_preemptions(kv_connector_metadata)
```

`PreemptionDrainAdapter` cancels unsubmitted reservations, synchronizes armed
gather events, releases their leases, and only then abandons the invisible
publication journal. If synchronization fails it retains the lease and raises;
the worker must terminate rather than return to a forward that could overwrite
blocks whose read status is unknown.

Same-stream ordering alone is **not** the safety contract. A future model
runner, connector progress thread, graph replay, or separate low-priority copy
stream can change stream relationships, while allocator reclamation is
host-side. A recorded completion event plus explicit query/synchronize is the
portable proof that no DMA or kernel still reads the physical block.

## Required failure behavior

- Capacity exhaustion and block overlap abandon caching without waiting.
- A launch failure before CUDA submission cancels the reservation immediately.
- Once submitted, an armed lease is never cancelled.  Abort synchronizes its
  fence before releasing it.
- A failed fence query/synchronize retains the lease and is fatal to that
  worker; recycling blocks with unknown CUDA-read status would corrupt a later
  request.
- Connector shutdown calls `abort_all()` before destroying staging arenas.
- DCP ranks decide cache publication independently, but the manifest remains
  invisible until the existing all-rank quorum contract succeeds.

## Why not refcount `BlockPool` directly?

Adding an external refcount beneath `KVCacheManager` would touch allocation,
prefix caching, eviction order, hybrid cache groups, preemption, and reset
paths.  It would also duplicate a supported connector lifecycle already used
for asynchronous KV sends.  Delaying the finished request's existing block
table is the narrowest safe seam and is sufficient because streaming copies
occur while the request still owns its blocks.
