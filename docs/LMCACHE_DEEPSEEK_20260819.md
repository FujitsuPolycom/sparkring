# LMCache with DeepSeek-V4-Flash-0731 on the four-Spark ring, 2026-08-19

Status: **blocked by an engine defect, with the cache tier itself
proven functional.** LMCache stores this model's key-value state and
restores it from a filesystem tier across a complete process
teardown; the serving path then fails inside the engine's
key-value-load recovery code. The serving stack runs without LMCache.

## What was established

- **Registration works.** One LMCache multiprocess server per rank,
  started inside the engine's own container, registers the engine's
  key-value cache: `Registered KV cache ... with 170 layers`. The
  layer count matches the model's heterogeneous set plus the three
  DSpark draft caches that speculation adds.
- **Storage works.** A 52,623-token request produced 25 store events
  and 5.8 GB of filesystem-tier objects.
- **Cross-restart restore works.** After removing the containers
  entirely (engine, cache server, its memory tier, and the engine's
  native prefix cache all destroyed), a fresh deployment's startup
  scan primed 6.49 GB in 780 objects, and replaying the identical
  prompt retrieved **410 of 410 keys from the filesystem tier, none
  from memory**, in 576.7 ms. No surviving in-memory state can
  explain that restore.

## The defect

Applying the restored blocks fails in the engine's scheduler:

```
vllm/v1/core/sched/scheduler.py, _update_requests_with_invalid_blocks
    (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
ValueError: too many values to unpack (expected 1)
```

The key-value-load failure path unpacks exactly one block-identifier
list, which holds only for models with a single key-value cache
group. This checkpoint is heterogeneous: latent attention, the
sparse indexer, the sliding-window compressor, and the speculative
draft caches occupy separate groups, so the call returns one list per
group. Any external key-value connector that reports an invalid block
on this model reaches this line, and the engine core dies.

The failure is in the engine, not in the cache: the restore had
already completed successfully when it fired.

## Attribution caution recorded for reuse

A same-process replay measured 36-40x faster than cold with
byte-identical output, which looks like cache success and is not
evidence for it. Engine metrics for that run showed
`external_prefix_cache_hits_total = 0` against 47,738 queries while
the engine's own prefix cache reported 47,104 hits: the speedup was
native prefix caching. Only a measurement that destroys the engine's
own cache, as above, attributes a restore to the external tier.

## Configuration that reached this point

Per rank, inside the engine container so that server and engine share
a lifecycle and no server can outlive the engine whose device memory
it maps: chunk size 256 matching the engine block size, an 8 GiB lazy
memory tier, and a bounded `fs_native` filesystem tier with direct
I/O. Engine side: the multiprocess connector with both tiers listed,
`kv_load_failure_policy` recompute, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`.

Three conditions had to hold before registration succeeded:

- **The server must run inside the engine's container.** A server in
  a separate container cannot import the engine's device memory:
  registration fails with `cudaErrorInvalidResourceHandle`.
- **The registration deadline must exceed the registration time.**
  Registering 170 layers against a 32 GiB per-rank reservation took
  about 75 seconds, past the 60-second default; the deadline is now
  300 seconds. A deployment with a smaller reservation registers
  faster.
- **The heartbeat guard must be patched.** In the installed package,
  `if self._heartbeats is not None:` guards the heartbeat thread
  against an empty dictionary, so the thread never starts and the
  server reaps a healthy engine after roughly 150 seconds. Stores
  continue while lookups silently stop.

## What would unblock it

The scheduler path must handle one block-identifier list per
key-value cache group. A deployment of this model on two Sparks runs
LMCache on a different engine tree, so a tree that resolves this is
the other candidate; comparing the two scheduler implementations is
the next step. Neither is attempted here.
