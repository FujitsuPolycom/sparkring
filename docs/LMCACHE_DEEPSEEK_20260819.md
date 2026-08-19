# LMCache with DeepSeek-V4-Flash-0731 on the four-Spark ring, 2026-08-19

Status: **blocked in the cache library's read path, with storage and
cross-restart persistence proven functional.** LMCache stores this
model's key-value state and restores it from a filesystem tier across
a complete process teardown. Loading those restored bytes into device
memory then fails on a buffer-size mismatch, every block is marked
invalid, and the engine recomputes the prompt in full. The serving
stack runs without LMCache.

Two defects were found. The first is fixed and its fix is verified
here; the second is open and is the current blocker.

## What was established

- **Registration works.** One LMCache multiprocess server per rank,
  started inside the engine's own container, registers the engine's
  key-value cache: `Registered KV cache ... with 170 layers`. The
  layer count matches the model's heterogeneous set plus the DSpark
  draft caches that speculation adds.
- **Storage works.** A 52,623-token request produced 28 store events
  and 3.2 GB of filesystem-tier objects.
- **Cross-restart persistence works.** After removing the containers
  entirely (engine, cache server, its memory tier, and the engine's
  native prefix cache all destroyed), a fresh deployment's startup
  scan primed 3.41 GB in 410 objects, and replaying the identical
  prompt retrieved **410 of 410 keys from the filesystem tier, none
  from memory**, in 606.4 ms. No surviving in-memory state can
  explain that restore.
- **The recompute fallback works.** With the scheduler defect below
  fixed, a failed load is reported and rescheduled rather than
  killing the engine core: `Recovered from KV load failure: 1
  request(s) rescheduled (52480 tokens affected)`.

## Fixed defect: single-group unpack in the load-failure path

Applying restored blocks failed in the engine's scheduler:

```
vllm/v1/core/sched/scheduler.py, _update_requests_with_invalid_blocks
    (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
ValueError: too many values to unpack (expected 1)
```

The key-value-load failure path unpacked exactly one block-identifier
list, which holds only for models with a single key-value cache
group. This checkpoint is heterogeneous, so the call returns one list
per group, and any external connector reporting an invalid block on
this model killed the engine core.

The deployed fix iterates positions across all groups, intersects each
group's block at a position against the invalid set, and evicts the
tail of every group when eviction is required. With it applied on all
four ranks the same condition now produces the recovery message quoted
above.

## Open defect: a uniform buffer for a non-uniform cache

The cache server aborts every retrieve:

```
lmcache/v1/distributed/storage_manager.py, read prefetched results
ValueError: Size mismatch: memory_obj nbytes=985664,
                           gpu_buffer nbytes=15644672
```

The worker then reports `21525 block(s) marked invalid for scheduler
recompute`, and the engine recomputes the whole prompt. Every replay
repeats this, so no request ever benefits from the cache.

The cause is visible in the server's own group inventory. This model
registers five kernel groups whose geometry does not agree:

| Groups | Layers | Tokens per block | Head size | Element size |
|---|---:|---:|---:|---:|
| 0-2, latent attention | 23 each | 64 | 584 | 1 |
| 3, sparse indexer | 21 | 4 | 512 and 2048 | 4 |
| 4, sliding-window compressor | 20 | 8 | 1024 | 4 |

Block sizes of 4, 8, and 64 tokens and element sizes of 1 and 4 bytes
coexist in one registration. The read path sizes one device staging
buffer per transfer against a single geometry, so the buffer it
allocates and the stored object it is asked to receive disagree and
the transfer raises before any bytes land.

The write path has no such assumption: it stores each group correctly,
which is why all 410 objects are found and read back from the
filesystem tier. Only the read-into-device-memory path is affected.

This is a property of the cache library and the checkpoint, not of the
hardware, the fabric, or the deployment configuration. A model with a
single homogeneous key-value group does not reach it.

## Attribution cautions recorded for reuse

Two different signals have each falsely indicated success on this
work. Both are recorded because each is individually convincing.

- **Speed is not attribution.** A same-process replay measured 36-40x
  faster than cold with byte-identical output. Engine metrics for that
  run showed `external_prefix_cache_hits_total = 0` against 47,738
  queries while the engine's own prefix cache reported 47,104 hits:
  the speedup was native prefix caching. Only a measurement that
  destroys the engine's own cache attributes a restore to the external
  tier.
- **A hit counter is not a completed load.**
  `external_prefix_cache_hits_total` counts lookup matches, not bytes
  delivered. A run reporting 104,960 hits against 105,246 queries,
  with the engine's native counter at zero, was recomputing every
  token: the lookups matched and the transfers then failed. Confirm
  that the request completes before reading any cache counter as
  success.

## Configuration that reached this point

Per rank, inside the engine container so that server and engine share
a lifecycle and no server can outlive the engine whose device memory
it maps: chunk size 256 matching the engine block size, an 8 GiB lazy
memory tier, and a bounded `fs_native` filesystem tier. Engine side:
the multiprocess connector with both tiers listed,
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
  continue while lookups silently stop. The patch is deployed; no run
  recorded here stayed idle long enough to exercise it, so its effect
  is unverified.

## The build in use is not stock

The installed package reports itself as `lmcache 0.5.2+glm52dcp4.1`: a
build carrying local changes made for a GLM deployment, whose
key-value cache is a single homogeneous group. The size check that
raises is a correct guard in
`lmcache/v1/gpu_connector/gpu_ops.py`, comparing the stored object
against the buffer it was handed; the sizing decision that produces
the disagreement happens before it. Whether stock LMCache at the same
version sizes per group is not established here, and the local suffix
means the failure cannot be attributed upstream on this evidence.

## What would unblock it

Two candidates, neither attempted here:

1. **Establish whether stock LMCache has the same limitation.** The
   installed build is a local GLM-oriented fork, so the first question
   is whether an unmodified build of the same version, or a newer one,
   sizes the staging buffer per kernel group. This is a packaging
   question before it is a patching question.
2. **Compare against the two-Spark deployment's engine tree.** That
   deployment runs LMCache against a different tree; if its read path
   already handles per-group geometry, it names the change directly.

Patching the installed build is the fallback if neither resolves it,
and would mean sizing the staging buffer per kernel group rather than
once per transfer.
