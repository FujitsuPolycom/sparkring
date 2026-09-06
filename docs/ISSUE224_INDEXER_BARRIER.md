# Issue 224: sparse-indexer publication barrier race

Status: implemented and GPU-tested. The publication race reproduced on all four
GB10 GPUs; the fix prevented it in all 80 probes. The patched indexer passed 15
GPU correctness tests and 2,000 graph replays with concurrent copies. A full
model soak was not performed. See the
[bounded GPU evidence](../performance/records/glm53-flash/issue224-dgx4-validation.md).

## Finding

The strongest source-backed explanation is GPU deadlock in the fused DSA
indexer's cooperative top-k merge. The executor stack is downstream of it.

Affected source: B12X commit
`9ae41c5cb9935d740456479954b0089f80bd2ef2`, file
`b12x/attention/dsa_indexer/fused_indexer.py`.

The defect is in `_fused_group_barrier()` at lines 377–386. Its leader publishes
arrival before synchronizing the rest of its thread block:

```python
if tx == Int32(0):
    red_add_global_release_i32(arrival_ptr, Int32(1))
    spin_wait_global_ge_i32(arrival_ptr, (phase + Int32(1)) * ctas_per_group)
cute.arch.sync_threads()
```

`_coop_wide_round()` and `_coop_narrow_round()` publish global histogram bins
from multiple warps and then call this helper. There is no block-wide barrier
between those publications and the leader's arrival. A release fence on thread
0 orders that thread's operations; it cannot make an unscheduled publishing
warp finish. NVIDIA documents this distinction in the
[CUDA synchronization and memory-fence rules](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#memory-fence-functions).

The barrier after the spin only protects consumers inside the same block.
Another block can already have passed its own barrier and started reading the
global histogram while the delayed publisher is still running.

## Why this becomes a permanent GPU wait

The cooperative merge computes a pivot from the histogram independently in each
block. It conditionally stops refining when `bin_count == remaining_k`.
Inconsistent histogram snapshots can therefore change the number of subsequent
barriers each block executes.

One legal schedule, using three blocks and top-k 512:

1. Block A contributes 512 candidates in a lower pivot bin. Block B contributes
   one higher candidate, but its publishing warp is delayed. Block C has no
   candidates. Total candidate count is 513.
2. All three block leaders announce arrival. A reads an incomplete histogram
   with 512 lower candidates and concludes that refinement is done.
3. B's delayed publication completes. B now sees one higher candidate, leaving
   511 slots to select from the lower bin's 512 candidates. B needs another
   radix round.
4. A and C do not enter that round. B increments the cumulative arrival count
   from 3 to 4 and spins waiting for 6. The missing arrivals cannot occur.

The model collapses matching coarse/fine bins into two abstract bins and models
one leader plus one representative nonleader warp per block. It executes an
AST-lowered copy of the actual barrier helper; CUDA scheduling and code generation
are not executed. This is evidence of a reachable protocol failure, not a
hardware reproduction or a measurement of its production frequency.

The accompanying standalone model uses only Python's standard library. Run it
against a clean checkout of the affected B12X pin, then apply the candidate patch
and repeat:

```bash
python repro_indexer_barrier.py /path/to/b12x/b12x/attention/dsa_indexer/fused_indexer.py --expect deadlock
git -C /path/to/b12x apply /path/to/issue224-b12x-barrier.patch
python repro_indexer_barrier.py /path/to/b12x/b12x/attention/dsa_indexer/fused_indexer.py --expect complete
```

The evidence JSON records both traces: before the change, arrival is 4 with a
block waiting for 6; after the change, arrival reaches 6 with no pending actors.

## Full response path

```text
EngineCore.step_with_batch_queue()
  -> MultiprocExecutor.execute_model() / sample_tokens()
  -> worker main thread submits model GPU work
     -> B12xSparseIndexer.forward()
     -> dsa_indexer.run() -> fused paged indexer
     -> cooperative histogram merge can deadlock
  -> AsyncGPUModelRunnerOutput enqueues output copy
     -> copy stream waits for the model stream
  -> WorkerProc.handle_output() queues the asynchronous output object
     -> worker main thread returns to RPC dequeue
  -> async_output_busy_loop() -> enqueue_output() -> get_output()
     -> async_copy_ready_event.synchronize() cannot finish
  -> response never gets enqueued
  -> EngineCore waits in get_response()
```

Relevant vLLM locations at `e02b174693e13859de61811b5e8cd13d5308e259`:

- `vllm/v1/engine/core.py`: `step_with_batch_queue()` schedules ahead, then drains
  the oldest result through `future.result()`.
- `vllm/v1/executor/multiproc_executor.py`: `handle_output()` sends asynchronous
  results to a separate thread. `enqueue_output()` must resolve `get_output()`
  before publishing a response. All main worker threads can be idle during this
  failure.
- `vllm/v1/worker/gpu_model_runner.py:326`: the output copy stream waits for the
  current model stream. Line 356 synchronizes the output-completion event.

Other ranks can be waiting in later GPU collectives when one rank stops making
progress. The specific kernel running on each rank still requires a GPU trace.
GPU utilization alone does not identify a kernel or prove this attribution.

## Regression evidence

The earlier image's receipt pins B12X to
`6255090a03b12c3f7d552102a02fac0b542fb8c9`, while the affected operator image
pins `9ae41c5cb9935d740456479954b0089f80bd2ef2`.

B12X commit `357576e6d49a2d9fbf623cd73542826fdf55bb8e` introduces the separate
12/12/8-bit cooperative merge and its conditional refinement. It is not an
ancestor of the earlier pin and is an ancestor of the affected pin. The earlier
merge also has publication-order concerns, but its main refinement loop uses a
fixed four rounds; it does not use this new early-exit protocol. Absence of
observed hangs in the earlier image is not proof that it is race-free.

The affected GB10 profile's `attention.dsa_indexer` entry contains only
`backend: native`. Missing `fused_merge` resolves to `auto`, which now resolves
to cooperative for multi-block groups. GLM's 32-head/top-k-512 shape permits the
fused path for decode plans up to 16 rows on SM121. With 48 SMs and a 16-row
plan, the scratch planner assigns three blocks per group. Request concurrency
and query-row count are not interchangeable, especially with speculation.

This makes the failure path reachable under the recorded default composition.
The exact live plan and group shape at each reported hang are not available.

## Corrections to the original diagnosis

The exact vLLM comparison from `22ffe140` to `e02b1746` leaves the executor,
shared-memory queue, and EngineCore files unchanged. Their source already has:

- Five-second shared-memory rechecks, independent of notification delivery.
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, default 300, passed to both execution
  and sampling RPCs. `collective_rpc()` computes a deadline and passes the
  remaining time to `dequeue()`.

A stack in `poll()` cannot establish an indefinite poll. A supervisor restarting
after 60 seconds can prevent the 300-second timeout from being observed. If the
same RPC is confirmed to remain blocked beyond its configured deadline, inspect
the actual container files, environment, and closure's method/deadline; that is
additional evidence not explained by the source-pinned timeout behavior.

## Candidate fix and isolation experiment

Add `cute.arch.sync_threads()` at entry to `_fused_group_barrier()`, before the
leader publishes arrival. Keep the existing trailing barrier. This makes every
publishing warp participate before peers are allowed to consume the histogram.
Bump the fused indexer's `KernelCompileSpec` revision from 1 to 2 so existing
cached compiled artifacts do not reuse the old protocol.

The kernel patch contains those changes only. The image builder applies
`runtime/glm53-flash-jj-r8-gb10/patch_indexer_barrier.py` to the exported B12X
source before creating its source manifest. The transform checks both input
and output hashes and records its own digest in the build receipt. It rejects
unexpected source rather than applying a speculative replacement. The output
uses LF line endings; the original GPU test file used mixed line endings.
Both have identical Python source after newline normalization.

The [published child image](../runtime/glm53-flash-jj-r8-gb10/hotfix/README.md)
includes the fix; users do not need to rebuild it. Existing image digests are
unchanged, so restarting an old image alone does not install the update.

For a targeted A/B run, pass `B12X_FUSED_INDEXER=0` inside every worker container
before startup. The pinned `dsa_indexer/scratch.py` recognizes this switch and
selects the tiled decode route. It bypasses the suspect kernel while retaining
SparkCache, speculation, and the transport configuration. The SparkRing launcher
patch forwards this variable and validates 0/1; the default remains 1. Existing
installed launchers do not forward it automatically.

Use a separate `JIT_CACHE_NAMESPACE` for that run so captured/compiled model
artifacts cannot hide the changed dispatch. Apply the same setting to all ranks
through the managed coordinated restart procedure. Changing the environment in
an already-running container cannot change its existing plans and graphs.

If a stall recurs after installing the fix, retain full worker-thread dumps,
image/source hashes, and the selected indexer route. Capture every worker thread
twice, ten seconds apart, and include EngineCore, executor-timeout errors, and
SparkCache metrics. GPU kernel coverage does not establish that every possible
EngineCore stall has the same cause.

The regular indexer launch also assumes co-resident blocks without requesting
cooperative launch. That is a separate forward-progress risk requiring occupancy
and concurrent-stream validation; the candidate patch does not claim to solve
all possible kernel hangs.
