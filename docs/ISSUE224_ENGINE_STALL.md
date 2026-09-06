# Running-request stall: evidence and recovery detection

The [source-level indexer investigation](ISSUE224_INDEXER_BARRIER.md) identifies
a histogram-publication race and includes a GPU-tested kernel fix and targeted
isolation experiment. This document covers the fallback liveness detector.

Status: implemented for offline output-stall detection. The separate kernel
fix addresses the confirmed publication race behind the suspected deadlock in
[issue #224](https://github.com/FujitsuPolycom/sparkring/issues/224). No full model
soak or deployment of the fix to serving containers is claimed.

## Source evidence

The operator pins identify the affected vLLM commit as
`e02b174693e13859de61811b5e8cd13d5308e259` and the earlier comparison commit as
`22ffe1401ca9bd3e4503e62de7b414deca7661a1`, both available in
`https://github.com/FujitsuPolycom/vllm`.

Comparing those exact commits shows no differences in
`vllm/v1/executor/multiproc_executor.py`,
`vllm/distributed/device_communicators/shm_broadcast.py`, or
`vllm/v1/engine/core.py`. The 15 changed files concern B12X attention/model
paths, associated tests, and argument handling. Timing changes can expose a
pre-existing race, so this does not exonerate the executor.

The affected queue already caps shared-memory reader waits at five seconds
(`SHM_READER_RECHECK_INTERVAL_MS`). A stack sampled in `poll()` does not show
that the individual poll is indefinite. Lost notification alone should not
permanently hide a published shared-memory slot with this source.

With async scheduling enabled, `WorkerProc.handle_output()` puts outputs on
`async_output_queue`. A separate `async_output_busy_loop()` invokes
`enqueue_output()`, which calls `AsyncModelRunnerOutput.get_output()` before
publishing the response. The main worker can therefore wait for its next RPC
while its response thread is still blocked. Main-thread-only stacks cannot
establish that a response was produced or lost.

## Discriminating evidence to collect during a hang

Capture all Python threads in EngineCore and every TP worker, twice at least
ten seconds apart, before stopping the stack. Use `py-spy dump --pid PID`
inside each relevant process namespace. Include native frames if supported.
Preserve the full output, not just MainThread.

Check these possibilities in order:

1. Async output is waiting for CUDA completion. A response thread in
   `get_output()` or event synchronization supports this; repeated stacks and
   GPU/stream evidence are needed to identify the dependency that cannot finish.
2. Async output failed before publishing. A missing output thread and an
   associated traceback support this. Inspect logs from all ranks; the process
   itself may still be alive.
3. Transport lost a response or queue state diverged. This requires evidence
   that the relevant response was published, plus queue/slot and rank identity.
   An idle main thread alone is insufficient.

Preserve full image digests and hashes of the three source files above from
the running containers. A stale version string or JIT namespace is insufficient
to establish the actual runtime source. Record async-output stacks, metrics,
and logs together so they can be correlated with the same stall.

## Detection and recovery

The rank-zero monitor detects sustained running requests without output batches
using `vllm:iteration_tokens_total_count`. The separate 300-second default
`SPARKRING_LIVENESS_OUTPUT_SECONDS` must exceed legitimate prefill and restore
gaps. This standard metric counts output-bearing batches, not every engine
step. It is a fallback heuristic for the single-engine TP4 profile.

The offline regression feeds fresh unchanged metrics with three running
requests for 300 seconds. Before the change, liveness stays HTTP 200. After
the change, it returns HTTP 503 with `engine_output_stall`. Other checks cover
progress, idle periods, counter reset, recovery, missing metrics, invalid
timeouts, and independent prefill grace. No GPU race is reproduced by these
tests.

Rebuild the operator wrapper to deploy the monitor; editing a runtime setting
alone cannot update an existing image. This patch does not change published
image pins or qualification receipts. Validate the rebuilt image with both
long healthy prefills/restores and injected output stalls before using its
signal for unattended recovery. Follow the managed deployment's coordinated
stop/recovery procedure; this monitor only reports health.

Do not automatically replay a timed-out model step. The executor's responses
are ordered without per-call IDs, and a partially completed step can mutate KV
and speculative state. A future executor timeout must invalidate the executor
and propagate failure through the established recovery path, including pending
futures. It needs separate tests for late responses and partial multi-rank
completion; adding a retry to `get_response()` is not a safe fix.
