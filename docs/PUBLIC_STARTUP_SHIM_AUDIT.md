# Public multi-node startup shim audit

Date: 2026-07-29

Upstream inspected: `vllm-project/vllm@fcc614141e5e9ab18cb304c476f7feed2a9552e3`

Executable hash target: the post-overlay files installed by the pinned
SparkRing faststart image, not the unmodified upstream checkout.

## Outcome

The historical follower shim is unnecessary when pinned vLLM is launched
through its intended multi-node headless path.  No assertion guard is
published.  Instead, `runtime/public-headless-abi-gate.py` attests the two
upstream source files that define this contract and verifies the control-flow
invariants before vLLM starts.

## Follower `collective_rpc`

Pinned upstream `vllm/v1/executor/multiproc_executor.py` constructs the
executor-side broadcast writer and its complete response queue list only when
`node_rank_within_dp == 0`. On follower nodes:

- `self.rpc_broadcast_mq` remains `None`;
- `self.response_mqs` remains empty;
- local worker processes receive the leader's remote broadcast through their
  own `WorkerProc.rpc_broadcast_mq`.

Consequently, deleting the assertion or returning `[]` cannot implement a
correct follower RPC. Returning an empty result is specifically unsafe for
callers that aggregate worker values; it previously surfaced as a downstream
`min()` failure in the reference development history.

More importantly, a correctly launched follower never needs such a local RPC.
Pinned `vllm/entrypoints/cli/serve.py::run_headless` constructs
`MultiprocExecutor(..., monitor_workers=False)`, blocks in
`start_worker_monitor(inline=True)`, and returns before
`CoreEngineProcManager` can be constructed.  The leader EngineCore owns KV
profiling and broadcasts the resulting worker calls to every rank.

The executable gate pins both that branch and the complementary queue
ownership in `multiproc_executor.py`. The pinned hashes are taken after the
public overlay is applied because that overlay also modifies these files.
Source drift fails startup with exit 78 instead of installing a compatibility
guess.

## Empty KV configuration

Pinned `vllm/v1/core/kv_cache_utils.py` has two empty-list hazards:

- `generate_scheduler_kv_cache_config()` indexes `kv_cache_configs[0]`;
- `get_kv_cache_configs()` computes `min()` across generated configs.

The correct empty result is not derivable from those functions alone because
`EngineCore` immediately consumes `num_blocks`, `kv_cache_groups`, and cache
capacity from the scheduler configuration. A fabricated empty
`KVCacheConfig` could let startup continue with semantically wrong allocator
state.

These hazards are unreachable on a correctly launched follower: only the
leader constructs EngineCore and generates the global scheduler
configuration. They remain useful diagnostics because encountering either one
means the launch topology or an upstream ABI has drifted.

## Reopen condition

The offline ABI contract is closed. A 2026-07-29 partial four-rank bring-up
proved one leader plus three headless followers through distributed
initialization, full model/MTP loading, B12X prewarm, and KV allocation without
entering follower `collective_rpc`. Live acceptance still must prove the
corrected startup reaches API readiness and serves a deterministic request.

If that acceptance run ever reaches follower `collective_rpc`, stop and capture
the call sequence. Do not replace the assertion with an empty return to make
startup appear to progress.
