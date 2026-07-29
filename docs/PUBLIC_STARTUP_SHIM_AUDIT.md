# Public multi-node startup shim audit

Date: 2026-07-29  
Upstream inspected: `vllm-project/vllm@fcc614141e5e9ab18cb304c476f7feed2a9552e3`

## Outcome

The two startup shims described by the historical artifact inventory are not
safe to recreate as assertion guards alone. No patch was published from this
audit.

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

A public fix needs an explicit executor-to-local-worker request channel (or an
upstream-supported forwarding API), plus response ownership and timeout
semantics. It must not inject a follower-originated request into the leader's
world-wide scheduler broadcast stream.

Required tests before a patch is admissible:

1. one leader plus three headless followers, one local worker per node;
2. follower-originated RPC returning one result per intended local worker;
3. `unique_reply_rank`, non-blocking futures, timeout and worker-error paths;
4. no duplicate execution on remote ranks;
5. startup KV profiling and `initialize_from_config` complete on all ranks.

## Empty KV configuration

Pinned `vllm/v1/core/kv_cache_utils.py` has two empty-list hazards:

- `generate_scheduler_kv_cache_config()` indexes `kv_cache_configs[0]`;
- `get_kv_cache_configs()` computes `min()` across generated configs.

The correct empty result is not derivable from those functions alone because
`EngineCore` immediately consumes `num_blocks`, `kv_cache_groups`, and cache
capacity from the scheduler configuration. A fabricated empty
`KVCacheConfig` could let startup continue with semantically wrong allocator
state.

This must be resolved together with the follower RPC ownership above: capture
the real per-node `kv_cache_specs` and `available_memory` cardinalities in a
headless four-node startup, then define which executor owns the global
scheduler config and which followers only initialize local tensors.

## Reopen condition

Implement these patches only after either:

- upstream vLLM publishes a supported headless multiprocess forwarding path;
  or
- a model-down four-node probe captures the exact follower call sequence and a
  local request/response channel can be tested independently.

Until then the public entrypoint's capability gate and the repository status
must continue to report startup correctness as unresolved. Do not replace the
assertion with an empty return to make startup appear to progress.
