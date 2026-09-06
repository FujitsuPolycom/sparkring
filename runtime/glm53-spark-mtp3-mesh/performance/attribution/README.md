# Request cache attribution at scheduler boundaries

Status: **implemented** with CPU scheduler-seam coverage. The MTP3 profile
remains **research-only**; rebuilt-image validation is required.

`patch_scheduler.py` applies a byte-checked transform after the recurrent
checkpoint payload has been installed. The payload files and their manifest
are immutable inputs. The image installer first verifies every checkpoint
ownership dependency, applies the transform, and changes only the scheduler
entry to the transform's expected output hash. The image receipt records both
transform hashes and verifies the installed scheduler file.

## Connector interface

Instrumentation is active only when the connector sets
`request_cache_events_enabled = True`, and calls its optional method:

```python
record_request_cache_event(request, event, **fields)
```

SparkCache may enable that property with `SPARK_CONTEXT_CACHE_TRACE_REUSE=1`.
Connectors without the property receive no callbacks or dispatch metadata.
The runtime does not add Prometheus labels or public API fields. The connector
owns bounded request ledgers, aggregate counters, and opt-in request traces.

Each event includes `preemptions`, copied from `request.num_preemptions`.
Token counts refer to the original prompt and exclude generated tokens.

| Event | Fields | Authoritative boundary |
|---|---|---|
| `admitted` | `local_tokens`, `external_tokens`, `lease_attached`, `source` | Successful block allocation after local/remote reconciliation; `source` is `gpu_lease` or `prefix_lookup` |
| `restore_finalized` | `success`, `valid_prefix_tokens` | All-worker receive finalization after failure handling and the full-hit sampling-token adjustment |
| `prompt_step_completed` | `start_token`, `end_token`, `stale=False` | Model output that passes invalid-load, abort, stale-output, and attempt-generation checks |
| `preempted` | Common fields only | Request counters reset and preemption generation incremented |
| `finished` | `status` as the request-status enum name | Terminal request cleanup, before the connector's finish hook |

The admission event describes reuse accepted for allocation, not completed
inference. External reuse remains provisional until receive finalization.
The connector must reconcile a failed restore against `valid_prefix_tokens`;
a failure does not increment the preemption generation. A replacement lookup
can therefore admit a prefix in the same generation. An ordinary resume after
an asynchronous load does not emit another admission event. Lease attribution
survives deferred block allocation and is emitted once allocation succeeds;
its saved generation cannot be reused after preemption.

A full external prompt restore can materialize every prompt token while the
scheduler recomputes its final token to produce sampling logits. Restored state
span and external prompt tokens reused are separate quantities. A resident GPU
lease is local reuse, even when another request originally populated that
lease through persistent restoration.

## Completed prompt work

The scheduler snapshots each dispatched token range before advancing request
counters. Each range is clipped to the original prompt length and retains its
preemption generation. Matching accepted output emits that saved range once;
subsequent scheduling or counter resets cannot change its endpoints.

A completed decode step can emit an empty prompt range. This allows the
connector to commit pending prefix reuse after a resumed request completes
inference without additional prompt computation. It contributes zero prompt
compute tokens. Stale, failed, or aborted output earns no completed-work
credit. These counters do not measure discarded GPU execution or kernel time.

Cumulative prompt work and reuse across preemption attempts can exceed the
original prompt length. Do not derive actual computation by subtracting cache
offers from the prompt length. Request completion and missing-event handling
remain connector responsibilities.

## Offline validation

```bash
python -m pytest runtime/glm53-spark-mtp3-mesh/performance/attribution -q
```

Tests execute transformed scheduler methods and admission/output branches.
They cover partial local tails, lease adoption, prompt clipping, asynchronous
counter advancement, duplicate completion, failed restores, stale output,
preemption, terminal cleanup, and connectors without instrumentation. They do
not qualify CUDA execution, cache restoration, or hardware serving behavior.
