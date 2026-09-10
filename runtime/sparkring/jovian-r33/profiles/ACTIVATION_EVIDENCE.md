# R33 activation evidence

Status: **research-only**. An activation receipt is valid only when every
reported value is derived from an immutable input listed below. Configuration
values prove admission only; they do not prove that a runtime path executed.

| Receipt claim | Required evidence | Deterministic extraction |
|---|---|---|
| Image identity | `docker inspect` captured after container start | Compare `.Image` with `image.image_id` on every rank. |
| Loaded NCCL 2.31.2 | Worker process maps and the image artifact lock | Resolve every mapped `libnccl.so*`, hash the resolved file, and compare it with the locked `libnccl.so.2.31.2` artifact. |
| Both host PCIe domains carried NCCL | `NET/IB RouteFinal` records plus rank-local sysfs PCI addresses | Map each distinct `hca=` value to `/sys/class/infiniband/<hca>/device`; require successful routes through HCAs whose PCI addresses use both host domains. `NCCL_IB_HCA` alone is insufficient. |
| CUDA graph sizes | Engine configuration log plus completed graph-capture log | Parse `cudagraph_capture_sizes` from the initialized engine and require capture completion. The ordered set must equal the selected profile. |
| InstantTensor | Completed `Loading safetensors using InstantTensor loader` progress record | Require the terminal progress record for each worker. `LOAD_FORMAT=instanttensor` alone is insufficient. |
| MTP3 | Initialized engine configuration and decode result metadata | Require `method='mtp'`, `num_spec_tokens=3`, and at least one completed decode request with observed accepted draft tokens. |
| mHC token sharding | `GLM_MHC_PREFILL` diagnostics | Launch with `VLLM_GLM53_MHC_PREFILL_DIAGNOSTICS=1`; require `rows=8192 owner_rows=2048` on every TP4 rank. |
| SIRCL collectives | Atomic graph status snapshots before and after requests | Sum positive submitted-sequence or event-count deltas from `SPARK_TP4_GRAPH_STATUS_PATH`. Require zero native health failures. |
| One-million-token admission and KV capacity | Initialized engine and KV-cache log records | Require `max_seq_len=1048576`, parse `GPU KV cache size`, and retain the complete source lines. |
| Output correctness | Request harness receipt | Record request input hashes, response hashes, completion state, and the oracle or invariant used by the harness. |
| Long-prefill liveness | Bounded request harness receipt and logs | Require at least two completed prompts of 32,768 tokens or more, no `sample_tokens` timeout, and no fatal engine error during the bounded interval. |
| SparkCache disabled | Container command, environment, and imports | Require `SPARKCACHE_ENABLED=0`, no KV-transfer connector argument, and no SparkCache capture or restore log record. |

## Continuation-prefill coalescing

The locked R33 source composition does not implement B12X continuation-prefill
coalescing. Its Kimi GDN cache specification reserves one B12X prefill
checkpoint, and its B12X API accepts one checkpoint destination per request.
vLLM commit `8133d8210f5c8f71389add06e4e0033ee7f13b71` removes redundant
scheduler split points only when every recurrent group already exports a
multi-checkpoint set. Capacity one makes that branch ineligible. The source
does not read `VLLM_B12X_KDA_PREFILL_COALESCING`.

The activation verifier field `continuation_coalesced_groups` therefore has no
valid producer in this source composition. Do not populate it from the
environment or infer it from aggregate throughput. A compatible source port
requires the four-checkpoint B12X export contract, the vLLM scheduler and
worker ownership contract, and one of these diagnostics:

1. Add a bounded diagnostic counter at the scheduler branch that removes an
   exported checkpoint from `reuse_stops`, then persist its value in a status
   file.
2. Capture the scheduler's chosen chunk boundaries and exported checkpoint
   positions in a bounded diagnostic record, then require an 8,192-token
   continuation chunk whose internal checkpoint avoided an extra split.

The B12X and vLLM changes alter both source identities and require a rebuilt
image receipt. Until that image exists, an activation can qualify the model,
transport, mHC path, and liveness, but it cannot claim that
continuation-prefill coalescing is implemented or executed.

## Collector behavior

A collector must retain every source artifact beside the generated activation
receipt and include its SHA-256 digest. It must fail when a source is missing,
ambiguous, stale, or inconsistent across ranks. It must reject environment-only
claims, aggregate counters without before-and-after snapshots, and logs that do
not identify the worker rank. The collector must write the receipt only after
all checks pass; a partial run remains a set of unqualified source artifacts.
