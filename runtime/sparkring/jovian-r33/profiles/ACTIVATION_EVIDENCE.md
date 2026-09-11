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

Status: **implemented, GPU qualification pending**. The vLLM source composition
identified by tree `f85a62b998b80ef3ae14799115d1b47398ede46e` carries sparse
checkpoint plans through scheduling, allocation, worker metadata and Kimi GDN
execution. It preserves R33's packed FlashKDA metadata path. The B12X path uses
fixed four-column checkpoint metadata and requires a B12X package that supports
four transactional checkpoint exports.

`VLLM_B12X_KDA_PREFILL_COALESCING=1` admits the path only for GLM5Next BF16,
model runner V2, TP4 with DCP1, DCP2 or DCP4, PP1/DP1, an 8,192-token scheduler
budget, aligned recurrent caching with retention interval zero, and static MTP3
or no speculation. Unsupported configurations fail during initialization.
Cache hits, resumed requests, preempted requests, asynchronous external loads,
multimodal requests and concurrent service use the ordinary scheduler path.

Set `VLLM_B12X_KDA_PREFILL_COALESCING_LOG_LIMIT` to a positive integer on every
rank. A qualifying activation must capture a bounded record with the request
identifier, an 8,192-token span and the checkpoint token positions selected on
every rank. The default value is zero and produces no per-request records.
Environment values alone do not prove execution. The TP4 template sets this
limit to four; TP2 disables both coalescing and its diagnostics. The SparkCache
overlay inherits TP4 admission settings, but external-load requests still use
the ordinary scheduler path.

The source patch and complete changed-file manifest are packaged as
`vllm-source-composition.patch` and
`vllm-source-composition-manifest.json`. Their hashes must match the image's
source lock. The prefix-hit metadata fix in tree
`f85a62b998b80ef3ae14799115d1b47398ede46e` has package verification; model
qualification is pending. A bounded TP4 run of tree `4f1813fc` demonstrated
mHC and coalescing execution but failed on a prefix-cache-hit request. That
result proves execution only for that source tree and does not establish
correctness, long-prefill liveness, throughput, or qualification of the fixed
composition. Require matching native, wheel and activation receipts from the
fixed image, including completed prefix-cache-hit requests.

## TP2 activation evidence

TP2 is unqualified. Its template sets
`VLLM_B12X_KDA_PREFILL_COALESCING=0`; the activation receipt must report integer
zero for `continuation_coalesced_groups` on both ranks. Retain the initialized
configuration and bounded request diagnostics that establish this value.

The committed mHC source admits TP2/DCP1 at a 4,096- or 8,192-token prefill
ceiling. Extract each rank's `mhc_prefill_rows` from the `rows` field of its
`GLM_MHC_PREFILL` record and compare it with the initialized scheduler ceiling.
Both ranks must report the same ceiling. Require a positive sharded-call count
and `mhc_owner_rows` containing half that ceiling: 2,048 or 4,096, respectively.
The 8,192-row TP4 evidence in the table cannot be reused as TP2 evidence.
Retain the same image, NCCL, graph, loader, MTP and correctness evidence, using
RoCEnante collective activity for TP2 instead of SIRCL snapshots. Passing the
offline validator with synthetic test fixtures does not qualify model serving.

## Collector behavior

For a bounded restore-failure test, the managed site may specify
`cache_diagnostics` with `namespace` set to an isolated copy's safe name,
`access_mode` set to `restore-only`, and integer `trace_reuse` set to `1`.
This option requires the R33 `tp4-dcp1-sparkcache` profile and image receipt.
The renderer disables asynchronous capture and startup cache clearing, and
enables reuse traces on every rank while preserving the pinned libraries and
buffer limits. It uses an explicit empty `SPARKCACHE_CLEAR_ONCE`; a word such
as `none` would be a clear token, not a disabled setting. The namespace
must differ from the image's default. Save these settings in the site before
rendering so the installer can reproduce them; do not edit rank environments.
Restore-only mode disables publication through the connector; it does not make
the host cache directory a read-only filesystem. A diagnostic run cannot supply
the capture evidence required for full cache qualification.

A collector must retain every source artifact beside the generated activation
receipt and include its SHA-256 digest. It must fail when a source is missing,
ambiguous, stale, or inconsistent across ranks. It must reject environment-only
claims, aggregate counters without before-and-after snapshots, and logs that do
not identify the worker rank. The collector must write the receipt only after
all checks pass; a partial run remains a set of unqualified source artifacts.
