# SparkCache

Persistent NVMe context cache for GLM-5.2 on the switchless four-Spark
DCP4 ring. SparkCache restores a served context after a full runtime
restart in **~1.4 s instead of ~30 s** (15-24x), so a returning
conversation skips re-prefill.

**SparkCache is not the LMCache project.** It is an original
implementation of vLLM's KV-Connector-V1 interface — the same standard
plug-in point LMCache uses — with a design specific to this hardware.

## The one idea

DCP4 already splits every context across the four Sparks: rank *r* holds
tokens where `position % 4 == r`. SparkCache simply writes each rank's own
shard to that rank's own NVMe. A restore is four parallel local reads,
zero network traffic — the data never has to be gathered onto one node.

At ~32 KB/token a 32K context is ~1 GB total but only ~262 MB per rank;
each drive reads that in ~18 ms. The restore cost is dominated by
verification and GPU write, not disk.

## Guarantees (fail-closed)

- Content-addressed chunks, per-record SHA-256, identity pinning
  model/quant/TP/DCP/shard-rank/chunk geometry. An entry can only restore
  into the exact configuration that wrote it.
- **Quorum admission:** each rank reports the digests it has *verified* it
  can serve; the scheduler offers a restore only when all four ranks
  confirm. A rank whose integrity sweep finds damage stops confirming, so
  a corrupted entry is silently withdrawn from service.
- **Integrity sweep** at worker startup and after every store verifies
  each held entry off the request path.
- On a load failure the request finishes cleanly (no wrong output ever
  served) and the entry self-heals: it is invalidated, its corrupt chunks
  purged, and the next identical request re-prefills and republishes.
- No physical-slot coordinates, block tables, CUDA pointers, or transport
  sequence numbers are ever persisted — only logical, portable records.

## Files

| File | Role |
|---|---|
| `spark_context_cache_connector.py` | the KV-Connector-V1 connector (store, restore, quorum, sweep) |
| `spark_context_cache_codec.py` | pure DCP shard math + record packing (no vllm/torch) |
| `spark_context_cache_store.py` | loader shim for the storage engine |
| `../../experiments/persistent_context_cache/cache_manifest.py` | fail-closed content-addressed manifest engine |

## Enabling

Set `SPARK_CONTEXT_CACHE_ENABLE=1` and pass a `--kv-transfer-config` that
names `spark_context_cache_connector`; the serve script does this behind
`SPARK_CONTEXT_CACHE_ENABLE`. Rank-local store lives at
`/cache/context`. Without the flag, serving is byte-identical to the
no-cache runtime.

## Measured (2026-07-28, live, four DGX Sparks)

- Store (fresh prefill): 32.9 s, committed on all four ranks
- Restore after full restart: 2.11 s cold, 1.34-1.42 s warm (15-24x)
- Concurrency stress: 16 mixed requests, zero failures, cached ~10x faster
  than novel prefills

## Known limit

An adversarially-timed on-disk corruption landing in the window between an
integrity sweep and the next request costs one failed request before the
entry retires and re-prefills (no wrong output is served). Closing that
same-request case needs pre-admission re-verification and is the next
design item; the runtime's own `recompute` fallback path is unsound under
DCP4 + async scheduling and is not used.
