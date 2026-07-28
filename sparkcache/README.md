# SparkCache

Persistent NVMe context cache for vLLM with decode-context-parallel (DCP)
sharding. SparkCache restores a served context after a full runtime restart
in **~1.4 s instead of ~30 s** (15-24x measured), so a returning
conversation skips re-prefill entirely.

Built for and measured on SparkRing's switchless four-node DGX Spark ring —
but the cache itself never touches the network, so **it works identically
on ordinary switched clusters**.

**SparkCache is not the LMCache project.** It is an original implementation
of vLLM's KV-Connector-V1 interface — the same standard plug-in point
LMCache uses — with a design specific to DCP-sharded serving.

## Dependency surface

- **vLLM V1** with the KV-Connector-V1 interface (`KVConnectorBase_V1`)
- **Decode context parallelism** (`--decode-context-parallel-size N`)
- A local filesystem path per rank (NVMe strongly recommended)
- **Any interconnect — or none.** No RDMA, no switch topology, no network
  assumptions: the cache does zero cross-node traffic by construction.

## The one idea

DCP already splits every context across ranks: rank *r* holds the KV for
token positions where `position % N == r`. SparkCache simply writes each
rank's own shard to that rank's own disk. A restore is N parallel local
reads — the KV never has to be gathered anywhere.

At ~32 KB/token (GLM-5.2 with MLA + indexer + MTP draft state) a 32K
context is ~1 GB total but only ~262 MB per rank on a 4-way ring; each
drive reads that in ~18 ms. Restore cost is dominated by verification and
GPU writes, not disk. Contrast with single-store designs, which must move
the full context across the network to one place and read it back through
one drive.

## Guarantees (fail-closed)

- Content-addressed chunks, per-record SHA-256, identity pinning
  model/quant/TP/DCP/shard-rank/chunk geometry. An entry can only restore
  into the exact configuration that wrote it.
- **Quorum admission:** each rank reports the digests it has *verified* it
  can serve; the scheduler offers a restore only when every rank confirms.
  A rank whose integrity sweep finds damage stops confirming, so a
  corrupted entry is silently withdrawn and the request re-prefills.
- **Integrity sweep** at worker startup and after every store verifies
  each held entry off the request path.
- On a load failure the request finishes cleanly (no wrong output ever
  served) and the entry self-heals: invalidated, corrupt chunks purged,
  next identical request re-prefills and republishes.
- No physical-slot coordinates, block tables, CUDA pointers, or transport
  sequence numbers are ever persisted — only logical, portable records.

## Files

| File | Role |
|---|---|
| `spark_context_cache_connector.py` | the KV-Connector-V1 connector (store, restore, quorum, sweep) |
| `spark_context_cache_codec.py` | pure DCP shard math + record packing (no vllm/torch imports) |
| `spark_context_cache_store.py` | fail-closed loader shim for the storage engine |
| `persistent_context_cache/cache_manifest.py` | content-addressed manifest engine |
| `test_spark_context_cache_connector.py` | GPU-free connector suite (vLLM stubbed, CPU torch) |
| `persistent_context_cache/test_cache_manifest.py` | storage-engine suite |

Run the tests from this directory:

```bash
python -m pytest test_spark_context_cache_connector.py persistent_context_cache/test_cache_manifest.py -q
```

## Enabling

Set `SPARK_CONTEXT_CACHE_ENABLE=1` and pass a `--kv-transfer-config` that
names `spark_context_cache_connector` (the module must be importable, e.g.
this directory on `PYTHONPATH`). Rank-local store root defaults to
`/cache/context` (`SPARK_CONTEXT_CACHE_ROOT`). Without the flag, serving
is byte-identical to a no-cache runtime.

## Adopting on your (switched or switchless) cluster

The mechanism is general; three things are currently tuned to SparkRing's
GLM-5.2 deployment and are the porting surface:

1. **Identity strings** (`quantization_layout`, `rope_layout`, checkpoint
   ids): set these to describe *your* model/KV layout. They exist so an
   entry can never restore into a mismatched configuration — pick values
   that change whenever your KV bytes would.
2. **Layer classification** (`spark_context_cache_codec.classify_layer`):
   maps vLLM cache-layer names to record kinds (target KV, sparse-indexer
   state, speculative-draft KV). For a vanilla attention model this
   collapses to a single record kind; extend it if your model registers
   extra cache layers.
3. **Serve wiring**: how your launcher passes `--kv-transfer-config` and
   the env flags.

Nothing in the store/restore path assumes the SparkRing transport, RDMA,
or any particular interconnect.

## Measured (2026-07-28, live, four DGX Sparks, DCP4)

- Store (fresh prefill): 32.9 s, committed on all four ranks
- Restore after full restart: 2.11 s cold, 1.34-1.42 s warm (15-24x)
- Concurrency stress: 16 mixed requests, zero failures, cached ~10x faster
  than novel prefills

## Known limits

- The restore currently runs synchronously on the worker's main thread
  (multi-second for very large contexts). If your runtime enforces tight
  deadlines on in-flight async collectives, size them to cover your
  worst-case restore, or the stall can be misread as a hang. An
  asynchronous restore (request parks while a background thread loads and
  verifies; main-thread cost drops to sub-millisecond) is implemented and
  in validation, not yet landed here.
- An adversarially-timed on-disk corruption landing between an integrity
  sweep and the next request costs one failed request before the entry
  retires and re-prefills (no wrong output is served). Closing that
  same-request window needs pre-admission re-verification; the runtime's
  own `recompute` fallback proved unsound under DCP + async scheduling on
  this deployment and is not used.

## Status

Research pre-release, same caveats as the rest of SparkRing: no stable
API, flags and layouts may change without notice. Licensed Apache-2.0 with
the repository.
