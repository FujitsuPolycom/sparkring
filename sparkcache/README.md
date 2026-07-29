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
- The public compatibility target is
  `vllm-project/vllm@fcc614141e5e9ab18cb304c476f7feed2a9552e3`
  plus the two fail-closed patches in
  [`../runtime/patches/vllm`](../runtime/patches/vllm). They provide
  asynchronous-load rollback and the connector-specific safe VMM exemption.
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
- **Manifest-last publication:** immutable chunks are file-fsynced, linked,
  and directory-fsynced before the manifest is file-fsynced, linked, and
  directory-fsynced. Newly created directory entries are persisted too. A
  digest enters quorum only after every durability barrier completes.
- **Metadata-only startup discovery:** workers validate manifest identity,
  descriptor geometry, and referenced chunk existence/size without reading
  every cached payload at startup.
- **Quorum admission:** each rank reports structurally compatible digests;
  the scheduler offers a restore only when every rank confirms.
- **Restore is the payload-integrity boundary:** every selected chunk is
  read and SHA-256 verified before restored state is released.
- On a load failure the request finishes cleanly (no wrong output ever
  served) and the entry self-heals: invalidated, corrupt chunks purged,
  next identical request re-prefills and republishes.
- `sweep_integrity()` remains an explicit payload-reading diagnostic. It
  is not run after every store or during normal startup.
- No physical-slot coordinates, block tables, CUDA pointers, or transport
  sequence numbers are ever persisted — only logical, portable records.

## Files

| File | Role |
|---|---|
| `spark_context_cache_connector.py` | KV-Connector-V1 async store/restore, discovery, quorum, and diagnostics |
| `spark_context_cache_codec.py` | pure DCP shard math + record packing (no vllm/torch imports) |
| `spark_context_cache_store.py` | fail-closed loader shim for the storage engine |
| `spark_context_cache_native_placement.py` | attested native-placement adapter and parked-request state machine |
| `spark_context_cache_native_restore.py` | bounded parallel read/hash plus native placement orchestration |
| `spark_cache_native.py` | strict dependency-free ctypes ABI binding |
| `native/` | reproducible C++/CUDA native-placement source, CMake build, and ABI tests |
| `persistent_context_cache/cache_manifest.py` | content-addressed manifest engine |
| `test_spark_context_cache_connector.py` | GPU-free connector suite (vLLM stubbed, CPU torch) |
| `persistent_context_cache/test_cache_manifest.py` | storage-engine suite |

Run the tests from this directory:

```bash
python -m pytest \
  test_spark_context_cache_connector.py \
  persistent_context_cache/test_cache_manifest.py \
  test_spark_context_cache_native_placement.py \
  test_spark_context_cache_native_restore.py \
  native/tests/test_ctypes_binding.py \
  native/tests/test_layout_contract.py -q
```

The current public package passes **106 GPU-free tests**.

## Enabling

The `--kv-transfer-config` below is the enable switch. The public connector
does **not** consume a second `SPARK_CONTEXT_CACHE_ENABLE` flag: omit the
complete `--kv-transfer-config` argument to disable SparkCache. The module
must be importable (for example, put this directory on `PYTHONPATH`).

The pinned vLLM factory also requires
`--disable-hybrid-kv-cache-manager`, because this connector deliberately
does not advertise HMA support:

```bash
vllm serve /models/glm-5.2 \
  --disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{
    "kv_connector": "SparkContextCacheConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "spark_context_cache_connector",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
      "spark_cache_root": "/cache/context",
      "spark_cache_target_checkpoint_sha256": "<64 lowercase hex characters>",
      "spark_cache_draft_policy": "colocated_target",
      "spark_cache_store": true,
      "spark_cache_restore": true
    }
  }'
```

The target digest is mandatory and must identify the actual immutable
checkpoint contents (for example, hash an artifact manifest that pins every
weight shard). Mutable paths and tags such as `latest` are rejected.
Replacing weights in place requires a new digest and therefore selects a new
cache namespace.

For a separately loaded drafter, set
`"spark_cache_draft_policy": "separate"` and provide
`"spark_cache_draft_checkpoint_sha256": "<64 lowercase hex characters>"`.
For `colocated_target`, omit the draft digest: it is derived from the target
digest, and a conflicting value is rejected.

The rank-local store root defaults to `/cache/context`
(`SPARK_CONTEXT_CACHE_ROOT`). `spark_cache_store=false` and
`spark_cache_restore=false` independently suppress publication and lookup,
but the connector remains instantiated while `--kv-transfer-config` is
present.

Native direct placement is opt-in. Build `native/`, then set
`SPARK_CONTEXT_CACHE_NATIVE_RESTORE=1`,
`SPARK_CONTEXT_CACHE_NATIVE_LIBRARY` to the resulting shared library, and
`SPARK_CONTEXT_CACHE_NATIVE_LIBRARY_SHA256` to its SHA-256. The connector
fails closed on an absent library, hash mismatch, unsupported tensor
layout, or ABI mismatch.

## Adopting on your (switched or switchless) cluster

The mechanism is general; three things are currently tuned to SparkRing's
GLM-5.2 deployment and are the porting surface:

1. **Immutable checkpoint digests and layout identities**
   (`spark_cache_target_checkpoint_sha256`, optional separate-draft digest,
   `quantization_layout`, and `rope_layout`): checkpoint fields accept only
   lowercase SHA-256 values. Set them from an immutable artifact manifest,
   and change layout identities whenever equal token IDs would produce
   different persistent KV bytes.
2. **Layer classification** (`spark_context_cache_codec.classify_layer`):
   maps vLLM cache-layer names to record kinds (target KV, sparse-indexer
   state, speculative-draft KV). For a vanilla attention model this
   collapses to a single record kind; extend it if your model registers
   extra cache layers.
3. **Serve wiring**: how your launcher passes the exact
   `--kv-transfer-config` and `--disable-hybrid-kv-cache-manager` arguments.

Nothing in the store/restore path assumes the SparkRing transport, RDMA,
or any particular interconnect.

## Measured (2026-07-28, live, four DGX Sparks, DCP4)

- Store (fresh prefill): 32.9 s, committed on all four ranks
- Restore after full restart: 2.11 s cold, 1.34-1.42 s warm (15-24x)
- Concurrency stress: 16 mixed requests, zero failures, cached ~10x faster
  than novel prefills

## Feature status

| State | Capability |
|---|---|
| **Published here** | DCP-rank-local NVMe shards with zero cache network traffic |
| **Published here** | Target CKV, sparse-indexer state, and MTP draft-KV records |
| **Published here** | Content-addressed immutable chunks and manifest-last durable publication |
| **Published here** | Per-record SHA-256 plus model/quantization/TP/DCP/rank/geometry identity pinning |
| **Published here** | All-rank quorum admission, corruption withdrawal, clean miss/re-prefill, and self-healing invalidation |
| **Published here** | Portable logical records: no CUDA pointers, physical slots, block tables, or transport sequence numbers on disk |
| **Published + live v47** | Asynchronous restore: only the restoring request parks while background load and verification run |
| **Published + live v47** | Optional checksum-attested native direct placement with bounded parallel reads/hashing |
| **Published + live v47 candidate** | Ownership-safe snapshot followed by asynchronous pack/hash/write/manifest commit |
| **Published v48-next; not deployed** | Metadata-only startup discovery; payload hashing moves to the restore integrity boundary |
| **Planned** | Prefix-aware partial restore, chunk reuse when conversations grow, background-I/O preemption, and a leased/staged snapshot path |

## Known limits

- This directory now contains **v48-next source**. The active four-Spark
  deployment remains the checksum-pinned v47 bundle; metadata-only startup
  discovery is the only core connector change beyond that live bundle.
- The live reference runtime is
  `0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea.fi25dd814.cu132.20260626`.
  It contains a larger sparse-MLA/GB10 overlay that is not published here.
  The two public SparkCache patches reproduce their required semantics against
  official upstream at the pinned commit; byte-identical reproduction of the
  complete reference runtime is not claimed.
- A live 393K-token v47 store moved packing, hashing, NVMe writes, and
  manifest publication into the background, but the mandatory
  ownership-safe GPU-to-CPU snapshot still took **3.63-4.16 seconds per
  rank**. Background commit took **11.23-15.55 seconds**.
- That live commit-overlap probe served an unrelated decode correctly, but
  missed the strict interference budget: TTFT regressed by 1.30 s and
  post-first-token decode duration by 1.78 s. There was no full engine
  freeze and no completion-time integrity sweep, but activity-aware I/O
  preemption is still needed.
- A separate no-reload, carrier-first 32K probe isolated the foreground
  snapshot itself. The four-rank snapshot union was **1.164 s**; an unrelated
  1,023-total-token decode emitted both before and after it, zero requests
  waited, and API health remained 200. Its maximum inter-event gap during the
  snapshot (6.238 s) did not exceed the simultaneous-prefill gap immediately
  beforehand (6.874 s). Because no stream event landed inside the 1.164-second
  window, this resolves no *additional* snapshot stall above prefill
  interference; it does not claim a literally pause-free snapshot.
- Metadata-only discovery intentionally does not read same-size payload
  corruption at startup. Restore detects it before releasing state,
  withdraws the entry, and cleanly re-prefills.
- Matching remains exact full-prefix/full-span. Longest-stored-prefix
  restore for a grown conversation is designed but not integrated.
- The cache has no integrated capacity policy, TTL/LRU, or orphan-chunk
  collector yet.

## Status

Research pre-release, same caveats as the rest of SparkRing: no stable
API, flags and layouts may change without notice. The Python/native source
and GPU-free tests are reproducible here; v48-next still needs a sealed
four-node live gate before replacing v47. Licensed Apache-2.0 with the
repository.
