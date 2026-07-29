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

## The Concept

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
| `streaming/` | bounded write-behind prefill journal, native gather ring, block leases, idle progress worker, and preemption contract |
| `replication/` | bounded transactional buddy-replication protocol for optional 10 GbE diagonals |
| `runtime_patches/` | fail-closed attestation for the pinned vLLM delayed-free seam |
| `../docs/SPARKCACHE_STREAMING_SNAPSHOTS.md` | streaming-snapshot design, gates, and rollout plan |
| `test_spark_context_cache_connector.py` | GPU-free connector suite (vLLM stubbed, CPU torch) |
| `persistent_context_cache/test_cache_manifest.py` | storage-engine suite |

Run the tests from the repository root:

```bash
python -m pytest \
  sparkcache/test_spark_context_cache_connector.py \
  sparkcache/persistent_context_cache/test_cache_manifest.py \
  sparkcache/test_spark_context_cache_native_placement.py \
  sparkcache/test_spark_context_cache_native_restore.py \
  sparkcache/native/tests/test_ctypes_binding.py \
  sparkcache/native/tests/test_layout_contract.py -q
```

The command above is the focused storage/restore gate. The complete current
SparkCache suite (`python -m pytest sparkcache -q`) passed **407 tests with
1 skipped** on the GPU-free development host on 2026-07-29.

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

## Streaming write-behind status (v50/v51)

The current streaming path journals each completed 256-token region during
prefill. Sixteen chunks form one bounded gather batch. A two-slot mapped-host
ring gathers target CKV and sparse-indexer/MTP-colocated records, short block
leases prevent allocator reuse until the CUDA fence completes, and a
transactional writer publishes immutable objects before the manifest-last
visibility edge. Ring or writer pressure aborts only the optional cache
transaction; it must not delay inference.

A four-Spark v50 probe exposed one missing lifecycle edge. A 32,769-token
request (`max_tokens=1`) wrote all 128 rank-local chunk objects, but after the
model stopped producing work there were no more worker callbacks to call
`poll()`. No rank published a manifest while the system sat idle. A later
four-token request supplied the next callback and all four ranks committed
immediately, reporting final tails of **278.753-278.769 seconds**. This was
not a disk-throughput result: progress itself was incorrectly coupled to a
subsequent inference request.

The v51 candidate adds an event-armed worker progress thread. It:

- sleeps indefinitely with zero polls when no asynchronous work exists;
- wakes after a submitted offer and advances the complete adapter poll path
  while an inflight batch or committed/aborted terminal handoff remains;
- establishes the rank's CUDA device once in the progress thread;
- serializes foreground and background adapter state through one reentrant
  lock, including lost-wake clear/recheck;
- drains manifest publication, digest advertisement, terminal status, and
  optional timing without requiring another request; and
- stops in two phases before runtime/writer destruction.

Expected writer failures remain fail-open for serving and abort only cache
publication. Escaped CUDA/native ownership failures remain sticky and fatal
because silently continuing could allow a leased KV block to be reused.

**v51 live no-nudge validation passed on 2026-07-29.** One 32,769-token
completion with `max_tokens=1` (request
`cmpl-a691915a48161692-0-a89689cb`) published digest
`e298d60abc4d5f01f317a14e095c39231298618f0f573e3aa2da741423bc7dee`
without a follow-up inference request. All four ranks produced 128 chunks,
one manifest, one committed terminal record, and one timing record; payload
SHA verification was clean. Final tails were 1514.582, 1508.082, 1517.160,
and 1526.855 ms for ranks 0-3. The dominant fence took 1283.9-1287.9 ms,
the writer about 208-223 ms, and manifest publication about 12-18 ms. Every
rank ended with zero active contexts, leases, and tickets; background
progress remained alive with `error=null`.

The first workstation gate report falsely timed out because its clock was
about 0.44 seconds ahead of rank 0. Its subsecond Docker `--since` boundary
therefore excluded the request POST and worker records. Corrected observer
evidence is sealed in
`deliverables/evidence/sparkcache-v51-no-nudge-verified.json` in the private
validation tree (SHA-256
`0f8433ccae345956a0ee109448b6c6d79c172d798c6453be95088bca4e612005`);
the false timeout was an observer-window error, not a runtime failure.

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
| **Current source + live v50 evidence** | Metadata-only discovery plus transactional 256-token streaming journal, bounded block leases, and checksum-pinned two-slot mapped-host gather ring |
| **Current source + live v51 no-nudge pass** | Event-armed idle progress worker that completes ring/writer/manifest/terminal work without a later inference callback |
| **Implemented offline; carrier pending** | Transactional, credit-bounded buddy replication suitable for the two diagonal 10 GbE links |
| **Planned** | Prefix-aware partial restore and chunk reuse when conversations grow |

## Known limits

- This directory contains the **v51 streaming candidate source**. v50 proved
  the live gather/journal path but also proved that final publication could
  stall indefinitely when inference became idle. The v51 progress-worker fix
  has passed both offline tests and the four-Spark no-nudge validation. The
  remaining promotion work concerns restore equivalence, interference, and
  performance rather than idle publication progress.
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

Research pre-release, same caveats as the rest of SparkRing: no stable API,
flags and layouts may change without notice. The Python/native source and
GPU-free tests are reproducible here. v51 has published a fresh four-rank
manifest after a one-token completion with no second inference request,
passing its decisive idle-progress gate. Licensed Apache-2.0 with the
repository.
