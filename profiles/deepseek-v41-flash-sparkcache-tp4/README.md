# DeepSeek-V4.1-Flash with SparkCache on four Sparks

This profile serves `deepseek-ai/DeepSeek-V4.1-Flash`, published by DeepSeek,
at revision `dba1be0a40aa45a94ad051997016db3960a90277`. Status:
**Experimental** (`research-only`). The default image, release 2026.10.5's,
admits this profile
([release record](../../runtime/releases/dev-20261016-sparkcache-4a17f1f5198e-status038/README.md)).
On that image it installed on four Sparks, served requests and restored a
saved session after a restart; on the image composition's recheck build it
recovered from a corrupted rank's cache, re-saving the damaged entries
([record](../../performance/records/images/dev-20261016-sparkcache-4a17f1f5198e-status038-sparkcache-recovery-ring8-20261016.md)).
No performance is measured. [HARDWARE.md](HARDWARE.md) states the gates that
remain open for serving qualification.

The [configuration](config.json) keeps every serving setting of
`deepseek-v41-flash-tp4`, including TP4/DCP1, DSpark five-token speculation,
Engram tables on disk, 1,048,576-token context, 16 sequences and GPU memory
utilization 0.83. The added connector uses `deepseek-v41-b12x-hma`, embedded
draft policy `colocated_target`, and checkpoint identity
`0b447c086ca74be5e7ed9e98c424637f1f8015ee38fd1e767370415377ed401e`.
The [identity inputs and verifier](../../integrations/sparkcache/4a17f1f5198e/deploy/deepseek_v41/README.md)
bind the checkpoint files and draft settings. Verify local weights before use.

`--checkpoint 1017` serves DeepSeek's 2026-10-17 weights instead,
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170` (the same `config.json` and
weight index; every weight shard differs), as `DeepSeek-V4.1-Flash-1017-TP4`,
with the connector's checkpoint identity
`642667f82bb738e69d7ced6470fcdb4d48be378cd869e0243a1a4f9c89bf31dc` and the
cache root `/cache/deepseek-v41-flash-sparkcache-ab53c097d7ca-1bae0e108899`,
so the two revisions never share cache entries. Research-only: those weights
served under this profile's settings on four Sparks on 2026-10-17 (KV pool
11,308,736 tokens; a session restored across a restart) as a research
configuration
([record](../../performance/records/images/dev-20261016-sparkcache-4a17f1f5198e-status038-installer-fixes-ring8-20261017.md));
the entry itself has not been installed yet.

SparkCache stores full-history latent/indexer pages and encoder sliding-window
pages. It never stores request-private decoder/draft sliding windows (`replay`)
or compressor rings (`transient`). Restored spans end at least 128 tokens
before the prompt ends, aligned to 256 tokens. The engine recomputes the suffix;
Engram derives its history from request tokens and needs no persistent state.
The recorded layout has 17 groups: seven encoder windows, one full-history
latent/indexer group, one compressor ring and eight decoder/draft windows.
Decoder and draft both reserve 128 tokens. Scheduler and worker derive the
256-token boundary from the LCM of group block sizes, even when vLLM's
scheduler hash block is 128. H2 must verify their topology digests agree.

Publication uses `tail-cow-v2` with explicit `fast` CRC32 payload integrity,
minimum span 4,096 and maximum span 524,288 tokens. The independent root is
`/cache/deepseek-v41-flash-sparkcache-ab53c097d7ca-dba1be0a40aa`.
CUDA restore and asynchronous page capture are disabled. Corrupt entries
trigger recomputation. TP2 and DCP above 1 are unsupported.

## Storage and memory budget

The disk quota is 32 GiB per rank with a 28 GiB eviction low watermark. Estimated
persistent data is 890 bytes per token plus about 2.7 MB of encoder window
state before padding: roughly 0.47 GB at the maximum stored span. The quota
therefore holds roughly sixty-four such entries before metadata and padding;
this is sizing arithmetic, not a measured capacity guarantee.

The container keeps its base limit of 115,964,116,992 bytes and utilization
0.83. Two 1 GiB capture slots are configured but inactive while asynchronous
capture is off. The configured CUDA restore budget is 512 MiB per load lane
(two lanes); its arenas are also inactive with CUDA restore off. Synchronous
capture, Python restore buffers and filesystem page cache still consume GB10
unified memory, and cache I/O competes with Engram disk reads. Lowering GPU
utilization without measurements would shrink the KV pool without proving
stability. H1 records the actual pool and per-rank memory against the cache-off
profile on the same image; a reduction beyond 3%, memory pressure or an OOM
requires an explicit measured adjustment and another comparison. Enabling
either disabled path requires a separate qualification and memory budget.

CPU tests cover source pins, rendering, admission, identity, storage naming
and startup evidence. They establish no serving, CUDA, RDMA or performance
qualification. The repository distributes no weights or checkpoint license.
