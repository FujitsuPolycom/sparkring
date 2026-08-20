# LMCache key-value reuse for DeepSeek-V4-Flash-0731 on the four-Spark ring

Status: **unsupported.** The cache package installed in the deployment
image cannot map this checkpoint's key-value cache. Serving this model
without external key-value reuse is unaffected and is specified below.

## Why the installed package cannot map this checkpoint

Conditions: `lmcache 0.5.2+glm52dcp4.1` in the deployment image, one
multiprocess server per rank inside the engine container, four-rank
tensor-parallel serving of `deepseek-ai/DeepSeek-V4-Flash-0731` with
`--kv-cache-dtype fp8_ds_mla`.

Measurement: the server registers 170 layers and reports its kernel
group inventory. Five engine groups are registered, and group 0 carries
three distinct geometries:

| Group | Layers | Tokens per block | Head size | Element size |
|---|---:|---:|---:|---:|
| 0 | 21 | 64 | 584 | 1 |
| 0 | 21 | 64 | 132 | 1 |
| 0 | 20 | 2 | 584 | 1 |
| 1, 2 | 23 each | 64 | 584 | 1 |
| 3 | 21 | 4 | 512 and 2048 | 4 |
| 4 | 20 | 8 | 1024 | 4 |

Every retrieve aborts in the device-side read path with
`Size mismatch: memory_obj nbytes=985664, gpu_buffer nbytes=15644672`.
The worker marks 21,525 blocks invalid and the engine recomputes the
prompt in full.

Result: the package contains no reference to `deepseek_v4`, and its read
path (`lmcache/v1/gpu_connector/gpu_ops.py`) sizes one device staging
buffer per transfer with no notion of kernel groups.

Conclusion: a single buffer size cannot satisfy a group holding three
geometries. The support required for this checkpoint is absent from this
package. Upstream publishes a
[DeepSeek-V4-Flash recipe](https://docs.lmcache.ai/recipes/deepseek_v4_flash.html)
stating that multiprocess mode maps these groups without additional
configuration, so a package carrying that support is the required input.

## Behavior that is functional

The failure is confined to loading stored bytes into device memory.

- Registration: 170 layers, matching the model's heterogeneous set plus
  the speculative draft caches.
- Storage: a 52,623-token request produces 28 store events and 3.2 GB of
  filesystem-tier objects.
- Cross-restart persistence: after the engine, cache server, memory
  tier, and the engine's native prefix cache are all destroyed, a fresh
  deployment's startup scan primes 410 objects and retrieves 410 of 410
  keys from the filesystem tier, none from memory, in 659 ms.
- Load-failure recovery: the engine reschedules rather than terminating,
  reporting `Recovered from KV load failure`. This requires the
  multi-group block-identifier handling described below.

## Engine requirements for this checkpoint

These hold whether or not external key-value reuse is in use.

- **Cache dtype `fp8_ds_mla`.** The engine gates the layout on an exact
  string comparison against `fp8_ds_mla`, so `--kv-cache-dtype fp8`
  selects a generic path that declares a geometry differing from the one
  it allocates. Both values allocate identically; only the declaration
  differs, which is invisible to the engine's own kernels and incorrect
  for any external consumer.
- **Tokenizer mode `deepseek_v4`.**
- **Multi-group block identifiers in the load-failure path.**
  `KVCacheManager.get_block_ids` returns one list per key-value cache
  group. A single-list unpack terminates the engine core on the first
  invalid block. The deployed handling iterates positions across all
  groups and evicts the tail of every group when eviction is required.

Key-value capacity under a 32 GiB per-rank reservation is 4,382,668
tokens with speculative decoding at depth five and 4,581,351 tokens
without it, so speculation costs 4.5% of capacity. Capacity does not
vary with the cache dtype value.

## Validation method

An external key-value tier is functional only when a request served from
a destroyed engine satisfies all of: the request completes; its output
hash matches the store-phase hash for the same prompt; the engine's
native prefix-cache counter is zero, proving the engine's own cache
cannot account for the result; and the external hit counter is non-zero.

Counters and timings are corroboration and are not sufficient alone. Two
instrument properties make partial evidence misleading:

- `external_prefix_cache_hits_total` counts lookup matches, not completed
  transfers. A configuration that fails every transfer reports 104,960
  hits against 105,246 queries while recomputing every token.
- A replay within a live engine process is served by the engine's own
  prefix cache. Such a replay measures 36-40x faster than cold with
  byte-identical output while the external hit counter reads 0 against
  47,738 queries and the native counter reads 47,104.

## Conditions required for support

1. A cache package carrying DeepSeek-V4 kernel-group support, with the
   local GLM changes in `0.5.2+glm52dcp4.1` either shown to be
   unnecessary or reconciled with it.
2. The validation method above, satisfied in full.
3. Idle-interval verification of the heartbeat guard. In the installed
   package `if self._heartbeats is not None:` tests a dictionary for
   existence rather than contents, so the heartbeat thread does not start
   and the server reaps a live engine after roughly 150 seconds, during
   which stores continue and lookups stop. Corrected handling is
   deployed; no recorded interval is long enough to exercise it.
