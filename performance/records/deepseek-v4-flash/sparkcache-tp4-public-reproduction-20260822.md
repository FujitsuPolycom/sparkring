# DeepSeek-V4 SparkCache TP4 public-artifact reproduction

Lane: **public-functional**. Status: **qualified**. Evidence scope:
**live-validated reproduction of the qualified artifact contract**. Hardware:
four directly cabled NVIDIA DGX
Spark systems in a cycle, TP4/DCP1. Evidence scope: one fresh-root
cold-store, coordinated engine restart, external restore, semantic equality,
and post-restore canary. This record does not qualify another checkpoint,
image, SparkCache build, scheduler budget, topology, or performance claim.

## Conditions

| Input | Verified identity |
|---|---|
| Serving image | manifest `sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028`; local image ID `sha256:50036224411e5ef04d651730c56d794111991b37981ec76ca81b66ea7d35dae7` on all four ranks |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`; complete-manifest SHA-256 `bd6b0117ca28997acc9f22022814bb6bc50b5c3e1bc466d148b1d45067fe714f`; 74 files; 166,898,661,074 bytes on every rank |
| SparkCache wheel | `sparkcache-0.1.0a1-py3-none-any.whl`; SHA-256 `87c17d8dab5052f5a7833349dc9b99b76a3b6531ca6f0d3deff812f724fecdcc` |
| SparkCache source | tag `v0.1.0a1`; commit `5344192526d328b5cda3417c857b7ffb048fca8a`; deployable source-tree SHA-256 `342236206e35ad648fa9603c7259a6dd894da7a285be855b3dabbee73284f631` |
| vLLM overlays | scheduler SHA-256 `2f34aa9d65a495a86d814c90f654fbe1ff754cfdbecd204b98d513652ca3e06d`; vLLM config SHA-256 `71c4f9e622dd8b3d665f2a2b5fb932206516ddb82873ff89283c63aa80696005` |
| Serving contract | 524,288-token model limit; 32 GiB KV per rank; block size 256; 4,096-token scheduler budget; DSpark K5; cache profile `deepseek-v4-fp8-hma` |

The tagged launcher rejected a cache-disabled source inspection containing the
base quickstart's general `/cache` bind. The accepted source inspection used
the composition serving values, the read-only model bind, and no general
runtime-cache bind. The launcher supplied fresh, disjoint SparkCache and JIT
host directories.

## Measurement

Raw-record location: unavailable in a public artifact; exact responses,
inspections, logs, manifests, and cache inventories remain in the private
validation workspace. The gate ran once, so no repeatability distribution or
uncertainty estimate is available.

The miss gate generated a deterministic 73,774-token prompt and required the
exact response `SPARKCACHE_OK:9540`. Every physical rank committed the same
73,728-token reusable span with digest prefix `c17e6fbaefea` into a fresh
rank-local root.

| Rank | Snapshot | Background commit |
|---:|---:|---:|
| 0 | 346.7 ms | 3,787.0 ms |
| 1 | 433.9 ms | 3,641.0 ms |
| 2 | 367.9 ms | 3,888.5 ms |
| 3 | 421.5 ms | 3,807.8 ms |

All four containers were stopped and started together without changing or
removing the fresh cache roots. Startup discovery reported one offered
manifest and zero rejected manifests per rank. The tagged hit gate sent its
current-generation inventory canary before the long request.

| Rank | Verified restore |
|---:|---:|
| 0 | 452.7 ms |
| 1 | 534.8 ms |
| 2 | 433.3 ms |
| 3 | 495.4 ms |

## Result

The scheduler reported 73,814 external-cache queried tokens, 73,728
external-cache hit tokens, and zero native-prefix hit tokens. The repeated
request returned `SPARKCACHE_OK:9540`; the fresh post-restore canary returned
`SPARKCACHE_CANARY_OK`. Four current-generation rank reports and four
compatible digests were present, and rank-local capacity checks passed.

## Conclusion

The exact public image, checkpoint revision and complete manifest,
SparkCache release wheel, source tag, overlay chain, and TP4/DCP1 composition
restored durable prefix state after a coordinated restart on this four-Spark
system.

## Limitations

- The tagged launcher verified the source-bound overlay receipt before Docker
  creation but did not mount the receipt into the containers. Overlay
  attestation for this run is external inspection evidence rather than a
  self-contained in-container receipt.
- Raw host inspections, logs, and cache trees remain outside the public
  repository. This file retains sanitized identities and measurements only.
- One fresh-root cycle is not a production-reliability or performance matrix.
- Streaming snapshots, native direct restore, DeepSeek DCP2/DCP4, and another
  scheduler budget remain outside this evidence scope.

Raw-record location: unavailable in a public artifact; exact responses,
inspections, logs, manifests, and cache inventories remain in the private
validation workspace. The gate ran once, so no repeatability distribution or
uncertainty estimate is available.
