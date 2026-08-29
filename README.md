# SparkRing

> Repository branches are mutable. Pin an immutable commit when reproducing a
> deployment.

  
  SparkRing is a low-latency collective transport and vLLM-based
inference-serving stack for switchless clusters of NVIDIA DGX 'Spark' systems
powered by the GB10 Grace Blackwell Superchip.

SparkRing supports GB10 pairs and four-node rings. Six-node ring work is
research-only.

Models run as tensor-parallel deployments over the direct fabric without an
external Ethernet or InfiniBand switch. [SIRCL](https://github.com/FujitsuPolycom/sparkring/blob/main/docs/SIRCL.md) provides custom RDMA collectives
where applicable, CUDA-graph command rings support repeated decode
work, and [patched NCCL](https://github.com/FujitsuPolycom/sparkring/blob/main/spark_transport/nccl/README.md) handles communication outside SIRCL's supported paths.

The repository provides launch tooling, model profiles, test evidence, and [performance data](https://github.com/FujitsuPolycom/sparkring/tree/main/performance).

## Setup

Start with ssh to node0 and enough disk space for the intended model weights. 
Use the [bootstrap guide](docs/BOOTSTRAP.md). The model-independent `sparkring cluster
init` workflow enrolls nodes, discovers management and ConnectX-7 hardware,
generates the cluster inventory, and launches Ring Doctor before any model
profile is selected.

## Resources

- [Supported models and profiles](#profiles)
- [GLM-5.3 Flash TP4 with SparkCache](docs/GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md)
- [GLM-5.3 Flash TP4 with external caching disabled](docs/GLM53_FLASH_DFLASH2_BF16_TP4_QUICKSTART.md)
- [GLM-5.3 source-built image recipe](runtime/glm53-flash/BUILD.md)
- [Benchmark results](#benchmark-results)
- [Deployment prerequisites](docs/PREREQUISITES.md) — then choose a profile quickstart below

## Cluster sizes

- **2× DGX Spark — direct pair.** Models: DeepSeek-V4-Flash-0731 and
  Qwen3.8-27B EXL3 K5/K6. DeepSeek is compatible with SparkCache
- **4× DGX Spark — physical ring.** Models: GLM-5.2 EXL3 3.5-bpw,
  GLM-5.3 Flash with BF16 DFlash2, DeepSeek-V4-Flash-0731, and Qwen3.8-27B
  EXL3 K5/K6. Both GLM families and DeepSeek are compatible with SparkCache;
  Qwen with SparkCache is unsupported.
- **6× DGX Spark — physical ring.** **Research-only.** GLM and KIMI profiles
  are not part of the supported repository surface.

## Profiles

### GLM-5.3 Flash profiles

| Profile | Deployment | Context | Seqs | Batch | KV / cache | Start here |
|---|---|---:|---:|---:|---|---|
| GLM-5.3 Flash + BF16 DFlash2 — functionally qualified | 4 Sparks · TP4/DCP1 | 512K | 32 | 8,192 | 12 GiB/rank FP8 hybrid target cache | [Quickstart](docs/GLM53_FLASH_DFLASH2_BF16_TP4_QUICKSTART.md) |
| GLM-5.3 Flash + BF16 DFlash2 + SparkCache — functionally qualified | 4 Sparks · TP4/DCP1 | 512K | 32 | 8,192 | 12 GiB/rank FP8 target KV + 48 GiB/rank SparkCache | [Quickstart](docs/GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md) |

### Other model profiles

| Profile | Deployment | Context | Seqs | Batch | KV / cache | Start here |
|---|---|---:|---:|---:|---|---|
| GLM-5.2 EXL3 3.5-bpw | 4 Sparks · TP4/DCP4 | 1M | 16 | 4,096 | NVFP4 DS-MLA · 9.25 GB/rank | [Quickstart](docs/GLM52_35BPW_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | 2 Sparks · TP2/DCP1 | 1M | 32 | 4,096 | FP8 DS-MLA · 16 GiB/rank | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | 4 Sparks · TP4/DCP1 | 1M | 32 | 4,096 | FP8 DS-MLA · 16 GiB/rank | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | 2 Sparks · TP2/DCP1 | 1M | 32 | 8,192 | FP8 | [Quickstart](docs/QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | 4 Sparks · TP4/DCP1 | 1M | 64 | 8,192 | FP8 | [Quickstart](docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md) |
| GLM-5.2 EXL3 3.5-bpw + SparkCache | 4 Sparks · TP4/DCP4 | 1M | 16 | 4,096 | NVFP4 DS-MLA + SparkCache | [SparkCache compositions](recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | 2 Sparks · TP2/DCP1 | 1M | 32 | 4,096 | FP8 DS-MLA + SparkCache | [SparkCache compositions](recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | 4 Sparks · TP4/DCP1 | 1M | 32 | 4,096 | FP8 DS-MLA + SparkCache | [SparkCache compositions](recipes/sparkcache/README.md) |

Qwen with SparkCache has no published composition recipe or live cache evidence.
The GLM-5.3 context and sequence values are configured serving limits. The
functional qualification covers startup, semantic generation, runtime health,
and an 8,192-token persistent restore; it does not exercise a 512K request,
full-limit concurrency, throughput qualification, or soak behavior.
The public BF16 DFlash2 checkpoint is licensed CC BY-NC-ND 4.0 for research
and evaluation; review its model card before use.
The qualified GLM-5.3 community images are published by immutable digest in
the two quickstarts. Both guides also link the complete source-build recipes,
SBOM workflow, source commits, applied patches, and license record.
See the
[profile registry](docs/profiles/README.md) for recipe identities and evidence
scope.

## Benchmark results

### GLM-5.3 Flash research observation

**Research-only — 16K context, single observation.** The SparkCache-enabled
profile recorded 2,371 tok/s integrated-scout prefill and 36.06 tok/s sustained
C1 decode. No A/B baseline was measured. C4 and C8 were capacity-limited, so
their throughput values are invalid and excluded.

| Profile | Prefill | C1 decode | C8 decode | Highest valid decode | Coding peak |
|---|---:|---:|---:|---:|---:|
| [GLM-5.3 Flash + BF16 DFlash2 + SparkCache · 4 Sparks](performance/records/glm53-flash/sparkcache-dflash2-bf16-tp4-16k-run1-20260829.md) | 2,371 | 36.06 | — | C1: 36.06 | — |

### Other model profiles

**Qualified — 16K context.** All values are tokens per second. Prefill uses a
cold prompt with caching disabled. Decode uses unique, cold prompt contexts at
temperature 1.0; decode values are aggregate throughput across active streams.

| Profile | Prefill | C1 decode | C8 decode | Highest tested decode | Coding peak |
|---|---:|---:|---:|---:|---:|
| [GLM-5.2 EXL3 3.5-bpw · 4 Sparks](performance/records/glm-3.5bpw/normalized-base-20260822.md) | 671 | 20.15 | 64.13 | C8: 64.13 | 25.39 |
| [DeepSeek-V4-Flash DSpark · 2 Sparks](performance/records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.md) | 1,926 | 58.36 | 162.69 | C32: 307.13 | 59.31 |
| [DeepSeek-V4-Flash-0731 · 4 Sparks](performance/records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.md) | 2,488 | 68.84 | 265.16 | C32: 508.11 | 95.77 |
| [Qwen3.8-27B EXL3 K5/K6 · 2 Sparks](performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.md) | 1,367 | 29.50 | 142.20 | C8: 142.20 | 39.95 |
| [Qwen3.8-27B EXL3 K5/K6 · 4 Sparks](performance/records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.md) | 1,964 | 35.07 | 191.02 | C8: 191.02 | 48.46 |

See [benchmark results and throughput tables](docs/RESULTS.md) for full
matrices, sample counts, exact settings, and limitations.

## Architecture

Two-Spark profiles use one direct 200 Gb/s cable and patched NCCL. Four-Spark
profiles use a switchless `0-1-2-3-0` cable cycle. SIRCL serves its tested
collective paths; patched NCCL handles the remaining paths.

See [architecture](docs/ARCHITECTURE.md), [SIRCL](docs/SIRCL.md), and the
[deployment prerequisites](docs/PREREQUISITES.md).

## Repository map

| Path | Purpose |
|---|---|
| `spark_transport/` | Native transport and vLLM adapters |
| `runtime/` | Pinned runtime inputs and builders |
| `scripts/` | Site validation, preflight, launch, and evidence tooling |
| `recipes/` | Machine-readable serving recipes |
| `performance/` | Benchmark methods, records, and sanitized receipts |
| `docs/` | Profile procedures, architecture, prerequisites, and evidence |

## Acknowledgements

SparkRing builds on work from vLLM, NVIDIA NCCL, B12X, SparkInfer, LMCache,
ExLlamaV3, and the
[local inference community](https://github.com/local-inference-lab/).

Detailed third-party attribution is in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
