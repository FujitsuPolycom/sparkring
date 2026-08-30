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
- [Evidence and results](#evidence-and-results)
- [Deployment prerequisites](docs/PREREQUISITES.md) — then choose a profile quickstart below
- [GLM-5.3 Flash quickstart routing](docs/GLM53_FLASH_QUICKSTARTS.md)

## Profiles

### GLM-5.3 Flash profiles

| Profile | Deployment | Context | Seqs | Batch | KV / cache | Start here |
|---|---|---:|---:|---:|---|---|
| GLM-5.3 Flash + BF16 DFlash2 | 4 Sparks · TP4/DCP1 | 512K | 32 | 8,192 | 12 GiB/rank FP8 KV | [Quickstart](docs/GLM53_FLASH_DFLASH2_BF16_TP4_QUICKSTART.md) |
| GLM-5.3 Flash + BF16 DFlash2 + SparkCache | 4 Sparks · TP4/DCP1 | 512K | 32 | 8,192 | 12 GiB/rank FP8 KV + 48 GiB nvme/rank SparkCache | [Quickstart](docs/GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md) |

Source-built GLM-5.3 paths are reviewed separately from the published BF16
DFlash2 composition:

| Runtime path | Status | Start here |
|---|---|---|
| External DFlash7 snapshot-v1 with vLLM `0b67266a` Python over retained `da4d7be6` extensions | **qualified** only for one exact local image and its recorded 131,072-token full-snapshot case | [Shortest qualified start](docs/GLM53_DFLASH7_PYTHON_OVERLAY_SPARKCACHE_TP4_QUICKSTART.md#shortest-qualified-start) |
| Adaptive embedded MTP with live-tensor B12X KDA | **implemented**, not qualified | [Adaptive-MTP quickstart](docs/GLM53_B12X_KDA_ADAPTIVE_MTP_SPARKCACHE_TP4_QUICKSTART.md) |
| Source-built vLLM `e10536a` profiles | **implemented**, not qualified | [e10536a quickstart](docs/GLM53_E10536A_SPARKCACHE_TP4_QUICKSTART.md) |

The [GLM-5.3 routing guide](docs/GLM53_FLASH_QUICKSTARTS.md) explains source
ancestry, image construction, immutable registry pulls, local archive fanout,
profile resolution, and evidence boundaries.

For the 20 GiB FP8 KV pool, GLM hybrid allocation does not scale linearly from
the reported 916,676-token capacity. A no-cache C6 × 128K observation admitted
one request at a time and serialized completions. C2 × 128K is the only
observed safe candidate pending live CUDA qualification. C8 × 64K and
C16 × 32K are planned and unqualified; C16 × 128K is unsupported unless GPU
trunk pages are shared or KV capacity increases.

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

The public BF16 DFlash2 checkpoint is licensed CC BY-NC-ND 4.0 for research
and evaluation; review its model card before use. The qualified BF16 DFlash2
community image is published by immutable digest. The DFlash7 qualification
belongs to a local image ID and has no published OCI digest.
See the [profile registry](docs/profiles/README.md) for recipe identities and evidence scope.

## Evidence and results

Evidence records stay beside their methods, exact inputs, and limitations.
See [results](docs/RESULTS.md) for qualified throughput tables and
[`performance/records/`](performance/records/) for research-only and
artifact-specific observations. A runtime status or green CPU-only test does
not establish throughput, output quality, or live serving behavior.

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
