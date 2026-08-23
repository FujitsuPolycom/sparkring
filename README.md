# SparkRing

> **Alpha software.** Interfaces and documentation may change. Pin a commit if
> you depend on a specific repository state.

SparkRing is a low-latency collective transport and vLLM-based
inference-serving stack for switchless clusters of NVIDIA DGX Spark systems
powered by the GB10 Grace Blackwell Superchip.

SparkRing supports GB10 pairs and four-node rings.

Models run as tensor-parallel deployments over the direct fabric without an
external Ethernet or InfiniBand switch.

The [Switchless Inference RDMA Collective Layer (SIRCL)](docs/SIRCL.md)
provides custom RDMA collectives and CUDA-graph command rings for qualified
paths.

[Patched NCCL](spark_transport/nccl/README.md) handles communication outside
SIRCL's supported paths.

The repository provides launch tooling, model profiles, and measured results.
Model-specific policy belongs to profiles rather than the transport.

## Start here

- [Supported models and profiles](#profiles)
- [Benchmark results and throughput tables](docs/RESULTS.md)
- [Deployment prerequisites](docs/PREREQUISITES.md) — then choose a profile quickstart below

## Cluster sizes

- **2× DGX Spark — direct pair.** Models: DeepSeek-V4-Flash DSpark and
  Qwen3.8-27B EXL3 K5/K6. DeepSeek is compatible with SparkCache.

- **4× DGX Spark — physical ring.** Models: GLM-5.2 EXL3 3.5-bpw,
  DeepSeek-V4-Flash-0731, and Qwen3.8-27B EXL3 K5/K6. GLM and DeepSeek are
  compatible with SparkCache; Qwen with SparkCache is unsupported.

## Profiles

TP is tensor parallelism; DCP is decode-context parallelism. Capacity values
state maximum context tokens and maximum concurrent sequences.

| Profile | Topology | Status | Start here |
|---|---|---|---|
| GLM-5.2 EXL3 3.5-bpw | four-Spark cycle, TP4/DCP4 | qualified at 262,144 tokens/eight sequences; 1,048,576 tokens/16 sequences not qualified | [Quickstart](docs/GLM52_35BPW_QUICKSTART.md) |
| DeepSeek-V4-Flash DSpark | two-Spark pair, TP2/DCP1 | implemented and benchmarked; SIRCL unsupported | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | four-Spark cycle, TP4/DCP1 | implemented and benchmarked; SIRCL width 4096 research-only | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | two-Spark pair, TP2/DCP1 | implemented and benchmarked through eight concurrent requests | [Quickstart](docs/QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | four-Spark cycle, TP4/DCP1 | implemented and benchmarked through eight concurrent requests; SIRCL unsupported | [Quickstart](docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md) |
| GLM-5.2 EXL3 3.5-bpw + SparkCache | four-Spark cycle, TP4/DCP4 | restore qualified at 262,144 tokens/eight sequences; 1,048,576 tokens/16 sequences not qualified | [SparkCache compositions](recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | two-Spark pair, TP2/DCP1 | restore qualified at 131,072 tokens/six sequences; 1,048,576 tokens/32 sequences not qualified | [SparkCache compositions](recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | four-Spark cycle, TP4/DCP1 | restore qualified at 524,288 tokens/32 sequences; 1,048,576 tokens/32 sequences not qualified | [SparkCache compositions](recipes/sparkcache/README.md) |

Qwen with SparkCache is unsupported. See the
[profile registry](docs/profiles/README.md) for recipe identities and evidence
scope.

## Architecture

Two-Spark profiles use one direct 200 Gb/s cable and patched NCCL. Four-Spark
profiles use a switchless `0-1-2-3-0` cable cycle. SIRCL serves its qualified
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
