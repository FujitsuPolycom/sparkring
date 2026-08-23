# SparkRing

> **This repository is changing rapidly.** Documentation, profiles, and
> branch history are being restructured as SparkRing's scope narrows to
> switchless multi-Spark collective transport and serving. Published
> branches may be rebased and documents may be moved, renamed, or
> replaced without a deprecation period. Pin a commit if you depend on a
> specific state of this tree.

SparkRing is a low-latency collective transport and vLLM-based
inference-serving stack for switchless clusters of NVIDIA DGX Spark systems
powered by the GB10 Grace Blackwell Superchip.

SparkRing supports GB10 pairs, four-node rings, and Six-node rings (in dev).

Models run as tensor-parallel deployments over the direct fabric without an
external Ethernet or InfiniBand switch. [SIRCL](https://github.com/FujitsuPolycom/sparkring/blob/main/docs/SIRCL.md) provides custom RDMA collectives
for tested paths, CUDA-graph command rings support repeated decode
work, and [patched NCCL](https://github.com/FujitsuPolycom/sparkring/blob/main/spark_transport/nccl/README.md) handles communication outside SIRCL's supported paths.

The repository provides launch tooling, model profiles, test evidence, and [performance data](https://github.com/FujitsuPolycom/sparkring/tree/main/performance). Model-specific policy
belongs to profiles rather than the transport.

## Start here

- [Supported models and profiles](#profiles)
- [Benchmark results and throughput tables](docs/RESULTS.md)
- [Deployment prerequisites](docs/PREREQUISITES.md) — then choose a profile quickstart below

## Cluster sizes

- **2× DGX Spark — direct pair.** Models: DeepSeek-V4-Flash-0731 and
  Qwen3.8-27B EXL3 K5/K6. DeepSeek is compatible with SparkCache
- **4× DGX Spark — physical ring.** Models: GLM-5.2 EXL3 3.5-bpw,
  DeepSeek-V4-Flash-0731, and Qwen3.8-27B EXL3 K5/K6. GLM and DeepSeek are
  compatible with SparkCache; Qwen SparkCache support is Pending.
- **6× DGX Spark — physical ring.** **In dev. Target models: GLM | KIMI.**

## Profiles

| Profile | Model identity | Topology | Status | Start here |
|---|---|---|---|---|
| GLM-5.2 EXL3 3.5-bpw | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f` | four-Spark cycle, TP4/DCP4 | 1M context/16 sequences; benchmark results published | [Quickstart](docs/GLM52_35BPW_QUICKSTART.md) |
| DeepSeek-V4-Flash | `deepseek-ai/DeepSeek-V4-Flash-DSpark@913f0657a874f76844e2e91cbe706dbcaceeb6d7` | two-Spark pair, TP2/DCP1 | tested; benchmark results published; SIRCL unsupported | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` | four-Spark cycle, TP4/DCP1 | tested; benchmark results published; SIRCL width 4096 research-only | [Quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated@ab3a91a13813df8096cb4c1d560ed3669035d0cf` | two-Spark pair, TP2/DCP1 | 1,048,576-token profile; benchmark results through C8 | [Quickstart](docs/QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated@ab3a91a13813df8096cb4c1d560ed3669035d0cf` | four-Spark cycle, TP4/DCP1 | 1,048,576-token profile; benchmark results through C8; SIRCL unsupported | [Quickstart](docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md) |
| GLM-5.2 EXL3 3.5-bpw + SparkCache | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f` | four-Spark cycle, TP4/DCP4 | cache restore tested at 262K/eight sequences | [SparkCache compositions](recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` | two-Spark pair, TP2/DCP1 | cache restore tested at 131K/six sequences | [SparkCache compositions](recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` | four-Spark cycle, TP4/DCP1 | cache restore tested at 524K | [SparkCache compositions](recipes/sparkcache/README.md) |

Qwen with SparkCache has no published composition recipe or live cache evidence.
See the
[profile registry](docs/profiles/README.md) for recipe identities and evidence
scope.

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
