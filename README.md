# SparkRing

SparkRing is a low-latency collective transport and vLLM-based
inference-serving stack for switchless clusters of NVIDIA DGX Spark systems
powered by the GB10 Grace Blackwell Superchip.

Four 200 Gb/s ConnectX-7 links connect four DGX Sparks in a physical ring.
Models run as tensor-parallel deployments without an Ethernet or InfiniBand
switch in the inference fabric.

SparkRing routes qualified vLLM collectives through SIRCL, its custom RDMA
transport for the four-node ring. CUDA-graph command rings support repeated
decode work, while patched NCCL handles communication outside SIRCL's supported
paths.

The repository provides launch tooling, speculative-decoding integration,
model profiles, and explicit qualification evidence. Model-specific policy
belongs to profiles rather than the transport.

## Start here

| Goal | Documentation |
|---|---|
| Deploy the reproducible public default | [Four-Spark quickstart](docs/QUICKSTART.md) |
| Select another model or configuration | [Validated-profiles registry](docs/profiles/README.md) |
| Understand the transport and runtime | [Architecture](docs/ARCHITECTURE.md) |
| Inspect measured results | [Results and evidence boundaries](docs/RESULTS.md) |
| Prepare a new cluster | [Prerequisites](docs/PREREQUISITES.md) |
| Contribute code or documentation | [Contributor guide](CONTRIBUTING.md) and [agent guide](AGENTS.md) |

## Deployment profiles

| Profile | Status | Start here |
|---|---|---|
| GLM-5.2 EXL3 3.25-bpw with LMCache CS512 | Public-functional default; bounded clean-checkout live validation | [Quickstart](docs/QUICKSTART.md) |
| GLM-5.2 EXL3 R7 3.5-bpw | Accepted on one four-Spark appliance; rebuilds require qualification | [R7 quickstart](docs/EXL3_R7_QUICKSTART.md) |
| GLM-5.2 NF3 | Accepted deterministic alternative | [NF3 quickstart](docs/NF3_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | Functional operator launch; not shadow-qualified | [DeepSeek quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| Other models and widths | Maturity varies by profile | [Complete registry](docs/profiles/README.md) |

The registry owns model identities, revisions, configuration links, maturity,
and remaining qualification gates. This README does not duplicate those
contracts.

## Topology

```text
             management network
          |       |       |       |
         S0      S1      S2      S3

                  200 Gb/s
             S0 ========== S1
             ||             ||
    200 Gb/s ||             || 200 Gb/s
             ||             ||
             S3 ========== S2
                  200 Gb/s
```

Management traffic uses the ordinary LAN. RDMA inference traffic uses the four
direct ConnectX-7 links.

## Transport and vLLM integration

SIRCL is the Switchless Inference RDMA Collective Layer. It owns the persistent
RDMA sessions and graph-replayable command rings used by qualified collective
operations.

A checksum-pinned NCCL build provides the ring-safe fallback for communication
that SIRCL does not admit.

SparkRing integrates with vLLM through either the published runtime overlay or
the optional `sparkring_plugin` Python package. The plugin passes offline tests
but has not yet been validated on a live cluster.

Implementation details and admission mechanics are documented in:

- [SIRCL](docs/SIRCL.md)
- [System architecture](docs/ARCHITECTURE.md)
- [vLLM integration](spark_transport/integrations/vllm/README.md)
- [Machine-readable component status](docs/STATUS.json)

## Repository map

| Path | Purpose |
|---|---|
| `spark_transport/` | Native transport, probes, tests, and vLLM adapters |
| `sparkring_plugin/` | Optional pip-installable vLLM integration |
| `sparkcache/` | Persistent rank-local context cache |
| `runtime/` | Pinned runtime builders and published patches |
| `scripts/` | Site configuration, preflight, launch, and evidence tooling |
| `recipes/` | Machine-readable serving profiles |
| `docs/` | Architecture, procedures, profiles, results, and historical records |

## Offline contributor checks

These commands require neither a cluster nor a GPU:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.11.0"

python -m pytest spark_transport sparkcache runtime scripts -q
python -m pytest sparkring_plugin -q
ruff check --select E,F,W --ignore E501 --exclude runtime/patches .
```

## Evidence and maturity

Measured claims and their limitations are recorded in
[`docs/RESULTS.md`](docs/RESULTS.md). Component maturity is recorded in
[`docs/STATUS.json`](docs/STATUS.json). Experimental chronology and superseded
configurations belong in explicitly historical documents.

A published configuration is not automatically accepted. Each profile states
whether it is offline-validated, live-validated, accepted, research-only, or
unsupported.

## Acknowledgements

SparkRing builds on work from vLLM, NVIDIA NCCL, B12X, SparkInfer, LMCache,
ExLlamaV3, and the
[local inference community](https://github.com/local-inference-lab/).

Detailed third-party attribution is in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
