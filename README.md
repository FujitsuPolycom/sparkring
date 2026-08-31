# SparkRing

> Repository branches are mutable. Pin an immutable commit when reproducing a
> deployment.

SparkRing is a collective-communication and inference-serving stack for
switchless clusters of NVIDIA DGX Spark systems. It connects each system
directly to its neighbours and runs distributed workloads without an external
Ethernet or InfiniBand fabric switch.

The repository contains:

- cluster discovery, configuration, and diagnostics;
- native RDMA collectives and vLLM integration;
- patched NCCL support for communication outside the native transport;
- reproducible runtime builders and deployment profiles; and
- benchmark methods, records, and sanitized receipts.

## Get started

Start on the system that will serve as rank 0:

```bash
curl -fLO \
  https://raw.githubusercontent.com/FujitsuPolycom/sparkring/main/bootstrap.sh
less bootstrap.sh
bash bootstrap.sh
sparkring cluster init --size 4
sparkring doctor --verify
```

The bootstrap workflow enrolls nodes over the management network, discovers
ConnectX-7 hardware, writes a local cluster inventory, and checks the direct
fabric before any serving profile is selected.

Read the [bootstrap guide](docs/BOOTSTRAP.md) before applying network changes.
Then choose a deployment from the [profile registry](docs/profiles/README.md).

## Topologies

| Topology | Status | Fabric |
|---|---|---|
| Two-system pair | **implemented** | One direct 200 Gb/s link |
| Four-system cycle | **implemented** | Four direct links in a closed `0-1-2-3-0` cycle |
| Six-system cycle | **research-only** for serving | Six direct links in a closed cycle |

Each rank also needs a management-network connection for SSH, rendezvous, and
the rank-0 API. The management network is not an inference-fabric edge.

## Communication paths

[SIRCL](docs/SIRCL.md) provides persistent RDMA sessions and graph-replayable
collectives for supported four-rank shapes. [Patched
NCCL](spark_transport/nccl/README.md) handles other collective shapes and
phases. A deployment profile states which path it uses.

See [architecture](docs/ARCHITECTURE.md), [deployment
prerequisites](docs/PREREQUISITES.md), and [cable
qualification](spark_transport/CABLE_QUALIFICATION.md) for the system
contracts.

## Profiles and evidence

Model, runtime, topology, memory, and scheduler settings belong to deployment
profiles rather than the transport definition:

- [profile registry](docs/profiles/README.md);
- [runtime builders](runtime/README.md);
- [serving configuration templates](scripts/config/README.md);
- [benchmark results](docs/RESULTS.md); and
- [methods, records, and receipts](performance/README.md).

A successful offline test or process start does not establish serving
correctness, output quality, capacity, or performance. Each profile documents
its own evidence and limitations.

## Repository map

| Path | Purpose |
|---|---|
| `spark_transport/` | Native transport, patched NCCL, and framework adapters |
| `runtime/` | Pinned runtime inputs and image builders |
| `scripts/` | Cluster validation, launch, and evidence tooling |
| `recipes/` | Machine-readable serving compositions |
| `performance/` | Benchmark methods, records, and sanitized receipts |
| `docs/` | Architecture, prerequisites, profiles, and operator procedures |

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution and validation
guidance. Security reports belong in private GitHub security advisories as
described in [SECURITY.md](SECURITY.md).

## Acknowledgements

SparkRing builds on vLLM, NVIDIA NCCL, B12X, SparkInfer, LMCache, ExLlamaV3,
and work published by the [Local Inference
Lab](https://github.com/local-inference-lab/). Model-specific profiles identify
their exact upstream source, checkpoint, revision, and license.

Detailed third-party attribution is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
