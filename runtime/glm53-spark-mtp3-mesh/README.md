# GLM-5.3 Flash Spark with native MTP3 and hardware-forwarded mesh

Status: **research-only**. Bundle composition, site rendering, managed
host-fabric installation/supervision, and CPU checks are **implemented**.
The [managed functional record](../../performance/records/glm53-flash/spark-mtp3-managed-mesh-functional-20260905.md)
qualifies bounded installer, policy-scoped fault/recovery, post-recovery
readiness, and one persistent-cache recall case. Broader coverage remains
unqualified. The
[sampling-warmup image functional record](../../performance/records/glm53-flash/spark-mtp3-mesh-temperature-one-functional-20260905.md)
qualifies bounded native checks, four-rank startup/restart, and one persistent
recall restoration for its exact image. Broader cache/workload coverage and
failure containment remain unqualified.

This profile serves the `GLM-5.3-Flash-NVFP4-Spark` checkpoint with its built-in
multi-token predictor at depth three. It combines graph-native SIRCL,
dual-rail fused SIRCL, and a modified RoCEnante all-reduce over a four-node
physical ring. Opposite ranks communicate through hardware forwarding in an
intermediate ConnectX-7. No external draft checkpoint is required.

Follow the [operator quickstart](../../docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md).
It starts from a public checkout and image/model artifacts, explains the
private site schema and four-rank distribution, and links the managed
installation commands. No private experiment checkout or existing cache is
required. The [managed operations guide](MANAGED_MESH.md) creates the shared
authentication inputs and provides model-start, readiness, stop, and recovery
commands. Keep private site files and the health key outside this repository.
The [measurement record](../../performance/records/glm53-flash/spark-mtp3-mesh-20260905.md)
contains the bounded throughput observations and their limitations.

## Operator benchmark observations

Status: **research-only** measurements, not general performance guarantees.
The [throughput record](../../performance/records/glm53-flash/spark-mtp3-mesh-20260905.md)
identifies the measured source configuration and its differences from the
packaged image. C denotes concurrent requests; decode values are aggregate
output tokens per second across those requests.

| Context | C1 | C2 | C4 | C8 | C12 | C16 |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 48.2 | 75.8 | 112.2 | 168.8 | 193.4 | 231.3 |
| 32K | 49.9 | 76.8 | 119.0 | 164.6 | 197.3 | 222.7 |
| 64K | 43.0 | 76.4 | 119.0 | 165.8 | 192.3 | 220.9 |

Concurrency-one prefill scouts measured **2,703–2,787 prompt tokens/s** over
8K–128K contexts, with one observation per context. They are not a repeated
cold-cache benchmark.

The separate [Estonia accuracy record](../../performance/records/glm53-flash/spark-mtp3-country-recall-20260905.md)
reports **30/30 correct** at C8 on one repeated 133,208-token prompt, no
output-limit hits, and 1.96 s mean cache-primed TTFT. Its 23.8 tok/s figure
uses summed request times, not cluster wall time. Both records retain the
operator screenshots and metric definitions.

The [long-context needle hunt](../../performance/records/glm53-flash/spark-mtp3-needle-20260905.md)
passed **4/4** exact-value, revision, and cross-reference checks, reaching
**507,367 actual prompt tokens** on the published image.

## Composition

| Input | Contract |
|---|---|
| Model, MTP depth, graph shapes, mesh bundle, marker identity, cache identity | [`pins.json`](pins.json) |
| Linux/ARM64 image, vLLM, B12X kernels, SparkCache, native SIRCL | [`../glm53-flash-jj-r8-gb10/pins.json`](../glm53-flash-jj-r8-gb10/pins.json) |
| Topology and rank-local filesystem inputs | [`site.example.json`](site.example.json), [`fabric.example.json`](fabric.example.json) |
| Source-bound collective dispatch and health checks | [`glm53_rocenante_overlay`](../../spark_transport/experiments/glm53_rocenante_overlay/README.md) |
| Hardware-forwarding plan and native source marker | [`cx7_hairpin_diagonal`](../../spark_transport/experiments/cx7_hairpin_diagonal/README.md) |
| Modified RoCEnante communication package | [`third_party/b12x_roce`](../../third_party/b12x_roce/README.md) |

The model runtime and kernels come from the pinned parent image. The managed
profile requires the [published child image](IMAGE_BUILD.md), which adds
the verified transport bundle, managed source marker, and temperature-one
readiness helper. Pull the immutable reference in [public-image.json](public-image.json)
and use [image-receipt.json](image-receipt.json) for rendering and installation.
Local source reproduction is optional; distribute identical verified bytes
to all ranks. The canonical transport bundle remains mounted read-only.
Image construction alone does not qualify serving or host-device setup.

## Offline interfaces

Run from the repository root. For optional source reproduction,
`base-sircl` must be the complete bundle extracted from the pinned **parent**
operator image, including its native library. For the published-image
quickstart, skip composition and render using its extracted canonical bundle.

```bash
python3 runtime/glm53-spark-mtp3-mesh/profile.py bundle \
  --base-sircl build/base-sircl \
  --output build/mtp3-mesh-bundle

python3 runtime/glm53-spark-mtp3-mesh/profile.py render \
  --site /srv/sparkring/site/mtp3-mesh.json \
  --bundle /srv/sparkring/artifacts/mtp3-mesh-bundle \
  --image-receipt runtime/glm53-spark-mtp3-mesh/image-receipt.json \
  --output build/mtp3-mesh-launch
```

Both commands reject mismatched pins. Rendering creates rank environment
files, the model launcher, site/topology copies, and `fabric-plan.json` with
reviewable route, neighbor, traffic-control, and source-marker argument
arrays. Neither command contacts hosts, installs networking, or starts a
model. Output directories must not already exist.

The examples use documentation and benchmark-only addresses and synthetic
MAC addresses. Replace them with a verified inventory outside version
control; they are not a deployable site.

## Managed host interface

Follow [managed mesh installation and operation](MANAGED_MESH.md).
`managed_install.py` validates a pre-created stopped rank container and
installs root-owned source, private authentication inputs, and systemd units.
`managed_cluster.py` coordinates four-rank readiness, model startup,
model-stop barriers, cleanup, and explicit recovery.

The service configures and monitors paths that let opposite Sparks
communicate through an intermediate NIC in the ring. Model startup requires
all four ranks to agree that the fabric is ready. An unhealthy fabric stops
dependent serving; recovery is an explicit operator action. The intermediate
NIC forwards packets in hardware, not through a helper's CPU loop. A shared
key and deployment identifier authenticate peers. Model and mesh units do
not restart automatically.

Network cleanup removes only exact created-and-owned artifacts. Existing
matching objects are adopted and preserved; conflicting state fails closed.
Hot marker replacement beneath active RC QPs and unattended high availability
are **unsupported**.

The generic topology schema's external helper locator is not the managed
service interface. Do not execute its generic apply plan. The bounded
`inspect_fabric.py --minimum-remaining` interface and `--run-seconds`
marker mode are diagnostic tools for isolated bounded tests, not the managed
serving lifecycle. Use authenticated managed readiness for serving.

## Cache identity

Native MTP uses the target checkpoint as the draft identity. The profile sets
SparkCache's `draft_policy=separate` because that describes the registered
state layout; it does not request an external model. A dedicated namespace
prevents restoration of entries tagged for an external DFlash checkpoint.
Do not relabel those entries to avoid cache misses. Persistent restore under
the native-MTP identity requires its own qualification.

## Tests

```bash
python3 -m pytest runtime/glm53-spark-mtp3-mesh \
  spark_transport/experiments/glm53_rocenante_overlay \
  spark_transport/experiments/cx7_hairpin_diagonal -q
```

These are offline contract tests. They do not allocate a GPU, prove host
hardware forwarding, or qualify live model throughput.
