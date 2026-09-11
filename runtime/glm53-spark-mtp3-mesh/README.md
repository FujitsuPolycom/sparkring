# GLM-5.3 Flash Spark with MTP3 and hardware-forwarded mesh

Status: **research-only** profile. Bundle composition, site rendering, managed
host services, and CPU checks are **implemented**. The published image has
[qualified bounded GPU, serving, restart, and persistent-cache checks](../../performance/records/glm53-flash/spark-mtp3-compute-stream-safety-20260906.md).
The record identifies the exact image, measurements, and scope; it does not
establish in-flight collective failure containment or unattended availability.

This profile serves the `GLM-5.3-Flash-NVFP4-Spark` checkpoint with its built-in
multi-token predictor at depth three. The predictor uses a separate runtime-
NVFP4 proposal head with BF16 activations; the target/verifier head retains its
BF16 checkpoint representation. It combines graph-native SIRCL, dual-rail
fused SIRCL, and a modified RoCEnante all-reduce over a four-node physical
ring. Opposite ranks communicate through hardware forwarding in an intermediate
ConnectX-7. No external draft checkpoint or DFlash model is required.

At TP4, the proposal head adds approximately 85.08 MiB of packed NVFP4 values
and scales per rank. The BF16 target/verifier head remains allocated. The
proposal-head figure is an added allocation, not a net memory reduction.

Follow the [operator quickstart](../../docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md).
It starts from a public checkout and image/model artifacts, explains the
private site schema and four-rank distribution, and links the managed
installation commands. No private experiment checkout or existing cache is
required. The [managed operations guide](MANAGED_MESH.md) creates the shared
authentication inputs and provides model-start, readiness, stop, and recovery
commands. Keep private site files and the health key outside this repository.
The [proposal-head comparison](../../performance/records/glm53-flash/spark-mtp3-nvfp4-proposal-head-20260905.md)
contains the bounded throughput observations for this head configuration and
their limitations.

## Operator benchmark observations

The [compute matrices](../../performance/records/glm53-flash/spark-mtp3-compute-matrices-20260905.md)
record full C1/C2/C4/C8/C12/C16 results for the proposal head, loader/RNG fixes,
shared MoE scales, and top-k selector. Column averages give each of the
8K/32K/64K contexts equal weight. The records identify their measured images;
they do not substitute for validation of the combined stream-safety image.

The [consolidated validation report](../../performance/records/glm53-flash/spark-mtp3-validation-summary-20260905.md)
collects the completed tests, three-pass prefill measurements, and remaining
work. Use that report to avoid repeating checks already covered by receipts.

Status: **research-only** measurements, not general performance guarantees.
The [proposal-head comparison](../../performance/records/glm53-flash/spark-mtp3-nvfp4-proposal-head-20260905.md)
reports three proposal-head repetitions and two controls at 8K. C denotes
concurrent requests; decode values are
mean aggregate output tokens per second across those requests.

| Context | C1 | C2 | C4 | C8 |
|---:|---:|---:|---:|---:|
| 8K | 51.6 | 76.9 | 120.8 | 168.8 |

Relative to a shared-BF16-proposal-head control with the same CUDA version,
B12X kernels, metadata reuse, and dense-kernel integration, C1 improved 8.22%
in raw output throughput and 4.90% in
acceptance-normalized sequence steps/s. C2/C4/C8 results were mixed. Repeated
prefill means moved by no more than 0.36% over 8K–128K contexts, so no prefill
gain is claimed. The
[broader matrix](../../performance/records/glm53-flash/spark-mtp3-mesh-20260905.md)
records measurements for its explicitly identified image and serving settings.
The head-specific control preserves the verifier implementation; the complete
CUDA 13.3/B12X `b58f34ea` image changes other target computation relative to
the configuration recorded in that matrix, so cross-image gains cannot be assigned only to the
proposal head or dense kernels.

The published image's
[compute-equivalence record](compute-image-equivalence.json) matches every
vLLM, B12X, and SparkCache package file and selected environment entry to the
serving image identified in that record. This establishes compute-package
content equivalence, not end-to-end performance equivalence: the mounted
transport differs outside that comparison. The published image's separate
[runtime validation](../../performance/records/glm53-flash/spark-mtp3-compute-stream-safety-20260906.md)
includes an 18-cell matrix, two focused repetitions, native and GPU stream
checks, an idle rank-loss test, restart, and a verified persistent-prefix restore.

The separate [Estonia accuracy record](../../performance/records/glm53-flash/spark-mtp3-country-recall-20260905.md)
reports **30/30 correct** at C8 on one repeated 133,208-token prompt, no
output-limit hits, and 1.96 s mean cache-primed TTFT. Its 23.8 tok/s figure
uses summed request times, not cluster wall time. The Estonia record includes
the operator screenshot and metric definitions.

The [long-context needle hunt](../../performance/records/glm53-flash/spark-mtp3-needle-20260905.md)
passed **4/4** exact-value, revision, and cross-reference checks, reaching
**507,367 actual prompt tokens** on serving image
`sha256:26273b8e358df139ae913610a5d43084ff0fd08aafe282ef633a3bc74afefe47`,
as recorded in that report. It does not identify the image pinned by
`public-image.json`.

## Composition

| Input | Contract |
|---|---|
| Model, MTP depth, graph shapes, mesh bundle, marker identity, cache identity | [`pins.json`](pins.json) |
| Linux/ARM64 parent image, SparkCache, and native SIRCL | [`../glm53-flash-jj-r8-gb10/pins.json`](../glm53-flash-jj-r8-gb10/pins.json) |
| CUDA 13.3, GLM metadata/loader/RNG integration, B12X scale sharing and selector overrides, NVFP4/BF16 proposal head | [`pins.json`](pins.json), [`IMAGE_BUILD.md`](IMAGE_BUILD.md) |
| Topology and rank-local filesystem inputs | [`site.example.json`](site.example.json), [`fabric.example.json`](fabric.example.json) |
| Source-bound collective dispatch and health checks | [`glm53_rocenante_overlay`](../../spark_transport/experiments/glm53_rocenante_overlay/README.md) |
| Hardware-forwarding plan and native source marker | [`cx7_hairpin_diagonal`](../../spark_transport/experiments/cx7_hairpin_diagonal/README.md) |
| Modified RoCEnante communication package | [`third_party/b12x_roce`](../../third_party/b12x_roce/README.md) |

The managed profile requires the [published child image](IMAGE_BUILD.md). It
retains the parent runtime while adding CUDA 13.3, the uniform MTP3
metadata and loader/RNG integration, B12X revision `ef308bac` with selector
overrides, the runtime-NVFP4/BF16
proposal head, the verified transport bundle, the managed source marker, and
the temperature-one readiness helper. The verifier remains BF16. Pull the
immutable reference in [public-image.json](public-image.json)
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
binds compute lock `139f3670` and transport bundle `69313e19` so the
NVFP4-proposal-head composition cannot restore entries from another computation.
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
