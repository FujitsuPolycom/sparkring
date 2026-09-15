# Mesh indexer publication-barrier image

Status: research-only serving artifact with qualified bounded indexer GPU tests.
The child image preserves the published MTP3 mesh parent's compute,
CUDA 13.3, transport bundle, marker, and native libraries. Its only executable
runtime change adds block-wide synchronization before histogram publication
and increases the fused-indexer compile revision from 1 to 2.

## Artifact identities

Tag: `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache:20260906-mtp3-mesh-indexer-barrier`

Immutable reference:
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:7aa57ed1901b83c8581c53cb233db930bb1a133d74be01168d967b9dd56b3e49`

Config image ID:
`sha256:995a42b4b08f525813c5783f78f34571b6bd1c197fccee53617ff8060574a16d`

Parent reference:
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:67dc0ae453baaae6831ccec1d259b4ef8b236a8b0dc9f747d901b95c66ec1987`

The image retains all 89 parent layers and adds two layers totaling 637,767
uncompressed bytes. Its B12X selector retains the mesh parent's PR316 changes;
it is not replaced with the SIRCL operator image's selector.

The compute source lock is
`57b6438ca014a493380fb874a84977d35b0983e9ee224a62ade3415fd9c7ce46`.
The image source receipt is
`030e9f3749672d186b66e41336980af1abb5c207f734feafa677ba82456c5d43`.
The unchanged transport bundle is
`69313e19e881ec93e9ed3bd150d2f24fc6b444488ac729a69f45d038e2243500`.

## Conditions

Validation used one NVIDIA GB10 SM121 on 2026-09-06, the image above, and
PyTorch 2.13.0+cu130. Tests ran in isolated containers without model weights,
serving traffic, or replacement of the running telemetry stack.

## Measurement

The delayed-publisher probe executes the mesh parent's barrier helper in a
three-block kernel, then compares it with an entry-synchronized variant.
Ten eager and ten graph-replay runs per variant check complete publication.
The probe executes only one barrier and does not deliberately hang the GPU.

The built child image runs fifteen fused-indexer correctness cases and 2,000
graph replays, checking exact selected-index sets, top-k scores within 0.01
absolute tolerance, and cleared merge state. Workloads use 32 heads, top-k 512,
3/4/8/16 query rows, lengths alternating between 4,097 and 65,536 tokens
(200,000 for the four-row case), and concurrent 64 MiB device copies.
The harnesses and their dependency requirements are documented in SparkRing's
`performance/harnesses/indexer_barrier/` directory on main.

## Result

- Parent helper: incomplete publication in 20/20 probes.
- Entry synchronization: complete publication in 20/20 probes.
- Built mesh image: fifteen GPU correctness tests and 2,000 graph replays passed.
- Local mesh suite: 348 tests passed; three documented platform skips.
- Unmodified mesh content verifier passed before and after installation.
  It verifies 385 B12X files, the 24 vLLM overrides, CUDA components, and the
  transport/marker/native-library contracts.
- External verification passed the parent-layer, image-label, source-receipt,
  and embedded-content checks.

## Conclusion

The barrier correction prevents the measured publication race in the mesh
parent's selector. The built image preserves the tested numerical behavior
and completes the bounded graph stress cases. These results support a model
qualification run, not an assertion that every serving stall is eliminated.

## Limitations and deployment boundary

No four-rank model startup, collective execution, actual SparkCache restore,
or serving soak was performed with this child image. The running telemetry
image was not patched or replaced. Its additional SparkCache changes are not
included in this artifact.

The managed profile renderer still pins the parent image and compute lock.
Do not bypass its receipt checks or pass this child's receipt as though it
were the parent's. Managed deployment requires a profile update binding the
child image, source lock, and coordinated startup inputs, followed by an
authorized stop/start window. The artifact is available for that validation;
this change does not silently promote it into the default managed profile.

The embedded profile records cache namespace
`glm53-spark-df116c4f-mtp3-nvfp4-a16-c57b6438c-mesh69313e19-tail-cow-v2`.
Use a distinct JIT namespace for qualification. Do not rename existing cache
entries into the source-bound namespace.

## Rebuild and verify

Place `Dockerfile`, `install.py`, and `patch_indexer_barrier.py` in an empty
context. Run `install.py --output /probe/build-input.json` in an isolated
container of the pinned parent with the context mounted at `/probe`. Build
with that result's `source_receipt_sha256` and `source_lock_sha256` supplied
as `SOURCE_RECEIPT_SHA256` and `COMPUTE_SOURCE_LOCK_SHA256`; set
`SPARKRING_REVISION` to the source commit. The published build uses
`8e54e22477907935c92c52a1c3d7a68e364bd776`.

The installer preserves the parent source receipt, records the checked source
transform, and updates compute package maps and the embedded profile. The
existing mesh verifier runs before and after installation without modified
verification logic. External verification uses the exact image ID, parent
config ID, source-receipt hash, and bundle hash recorded above.
