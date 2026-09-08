# GLM TP4 source image

**Status: research-only.** This recipe prepares an ARM64 image containing
continuation-prefill coalescing, token-sharded mHC, DCP owner selection, and
dual-domain NCCL. Source preparation and CPU receipt checks are implemented.
The image produced by this recipe still requires an ARM64 build and serving
qualification; measurements from another image do not qualify this build.

The recipe extends the existing native-MTP3 mesh deployment. It does not
replace site rendering, ASIC forwarding configuration, authenticated marker
ownership, or the managed host lifecycle.

## Composition

`glm53-tp4-lock.json` identifies every public source base, packaged patch,
resulting Git tree, installed package inventory, retained native library,
startup helper, and runtime profile. Full hashes in that file are authoritative.

| Component | Source and behavior |
|---|---|
| ARM64 parent | Digest-pinned `sparkring-glm53-sparkcache` image; supplies the CUDA toolchain, Torch, compiled vLLM dependencies, and native mesh bundle |
| vLLM | Public source base `2a979314dc97b03173a0a76fc15664ec924db32b` plus `patches/vllm.patch`; reproduces tree for revision `8f8ea47be212bbdd91b2172d5958ea2aae2b0e50` |
| B12X | Public source base `85a08f47750db333a33ab3eae245a0a08452d04c` plus `patches/b12x.patch`; reproduces checkpoint-export and kernel sources at `0b6d61c37c87ae49d2f9d20d38b9da023146e243` |
| NCCL | Public NVIDIA source base `73cf112295c33aee2b895f329f592f2a9b4b0f97` plus the cumulative `patches/nccl.patch`; preserves switchless routing and independent PCIe-domain discovery |
| SparkCache | Source and native snapshot library are installed to reproduce package dependencies; the serving connector is disabled in both declared profiles |
| Startup | Four source-pinned helpers under `startup/`; the warmup request uses thinking with low reasoning effort and temperature 1 |

The profiles select TP4/DCP1 or TP4/DCP4 with native MTP3, continuation
coalescing, and mHC prefill sharding. Compact index-cache gathering and the
SparkCache connector are disabled. DCP2 is not a declared profile.

The parent retains 1,985 generated support files and native libraries after
source replacement. Their full inventory, including 15 compiled vLLM
libraries, matches the installed-state evidence for the measured source
composition. Torch, Triton, CUTLASS DSL, Transformers, and FlashInfer versions
are checked independently. This establishes file identity; a rebuilt NCCL
library and complete serving behavior still require verification.

## Prepare and build

Run preparation from the repository root with Python 3.12 or later and Git.
Preparation downloads the exact public source bases, verifies patch hashes,
applies patches to an index, and checks complete resulting Git trees. It
does not invoke Docker or contact inference hosts.

```bash
python runtime/sparkring/source_image/prepare_image.py \
  --output /tmp/sparkring-glm-image-context \
  --source-cache /tmp/sparkring-glm-image-sources
```

Both paths must initially be absent. To prepare another context from the same
sources, use a different output directory and `--reuse-source-cache`. Reuse
checks the base commit, complete patched index tree, unstaged changes, and
untracked files before accepting any source directory.

Build on an ARM64 Docker host with the parent image available:

```bash
docker build --platform linux/arm64 --network none \
  -t sparkring-glm53-tp4-source \
  /tmp/sparkring-glm-image-context
```

The build compiles NCCL and the native snapshot library without GPU access.
Python packages install offline without dependency resolution. NCCL's
CPU routing compatibility test runs before compilation. The resulting NCCL
library must match the measured SHA-256 in the lock. A mismatch stops the
build; do not replace that hash solely to make the check pass. Record compiler
and linker differences, compare source and binary behavior, and qualify the
rebuilt library before accepting another binary identity.

The container path `/opt/sparkcache-jj-runtime` and its manifest schema names
are retained compatibility interfaces for source installation and verification.
They do not enable SparkCache serving.

## Verify and use the existing installer

Create a CPU-only receipt for the selected profile:

```bash
python runtime/sparkring/source_image/verify_image.py \
  --image sparkring-glm53-tp4-source \
  --context /tmp/sparkring-glm-image-context \
  --profile tp4-dcp1-mtp3-prefill \
  --output /tmp/sparkring-glm53-tp4-dcp1-receipt.json
```

Verification first compares the stopped image's verifier scripts, lock and
manifest against the trusted prepared context and local recipe. It then runs
a read-only, network-disabled container without GPU devices, importing only
the verified scripts from a read-only mount with isolated Python.
It checks parent layers, source-lock bytes, complete installed package maps,
runtime library identities, dependency metadata, and startup helper identity.
The receipt's image reference is a **local image config ID**, not a registry
manifest digest. It is not evidence of a registry publication or GPU test.

Select the matching `runtime_profile` in the private mesh site configuration
and pass this explicit image receipt through the existing deployment suite.
The renderer and managed installer validate the same lock and selected
profile. Keep the canonical public image receipt unchanged. Follow
`runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md` for host lifecycle and startup
ownership; replacing a running model remains a separate deployment action.

Use the [TP4 prefill quickstart](../../../docs/GLM53_TP4_PREFILL_QUICKSTART.md)
for the DCP1 setup sequence and explicit DCP4 alternative.

## Validation

```bash
python -m unittest discover \
  -s runtime/sparkring/source_image -p test_source_image.py -v
```

The CPU suite rejects altered package counts, hashes, revisions, dependency
versions, warmup bytes, runtime identities, malformed receipts, and registry
references presented as local IDs. Full-model qualification must additionally
show feature activation, correctness, and prefill/decode measurements on the
same generated image and selected profile.

Source licenses and notices remain those of each upstream repository.
Packaged patches preserve upstream notices; the parent image also contains
third-party CUDA and framework components with their own terms. See the
repository's `THIRD_PARTY_NOTICES.md` before redistribution.
