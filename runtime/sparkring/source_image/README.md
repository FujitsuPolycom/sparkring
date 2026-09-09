# Shared GLM source image

**Status: research-only.** Source preparation, profile selection, and CPU
verification are implemented. This recipe prepares one ARM64 image for the
listed GLM TP2 and TP4 profiles, including optional SparkCache. The image is
published with five passing CPU profile receipts. Full GPU serving tests
remain required; results from reference deployments do not qualify it.

The ring profiles use the existing native-MTP3 mesh deployment, including
site rendering, ASIC forwarding, authenticated marker ownership and the
managed host lifecycle. The switched and TP2 profiles use their dedicated
launchers described below.

## Composition

`glm53-tp4-lock.json` identifies every public source base, packaged patch,
resulting Git tree, installed package inventory, retained native library,
startup helper, and runtime profile. Full hashes in that file are authoritative.

| Component | Source and behavior |
|---|---|
| ARM64 parent | Digest-pinned `sparkring-glm53-sparkcache` image; supplies the CUDA toolchain, Torch, compiled vLLM dependencies, and native mesh bundle |
| vLLM | Public base `2a979314dc97b03173a0a76fc15664ec924db32b` plus `patches/vllm.patch`; reproduces `17bd258075f44dda8b405f384732f3c78d03f308`, including four-checkpoint coalescing, TP2/TP4 mHC admission, DCP owner exchange, and hybrid restore recovery |
| B12X | Public base `85a08f47750db333a33ab3eae245a0a08452d04c` plus `patches/b12x.patch`; reproduces `2883a5df65a7ea3cb6e82abb63d1448dd3154887`, including four-checkpoint kernels and profile-selected managed loading |
| NCCL | Public NVIDIA source base `73cf112295c33aee2b895f329f592f2a9b4b0f97` plus the cumulative `patches/nccl.patch`; preserves switchless routing and independent PCIe-domain discovery |
| SparkCache | `d0cf7296062ec8b4d17d65cd05a416d509e80bd8`, reproduced from its public base and packaged patch; includes capture-job read leases, request cleanup, private restore admission, and the exact common-vLLM source contract |
| Startup | Four source-pinned helpers under `startup/`; six C1 sampler requests and three C2 filtered pairs when concurrency permits, with validated SSE completion/usage and allocation-aware scheduler liveness |
| TP2 transport | Exact adaptive-grid RoCEnante source under `runtime/transport_profiles/`; selected before B12X import, with independent proxy/kernel and file verification |

| Profile | Checkpoint and parallelism | Communication and cache |
| --- | --- | --- |
| `tp4-dcp1-mtp3-prefill` | NVFP4-Spark; TP4/DCP1, MTP3, 1,048,576-token request limit | Weighted mesh and dual-domain NCCL; SparkCache disabled |
| `tp4-dcp4-mtp3-prefill` | NVFP4-Spark; TP4/DCP4, MTP3, 1,048,576-token request limit | Mesh owner exchange with fused endpoints and dual-domain NCCL; SparkCache disabled |
| `tp4-dcp1-mtp3-sparkcache` | NVFP4-Spark; TP4/DCP1, MTP3, 1,048,576-token request limit | Mesh and dual-domain NCCL; bounded asynchronous capture and verified restore |
| `glm53-flash-spark-tp2-mtp3` | NVFP4-Spark; TP2/DCP1, MTP3, 262,144-token request limit, 8.75 GiB KV per rank | One physical DAC using both host domains; managed loading; SparkCache disabled |
| `glm53-flash-spark-tp4-switched-mtp3` | NVFP4-Spark; TP4/DCP1, MTP3, 1,048,576-token request limit | Ordinary NCCL over operator-selected connected HCAs; custom transports and SparkCache disabled |

All listed profiles enable coalescing and mHC prefill sharding. Compact index
cache is disabled. DCP2 is supported by the prefill source but has no declared
launch profile in this lock. TP2 keeps sequential KDA execution; TP4 retains
its side-stream setting. The two communication implementations occupy separate
paths and share the installed model kernels.

The SparkCache profile uses two 512-MiB capture slots, two restore workers,
256 MiB of restore arenas, and an 8-GiB disk-cache limit per rank. Its maximum
persisted span is 65,536 tokens; longer prompts remain eligible for ordinary
inference. The source contract refuses mismatched vLLM ownership semantics.
SparkCache on the NVFP4-Spark TP2 profile is unsupported pending dedicated
validation.

The parent supplies generated support files and 15 compiled vLLM libraries.
The source lock records their complete retained inventory. Torch, Triton,
CUTLASS DSL, Transformers, and FlashInfer versions
are checked independently. This establishes file identity; a rebuilt NCCL
library and complete serving behavior still require verification.

## Prepare and build

Run preparation from the repository root with Python 3.12 or later and Git.
Preparation downloads the exact public source bases, verifies patch hashes,
applies patches to an index, and checks complete resulting Git trees. It
does not invoke Docker or contact inference hosts.

Download the pinned native input bundle from the
[SM121 native-files prerelease](https://github.com/FujitsuPolycom/sparkring/releases/tag/native-runtime-sm121-aa8fa11831af).
Preparation verifies its complete archive and library hashes before creating
the image context.

```bash
curl --fail --location \
  https://github.com/FujitsuPolycom/sparkring/releases/download/native-runtime-sm121-aa8fa11831af/native-runtime-files-20260908.tar \
  --output /tmp/native-runtime-files-20260908.tar
python runtime/sparkring/source_image/prepare_image.py \
  --output /tmp/sparkring-glm-image-context \
  --source-cache /tmp/sparkring-glm-image-sources \
  --native-files /tmp/native-runtime-files-20260908.tar
```

Both paths must initially be absent. To prepare another context from the same
sources, use a different output directory and `--reuse-source-cache`. Reuse
checks the base commit, complete patched index tree, unstaged changes, and
untracked files before accepting any source directory.

Build on an ARM64 Docker host with the parent image available:

```bash
docker build --platform linux/arm64 --network none \
  -t sparkring-glm53-source \
  /tmp/sparkring-glm-image-context
```

The command above reuses the exact pinned native libraries. Omitting
`--native-files` selects the optional native source-rebuild path, which
requires separate binary and GPU qualification. That path compiles NCCL and
the native snapshot library without GPU access.
NCCL uses 16 parallel compile jobs. Its compile-only diagnostic entrypoint
accepts `--jobs` to select a count from 1 through 64.
The compiler wrapper assigns a distinct source-derived random seed and keeps
compilation intermediates beside their explicit object outputs. Dependency
scans and device linking retain separate temporary handling. NVIDIA documents
this filename behavior for
[`--objdir-as-tempdir`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#file-and-path-specifications).
Python packages install offline without dependency resolution. NCCL's
CPU routing compatibility test runs before compilation. The resulting NCCL
library must match the measured SHA-256 in the lock. A mismatch stops the
build; do not replace that hash solely to make the check pass. Record compiler
and linker differences, compare source and binary behavior, and qualify the
rebuilt library before accepting another binary identity.

The image also installs the TP2 transport bundle, import hook, and profile
assets. Source preparation verifies their hashes from this checkout; no
private source archive or running container is needed. The transport hook is
inactive unless the selected profile enables it.

The container path `/opt/sparkcache-jj-runtime` and its manifest schema names
are retained compatibility interfaces for source installation and verification.
They do not enable SparkCache serving.

## Download the published image

The [publication record](publication.json) binds the generic `sparkring`
repository, immutable image digest, source lock and five CPU profile checks.
Status: **research-only**; these checks do not establish GPU serving results.

```bash
SPARKRING_IMAGE=ghcr.io/fujitsupolycom/sparkring@sha256:f25cdb6bf7df85754ea5c62445b139dac3143bc6e910a0442d681b6e63c31c4b
docker pull "$SPARKRING_IMAGE"
```

Prepare the matching context with the pinned native files below, then run the
image verifier with `--image "$SPARKRING_IMAGE"`. Preparation fetches source
inputs and creates verification artifacts; a downloaded image does not require
`docker build`. The verifier also inspects the declared parent image locally:

```bash
docker pull ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:11a556a54041fd823d152a7f051ac4f7c617dc539030df26e93008392fee0746
```

Use a checkout whose source lock matches `source_lock_sha256` in the
publication record. The verifier rejects mismatched source or profile assets.
The publication record is distribution evidence; use the generated profile
receipt for launch admission.

## Use pinned native files

To reuse the exact NCCL and snapshot libraries declared by `native_files` in
the source lock, supply that archive during preparation:

```bash
python runtime/sparkring/source_image/prepare_image.py \
  --output /tmp/sparkring-glm-image-context \
  --source-cache /tmp/sparkring-glm-source-cache \
  --native-files /tmp/native-runtime-files-20260908.tar
```

This selects `pinned` native mode. Preparation checks the archive hash and
size, its exact six-member inventory, every member's bytes and header policy,
and both libraries' AArch64 ELF headers and existing runtime hashes. The
retained NCCL archive must match its trusted digest and source tree. SparkCache's
native Git subtree is recomputed from the actual source bytes and executable
modes; Python-only changes outside that subtree do not change the native ABI
proof. The archive's original source-lock reference remains provenance, not
the identity of the newly prepared composition.

Installation writes only the two fixed runtime library paths and their
licenses/provenance under `/opt/sparkring/native-files`. All destinations are
checked before writing. NCCL and snapshot compilation are skipped in this mode;
their source archives remain available for inspection. The manifest, installed
state, and CPU image receipt bind the selected mode and exact native artifact.
Verification rejects changed archives, installed files, metadata, or source
identities. It makes no byte-identical rebuild claim and does not qualify the
new shared image's GPU behavior.

Omitting `--native-files` retains `compile` mode, including both native builds
and their strict output-hash guards. An unsuccessful compile never falls back
to pinned files automatically.

## Verify and use the existing installer

Create a CPU-only receipt for the selected profile:

```bash
python runtime/sparkring/source_image/verify_image.py \
  --image sparkring-glm53-source \
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

Use the [GLM-5.3 Flash TP4 Ring quickstart](../../../docs/GLM53_TP4_PREFILL_QUICKSTART.md)
for DCP1 or DCP4. To enable the bounded cache configuration, verify and select
`tp4-dcp1-mtp3-sparkcache` in both the receipt and private mesh site.
The [NVFP4-Spark TP2 guide](../../profiles/glm53-flash-spark-tp2/README.md)
uses the same image with a separate manual launcher. Each launch path verifies
the shared sources before entering their profile-specific startup code.

The [switched TP4 quickstart](../../../docs/GLM53_SWITCHED_TP4_QUICKSTART.md)
uses ordinary NCCL and the generic sampling warmup. Switched deployments are
provided as-is. This profile has not been validated on switched hardware.

DeepSeek, Qwen, and EXL3 still use their documented model-family builders.
This GLM recipe does not establish a universal native runtime for those
families.

## Validation

```bash
python -m pytest runtime/sparkring/source_image runtime/profiles runtime/transport_profiles -q
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
