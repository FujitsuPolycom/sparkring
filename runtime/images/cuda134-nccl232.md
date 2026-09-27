# CUDA 13.4.2 and NCCL 2.32.3 toolchain layer

Status: **research-only**. The toolchain layer adds the CUDA 13.4.2 toolkit and
runtime libraries and a patched NCCL 2.32.3 library to a SparkRing serving image,
without rebuilding its PyTorch, vLLM or FlashInfer native binaries. The
[installer images](#installer-image-assembly) carry this layer. The layer itself has no
serving qualification; each profile states its own evidence scope.

Two preparers produce it. Neither invokes Docker:

| Preparer | Input | Output |
|---|---|---|
| [toolchain_context.py](toolchain_context.py) | [cuda134-nccl232.json](cuda134-nccl232.json) and the NCCL source archive | CUDA toolkit stage, NCCL build script, and CUDA-only, NCCL-only or combined layers over the lock's parent |
| [toolchain_assembly.py](toolchain_assembly.py) | [cuda134-nccl232-installer.json](cuda134-nccl232-installer.json), a built toolkit stage, the NCCL build output and a software-layer parent | Combined layer over that parent |

[cuda134-nccl232.json](cuda134-nccl232.json) pins the ARM64 CUDA base,
toolkit/compatibility package versions, NCCL source archive and routing patch,
and a parent image, SparkRing `shared-2026.09.4-rc.4`. Toolkit package 13.4.2-1
updates the CUDA 13.4.1 base. The NVIDIA compatibility libraries are contained in
the image; no host driver is installed.

The retained PyTorch wheel was built against CUDA 13.0. The layer selects the
CUDA 13.4 runtime/toolkit and compiles NCCL with that toolkit; it does not
relabel or rebuild PyTorch, vLLM or FlashInfer native binaries. Their
compatibility on SM121 is part of each image's qualification.

## Build from source

Check out the NCCL revision in the lock and export it without working-tree
changes, using LF Git objects:

```bash
git clone https://github.com/NVIDIA/nccl.git /var/tmp/nccl-toolchain-source
git -c core.autocrlf=false -C /var/tmp/nccl-toolchain-source archive --format=tar \
  -o /var/tmp/nccl-v2.32.3.tar 12df1a11afad322be5a204a2db890161cbf8131d
python runtime/images/toolchain_context.py \
  --nccl-source /var/tmp/nccl-v2.32.3.tar \
  --output /var/tmp/sparkring-toolchain-context
docker build --build-arg BUILD_JOBS=2 \
  -t sparkring:rc4-cuda1342-nccl2323 /var/tmp/sparkring-toolchain-context
```

Use an unused output directory. `--variant cuda` and `--variant nccl` prepare
separate controls over the same parent. The combined variant is the default.
The NCCL build runs the GPU-free routing-handle test and compiles SM121 code.
CUDA package dependencies are resolved during the build; the complete installed
toolkit is hashed into the resulting image receipt rather than assumed to be
byte-reproducible from the package version alone.

The cuBLAS packages are pinned explicitly because NVIDIA's 13.4.1 base holds
their versions. Updating the toolkit to 13.4.2 also requires cuBLAS 13.8.0.4;
the build permits changing those two held packages inside the image.

To build the toolkit stage and NCCL separately, build only the reusable compiler
stage, then compile NCCL in a resource-limited container. From a prepared
context directory:

```bash
docker build --target toolchain_builder -t sparkring:cuda1342-nccl-builder .
mkdir /var/tmp/nccl232-artifacts
docker run --name sparkring-nccl232-build --cpus=2 --memory=12g --memory-swap=12g \
  -e BUILD_JOBS=2 -v "$PWD":/inputs:ro \
  -v /var/tmp/nccl232-artifacts:/output \
  --entrypoint bash sparkring:cuda1342-nccl-builder /inputs/build-nccl.sh
tar -C /var/tmp/nccl232-artifacts -czf /var/tmp/nccl232-artifacts.tar.gz .
```

This build needs no GPU. The output includes NCCL libraries, headers, licenses,
input lock/patch, compiler version, package inventory and the library digest.
The compiler image supplies the CUDA toolkit for assembly.

## Installer image assembly

[cuda134-nccl232-installer.json](cuda134-nccl232-installer.json) records the
toolkit stage and NCCL build that the installer images use: toolkit stage image
configuration `sha256:70b4f8d44a2a…`, its `docker save` export, the NCCL build
archive and the `libnccl.so.2.32.3` digest. `toolchain_source_commit` names the
source revision whose `toolchain_context.py` generated that stage and build
script; the `toolchain_context.py` in this directory generates the same
`cuda134` and `toolchain_builder` stages and the same `build-nccl.sh`. The
lock's `parent` is empty; assembly records the actual software-layer parent.

```bash
python runtime/images/toolchain_assembly.py \
  --parent-release SOFTWARE_LAYER_LABEL \
  --parent-image sha256:SOFTWARE_LAYER_IMAGE_ID \
  --parent-receipt external-base-installed.json \
  --parent-tag sparkring:SOFTWARE_LAYER_TAG \
  --nccl-artifacts /var/tmp/nccl232-artifacts.tar.gz \
  --version-label IMAGE_VERSION_LABEL \
  --output /var/tmp/sparkring-toolchain-assembly
docker build -t sparkring:TOOLCHAIN_TAG /var/tmp/sparkring-toolchain-assembly
docker run --rm --network none --read-only sparkring:TOOLCHAIN_TAG verify
```

`--parent-receipt` is `/opt/sparkring/receipts/external-base-installed.json`
copied out of the parent image. The toolkit stage is referenced through the local
tag `sparkring:installer-toolkit-70b4f8d4`, or `--toolkit-tag`; tag the recorded
image configuration before building. Assembly refuses an NCCL archive or library
whose digest differs from the lock and an archive entry that leaves its directory.
The build checks the parent receipt and NCCL library digests before sealing the
toolchain receipt, and also installs [toolchain_gpu_smoke.py](toolchain_gpu_smoke.py)
as `/opt/sparkring/toolchain/gpu_smoke.py`, a single-GPU import, matrix
multiplication and PTX JIT probe.

The published image `dev-20260924-cuda1342-nccl2323-status031` uses this
assembly with parent release label `installer-status031-software-46ecfc996597`,
parent image `sha256:01869ac593003b144416828b000b33febb626179dc901d14907ce6a202150373`,
parent receipt SHA-256 `1354e3d0270b297dafafd39e5341210f4d0e7bd908b15b0026531c4e0350da62`
and version label `shared-2026.09.4-rc.5-candidate.20260924-status031`; with
those inputs the preparer writes the Dockerfile and `toolchain.json` of that
image's build context byte for byte. Its toolchain receipt SHA-256 is
`d72be89ed1714ce3739a8236dc1c64d34e73f795b5f28997b6c7aff1a34d72f7`, as its
[installer image lock](../releases/dev-20260924-cuda1342-nccl2323-status031/installer-image.json)
records.

## Verification and serving boundary

The entrypoint defaults to `verify`. It verifies the inherited parent receipt,
the copied toolkit and the NCCL library against a separate toolchain
receipt. The parent's owned files and receipt remain unchanged. `serve` verifies
both inventories, then starts the parent's vLLM CLI with the selected library
for both PyTorch and vLLM. Inherited profile NCCL overrides are replaced
before the serving process is executed; unrelated preloads are preserved.

CUDA library selection covers both the ELF loader and Torch's internal
vendor-library search. The layer preloads the selected toolkit runtime/math
libraries and supplies a namespace under `/opt/sparkring/toolchain/python`
whose `nvidia/cu13/lib` points to the toolkit and `nvidia/nccl/lib` points to
the selected NCCL directory. Torch can explicitly open NCCL by its vendor
package path even when another version is preloaded. Both aliases are needed
for the combined layer. This leaves inherited wheel
payloads and metadata intact while preventing their bundled CUDA 13.0 libraries
from mixing with the selected CUDA 13.4 set. The aliases are recorded in the
toolchain receipt. The entrypoint re-executes with normalized loader settings
before verification, including when a profile supplies an empty `PYTHONPATH`
or an inherited NCCL override. Library-selection normalization is idempotent.

Do not bypass this entrypoint with the parent's `external-base.py` when testing
the layer. A profile adapter must explicitly invoke the toolchain entrypoint
and retain its rank, model, cache and transport settings. Use a distinct JIT-cache
directory for each comparison arm and verify the actual mapped CUDA/NCCL paths
in the worker processes. The image does not upgrade a running container.

Qualifying an image that carries the layer covers: parent inventory, Python/native
imports, selected CUDA runtime and driver versions, SM121 JIT execution,
changing-payload TP2/TP4 AllReduce/AllGather/ReduceScatter, graph replay,
direct-peer routing and PCI-domain traffic attribution, then bounded model
checks. SparkCache publication and restore compatibility are outside this
layer's checks. SIRCL and RoCEnante dispatch must remain matched when
attributing NCCL performance.

The GPU-free import regression in `runtime/images/test_toolchain_imports.py`
runs inside an image that carries the layer and checks library mappings after
each media, Torch, vLLM and B12X import. It skips on hosts without the installed
toolchain.

## Routing patch

The [NCCL 2.32.3 patch](../../spark_transport/nccl/nccl-2.32.3-dual-pci-domain.patch)
ports the four-IPv4-GID listener format, PCI-root-preserving subnet preference,
final-QP diagnostics and optional Tree/PAT connection guard. The extension
preserves the listener prefix and bounds the handle at 112 bytes. All added
policy flags default off. Existing profiles select their tested network policy.
The 2.32.3 PAT initialization is retained after the optional ring-only guard.
Do not layer another SparkRing switchless patch over this cumulative patch.
