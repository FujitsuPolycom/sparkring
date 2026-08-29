# Build the GLM-5.3 Flash ARM64 runtime

Status: **research-only**. The builder constructs and verifies a runtime from
public, immutable inputs. Its output has not passed the four-Spark TP4/DCP1
qualification recorded for the rank-local images in `pins.json`.

The image contains vLLM, B12X, InstantTensor, CUDA runtime libraries, and a
source-built NCCL 2.30.7 library for a switchless four-Spark cycle. It does not
contain the GLM-5.3 target checkpoint, the DFlash2 checkpoint, site addresses,
credentials, or persistent context data.

## Build host

Use a Linux ARM64 host with Docker BuildKit, Git, Python 3, internet access,
and at least 250 GiB of free local storage. A DGX Spark can compile the image;
the build does not require stopping a running service but competes for CPU,
memory, storage, and network bandwidth. Do not build on a serving rank during a
latency-sensitive workload.

The builder rejects uncommitted changes to its tracked inputs. Start from an
immutable SparkRing revision:

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git sparkring
git -C sparkring checkout --detach <immutable-sparkring-revision>
```

Replace the documentation token with the revision containing this builder.
Run from any directory and write the machine-specific receipt outside the
checkout:

```bash
IMAGE='sparkring-glm53-runtime:da4d7be-source-arm64' \
BUILD_RECEIPT="$PWD/glm53-runtime-image-receipt.json" \
bash sparkring/runtime/glm53-flash/build-image.sh
```

`build-image.sh` performs these checks before returning:

1. clones vLLM, B12X, and NCCL at the commits and Git trees in `pins.json`;
2. downloads the InstantTensor source distribution and verifies its SHA-256;
3. applies the source-bound SparkRing NCCL patch and verifies the patched Git
   tree;
4. builds vLLM and B12X for Linux ARM64 and CUDA SM121;
5. builds NCCL from source for SM121;
6. assembles the runtime without model weights;
7. verifies the image platform, OCI labels, Python imports, source receipt,
   and NCCL binary; and
8. writes a `sparkring-glm53-runtime-image/v1` receipt.

The build receipt assigns **implemented** status to that image ID. Four-rank
serving remains unqualified until the same registry digest is pulled on every
rank and passes the cache-enabled and cache-disabled procedures in `docs/`.

## Build inputs

`pins.json` is the canonical machine-readable contract. Its public build
section records:

- the build-only PyTorch ARM64 image and distributed CUDA runtime parent by
  digest;
- vLLM and B12X commits and Git trees;
- NVIDIA NCCL tag `v2.30.7-1`, commit, base tree, SparkRing patch SHA-256, and
  patched tree;
- the InstantTensor source URL and SHA-256; and
- toolchain and CUDA architecture settings.

Changing any source, parent image, patch, compiler input, or architecture
creates a different implemented image and requires a separate live receipt.
