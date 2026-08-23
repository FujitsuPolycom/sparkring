# Qwen3.8-27B public ARM64 runtime builder

This directory builds a local ARM64/SM121 image for the four-Spark
Qwen3.8-27B EXL3 K5/K6 profile. Every source and patch comes from a public,
immutable input. The build does not use the maintainer-held runtime archive
described by the performance record.

Status: **experimental, offline-validated builder; live validation pending.** Offline tests cover
pin handling, prepared-source receipts, the container contract, and runtime
verification logic. A locally produced image has no four-rank evidence until
that exact image ID completes the Qwen startup and functional checks.

## Why no published image is required

The runtime can be built once on an ARM64 DGX Spark, exported with
`docker save`, and loaded on the other three ranks. Loading the same OCI archive
gives every rank the same image ID without requiring a public registry. A
published image would shorten setup time, but it is not an input to this
builder.

The final image contains the Python environment, source trees, Qwen chat
template, patched NCCL, and the tracked rank launcher under `/ws`. It contains
no model weights, site addresses, credentials, rank environment, benchmark
artifacts, or external key-value cache.

## Public inputs

[`pins.json`](pins.json) records the complete input set. The important pins
are:

| Input | Identity |
|---|---|
| CUDA devel parent | `nvcr.io/nvidia/cuda@sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d` |
| Companion recipe | `FujitsuPolycom/qwen38-spark-pair@b9e1031b80b6f3f64bfc75ae3922322f56954fd6` |
| vLLM | `FujitsuPolycom/vllm@229effc810ee6b8112f661472f6aace4eb8c787d` |
| ExLlamaV3 | `turboderp-org/exllamav3@5f3c537ca9d89893d771256f5c43c93656553fbb` plus the companion ARM64 patch |
| Torch | `2.12.0+cu132` from the public PyTorch CUDA 13.2 index |
| B12X | `1.2.4` from PyPI |
| NCCL | `73cf112295c33aee2b895f329f592f2a9b4b0f97` plus the two tracked switchless-cycle patches |

The companion freeze contains two local-path requirements and LMCache. The
prepared context removes those three entries: ExLlamaV3 and vLLM build from
their pinned public trees, and the base Qwen profile does not include LMCache.
Any other local path, editable requirement, or Git URL in that freeze is a hard
error.

## Build

Build on an ARM64 host. The source builds are compute- and disk-intensive;
vLLM can take several hours on a DGX Spark.

```bash
parent='nvcr.io/nvidia/cuda@sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d'
docker pull "$parent"
parent_id="$(docker image inspect --format '{{.Id}}' "$parent")"

BASE_IMAGE="$parent" \
BASE_IMAGE_ID="$parent_id" \
BASE_IMAGE_LICENSES='LicenseRef-NVIDIA-Deep-Learning-Container' \
IMAGE='sparkring-qwen38:arm64-sm121' \
  bash ./runtime/qwen38/build-image.sh
```

The wrapper rejects a different parent reference, a mismatched resolved image
ID, dirty or untracked builder inputs, source commit/tree drift, patch-hash
drift, and a patched NCCL binary whose SHA-256 differs from
`e69a8c240f45d10166bcd901d99db78bb63147adda66e586d8dd505c6d608b54`.

NCCL builds in an untouched stage from the CUDA parent used by the known
artifact. The serving stage installs CUDA 13.2. ExLlamaV3 builds with
`TORCH_CUDA_ARCH_LIST=12.1`; vLLM builds with the required family target
`TORCH_CUDA_ARCH_LIST=12.0f`.

## Verify

The image build runs the identity and import verifier without loading a model.
On the build host, rerun the same check with GPU access:

```bash
docker run --rm --gpus all \
  --entrypoint /ws/venv/bin/python \
  sparkring-qwen38:arm64-sm121 \
  /ws/runtime/verify_runtime.py --imports
```

The verifier checks the vLLM, patched ExLlamaV3, and patched NCCL Git trees;
the NCCL library and chat-template hashes; installed Torch, torchvision, B12X,
and Ray versions; the CUDA compiler version; and optional native imports.
The Containerfile runs it with `--imports`, so vLLM import and the
`exllamav3_ext.exl3_gemm` ABI are image-build gates as well as post-build
checks.

Extract the build's source receipt and verify that its SHA-256 matches the OCI
label:

```bash
IMAGE=sparkring-qwen38:arm64-sm121
docker run --rm \
  -e LD_PRELOAD= -e VLLM_NCCL_SO_PATH= \
  --entrypoint cat "$IMAGE" \
  /ws/runtime/source-receipt.json > qwen38-source-receipt.json
receipt_sha=$(sha256sum qwen38-source-receipt.json | awk '{print $1}')
label_sha=$(docker image inspect --format \
  '{{index .Config.Labels "org.sparkring.source-receipt-sha256"}}' "$IMAGE")
test "$receipt_sha" = "$label_sha"
```

Retain the receipt, hash, and immutable image ID as the identity of the local
build.

The image command is `sleep infinity` because one identical image serves two
roles: a persistent rank container and a verified runtime carrier. Starting a
rank remains an explicit operator action through `/ws/qwen38_dgx4_serve.sh`;
loading the image never starts a model, claims a GPU, or replaces an existing
service.

## Distribute one built image

Export the image once and record the archive hash:

```bash
docker save --output qwen38-runtime.oci.tar sparkring-qwen38:arm64-sm121
sha256sum qwen38-runtime.oci.tar > qwen38-runtime.oci.tar.sha256
```

Copy that archive to the other ranks, verify its SHA-256 there, then load it:

```bash
sha256sum --check qwen38-runtime.oci.tar.sha256
docker load --input qwen38-runtime.oci.tar
docker image inspect --format '{{.Id}}' sparkring-qwen38:arm64-sm121
```

All four ranks must report the same image ID. Registry push and pull are an
optional replacement for this save/load step.

Mount only the mutable inputs beneath the baked `/ws` tree: the model at
`/ws/model/Qwen3.8-27B-EXL3-K5K6-hydrated`, one local rank environment at
`/ws/rank.env`, and writable cache/log directories at `/ws/cache` and
`/ws/logs`. Do not bind-mount a host directory over all of `/ws`, because that
would hide the baked environment and source trees required by the launcher.

The image's default command is `sleep infinity`, so loading or accidentally
starting the image does not launch a model. The canonical quickstart creates
the rank containers directly with an explicit launcher entrypoint and `--run`;
do not create a separate idle set first.

Continue with the repository's Qwen four-Spark quickstart after recording the
locally built image ID. Building, saving, loading, or starting an image changes
host state. Starting it can stop an existing model service.

## Evidence boundary

The source inputs are public, but the resulting image is a distinct artifact.
The maintainer-held TP4 measurements do not transfer to it. Record the image
ID, source receipt, model hashes, startup result, functional checks, and
post-run health before describing the image as live-validated.
