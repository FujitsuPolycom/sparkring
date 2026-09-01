# GLM-5.3 Flash R8 runtime for GB10

This directory builds and runs one Linux/ARM64 image for GLM-5.3 Flash on four
NVIDIA GB10 systems. The image combines the Jovian Judgement R8 scheduler,
BF16 DFlash2 speculation, B12X kernels, switchless NCCL, fastsafetensors, and
SparkCache. One image supports TP4 with DCP1, DCP2, or DCP4.

Local Inference Lab supplies the model quantization and the primary runtime
work that makes this profile practical:

- [`local-inference-lab/GLM-5.3-Flash-NVFP4`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4)
  is the target checkpoint;
- [`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement)
  is the source of the Jovian Judgement GLM runtime and scheduler work;
- [`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x)
  supplies the GB10 kernel integration.

The external BF16 draft is
[`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
Exact revisions and source-tree hashes are in [`pins.json`](pins.json).

## Use the runtime

Follow the
[`GLM-5.3 R8 GB10 quickstart`](../../docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md)
to obtain or build the image, distribute it once through the direct fabric,
and start four ranks. [`runtime.env.example`](runtime.env.example) exposes the
model paths, image identity, DCP degree, context limit, scheduler budget, KV
allocation, speculation, cache limits, network interfaces, and ports.

The launcher defaults to:

| Setting | Value |
|---|---:|
| topology | TP4/DCP4 |
| maximum model length | 1,048,576 tokens |
| batched-token budget | 8,192 tokens |
| sequences | 16 |
| FP8 KV allocation | 24 GiB for the default DCP4 profile; 26 GiB for DCP1; 30 GiB for DCP2 |
| DFlash2 depth | 7 |
| SparkCache publication | complete `snapshot-v1` objects |

DCP1 resolves to one-token KV interleaving without full-CKV gather. DCP2 and
DCP4 resolve to four-token KV interleaving with full-CKV gather. Operators can
change every value in the environment file without rebuilding the image.
`SPARKCACHE_ENABLED=0` omits the persistent connector while retaining vLLM's
GPU prefix cache; `SPARKCACHE_ENABLED=1` enables both layers.

## Build from pinned source

The builder accepts clean checkouts at the exact vLLM and SparkCache commits
recorded in `pins.json`. It verifies the commits, trees, package subtrees,
runtime files, parent image, native extensions, and SparkCache CUDA placement
library before producing an image.

```bash
python runtime/glm53-flash-jj-r8-gb10/build_image.py \
  --vllm-source /source/vllm \
  --sparkcache-source /source/sparkcache \
  --output-image sparkring-glm53-jj-r8-sparkcache:local-arm64 \
  --receipt ./glm53-r8-build-receipt.json
```

The build does not include model checkpoints, site addresses, SSH
credentials, or persistent cache data.

## Published image

Pull the immutable Linux/ARM64 image:

```text
ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:380283a506aeb8f9d486a3c64cd738e44268c3cc21590913ea9e4685869f256a
```

Its local Docker image ID is
`sha256:b3a13d8003e7de30d7737fd33c8307404e506ba570240819ec7eb4f5c611400f`.
Construction, direct-fabric distribution, profile smoke tests, historical
deep-context evidence, and limitations are recorded in
[`public-image-receipt.json`](public-image-receipt.json),
[`PUBLIC_IMAGE_VALIDATION.md`](PUBLIC_IMAGE_VALIDATION.md), and the
[`deep-context record`](../../performance/records/glm53-flash/dcp1-deep-context-boundary-20260831.md).

Run the offline contracts with:

```bash
python -m pytest runtime/glm53-flash-jj-r8-gb10 -q
```
