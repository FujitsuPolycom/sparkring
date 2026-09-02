# GLM-5.3 Flash GB10 operator image

This directory builds and runs one Linux/ARM64 image for GLM-5.3 Flash on four
NVIDIA GB10 systems. The image combines the Local Inference Lab GLM-5.3 vLLM
runtime recorded as `Jovian Judgement Community R10`, BF16 DFlash2
speculation, B12X kernels, switchless NCCL, fastsafetensors, and SparkCache.
One image supports TP4 with DCP1, DCP2, or DCP4.

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
[`GLM-5.3 GB10 quickstart`](../../docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md)
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
| shared GPU-prefix retention | up to 300 seconds |

DCP1 resolves to one-token KV interleaving without full-CKV gather. DCP2 and
DCP4 resolve to four-token KV interleaving with full-CKV gather. Operators can
change every value in the environment file without rebuilding the image.
`SPARKCACHE_ENABLED=0` omits the persistent connector while retaining vLLM's
GPU prefix cache; `SPARKCACHE_ENABLED=1` enables both layers.

With the connector enabled, `SPARKCACHE_ACCESS_MODE=read-write` restores and
publishes persistent entries. `restore-only` reuses compatible entries but
does not capture or publish new prompt state. Missing entries are computed by
vLLM normally. `store-only` and `disabled` are diagnostic modes.

`SPARKCACHE_CACHE_NAMESPACE` selects rank-local storage and JIT directories.
The template uses the semantic default `glm53-flash-dcp4-snapshot-v1`. The
directory name is not part of SparkCache's content identity or stored format;
changing it selects a different root and therefore discovers a different set
of stored entries.

Set `SPARKCACHE_ASYNC_PAGE_CAPTURE=1` to capture complete manager-page
snapshots through the bounded CUDA ring. `SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES`
defaults to 8 GiB for DCP1, 5 GiB for DCP2, and 3 GiB for DCP4. DCP4 with two
3 GiB slots is **qualified** for asynchronous publication of the recorded
124,928-token, 231.8 MiB-per-rank snapshot. The image also restored retained
900K and 1M entries, but did not asynchronously publish entries at those
sizes. Larger asynchronous publication and DCP1/DCP2 asynchronous capture
have no live qualification record. Asynchronous capture requires
`snapshot-v1`; page-tail publication is unsupported in this image.

## Build from pinned source

The builder accepts clean checkouts at the exact vLLM and SparkCache commits
recorded in `pins.json`. Build the attested SparkCache snapshot library on an
ARM64 CUDA 13 host before invoking the image builder:

```bash
cmake -S /source/sparkcache/sparkcache/native \
  -B /source/sparkcache/sparkcache/native/build-cuda \
  -G Ninja -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build /source/sparkcache/sparkcache/native/build-cuda \
  --target spark_cache_snapshot
```

The image builder verifies commits, trees, package subtrees, runtime files,
the parent image, retained native extensions, and both SparkCache CUDA
libraries before producing an image.

```bash
python runtime/glm53-flash-jj-r8-gb10/build_image.py \
  --vllm-source /source/vllm \
  --sparkcache-source /source/sparkcache \
  --snapshot-library /source/sparkcache/sparkcache/native/build-cuda/libspark_cache_snapshot.so \
  --output-image sparkring-glm53-sparkcache:local-arm64 \
  --receipt ./glm53-build-receipt.json
```

The build does not include model checkpoints, site addresses, SSH
credentials, or persistent cache data.

## Published image

Pull the immutable Linux/ARM64 image:

```text
ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:bc7d079f16ff4a418669c58c5250f2da52e989a0c5805569ba9429d41b765f65
```

Its local Docker image ID is
`sha256:35f397668c01075d0bdd28bbdb3398afd3744df6086646c6f68bcf7ebe7f918f`.
Construction, direct-fabric distribution, profile smoke tests, historical
deep-context evidence, and limitations are recorded in
[`async-capture-image-receipt.json`](async-capture-image-receipt.json),
[`ASYNC_CAPTURE_IMAGE_VALIDATION.md`](ASYNC_CAPTURE_IMAGE_VALIDATION.md), and the
[`deep-context record`](../../performance/records/glm53-flash/dcp1-deep-context-boundary-20260831.md).

Run the offline contracts with:

```bash
python -m pytest runtime/glm53-flash-jj-r8-gb10 -q
```
