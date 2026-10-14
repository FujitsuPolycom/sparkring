# What is a SparkRing image?

A SparkRing image is a Docker container image with the inference software for
Linux ARM64 GB10 hosts. It is not an operating-system image: pulling it neither
configures networking nor starts a model. `sudo sparkring install` pulls and
runs the right image for you ([Install SparkRing](install.md)).

| Image | Used by | Registry reference | Image ID |
|---|---|---|---|
| Installer image, release 2026.10.4 | `sparkring install` without `--image`, for the 15 profiles its lock lists | `ghcr.io/fujitsupolycom/sparkring@sha256:e21ba936d4fd6bfeb6e7a651e54a03ac461bf4e1e6f5721c1a34845c80acd76f` | `sha256:3cc9a81fd344fe52fe9cc261b46dc051e3c3f494e51b8142846a5f854835c502` |
| Installer image, release 2026.10.2 | `sparkring install --image 2026.10.2`, the rollback | `ghcr.io/fujitsupolycom/sparkring@sha256:4fffc4dc3074d5539f4e9d3a013ff9ef4e0be570a95b74d4646ee341da1f6911` | `sha256:d52737a109e083d3eef05c0fc0db4d09fc1bf34485a13467382130963ecfe66b` |
| Installer image, release 2026.10.1 | `sparkring install --image 2026.10.1`, and [Compose](compose.md) exports | `ghcr.io/fujitsupolycom/sparkring@sha256:71d410571407fef3ce2959c6d392f5a2c3f44b757b856e853a71f6e3295620ad` | `sha256:aba309e4610c711fda219ed7478a1d68d9bf16dfbd83a0653e32afcbd8f0106f` |
| Shared 2026.09.3 image | The [manual setup](setup.md) profiles whose release is `shared-2026.09.3` | `ghcr.io/fujitsupolycom/sparkring@sha256:2375f876bc9ea065e85ae10cebad7a8db8a2ec0e6862b4441c269c5bf56365c6` | `sha256:bc16a9819d853b42c28823c9c937638b545787a7d305917ff00f2ff902d04855` |

Installer images are built in layers, each on the one below it in the table:
release 2026.10.4's image is 2026.10.2's with four layers added, and
2026.10.2's is 2026.10.1's with five. Each is a 14 GiB download and 30 GiB
unpacked, but a Spark that already holds the parent image downloads only the
added layers: 3 MiB for 2026.10.4 over 2026.10.2, 8 MiB for 2026.10.2 over
2026.10.1.

`sparkring images` lists every installer image this package can run, the
default first, with the GitHub release that published it, its transports and
the profiles it runs. `--image` accepts a release tag (`2026.10.2`), a full
image name, or a part of the name that only one image has
([Another image](install-reference.md#another-image)).

## Installer image

Release 2026.10.4, image `dev-20261013-dcp2prefill-status038-sircl-tp6`. It
adds these layers to the [2026.10.2 image](#2026102-installer-image):

| Layer | Adds |
|---|---|
| TP2/DCP2 prefill admission | vLLM sources that let GLM-5.3-Flash prefill on two Sparks with decode-context parallelism 2 ([composition](../../runtime/images/compositions/dcp2-prefill/README.md)) |
| runtime-status 0.3.7 | The dashboard's vLLM statistics logger ([derived_layer.py](../../runtime/images/derived_layer.py)) |
| runtime-status 0.3.8 | The statistics logger and the resident quantization facts of GLM-5.3 on six Sparks ([derived_layer.py](../../runtime/images/derived_layer.py)) |
| SIRCL 0.3.2, rebuilt | SIRCL ring sessions that admit GLM-5.3 on six Sparks; native libraries unchanged ([SIRCL layer](../../runtime/images/installer-images.md#sircl-layer)) |

On a fabric that `sudo sparkring setup` recorded, `sparkring install` runs
profiles on this image with SIRCL ring sessions and NCCL off, and on the
prepared RoCE transport elsewhere
([transport and receipts](install-reference.md#transport-and-receipts)).
The GLM-5.3-Flash two- and four-Spark profiles and `qwen38-flash-next-tp2`
install their CSF checkpoints by default
([default checkpoint by profile](../../profiles/glm53-checkpoints.md#default-checkpoint-by-profile)).

Status: **implemented**, release maturity candidate. Three profiles were
installed and served on this image; the others passed the installer's image
checks without being started. The
[release record](../../runtime/releases/dev-20261013-dcp2prefill-status038-sircl-tp6/README.md)
states the evidence and limitations, and the
[installer image lock](../../runtime/releases/dev-20261013-dcp2prefill-status038-sircl-tp6/installer-image.json)
(`sparkring-installer-image/v3`) lists its 15 profiles and pins its identity.

## 2026.10.2 installer image

Release 2026.10.2, image
`dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`: the
parent of the 2026.10.4 image and its rollback (`--image 2026.10.2`). It runs
profiles on the same transports. Over the 2026.10.1 image it adds:

- the vLLM and B12X sources that read GLM-5.3-Flash's CSF checkpoint;
- SIRCL 0.3.2 ring sessions and their two prebuilt native libraries
  ([SIRCL layer](../../runtime/images/installer-images.md#sircl-layer));
- libsircl 0.6.0, SIRCL's NCCL-API library, built from `spark_transport/libsircl`
  ([libsircl layer](../../runtime/images/installer-images.md#libsircl-layer));
- the GLM-5.3 vLLM plugins `glm_dsa_indexer_split`, `glm53full_speedups` and
  `glm_dcp_decode_comm`;
- runtime-status 0.3.6, whose dashboard names the collective transport.

Its [lock](../../runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/installer-image.json)
lists 13 profiles; the
[release record](../../runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/README.md)
states its layers, evidence and limitations.

## 2026.10.1 installer image

Release 2026.10.1, image `dev-20261004-kraken-cuda1342-nccl2323-status034`:
the base of the two images above and the image of Compose exports. It has no
SIRCL layer, so its deployments run on the prepared RoCE transport. Its
[lock](../../runtime/releases/dev-20261004-kraken-cuda1342-nccl2323-status034/installer-image.json)
(`sparkring-installer-image/v2`) lists nine profiles.

| Component | Purpose |
|---|---|
| `eugr/spark-vllm-b12x` nightly-20261001 base | Torch 2.13.0 for CUDA 13.0, FlashInfer 0.7.1 and vLLM's compiled extensions, built for GB10 (SM121a) |
| vLLM and [B12X](https://github.com/local-inference-lab/b12x) sources | Local Inference Lab's `integration/karmic-kraken-beta` branches merged with SparkRing's changes ([composition record](../../runtime/images/compositions/external-kraken-20261004/README.md)) |
| CUDA 13.4.2 and NCCL 2.32.3 | CUDA runtime and the NCCL library the installer selects |
| Prepared RoCE transport (`tp2-rocenante-adaptive-prepared`) | Collectives over RoCE through B12X's RoCE communication package, RoCEnante, on two- and four-Spark profiles ([prepared transport](../../integrations/vllm/rocenante_prepared/README.md)). All Sparks of a deployment must run the same image |
| runtime-status 0.3.4 | The [model status page](dashboard.md) at `/v1/sparkring/status/view` on the model API port |
| SparkRing vLLM and B12X changes | B12X [selection-cache reconciliation](../../integrations/b12x/selection_cache/README.md), the [tool-result contract](../../integrations/vllm/tool_choice_contract/README.md), the shared-memory reader window that `--save-cpu` sets ([serving settings](install-reference.md#serving-settings)), MiMo-V2.6 vision attention sinks ([derive_mimo_vision.py](../../runtime/images/derive_mimo_vision.py)), and Qwen decode and host-to-device staging fixes |

The [publication record](../../runtime/releases/dev-20261004-kraken-cuda1342-nccl2323-status034/publication.json)
names the base image and the layers.

## How the installer starts an image

`sparkring install` starts an installer image with its entrypoint, a per-rank
runtime-binding file, the NCCL 2.32.3 library paths and a seccomp policy that
allows `io_uring` (`runtime/common/loader-seccomp.json`). Run installer images
with `sparkring install` or [Compose](compose.md); the manual profile
launcher, `runtime/common/toolchain_profiles.py`, refuses them.

## Shared 2026.09.3 image

The image of the [2026.09.3 release](../../runtime/releases/shared-2026.09.3/README.md);
its [component record](../../runtime/releases/shared-2026.09.3/components.md)
lists sources and libraries.

| Component | Purpose |
|---|---|
| ARM64 userspace and GPU dependencies | CUDA 13.3 and PyTorch 2.13.0 |
| Patched vLLM and B12X | Serving, model loading, kernels and model-specific integrations |
| SparkCache and native cache libraries | Persistent KV cache, when a profile enables it |
| NCCL, RoCEnante and SIRCL integrations | Collectives, as the profile selects |
| Isolated SGLang environment and Mia adapter | A separate engine with its own Python dependencies and NCCL |
| Verification helpers, receipts, contracts and notices | Installed-file identities, integration checks and attribution |

Components are off unless the profile enables them, and a profile uses one
serving engine. Some kernels compile at startup.

## Supplied separately

- Host Linux, NVIDIA driver, Docker and NVIDIA Container Toolkit.
- Cables, addresses, routes, RDMA configuration and host services.
- The complete model checkpoint on every rank.
- Private rank configuration, model paths and writable cache storage.
- Host-side scripts from the matching SparkRing checkout or package.

Model files and cache entries live outside the container, so replacing a
container keeps them.

## Names used in setup

| Term | Meaning |
|---|---|
| Checkout | Host-side scripts, profiles and documentation at one Git revision |
| Profile | Model, topology, serving settings and image selection |
| Checkpoint | Model files at a pinned repository revision |
| Site | Your SSH targets, addresses, interfaces and storage paths |
| Registry digest | Immutable reference for `docker pull` |
| Local image ID | Docker's image configuration ID; differs from the registry digest |
| Receipt | A record of specific identity or verification checks |

Do not pick an image by tag alone. For manual setup,
`python3 scripts/sparkring.py setup show PROFILE_ID` prints the image and
checkpoint that a profile's own record names, and the profile guide pulls the
image and checks its local image ID. For an installer profile that record
names the 2026.10.1 image; `sparkring install` uses its default image or the
one `--image` selects instead.
