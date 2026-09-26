# What is a SparkRing image?

A SparkRing image is a Docker container image for Linux ARM64 GB10 hosts that
holds a prepared inference software stack. It is not an operating-system
image: pulling it neither configures networking nor starts a model.
`sudo sparkring install` pulls and runs the right image for you; see
[Install SparkRing](install.md). This page lists what each image contains.

| Image | Used by | Registry reference | Image ID |
|---|---|---|---|
| Installer image | The six `sparkring install` profiles | `ghcr.io/fujitsupolycom/sparkring@sha256:451c5e23a90e0df2fc904e8851aab12c3ec9ffdcd1258b6f14cf502222e46b5f` | `sha256:4100e1d2bd038f885d92f8c0021d482b23f9a38a003cd7bfc3e700e7e0afa971` |
| Shared 2026.09.3 image | The [manual setup](setup.md) profiles and others on release `shared-2026.09.3` | `ghcr.io/fujitsupolycom/sparkring@sha256:2375f876bc9ea065e85ae10cebad7a8db8a2ec0e6862b4441c269c5bf56365c6` | `sha256:bc16a9819d853b42c28823c9c937638b545787a7d305917ff00f2ff902d04855` |

Each profile's `profile.json` names its image release; other profiles use
other releases.

## Installer image

Development image, tag `dev-20260925-qwendecode-cuda1342-nccl2323-status031`.
The download is 14.2 GiB and the unpacked image 29.5 GiB. Its
[installer image lock](../../runtime/releases/dev-20260925-qwendecode-cuda1342-nccl2323-status031/installer-image.json)
lists the six profiles and pins its identity; its
[publication record](../../runtime/releases/dev-20260925-qwendecode-cuda1342-nccl2323-status031/publication.json)
names the parent image and the added layer.

| Component | Purpose |
|---|---|
| `eugr/spark-vllm-b12x:nightly-20260924` base | vLLM with B12X kernels and loaders for GB10 |
| CUDA 13.4.2 and NCCL 2.32.3 | CUDA runtime and the NCCL library the installer selects |
| Paced RoCEnante transport (`tp2-rocenante-adaptive-prepared`) | Collectives whose forwarded-path send window bounds traffic relayed by a ring node |
| Runtime-status dashboard 0.3.1 | `/v1/sparkring/status/view` on the model API port |
| Qwen decode layer | Skinny-GEMM plans for BF16 projections on GB10, and the `VLLM_QWEN4_EXP_MXFP8_HC` setting, off unless a profile sets it |

`sparkring install` starts it with the image's entrypoint, a per-rank
runtime-binding file, the NCCL 2.32.3 library paths and a seccomp policy that
allows `io_uring` (`runtime/common/loader-seccomp.json`). The manual Qwen
launcher, `runtime/common/qwen_flash_next.py`, refuses this image; use
`sparkring install` or [Compose](compose.md).

## Shared 2026.09.3 image

See the [2026.09.3 release](../../runtime/releases/shared-2026.09.3/README.md).

| Component | Purpose |
|---|---|
| ARM64 userspace and GPU dependencies | CUDA 13.3 and PyTorch 2.13.0 |
| Patched vLLM and B12X | Serving, model loading, kernels and model-specific integrations |
| SparkCache and native cache libraries | Persistent KV cache, when a profile enables it |
| NCCL, RoCEnante and SIRCL integrations | Collectives, as the profile selects |
| Isolated SGLang environment and Mia adapter | A separate engine with its own Python dependencies and NCCL |
| Verification helpers, receipts, contracts and notices | Installed-file identities, integration checks and attribution |

Components are off unless the profile enables them, and a profile uses one
serving engine. Some kernels still compile at startup. The
[component record](../../runtime/releases/shared-2026.09.3/components.md) lists
sources and libraries.

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

`python3 scripts/sparkring.py setup show PROFILE_ID` prints a profile's image
and checkpoint from the repository records; the profile guide then pulls the
image and checks its local image ID. Do not pick an image by tag alone.
