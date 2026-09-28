# What is a SparkRing image?

A SparkRing image is a Docker container image for Linux ARM64 GB10 hosts that
holds a prepared inference software stack. It is not an operating-system
image: pulling it neither configures networking nor starts a model.
`sudo sparkring install` pulls and runs the right image for you; see
[Install SparkRing](install.md). This page lists what each image contains.

| Image | Used by | Registry reference | Image ID |
|---|---|---|---|
| Installer image | The nine `sparkring install` profiles | `ghcr.io/fujitsupolycom/sparkring@sha256:6459a148c95ef9de3730eaaa242016e3493a095005c3bc78c0b93d35ebc492e7` | `sha256:fb6be60ff426d17f69a287f1f488e22ee037cf7ff865b2cd3971cb16e3dca03c` |
| Shared 2026.09.3 image | The [manual setup](setup.md) profiles and others on release `shared-2026.09.3` | `ghcr.io/fujitsupolycom/sparkring@sha256:2375f876bc9ea065e85ae10cebad7a8db8a2ec0e6862b4441c269c5bf56365c6` | `sha256:bc16a9819d853b42c28823c9c937638b545787a7d305917ff00f2ff902d04855` |

Each profile's `profile.json` names its image release; other profiles use
other releases.

## Installer image

Development image, tag `dev-20260927-mimovision-cuda1342-nccl2323-status032`.
The download is 14.2 GiB and the unpacked image 29.5 GiB. Its
[installer image lock](../../runtime/releases/dev-20260927-mimovision-cuda1342-nccl2323-status032/installer-image.json)
lists the nine profiles and pins its identity; its
[publication record](../../runtime/releases/dev-20260927-mimovision-cuda1342-nccl2323-status032/publication.json)
names the parent image and the added layer. The
[installer image builders](../../runtime/images/installer-images.md) list the
builder of each layer in the chain.

| Component | Purpose |
|---|---|
| `eugr/spark-vllm-b12x:nightly-20260924` base | vLLM with [Local Inference Lab's B12X](https://github.com/local-inference-lab/b12x) kernels and loaders for GB10 |
| CUDA 13.4.2 and NCCL 2.32.3 | CUDA runtime and the NCCL library the installer selects |
| Paced RoCEnante transport (`tp2-rocenante-adaptive-prepared`) | Collectives whose forwarded-path send window bounds traffic relayed by a ring node |
| Runtime-status dashboard 0.3.2 | `/v1/sparkring/status/view` on the model API port; the header shows the served model name and the checkpoint architecture on its own line |
| Qwen decode layer | Skinny-GEMM plans for BF16 projections on GB10, and the `VLLM_QWEN4_EXP_MXFP8_HC` setting, off unless a profile sets it |
| Host-to-device staging fix | vLLM's `CpuGpuBuffer.copy_to_gpu` copies through fresh pinned memory, so a host buffer rewritten for the next step while its copy is still queued cannot change the copied data. Without it, Qwen with MTP, async scheduling and FULL CUDA graphs decoded about 0.2-0.5% of concurrent requests as token 8191 (` Register`) repeated ([#294](https://github.com/FujitsuPolycom/sparkring/issues/294)) |
| B12X selection-cache correction | When kernel tuning is shared across the ranks of a two- or four-Spark deployment, B12X reads its tuning cache once the ranks have reconciled it, so a restart reuses earlier tuning instead of measuring every kernel again ([integrations/b12x/selection_cache](../../integrations/b12x/selection_cache/README.md)). A DeepSeek-V4.1-Flash four-Spark restart became healthy in 200-225 s instead of 652-741 s |
| MiMo vision attention sinks | The MiMo-V2.6 vision encoder applies its per-head attention sinks in the softmax denominator, as the model was trained ([derive_mimo_vision.py](../../runtime/images/derive_mimo_vision.py)). With the sink added to each image's first key instead, MiMo read a red-and-blue test image as black and white |

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
