# What is a SparkRing image?

A SparkRing image is a Docker/OCI application container for Linux ARM64 GB10
hosts. It contains a prepared inference software stack. It is not a bootable
operating-system image. Pulling it neither configures networking nor starts a model.

`sparkring install` runs every installer profile on one installer image. Other
profiles, including the SparkCache profiles in the Qwen guides, select their own
image release, such as the shared 2026.09.3 image. Each profile's
`profile.json` names its image release and evidence scope.

## Included in the installer image

The installer image is
`ghcr.io/fujitsupolycom/sparkring@sha256:451c5e23a90e0df2fc904e8851aab12c3ec9ffdcd1258b6f14cf502222e46b5f`
(tag `dev-20260925-qwendecode-cuda1342-nccl2323-status031`, image configuration
`sha256:4100e1d2bd038f885d92f8c0021d482b23f9a38a003cd7bfc3e700e7e0afa971`). Its
[installer image lock](../../runtime/releases/dev-20260925-qwendecode-cuda1342-nccl2323-status031/installer-image.json)
lists the six installer profiles and pins its identity and receipts; its
[publication record](../../runtime/releases/dev-20260925-qwendecode-cuda1342-nccl2323-status031/publication.json)
names the parent image, `dev-20260925-cuda1342-nccl2323-status031`, and the
derived layer. The registry download is 14.2 GiB and the unpacked image
29.5 GiB. Status: **implemented**, a development image; registry verification
does not establish serving correctness, and each profile states its own
evidence scope.

| Component | Purpose |
|---|---|
| `eugr/spark-vllm-b12x:nightly-20260924` base | vLLM with B12X kernels and loaders for GB10 |
| CUDA 13.4.2 and NCCL 2.32.3 toolchain | CUDA runtime and the NCCL library the installer's container settings select |
| Paced RoCEnante transport (`tp2-rocenante-adaptive-prepared`) | Collectives whose forwarded-path send window bounds traffic relayed by a ring node |
| Runtime-status dashboard 0.3.1 | `/v1/sparkring/status/view` on the model API port |
| Qwen decode layer | Skinny-GEMM plans for BF16 projections on GB10 and the `VLLM_QWEN4_EXP_MXFP8_HC` setting, off unless a profile sets it |

`sparkring install` runs this image with the image's verified entrypoint, a
per-rank runtime-binding file, the NCCL 2.32.3 library paths and a seccomp
policy that permits `io_uring`. The manual Qwen launcher,
`runtime/common/qwen_flash_next.py`, refuses to create containers from this
image. Run the installer profiles through `sparkring install`.

## Included in the shared 2026.09.3 image

The [2026.09.3 release](../../runtime/releases/shared-2026.09.3/README.md) includes:

| Component | Purpose |
|---|---|
| ARM64 userspace and GPU dependencies | The main environment uses CUDA 13.3 and PyTorch 2.13.0 |
| Patched vLLM and B12X | Serving, model loading, kernels and model-specific integrations |
| SparkCache and native cache libraries | Persistent cache capability selected by a deployment profile |
| NCCL, RoCEnante and SIRCL integrations | Communication capabilities selected by a deployment profile |
| Isolated SGLang environment and Mia adapter | A separate engine with its own Python dependencies and NCCL |
| Verification helpers, receipts, contracts and notices | Installed-file identities, integration checks and component attribution |

The [release component record](../../runtime/releases/shared-2026.09.3/components.md)
and its inherited inventory identify the sources and libraries. Inclusion does
not imply that a component is enabled or that every model is qualified. One
profile selects one serving engine. Some kernels still prepare during startup.
The source records do not establish an offline rebuild of every inherited native
library. Other SparkRing releases have different payloads and evidence.

## Supplied separately

- Host Linux, NVIDIA driver, Docker and NVIDIA Container Toolkit.
- Cables, addresses, routes, RDMA configuration and required host services.
- The complete model checkpoint on every participating rank.
- Private machine/rank configuration, model paths and writable cache storage.
- Host-side operator scripts from the matching SparkRing checkout.

Bundled host-service assets still require the documented host installation;
their presence in a container does not install a host service. Model files and
cache entries live outside the disposable serving container.

## Names used in setup

| Term | Meaning |
|---|---|
| Checkout | Host-side scripts, profiles and documentation at a recorded Git revision |
| Profile | Model, topology, serving settings, image selection and evidence scope |
| Checkpoint | Model files from a separately pinned repository revision |
| Site | Your SSH targets, addresses, interfaces and storage paths |
| Registry digest | Immutable reference used by `docker pull` |
| Local image ID | Docker's image configuration identity, different from the registry digest |
| Receipt | A record of specific identity/verification checks; its scope matters |

Use `python3 scripts/sparkring.py setup show PROFILE_ID` from the checkout to
read a published profile's selected image and checkpoint. It reads repository
records only; it does not authenticate locally installed files. The selected
guide supplies the installed-image checks. Do not choose an image by package
name alone or mix one release's receipt with another release's container.
