# What is a SparkRing image?

A SparkRing image is a Docker/OCI application container for Linux ARM64 GB10
hosts. It contains a prepared inference software stack. It is not a bootable
operating-system image. Pulling it neither configures networking nor starts a model.

## Included in the shared serving image

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
