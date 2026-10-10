# What is a SparkRing image?

A SparkRing image is a Docker container image with the inference software for
Linux ARM64 GB10 hosts. It is not an operating-system image: pulling it neither
configures networking nor starts a model. `sudo sparkring install` pulls and
runs the right image for you ([Install SparkRing](install.md)).

| Image | Used by | Registry reference | Image ID |
|---|---|---|---|
| Installer image | The `sparkring install` profiles, the 13 its lock lists | `ghcr.io/fujitsupolycom/sparkring@sha256:4fffc4dc3074d5539f4e9d3a013ff9ef4e0be570a95b74d4646ee341da1f6911` | `sha256:d52737a109e083d3eef05c0fc0db4d09fc1bf34485a13467382130963ecfe66b` |
| 2026.10.1 installer image | `sparkring install --image 2026.10.1`, and [Compose](compose.md) exports | `ghcr.io/fujitsupolycom/sparkring@sha256:71d410571407fef3ce2959c6d392f5a2c3f44b757b856e853a71f6e3295620ad` | `sha256:aba309e4610c711fda219ed7478a1d68d9bf16dfbd83a0653e32afcbd8f0106f` |
| Shared 2026.09.3 image | The [manual setup](setup.md) profiles and others on release `shared-2026.09.3` | `ghcr.io/fujitsupolycom/sparkring@sha256:2375f876bc9ea065e85ae10cebad7a8db8a2ec0e6862b4441c269c5bf56365c6` | `sha256:bc16a9819d853b42c28823c9c937638b545787a7d305917ff00f2ff902d04855` |

Each profile's `profile.json` names its image release; other profiles use
other releases.

## Installer image

Development image of release 2026.10.2, tag
`dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`, built in five layers on
the [2026.10.1 installer image](#2026101-installer-image). The download is
14.2 GiB and the unpacked image 29.7 GiB; a Spark that holds
2026.10.1's image downloads the 6 layers it adds, 8.3 MiB.

| Layer | Adds |
|---|---|
| Kraken CSF sources | The vLLM and B12X sources that read GLM-5.3-Flash's CSF checkpoint ([derive_kraken_csf_sources.py](../../runtime/images/derive_kraken_csf_sources.py)) |
| SIRCL 0.3.2 | SIRCL ring sessions and their two prebuilt native libraries ([SIRCL layer](../../runtime/images/installer-images.md#sircl-layer)) |
| libsircl 0.6.0 | SIRCL's NCCL-API library, built from `spark_transport/libsircl` with its kernel packs and fail-stop mode; it creates a communicator on the current device when no context is current ([libsircl layer](../../runtime/images/installer-images.md#libsircl-layer)) |
| GLM-5.3 plugins | The vLLM general plugins `glm_dsa_indexer_split` 1.1.0, `glm53full_speedups` 1.1.0 and `glm_dcp_decode_comm` 2.0.1 ([derive_glm53_plugins.py](../../runtime/images/derive_glm53_plugins.py)) |
| runtime-status 0.3.6 | The status dashboard, which names the collective transport and the SIRCL version of the tensor-parallel group ([derived_layer.py](../../runtime/images/derived_layer.py)) |

- The [installer image lock](../../runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/installer-image.json)
  (`sparkring-installer-image/v3`) lists its 13 profiles and pins the
  image's identity.
- The [release record](../../runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/README.md)
  states its layers, evidence and limitations.

On a fabric that `sudo sparkring setup` recorded, `sparkring install` runs
profiles on this image with SIRCL ring sessions and NCCL off
([transport and receipts](install-reference.md#transport-and-receipts)), and
on the prepared transport elsewhere. Its GLM-5.3-Flash profiles install the
CSF checkpoint by default
([default checkpoint by profile](../../profiles/glm53-checkpoints.md#default-checkpoint-by-profile)).

## 2026.10.1 installer image

Development image of release 2026.10.1, tag
`dev-20261004-kraken-cuda1342-nccl2323-status034`: the parent and rollback
image of 2026.10.2 (`--image 2026.10.1`) and the image of Compose exports.
The download is 14.2 GiB and the unpacked image 29.7 GiB. It shares 32 of its
45 registry layers with `dev-20261001-kraken-cuda1342-nccl2323-status034`, so
a Spark that holds that image downloads the other 13, about 3.8 GiB.

- The [installer image lock](../../runtime/releases/dev-20261004-kraken-cuda1342-nccl2323-status034/installer-image.json)
  lists the nine profiles and pins the image's identity.
- The [publication record](../../runtime/releases/dev-20261004-kraken-cuda1342-nccl2323-status034/publication.json)
  names the base image and the two layers.
- The [composition record](../../runtime/images/compositions/external-kraken-20261004/README.md)
  lists the source commits, the merge decisions and the pinned build inputs.

`dev-20261001-kraken-cuda1342-nccl2323-status034`, built the same way from
SparkRing's `sparkring/kraken-beta-20261001` branches
([composition record](../../runtime/images/compositions/external-kraken-20261001/README.md)),
is the rollback image: `sudo sparkring install --profile PROFILE --image 2026.10.0`
installs a profile on it ([Another image](install-reference.md#another-image)).

| Component | Purpose |
|---|---|
| `eugr/spark-vllm-b12x` nightly-20261001 base | Torch 2.13.0 for CUDA 13.0, FlashInfer 0.7.1 and vLLM's compiled extensions, built for GB10 (SM121a) |
| vLLM and [B12X](https://github.com/local-inference-lab/b12x) sources | Local Inference Lab's Karmic Kraken beta branches (`integration/karmic-kraken-beta`) merged with SparkRing's changes: branches `sparkring/kraken-beta-20261004` of [FujitsuPolycom/vllm](https://github.com/FujitsuPolycom/vllm/tree/sparkring/kraken-beta-20261004) and [FujitsuPolycom/b12x](https://github.com/FujitsuPolycom/b12x/tree/sparkring/kraken-beta-20261004) |
| CUDA 13.4.2 and NCCL 2.32.3 | CUDA runtime and the NCCL library the installer selects |
| Prepared B12X RoCE transport bundle (`tp2-rocenante-adaptive-prepared`) | Collectives over RoCE through B12X's RoCE communication package (`b12x.comm.roce`, called RoCEnante), on profiles of two and four Sparks alike: the `tp2-` name is kept for compatibility, and when SIRCL is loaded it replaces this bundle's all-reduce slot. A send window bounds the traffic a ring node relays. A rank waits up to `B12X_ROCE_PEER_TIMEOUT_S` seconds (300 by default) for a late peer and logs waits over 5 s ([peer wait](../../integrations/vllm/rocenante_prepared/README.md#peer-wait), [#278](https://github.com/FujitsuPolycom/sparkring/issues/278)) |
| RoCE GID index per port | Each HCA uses the RoCE GID index of its fabric address, read at startup. NCCL still uses index 3, which the installer restores before a model starts ([RoCE GID index 3](install-reference.md#roce-gid-index-3)). Ranks of images with proxy ABI 5 and 6 refuse to connect, so all Sparks must run the same image ([GID index per port](../../integrations/vllm/rocenante_prepared/README.md#gid-index-per-port)) |
| Runtime-status dashboard 0.3.4 | [Status dashboard](dashboard.md) at `/v1/sparkring/status/view` on the model API port: settings, memory, transport and versions, with only the rows to check colored. The settings include the reasoning and tool-call parsers, the default chat template arguments and the shared-memory reader window |
| Qwen decode layer | Skinny-GEMM plans for BF16 projections on GB10; the `VLLM_QWEN4_EXP_MXFP8_HC` setting, off unless a profile sets it |
| Host-to-device staging fix | vLLM's `CpuGpuBuffer.copy_to_gpu` copies through fresh pinned memory, so a queued copy cannot pick up later writes to its host buffer. Without it, Qwen with MTP, async scheduling and FULL CUDA graphs decoded about 0.2-0.5% of concurrent requests as token 8191 (` Register`) repeated ([#294](https://github.com/FujitsuPolycom/sparkring/issues/294)) |
| B12X selection-cache reconciliation | When the ranks of a two- or four-Spark deployment share kernel tuning, B12X reads its tuning cache after they reconcile it, so a restart reuses earlier tuning instead of measuring every kernel again; this image's B12X does it in its preparation session ([selection cache](../../integrations/b12x/selection_cache/README.md)). On image `dev-20260927-b12xcache-cuda1342-nccl2323-status032`, a DeepSeek-V4.1-Flash four-Spark restart was healthy after 200-225 s with it, 652-741 s without |
| MiMo vision attention sinks | The MiMo-V2.6 vision encoder applies its per-head attention sinks in the softmax denominator, as the model was trained ([derive_mimo_vision.py](../../runtime/images/derive_mimo_vision.py)). With the sinks on each image's first key instead, MiMo read a red-and-blue test image as black and white |
| Tool-result contract | A Chat Completions request whose `tool_choice` is `required` or names a function, and whose output lacks a complete call, gets HTTP 400 if the token limit ended generation and HTTP 500 otherwise, not HTTP 200 without a tool call ([tool-result contract](../../integrations/vllm/tool_choice_contract/README.md), [#217](https://github.com/FujitsuPolycom/sparkring/issues/217)). `SPARKRING_TOOL_CHOICE_CONTRACT=0` in a profile's environment turns it off |
| Shared-memory reader window | vLLM's shared-memory readers poll for `SPARKRING_SHM_BUSY_LOOP_S` seconds after a read when that variable is set, and for one second otherwise ([derive_spin_wait.py](../../runtime/images/derive_spin_wait.py)). `sparkring install --save-cpu` sets it to 2 ms, and is refused on an image without it ([serving settings](install-reference.md#serving-settings), [#189](https://github.com/FujitsuPolycom/sparkring/issues/189)) |

This image carries no SIRCL layer, so its deployments run on the prepared
transport. A kraken-line image with the SIRCL layer
([SIRCL layer](../../runtime/images/installer-images.md#sircl-layer)) adds the
SIRCL package and its two prebuilt native libraries; `sparkring images` lists
`sircl` among its transports, and `sparkring install` runs profiles on it with
SIRCL ring sessions and NCCL off
([transport and receipts](install-reference.md#transport-and-receipts)).
The image that the
[`dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034` recipe](../../runtime/releases/dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034/README.md)
builds also has the vLLM and B12X sources that read GLM-5.3-Flash's CSF
checkpoint; on it, the GLM profiles of two and four Sparks install that
checkpoint by default
([default checkpoint by profile](../../profiles/glm53-checkpoints.md#default-checkpoint-by-profile)).

`sparkring install` starts the image with its entrypoint, a per-rank
runtime-binding file, the NCCL 2.32.3 library paths and a seccomp policy that
allows `io_uring` (`runtime/common/loader-seccomp.json`). Run it with
`sparkring install` or [Compose](compose.md); the manual profile launcher,
`runtime/common/toolchain_profiles.py`, refuses it.

## Shared 2026.09.3 image

See the [2026.09.3 release](../../runtime/releases/shared-2026.09.3/README.md).
The [component record](../../runtime/releases/shared-2026.09.3/components.md)
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

Do not pick an image by tag alone.
`python3 scripts/sparkring.py setup show PROFILE_ID` prints a profile's image
and checkpoint from the repository records; the profile guide then pulls the
image and checks its local image ID.
