# Installer image on LIL's Karmic Kraken beta branches

Status: **implemented**. These are the build inputs of installer image
`dev-20261001-kraken-cuda1342-nccl2323-status034`
([release lock](../../../releases/dev-20261001-kraken-cuda1342-nccl2323-status034/installer-image.json)).
The image has two layers over an external base.

| Layer | Image ID | Builder | Inputs |
|---|---|---|---|
| Base | `sha256:b094a6744a9b11da9366575745245c9c81d31c565030d1acd226c0018d1a774f` | `eugr/spark-vllm-b12x@sha256:141f46a4a2c3751798f16759cc859648784be430be852a84a21f0c4c427b4052` (nightly-20261001) | Local Inference Lab vLLM `dev/karmic-kraken` at `ab86b70734009c34e91db456bb96340c3faf3d4e`, B12X 1.5.0 at `e4084d2eef4932e0fa06db3f7a94deb83ad132d7`, Torch 2.13.0 for CUDA 13.0, FlashInfer 0.7.1, built for SM121a |
| Software | `sha256:4bf6cd655635a53eebe3685d2488b3cb91acf29b786e1365f2576379229ea77d`, composition `dfc4bae5c4facec45e60b37a19ee82795de741ebec77a3cf9f81b6f6818eda8c` | `external-arm64`, [external_context.py](../../external_context.py) | [inputs.json](inputs.json) |
| CUDA 13.4.2 and NCCL 2.32.3 | `sha256:9f02bcbfee89fa092f6edf85d5915bfd73f226bd0faf7d171fe32d7f7b8f1e84` | `cuda134-nccl232-assembly`, [toolchain_assembly.py](../../toolchain_assembly.py) | [toolchain-lock.json](toolchain-lock.json) |

## Sources

The software layer replaces the base's vLLM and B12X Python sources with
SparkRing's merges of Local Inference Lab's `integration/karmic-kraken-beta`
branches; the base's compiled vLLM extensions stay, because the merged vLLM
changes no file under `csrc/`, `cmake/` or `setup.py` relative to the base's
vLLM commit.

| Package | Source commit | Branch | Merged upstream commit |
|---|---|---|---|
| vLLM | `fe0a30fcb9afee42f4a3c23ca658657c6de5009e` | [FujitsuPolycom/vllm `sparkring/kraken-beta-20261001`](https://github.com/FujitsuPolycom/vllm/tree/sparkring/kraken-beta-20261001) | `4a379ed42881ee022aaf5d9ada554f9096e5acf3` |
| B12X | `6ca214e76877095d5c8c6d50c709a366cf35899f` | [FujitsuPolycom/b12x `sparkring/kraken-beta-20261001`](https://github.com/FujitsuPolycom/b12x/tree/sparkring/kraken-beta-20261001) | `b557d87850cc836268fd327ccd46eaa6a033cdbf` |

Each branch is SparkRing's source composition of the `dev-20260924` installer
image merged with the upstream commit. The vLLM branch also carries, as one
commit, the vLLM source edits that the derived layers of the
`dev-20261001-statusrows-cuda1342-nccl2323-status034` chain apply: TP2 HC
token-row ownership, Qwen decode plans, host-to-device staging, MiMo vision
attention sinks, the tool-result contract and the shared-memory reader window.
B12X's preparation session reads its selection cache only after the ranks
reconcile it, which the
[selection-cache integration](../../../../integrations/b12x/selection_cache/README.md)
adds to the other chain.

The integration assets (transports, features, SparkCache files, licenses and
the NCCL 2.31.2 library under `/opt/local-inference`) are exported from
`dev-20261001-statusrows-cuda1342-nccl2323-status034` (image ID
`sha256:490a668978e1b93886c13b63553594c9a2ded5514b01d737b57b1e12d3e83b5a`,
installed receipt `9e2086910544ab96bcace992178175b564b0b3385d1ad95a1893faa318c58b5f`).
The deployment add-ons and runtime-status 0.3.4 are the same artifacts that
image installs.

## Rebuild

The inputs manifest pins every artifact by SHA-256; the files it names are not
stored here. On an ARM64 Docker host that holds the base and
`dev-20261001-statusrows` images, in an empty directory:

1. `git archive --format=tar -o vllm-kraken-fe0a30fc.tar fe0a30fc vllm` from the
   vLLM branch, and `vllm-base-ab86b707.tar` from `ab86b707`; the same for B12X
   with `b12x` (`6ca214e7`, `e4084d2e`). Export with `core.autocrlf=false`.
2. Run [collect_base_inventory.py](programs/collect_base_inventory.py) in the
   base image with this directory mounted at `/candidate`, without network or GPU.
3. Run [export_installer_assets.py](programs/export_installer_assets.py) with the
   statusrows image ID and its receipt SHA-256 to write `assets/`.
4. Copy `runtime/images/external_base.py` and the five
   [Qwen prefill controller](../../../../integrations/vllm/qwen4_prefill/README.md)
   files into `controller/`; place the runtime-status wheel and source archive
   in `status/`, the add-on manifest and archive in `addons/`, and the
   `dev-20260924` vLLM and B12X source archives in `integration/`.
5. `python3 programs/make_inputs.py BASE_CONFIG_ID`, then
   `python3 runtime/images/external_context.py --manifest inputs.json --output CONTEXT`
   and `docker build --network none CONTEXT`.
6. Build the CUDA toolkit stage from
   [cuda134-nccl232.json](../../cuda134-nccl232.json)
   (`toolchain_context.py --variant cuda`, target `cuda134`), then run
   `toolchain_assembly.py` with [toolchain-lock.json](toolchain-lock.json), the
   software layer as parent and the NCCL build archive the lock pins.

The toolkit stage of this image was rebuilt from the pinned packages: its image
ID `sha256:0dd26ae86cf63c0d2b68a2ff3d59d34a1d7367f9408044e9e9ade7aa5d9a6115`
differs from the stage that
[cuda134-nccl232-installer.json](../../cuda134-nccl232-installer.json) records,
and its export was not archived. The NCCL library is the recorded
`libnccl.so.2.32.3`. Identical inputs produce identical context files
([external image composition](../../external-context.md)); Docker layer
metadata and image IDs of a rebuild differ, and no rebuild has been compared
with this image.
