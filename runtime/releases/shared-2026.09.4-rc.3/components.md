# Component sources and licenses

SparkRing integration code is Apache-2.0. Bundled components retain their own
terms; the image is not universally Apache-2.0. Model weights are not bundled.

The admitted vLLM and B12X source trees, complete patches, checksummed archives
and component notices are byte-identical to the assets published with
[`shared-2026.09.4-rc.1`](https://github.com/FujitsuPolycom/sparkring/releases/tag/shared-2026.09.4-rc.1).
Their immutable hashes are listed in [source-bundle.json](source-bundle.json).

| Component | Source or version | Terms |
|---|---|---|
| vLLM | LIL `af9e4dca109e0348323c0182e98a3aaf7282bfc3` plus the recorded SparkRing patch | Apache-2.0 |
| B12X | LIL `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` plus the recorded SparkRing patch | Apache-2.0 |
| SparkCache | Retained hybrid-cache integration with request-scope isolation | Apache-2.0 |
| CUDA Python / CUDA bindings | 13.3.1 public wheels | NVIDIA Software License; wheel notices retained |
| CUDA core | 1.0.1 public wheel | Apache-2.0; wheel notice retained |
| CUDA Pathfinder, CUTLASS DSL, PyTorch | Retained 1.8.1 / 4.6.2 / 2.13.0 | Respective package licenses and notices |
| SGLang and Mia adapter | Retained isolated foundation payload | Apache-2.0 and AGPL-3.0-or-later; inherited notices apply |
| NCCL, RoCEnante and SIRCL | Retained transport payloads and provenance | Component-specific terms; see the foundation inventory |

The prepared RoCEnante manifest is rebound only to the hashes of the installed
Kraken B12X preparation interfaces. The installed-runtime receipt owns that
manifest. Runtime flattening changes image-layer geometry, not component source
or licensing.
