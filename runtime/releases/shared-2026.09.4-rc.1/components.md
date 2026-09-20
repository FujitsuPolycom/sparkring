# Component sources and licenses

SparkRing integration code is Apache-2.0. Bundled components retain their own
terms; the image is not universally Apache-2.0. Model weights are not bundled.

| Component | Source or version | Terms |
|---|---|---|
| vLLM | LIL `af9e4dca109e0348323c0182e98a3aaf7282bfc3` plus [SparkRing patch](sources/vllm.patch) | Apache-2.0 |
| B12X | LIL `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` plus [SparkRing patch](sources/b12x.patch) | Apache-2.0 |
| SparkCache | Retained hybrid-cache integration with request-scope isolation; exact installed files are recorded in the image receipt | Apache-2.0 |
| CUDA Python / CUDA bindings | 13.3.1 public wheels | NVIDIA Software License; full wheel notices retained |
| CUDA core | 1.0.1 public wheel | Apache-2.0; full wheel notices retained |
| CUDA Pathfinder, CUTLASS DSL, PyTorch | Retained 1.8.1 / 4.6.2 / 2.13.0 | Respective package licenses and notices |
| SGLang and Mia adapter | Retained isolated foundation payload | Apache-2.0 and AGPL-3.0-or-later respectively; additional inherited notices apply |
| NCCL, RoCEnante and SIRCL | Retained transport payloads and provenance | Component-specific terms; see [foundation inventory](../shared-2026.09.0-rc.1/components.md) |

The [source manifest](sources/manifest.json) identifies complete patches and
checksummed archives for the admitted vLLM/B12X source trees. External test-harness
corrections are qualification artifacts, not unrecorded changes to these trees.
The inherited source bundle remains available with
[2026.09.0-rc.1](https://github.com/FujitsuPolycom/sparkring/releases/tag/shared-2026.09.0-rc.1),
including modified Mia source and notices. The exact inherited NCCL ARM-port
patch is not included; these records do not establish an offline rebuild of
every foundation library.

CUDA Python and bindings license text is retained under their respective
`/opt/venv/lib/python3.12/site-packages/*-13.3.1.dist-info/licenses/LICENSE`
paths. CUDA core retains its Apache text in
`cuda_core-1.0.1.dist-info/licenses/LICENSE`. The NVIDIA text has SHA256
`807b0df78905550dfe7c2ea11414cb74ea278eec67ad7aec271b4f0b7300cbd3`.
No notices are replaced by an Apache-only label.

Qwen prefill adaptations retain
[original-el8/Jason provenance](../../images/compositions/lil-r37-qwen-prefill/provenance.json).
RoCEnante retains [Luke and LIL attribution](../../../third_party/b12x_roce/provenance.json).
Source inclusion is not model qualification or a performance claim.
