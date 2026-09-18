# Shared runtime component provenance

Components retain their own licenses; this combined image is not licensed solely
under SparkRing's Apache-2.0 license. Model weights are not bundled.

| Role | Source identity | License |
|---|---|---|
| vLLM serving runtime | LIL `35bab057b1751a6076a457803bcc4b78809689cf` plus [complete patch](sources/vllm.patch); reconstructed tree `4190972dc2f5e8c5f62d072bfb2211c6b88ba81c61e323cb2e28719ee26bf79c` | Apache-2.0 |
| B12X kernels | LIL `a83336581a3a907076e60797df69ab66df5a2ff1` plus [complete patch](sources/b12x.patch); includes [PR394](https://github.com/local-inference-lab/b12x/pull/394) head `7530ea27d7d923dbc71a76a356922bd8b6b1611a` | Apache-2.0 |
| Qwen HC/coalescing and paired scoring origins | [Original-el8/Jason provenance](../../images/compositions/lil-r37-qwen-prefill/provenance.json); the release patches identify the Kraken adaptation and multimodal forwarding fix | Upstream notices retained |
| SparkCache | Integrated `76598d1d368d7a8fad2ebd03b7a0550e09eb7f99`; [cache composition](../../images/compositions/lil-r37-cache64/descriptor.json) | Apache-2.0 |
| Isolated SGLang | `e087e662ba1ac4ef7747537e2a9141085efd4561` and retained [integration](../../deepseek-v41-sglang/combined-image/README.md) | Apache-2.0 |
| Mia adapter | `e59e6eb67479aa68f6fa700c600dc90a0729b5ec`, with retained integration edits and complete source/notices | AGPL-3.0-or-later; retained MIT and Apache notices |
| NCCL, RoCEnante, SIRCL and dependencies | Unchanged native payloads and [inherited component inventory](../shared-2026.09.0-rc.1/components.md); [third-party notices](../../../THIRD_PARTY_NOTICES.md) | Component-specific terms |

The immutable [inherited source bundle](../shared-2026.09.0-rc.1/source-bundle.json)
supplies the unchanged foundation integrations, including complete modified Mia
source and notices. This release's vLLM/B12X manifest and patches supersede those
two components in that bundle. SparkRing integration and launcher source is in
the matching repository release tag. Benchmark results are not part of the
release qualification.

The exact inherited NCCL ARM-port patch bytes are not included; this is not an
offline source-only reconstruction claim for the entire foundation. The image's
`/opt/sparkring/receipts/native-installed.json` remains the installed-payload
authority. The release's external source index identifies the accepted sources;
older embedded metadata remains attributed to its original version.
