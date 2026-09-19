# Component sources and licenses

SparkRing's integration code is Apache-2.0. Other components retain their licenses;
the image is not universally Apache-2.0.

| Component | Source and terms |
|---|---|
| vLLM | LIL `35bab057b1751a6076a457803bcc4b78809689cf` plus the [complete patch](sources/vllm.patch); Apache-2.0 |
| B12X | LIL `a83336581a3a907076e60797df69ab66df5a2ff1` plus the [complete patch](sources/b12x.patch), including [PR #394](https://github.com/local-inference-lab/b12x/pull/394); Apache-2.0 |
| Qwen prefill integrations | [Original-el8/Jason provenance](../../images/compositions/lil-r37-qwen-prefill/provenance.json); upstream notices retained |
| SparkCache and Mia | Unchanged components and complete source records in the [foundation inventory](../shared-2026.09.0-rc.1/components.md); SparkCache Apache-2.0, Mia AGPL-3.0-or-later |
| NCCL, RoCEnante, SIRCL and dependencies | [Foundation inventory](../shared-2026.09.0-rc.1/components.md) and [third-party notices](../../../THIRD_PARTY_NOTICES.md); component-specific terms |

The [source manifest](sources/manifest.json) identifies full vLLM/B12X archives.
The inherited complete-source archive remains available with
[SparkRing 2026.09.0-rc.1](https://github.com/FujitsuPolycom/sparkring/releases/tag/shared-2026.09.0-rc.1),
including modified Mia sources and notices. The exact inherited NCCL ARM-port
patch is not included; these assets do not establish a complete offline rebuild
of every foundation library. Model weights are excluded and retain their terms.
