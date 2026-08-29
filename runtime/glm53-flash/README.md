# GLM-5.3 Flash runtime contract

Status: **qualified** for the exact source and artifact identities in
[`pins.json`](pins.json) when used by the SparkCache-enabled TP4/DCP1
composition. A rebuilt image has **implemented** status until it passes the
recorded live qualification.

The runtime uses the `local-inference-lab/vllm` GLM-5.3 implementation, B12X
kernels, SparkRing-patched NCCL 2.30.7, the public BF16 Inco DFlash2 drafter,
and SparkCache's `glm53-flash-hybrid` connector. SparkCache owns the derived
image recipe at `deploy/glm53_flash/Containerfile` in its source repository.
SparkRing owns the topology, launch, image-label, and artifact-attestation
contracts.

The source-complete public builder is documented in [`BUILD.md`](BUILD.md).
It fetches every source by immutable commit, builds NCCL from the NVIDIA
`v2.30.7-1` source with SparkRing's independently implemented switchless-cycle
patch, and emits an image receipt. The builder and its output are
**research-only** until one registry digest passes the four-rank checks. The
license and redistribution boundary is documented in
[`LICENSES/README.md`](LICENSES/README.md).

Before any rank starts, the runtime profile requires one exact image ID,
checks the source/profile labels, hashes the DFlash files and NCCL library,
checks the vLLM configuration postimage, and runs SparkCache's seven-file lease-contract
verifier against the installed vLLM files. The copied lease contract at
[`vllm-kv-block-lease-contract-da4d7be.json`](vllm-kv-block-lease-contract-da4d7be.json)
is a reviewable public copy of the contract required inside the image.

The full provenance, serving behavior, evidence, and limitations are in
[`docs/profiles/GLM53_FLASH_DFLASH2_BF16_TP4.md`](../../docs/profiles/GLM53_FLASH_DFLASH2_BF16_TP4.md).
