# GLM-5.3 Flash runtime contract

> Historical exact-artifact contract. The published Jovian Judgement
> r7-compatible image identities and operator environment are in
> [`runtime/glm53-flash-jj-r7-gb10/`](../glm53-flash-jj-r7-gb10/README.md).

Status: **qualified** for the exact source and artifact identities in
[`pins.json`](pins.json) when used by the SparkCache-enabled TP4/DCP1
composition. A rebuilt image has **implemented** status until it passes the
recorded live qualification.

The GLM runtime performance and correctness implementation comes primarily
from Local Inference Lab's
[Jovian Judgement vLLM source at `da4d7be6`](https://github.com/local-inference-lab/vllm/commit/da4d7be6c97434f6942292ed8abbf4b32dc44355).
[B12X at `2fcf23a0`](https://github.com/local-inference-lab/b12x/commit/2fcf23a0ce269be27b2e03fece73d46e90e6aeea)
supplies the Blackwell kernels and backend integration. The target is Local
Inference Lab's
[`GLM-5.3-Flash-NVFP4@520de24e`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/520de24eabf507659eaef7c70f14fd584527facc).
The external draft is
[`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410),
whose weights are BF16. It is not Local Inference Lab's separate
[MXFP8 DFlash2 checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8).

The source-complete public builder is documented in [`BUILD.md`](BUILD.md).
It fetches every source by immutable commit, builds NCCL from the NVIDIA
`v2.30.7-1` source with SparkRing's independently implemented switchless-cycle
patch, and emits an image receipt. The builder is **implemented**. The digest
recorded in `pins.json` is **qualified**; another output remains implemented
until its own digest passes the four-rank checks. The
license and redistribution boundary is documented in
[`LICENSES/README.md`](LICENSES/README.md).
Registry publication and its required SPDX SBOM are documented in
[`PUBLISHING.md`](PUBLISHING.md).

Before any rank starts, the runtime profile requires one exact image ID,
checks the source/profile labels, hashes the DFlash files and NCCL library,
checks the vLLM configuration postimage, and runs SparkCache's seven-file lease-contract
verifier against the installed vLLM files. The copied lease contract at
[`vllm-kv-block-lease-contract-da4d7be.json`](vllm-kv-block-lease-contract-da4d7be.json)
is a reviewable public copy of the contract required inside the image.

The full provenance, serving behavior, evidence, and limitations are in
[`docs/profiles/GLM53_FLASH_DFLASH2_BF16_TP4.md`](../../docs/profiles/GLM53_FLASH_DFLASH2_BF16_TP4.md).
