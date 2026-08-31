# GLM-5.3 Jovian Judgement r7-compatible public GB10 images

Status: **implemented and TP4 smoke-verified**, not generally qualified. Two
immutable Linux ARM64 images provide a cache-disabled base and an explicitly
configured SparkCache variant. Their exact identities, sources, and bounded
evidence are in [`artifacts.json`](artifacts.json).

Local Inference Lab's
[Jovian Judgement vLLM branch](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement)
is the primary source of GLM runtime performance and correctness. The images
use its community r7 source plus the exact connector seams identified by
[`FujitsuPolycom/vLLM@331573d`](https://github.com/FujitsuPolycom/vllm/commit/331573d20bd47e78327ed8d8b4d2e6d350bbb1ab).
[B12X at `6255090a`](https://github.com/local-inference-lab/b12x/commit/6255090a03b12c3f7d552102a02fac0b542fb8c9)
supplies the Blackwell kernels and backend integration.

The target is
[`GLM-5.3-Flash-NVFP4@520de24e`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/520de24eabf507659eaef7c70f14fd584527facc).
The external draft is
[`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410),
whose weights are BF16. It is not Local Inference Lab's separate
[MXFP8 DFlash2 checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8).

The `331573d` identity describes the active vLLM Python composition. The
images retain compiled native extensions from lower layers whose build
environment reports `VLLM_BUILD_COMMIT=3633d61c3c7b04bb4d598cadbdc342f3be40482d`
and whose intermediate source label is
`org.sparkring.native-parent.vllm=da4d7be6c97434f6942292ed8abbf4b32dc44355`.
A sorted SHA-256 manifest of every vLLM shared object was byte-identical
between the native parent and composed child. The images are not described as
source-built native `331573d` artifacts.

Both images inherit lower-layer labels
`org.glm53.dflash2.checkpoint-revision=b6d33aa93fc1ac5b23a88251a1c0ce0bfe2ad17c`
and `org.glm53.dflash2.mxfp8-quant-plumbing=v2`. Those labels record image
lineage and available quantization plumbing. They do not identify the mounted
draft. The active external draft identity is the BF16 `dc77ff1c` revision
above and is verified from its `config.json` and `model.safetensors` before
the container starts.

Use [`runtime.env.example`](runtime.env.example) and
[`launch-rank.sh`](launch-rank.sh) for the shortest configurable start. The
launcher selects `IMAGE_VARIANT=base` or `IMAGE_VARIANT=sparkcache`, verifies
the exact local image ID after the immutable pull, and marks settings that
differ from the smoke configuration `implemented-unqualified-configuration`.

Operator defaults are a 524,288-token request limit, 8,192 batched tokens, and
a 524,288-token SparkCache publication span. The image smoke used 262,144,
4,096, and a 262,144-token span. The larger defaults are implemented but have
no long-context capacity or throughput qualification.

Both images declare `ENTRYPOINT ["vllm", "serve"]`. The launcher therefore
passes `/models/target` as the first image argument. It does not pass another
`serve` token.

The operator procedure is
[`GLM53_JJ_R7_GB10_TP4_QUICKSTART.md`](../../docs/GLM53_JJ_R7_GB10_TP4_QUICKSTART.md).
The bounded C4 evidence is recorded in the
[`validation.json`](../../performance/receipts/glm53-flash/jj-r7-gb10-tp4-smoke-20260830/validation.json)
receipt and its
[`evidence record`](../../performance/records/glm53-flash/jj-r7-gb10-tp4-smoke-20260830.md).
