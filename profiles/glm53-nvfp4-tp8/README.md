# GLM-5.3 NVFP4 on eight Sparks

[GLM-5.3-NVFP4](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4/tree/b472e4ee53f6a9862da5486c56c6ca21be3dab70)
([Local Inference Lab](https://huggingface.co/local-inference-lab)'s NVFP4
quantization of the routed experts of Z.ai's [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3),
a 744B mixture-of-experts model with DeepSeek sparse attention; revision
`b472e4ee53f6`) on all eight Sparks of an eight-Spark ring, with decode-context
parallelism 4 and a 1M-token context. Node A serves the API on port 8015 as
`GLM-5.3-NVFP4-TP8`, with no API key. Status: Experimental.

The eight ranks reach each other through ConnectX relays, so only SIRCL ring
sessions carry the model's collectives, with NCCL off: one session for the
tensor-parallel group and one for each decode-context-parallel group of four
Sparks. No image lock in this package lists the profile: install it with
a development image lock (schema v3) whose image carries the SIRCL layer and
lists `glm53-nvfp4-tp8`:

```bash
sudo sparkring install --profile glm53-nvfp4-tp8 --image-lock LOCK
```

Each Spark holds the whole 433.0 GiB checkpoint; a blank Spark needs about
505 GiB free with the serving image and the compile cache allowance.
[Install SparkRing](../../docs/operations/install.md) covers requirements, the
fabric setup an eight-Spark ring needs, logs and recovery.

## Settings

| Setting | Value ([config.json](config.json)) | Source |
|---|---|---|
| Parallelism | TP8 with decode-context parallelism 4, on the eight positions of an eight-Spark ring | The served configuration below |
| Transport | SIRCL ring sessions for the tensor-parallel and both decode-context-parallel groups, NCCL off; the default tuning table's `cycle-8` row | SIRCL's [serving runbook](../../spark_transport/sircl/sparkring_sircl/vllm/RUNBOOK.md#serving-without-nccl) |
| Context / sequences / batch | 1048576 / 8 / 8192 | The served configuration below: 1M context, 8,192-token prefill chunks, up to 8 streams measured |
| KV allocation | 43 GiB per rank, `fp8_ds_mla` | The served configuration below |
| Weights | `--quantization modelopt_fp4` (the checkpoint's ModelOpt NVFP4 experts; attention, dense layers and shared experts in BF16), `--load-format b12x` for its per-expert scale files | The checkpoint's `hf_quant_config.json`; the GLM-5.3-Flash profiles load the same file layout with the B12X loader |
| Kernels | B12X attention and MoE backends | The GLM-5.3-Flash profiles |
| Parsers | `glm47` tool calls, `glm45` reasoning | The GLM-5.3-Flash profiles |
| Environment | The GLM-5.3-Flash profiles' variables without those only GLM-5.3-Flash's architecture reads | [`glm53-flash-nvfp4-spark-tp4`](../glm53-flash-nvfp4-spark-tp4/config.json) |

## Evidence and open items

A launcher outside this repository served this checkpoint revision at TP8
with decode-context parallelism 4 on SIRCL ring sessions with NCCL off, on
one eight-Spark ring with column gathers and 16 link slots of 512 KiB: 16K-token
prompts prefilled at 1,423 tokens/s, and decode ran 20.1 / 29.5 / 40.7 / 57.4
steps/s at 1 / 2 / 4 / 8 streams (one light run, 2026-10-07; no record of that
run is in `performance/records`). No installation of this profile has run.

What the profile does not pin, each to be measured on an eight-Spark ring
before the profile leaves research-only:

- Speculative decoding: the checkpoint's MTP layer, and the overrides that
  keep it unquantized, are not set, so the profile decodes without drafts.
- The served configuration's further kernel settings (MXFP8 indexer and target
  linears, W4A16 two-CTA experts, indexer prefill row splitting, the CKV gather
  capacity of 589,824 and the fused all-reduce with RMSNorm), its block size,
  CUDA graph capture sizes and GPU memory utilization.
- An image whose vLLM serves `GlmMoeDsaForCausalLM` on the B12X attention
  path: SIRCL's survey of image `aba309e4610c` records that its B12X
  attention class needs a source change for this model
  ([survey](../../spark_transport/sircl/sparkring_sircl/vllm/SURVEY.md#5-glm-53-at-tp8-with-dcp4-glmmoedsa)).
