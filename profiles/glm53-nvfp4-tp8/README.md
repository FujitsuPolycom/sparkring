# GLM-5.3 NVFP4 on eight Sparks

[GLM-5.3-NVFP4](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4/tree/b472e4ee53f6a9862da5486c56c6ca21be3dab70)
([Local Inference Lab](https://huggingface.co/local-inference-lab)'s NVFP4
quantization of the routed experts of Z.ai's [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3),
a 744B mixture-of-experts model with DeepSeek sparse attention; revision
`b472e4ee53f6`) on all eight Sparks of an eight-Spark ring, with decode-context
parallelism 4, two-token MTP speculative decoding and a 1M-token context.
Node A serves the API on port 8015 as `GLM-5.3-NVFP4-TP8`, with no API key.
Status: Experimental.

## Install

1. Build an image with the GLM-5.3 plugin layer and its lock (below). The
   installer refuses any other image.
2. Lock the GPU clocks on every Spark of the ring. The installer does not do
   this:

   ```bash
   sudo nvidia-smi -lgc 2418,2418
   ```

   The lock lasts until `sudo nvidia-smi -rgc` or a reboot. The measured
   results below ran with it.
3. Install:

   ```bash
   sudo sparkring install --profile glm53-nvfp4-tp8 --image-lock LOCK
   ```

Each Spark holds the whole 433.0 GiB checkpoint; a blank Spark needs about
505 GiB free with the serving image and the compile cache allowance.
[Install SparkRing](../../docs/operations/install.md) covers requirements, the
fabric setup an eight-Spark ring needs, logs and recovery.

### Image

The profile needs an image lock of schema `sparkring-installer-image/v3`
whose image carries the SIRCL layer and lists the vLLM plugins
`glm_dsa_indexer_split`, `glm53full_speedups` and `glm_dcp_decode_comm` in
`vllm_plugins`. The package carries two such locks, of the SIRCL 0.3.1, libsircl
and GLM-5.3 plugin image `27e9f75c0d09` ([image record](../../performance/records/images/dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-image-20261009.md)), for
Sparks that hold that image. [derive_glm53_plugins.py](../../runtime/images/derive_glm53_plugins.py)
adds the three plugins to a SIRCL and libsircl image and writes such a
lock from that image's v3 lock
([derived layers](../../runtime/images/installer-images.md#derived-layers)).
`sparkring install` refuses a lock without the plugins, because vLLM would
skip the plugins this profile names and serve without them.

## Settings

| Setting | Value ([config.json](config.json)) |
|---|---|
| Parallelism | TP8 with decode-context parallelism 4, on the eight positions of an eight-Spark ring |
| Transport | SIRCL ring sessions for the tensor-parallel and both decode-context-parallel groups, NCCL off; fused all-reduce, residual add and RMSNorm (`SIRCL_FUSED_NORM=1`) and column gathers (`SIRCL_COLUMN_GATHER=1`); 16 link slots of 512 KiB on the tensor-parallel session (below) |
| Context / sequences / batch | 1048576 / 16 / 8192 |
| KV cache | 43 GiB per rank, `fp8_ds_mla`; `--gpu-memory-utilization 0.91`, because vLLM's startup check of free memory refuses the default 0.92 by about 0.4 GiB on GB10 |
| Weights | `--quantization modelopt_fp4` (the checkpoint's NVFP4 routed experts); the BF16 dense linears quantized to MXFP8 at load (`--quantization-config`, except `*kv_b_proj` and `*.indexer.wk_weights_proj`) on `--linear-backend b12x`; MXFP8 LM head |
| Experts | B12X W4A16 (`VLLM_B12X_MOE_FP4_FORCE_A16=1`), two CTAs per SM for small batches, 4-bit activations for prefill calls of 1,536 tokens or more, host barrier reset off |
| Attention | B12X, with the compressed-KV gather for up to 589,824 tokens |
| Speculative decoding | MTP, two tokens, B12X attention, draft TP8, probabilistic draft sampling; the draft's linears MXFP8 and its experts W4A16 (except `*kv_b_proj` and `*.indexer.*`). `--hf-overrides` keeps the BF16 MTP layer 78 out of the target's NVFP4 quantization |
| vLLM plugins | `glm_dsa_indexer_split` (the DSA indexer's prefill rows split over the DCP groups, full launches) and `glm53full_speedups` (the latent projection split over TP8, row-parallel MTP `eh_proj`); `glm_dcp_decode_comm` loaded with its five items off (below) |
| Parsers | `glm47` tool calls, `glm45` reasoning |

`sparkring install`, the SIRCL launcher's `bundle --profile glm53-nvfp4-tp8`
and the [serving A/B runner](../../performance/harnesses/serving_ab/README.md)
read the two SIRCL switches from `config.json`
([SIRCL runbook](../../spark_transport/sircl/sparkring_sircl/vllm/RUNBOOK.md#adding-sircl-to-another-launcher)).
The 16 link slots of 512 KiB are SIRCL's own setting for eight ranks and the
default tuning table's `cycle-8` row; a tuning table that `sudo sparkring
fabric tune` measured on the ring replaces them with the link settings it
records.

`glm_dcp_decode_comm` ([its guide](../../integrations/vllm/glm_dcp_decode_comm/README.md))
changes how the DSA attention runs its decode-context-parallel collectives on
the SIRCL DCP sessions, with exact results. The profile sets its five item
flags (`GLM_DCP_DECODE_QUERY_PACK`, `GLM_DCP_DECODE_OVERLAP`,
`GLM_DCP_DECODE_WK_OVERLAP`, `GLM_DCP_DECODE_SELECTION_REUSE` and
`GLM_DCP_DECODE_A2A_FUSED`) to `0`; with every item off, its registration
patches nothing. Turning the items on is research-only: the plugin pins
SIRCL 0.3.2 and refuses at startup on any other SIRCL build, its
qualification is a run with the five flags and `GLM_DCP_DECODE_AUDIT=1`, and
no measurement of its effect on this profile exists.

The B12X linear backend matters most: without it vLLM selects
FlashInfer's CUTLASS MXFP8 kernel, and 1-stream decode falls from about 44 to
26 tokens/s.

## Evidence and open items

Conditions: image `af06e272` (the SIRCL 0.3.0 and libsircl image
`816c6d6a7e96` with the first two plugins of the plugin layer of
[derive_glm53_plugins.py](../../runtime/images/derive_glm53_plugins.py);
`glm_dcp_decode_comm` was not loaded),
checkpoint revision `b472e4ee`, this profile's settings with 8 sequences, one
eight-Spark ring, SM clocks locked at 2,418 MHz, started outside
`sparkring install`; 2026-10-09. The number of runs per cell is not
recorded, and no record of these runs is in `performance/records`.

| Context | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---:|---:|---:|---:|
| 0 | 48.0 | 69.6 | 103.2 | 151.2 |
| 16K | 44.4 | 62.9 | 93.2 | 129.0 |

Decode in output tokens/s. A 16K-token prompt prefilled at about 1,290
tokens/s.

Sixteen sequences (the profile's `--max-num-seqs`), on the same image with
everything else identical: decode steps/s at 1 to 8 streams stayed within 3 %
of 8 sequences (at 0K context 19.12 / 28.75 / 42.08 / 59.75 against 19.14 /
28.10 / 40.88 / 59.83 at 1 / 2 / 4 / 8 streams). Sixteen streams reached
216.9 tokens/s (86.3 steps/s) at 0K, 45 % above 8 streams; at 16K they gave
136.6 tokens/s, the same as 8 streams. The CUDA graph capture sizes already
reach 48 tokens (16 sequences of 3 tokens with two draft tokens).

On the image `27e9f75c0d09` (`sparkring-dev/kraken:csf-sircl-libsircl-plugins-dcp-20261009`:
SIRCL 0.3.1, libsircl snapshot `a3477af2` and all three plugins, with
`glm_dcp_decode_comm`'s items off), this profile's settings with 16
sequences, the same ring and clocks, started outside `sparkring install`:

| Context | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---:|---:|---:|---:|
| 0 | 51.6 | 73.9 | 106.9 | 153.7 |
| 16K | 46.4 | 63.4 | 94.8 | 135.2 |

Decode in output tokens/s. Time to first token: 11.77 s for a 16K prompt,
23.89 s for 32K.

Conclusion: the profile holds the fastest GLM-5.3 TP8 configuration with a
1M-token context measured on that ring as of 2026-10-09;
[glm53-nvfp4-tp8-dcp1](../glm53-nvfp4-tp8-dcp1/README.md) decodes faster with
a 524,288-token context. On the release image of 2026.10.2, `1a8c10354eb0`,
one installation passed the installer's functional checks and decoded
47.7 tokens/s at one stream and 16K context with GPU clocks not locked
([installer record](../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-installer-ring8-20261009.md));
the profile is not serving-qualified.

Open items:

- `--hf-overrides` restates the checkpoint's quantization settings from its
  `hf_quant_config.json`, which the
  [pin manifest](../checkpoints/local-inference-lab--GLM-5.3-NVFP4/b472e4ee53f6a9862da5486c56c6ca21be3dab70.json)
  pins, with the MTP layer's patterns added. Their equality with the
  `quantization_config` of `config.json`, which the repository does not hold,
  is checked by `SPARKRING_GLM53_NVFP4_CHECKPOINT=DIR python -m pytest
  runtime/common/test_glm_targets.py -k checkpoint` on a machine with the
  checkpoint.
- The default SIRCL tuning table's `cycle-8` row (1 MiB all-reduce capacity
  and dispatch ceiling, 28 KiB one-shot limit, 16 link slots of 512 KiB)
  applies to SIRCL 0.3.2 sessions and, as a compatible build the table
  lists, to SIRCL 0.3.1 sessions such as those of image `27e9f75c0d09`. The measured runs above gave the same
  capacity, dispatch ceiling and link slots to the SIRCL bundle (`--capacity`,
  `--dispatch` and container variables) and left the one-shot limit to the
  session, which derives 28 KiB on eight ranks. On an image whose SIRCL layer
  is 0.3.0, such as `af06e272`, the row does not apply and the installer's
  sessions take SIRCL's own capacity and dispatch ceiling.
- Block size and compilation settings are vLLM's defaults.
