# GLM-5.3 NVFP4 on eight Sparks without decode-context parallelism

[GLM-5.3-NVFP4](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4/tree/b472e4ee53f6a9862da5486c56c6ca21be3dab70)
([Local Inference Lab](https://huggingface.co/local-inference-lab)'s NVFP4
quantization of the routed experts of Z.ai's [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3),
a 744B mixture-of-experts model with DeepSeek sparse attention; revision
`b472e4ee53f6`) on all eight Sparks of an eight-Spark ring, at tensor
parallelism 8 without decode-context parallelism, with a 524,288-token context.
Node A serves the API on port 8015 as `GLM-5.3-NVFP4-DCP1-TP8`, with no API key.
Status: Experimental. It is a variant of
[glm53-nvfp4-tp8](../glm53-nvfp4-tp8/README.md), the default eight-Spark
GLM-5.3 profile, which keeps decode-context parallelism 4 and the 1M-token
context.

## When to choose it

| | This profile (DCP1) | glm53-nvfp4-tp8 (DCP4) |
|---|---|---|
| Context window | 524,288 tokens | 1,048,576 tokens |
| KV pool | 843,584 tokens | about 3.4M tokens |
| Decode at 16K context, 1 / 2 / 4 / 8 streams | 49.4 / 74.7 / 108.2 / 156.4 tok/s | 44.4 / 62.9 / 93.2 / 129.0 tok/s |
| Time to first token, 16K prompt | 10.7 s | 12.7 s |

Choose it for faster decode and prefill at contexts up to 512K tokens; choose
glm53-nvfp4-tp8 for longer contexts or more concurrent long requests. Both
rows' measurements and their conditions are below.

## Install

1. Use an image lock whose image carries the GLM-5.3 plugin layer and that
   lists this profile (below). The installer refuses any other image.
2. Lock the GPU clocks on every Spark of the ring, which the installer does
   not do (`sudo nvidia-smi -lgc 2418,2418`; `sudo nvidia-smi -rgc` or a
   reboot reverts it). The measured results below ran with it.
3. Install:

   ```bash
   sudo sparkring install --profile glm53-nvfp4-tp8-dcp1 --image-lock LOCK
   ```

Downloads, storage and the fabric are those of
[glm53-nvfp4-tp8](../glm53-nvfp4-tp8/README.md#install): the same checkpoint,
433.0 GiB on each Spark.

### Image

The profile needs an image lock of schema `sparkring-installer-image/v3`
whose image carries the SIRCL layer, lists the vLLM plugins
`glm_dsa_indexer_split` and `glm53full_speedups` in `vllm_plugins` and lists
this profile in `profiles`. No lock in this package does.
[derive_glm53_plugins.py](../../runtime/images/derive_glm53_plugins.py)
records such a lock from the plugin image
([derived layers](../../runtime/images/installer-images.md#derived-layers)).

## Settings

The settings of [glm53-nvfp4-tp8](../glm53-nvfp4-tp8/README.md#settings)
([config.json](config.json)) with these differences:

| Setting | This profile | glm53-nvfp4-tp8 |
|---|---|---|
| `--decode-context-parallel-size` | 1 | 4 |
| `--max-model-len` | 524288 | 1048576 |
| `--max-num-seqs` | 8, as measured | 16 |
| vLLM plugins | `glm_dsa_indexer_split`, `glm53full_speedups` | the same and `glm_dcp_decode_comm` with its items off |
| Served model name | `GLM-5.3-NVFP4-DCP1-TP8` | `GLM-5.3-NVFP4-TP8` |

The DSA indexer split runs at decode-context parallelism 1, 2 or 4.
`glm_dcp_decode_comm` serves only decode-context-parallel groups, which this
profile has none of. The SIRCL tensor-parallel session takes the default
tuning table's `cycle-8` row where the table names the image's SIRCL build,
as glm53-nvfp4-tp8 does.

## Evidence

Conditions: image `af06e272` (the SIRCL 0.3.0 and libsircl image
`816c6d6a7e96` with the `glm_dsa_indexer_split` and `glm53full_speedups`
plugins of the plugin layer), checkpoint revision `b472e4ee`, the settings
above with 8 sequences, SIRCL ring sessions with NCCL off, the fused
all-reduce and RMSNorm, a 1 MiB all-reduce capacity and dispatch ceiling and
16 link slots of 512 KiB, no SIRCL tuning table, SM clocks locked at
2,418 MHz, one eight-Spark ring; the serving A/B runner's arm `S+`, one
warm-up start and one measured start, at temperature 0 with at most 1,024
output tokens in 30 s cells; 2026-10-09, outside `sparkring install`. No
record of this run is in `performance/records`.

vLLM reported a KV pool of 843,584 tokens. Decode in output tokens/s:

| Context | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---:|---:|---:|---:|
| 0 | 49.5 | 84.7 | 117.5 | 170.8 |
| 16K | 49.4 | 74.7 | 108.2 | 156.4 |
| 32K | 53.1 | 76.6 | 109.9 | 168.1 |

Time to first token: 5.3-5.5 s for an 8K prompt, 10.7 s for 16K, 21.9 s for
32K.

Against glm53-nvfp4-tp8 at 16K context on the same image and ring
([its evidence](../glm53-nvfp4-tp8/README.md#evidence-and-open-items)):
decode 11-21 % faster at 1-8 streams and prefill about 17 % faster, with half
the context window and about a quarter of the KV pool.

Conclusion: at contexts up to 512K tokens this configuration decoded and
prefilled faster than glm53-nvfp4-tp8 in one measured start. No installation
of this profile has run; it is not serving-qualified.

Open items:

- One measured start; 16 sequences were measured only at decode-context
  parallelism 4.
- The checkpoint check of glm53-nvfp4-tp8's `--hf-overrides`
  (`runtime/common/test_glm_targets.py -k checkpoint`) covers this profile,
  which takes the same `--hf-overrides`.
