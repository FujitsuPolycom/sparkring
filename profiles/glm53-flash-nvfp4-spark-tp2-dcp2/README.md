# GLM-5.3-Flash CSF on two Sparks with decode-context parallelism 2 (research-only)

This profile runs
[GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)
(revision `dec48abd33ef`), created and published by Local Inference Lab, Inc.
and licensed under the Local Inference Lab License, Version 1.0.

Status: **Experimental** (machine status `research-only`). It is the CSF deployment of the pair profile
[`glm53-flash-nvfp4-spark-tp2`](../glm53-flash-nvfp4-spark-tp2/profile.json)
with vLLM's decode-context parallelism 2 (DCP2): the two Sparks of a pair split
the KV cache between them instead of each holding a full copy, so the same
memory holds twice the context tokens, and the pair serves 16 requests at once. It enables mHC prefill row
ownership through the [source-derived image](../../runtime/images/compositions/dcp2-prefill/README.md) and
leaves KDA prefill coalescing off. GLM-5.3-Flash thinks at effort `high` when a request does not choose.

## What the profile changes

Against the pair profile's `csf` checkpoint entry:

| Setting | This profile | `glm53-flash-nvfp4-spark-tp2` `csf` | Why |
|---|---|---|---|
| `--decode-context-parallel-size` | 2 | 1 | splits the KV cache over the pair |
| `--cp-kv-cache-interleave-size` | 4 | unset | GLM-5.3-Flash's attention under decode-context parallelism needs KV blocks of a multiple of 4 tokens; vLLM does not start with the default of 1 |
| `--max-num-seqs` | 16 | 8 | the split KV cache holds 16 requests at 128K |
| CUDA graph capture sizes | every multiple of 4 up to 64 | up to 32 | a decode step is 4 tokens per request with MTP3; steps of more than 8 requests would otherwise run without graphs |
| `VLLM_GLM53_MHC_PREFILL_SHARD` | 1 | 1 | source-patched vLLM and SIRCL pins admit TP2/DCP2 row ownership |
| `VLLM_B12X_KDA_PREFILL_COALESCING` | 0 | 1 | at DCP2 it cost 7 % of the KV cache and gave no measurable prefill gain ([record](../../performance/records/images/dcp2-prefill-e91876-glm53-flash-nvfp4-spark-tp2-dcp2-settings-20261013.md)) |
| default thinking effort | `high` | `high` | `serving_defaults`; the chat template's own default is `max` |
| `--kv-cache-memory-bytes` | 15,032,385,536 (14 GiB) per Spark | 10 GiB | see [Memory](#memory) |
| served model name | `GLM-5.3-Flash-CSF-DCP2-TP2` | `GLM-5.3-Flash-CSF-TP2` | clients can tell the two deployments apart |
| `cache_namespace` | `glm53-flash-csf-dcp2` | `glm53-flash-nvfp4-spark` | compile caches stay apart from the DCP1 deployment's |

Every other setting is the pair profile's `csf` setting: MTP3 with the CSF
draft settings (marlin draft experts, draft tensor parallelism 2,
probabilistic draft sampling, standard rejection, BF16 draft KV),
`--async-scheduling`, the B12X attention, MoE, linear and KDA-prefill
backends, W4A16 decode experts
(`VLLM_B12X_MOE_FP4_FORCE_A16=1`) with 4-bit-activation prefill
(`B12X_W4A16_A4_PREFILL_MIN_TOKENS=1536`) and small-M occupancy 2, FP8 KV,
the 2,048-token split pages, the 1,048,576-token context window and 8,192-token prefill chunks.

On SIRCL, the installer gives the decode-context-parallel group a SIRCL
session of its own (`SIRCL_GROUPS=tp,dcp`); on a pair that group holds the
same two ranks as the tensor-parallel group.

## Memory

A Spark's CPU and GPU share one 128 GB memory. The CSF checkpoint's weights
take 84.5 GiB per Spark. The profile keeps the pair profile's GPU memory
fraction (0.87) and context window and sets 14 GiB of KV cache per Spark,
which holds 4,126,165 tokens on a pair with KDA prefill coalescing off (3,835,395 with it on).
Under those text loads up to 512K-token
prompts, the lowest `MemAvailable` was 6.5 GiB on the API Spark and 8.0 GiB on
the other, about 4 GiB below the DCP1 profile's at 10 GiB of KV
([record](../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-glm53-flash-nvfp4-spark-tp2-dcp2-pair-20261013.md)). Image and video loads at this KV size are not measured;
the API Spark preprocesses every image and video.

## Install

The profile installs on an image whose lock lists it, such as the default
image of release 2026.10.4; `glm53-flash-tp2-dcp2` is its other name:

```bash
sudo sparkring install --profile glm53-flash-tp2-dcp2
```

`sparkring images --profile glm53-flash-nvfp4-spark-tp2-dcp2` lists every
image that runs it, and `--image NAME` selects another.

Such an image is built from the
[source composition](../../runtime/images/compositions/dcp2-prefill/README.md):
its vLLM is the pinned build `sparkring-kraken-bc9ea774-tp2-dcp2-prefill`,
which reads the CSF checkpoint and admits mHC row ownership at TP2/DCP2. The
installer refuses the profile on any other image, even one whose lock lists
it. A lock recorded from a local build of the composition is selected with
`--image-lock PATH`. The profile has no Compose exports. Building or
publishing the image does not qualify its serving behavior.

`VLLM_GLM53_MHC_PREFILL_SHARD=0` on both ranks turns mHC row ownership off.
Keep the 8192-token budget: mHC owner rows use vLLM's full-ceiling prepared
plan, preserving its numerical geometry. Independently preparing or tuning a
half-row plan is not covered by the exactness evidence. A 16384-token budget
also needs B12X to tune KDA prefill shapes it has not seen, which did not finish
within 37 minutes on a Spark.

## Evidence

**Measured, these settings:** hand-made containers on four pairs and installed starts, 2026-10-13
([record](../../performance/records/images/dcp2-prefill-e91876-glm53-flash-nvfp4-spark-tp2-dcp2-settings-20261013.md)). With mHC on, 16 requests and KDA coalescing off, prefill measured 1,885 / 1,976 / 1,973 tok/s
at 16K / 64K / 128K, against 1,878 / 1,942 / 1,934 for the earlier settings measured the same way. Aggregate
decode at 16 requests was 142-145 tok/s, against 109-112 at the earlier limit of 8. An installed start with KDA
coalescing also on measured prefill 2,151-2,217 tok/s on a different benchmark client, 5-11 % above the earlier
settings' installed start on that client. Every setting passed a seven-prompt correctness screen.

**Measured (historical, both prefill features off):** one DCP2 start and one of the DCP1 profile's `csf`
deployment ran on the same pair of an eight-Spark ring on installer image
`d52737a109e0` ([record](../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-glm53-flash-nvfp4-spark-tp2-dcp2-pair-20261013.md)). Both passed the seven functional checks and
a 256-request correctness screen with no degenerate, wrong or failed response.
Against DCP1, this profile held 2.00 times the KV tokens per byte of KV cache;
its decode ran 1-6 % fewer engine steps per second, and its prefill was 17 %
slower at 64K and 128K prompts and 27 % slower at 8K. The DCP2 containers were
the installer's DCP1 containers re-created with this profile's rendered
differences, not a `sparkring install` of this profile.

**Measured (component only):** the image's B12X kernels on an RTX 5090
produce bit-identical KDA outputs and retained states for coalesced/split
prefill, and bit-identical mHC outputs for full/owned rows with the serving
plan geometry. See the [test scope](../../integrations/vllm/dcp2_prefill/README.md).
