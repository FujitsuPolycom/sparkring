# GLM-5.3-Flash CSF on two Sparks with decode-context parallelism 2 (research-only)

This profile runs
[GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)
(revision `dec48abd33ef`), created and published by Local Inference Lab, Inc.
and licensed under the Local Inference Lab License, Version 1.0.

Status: **research-only**. It is the CSF deployment of the pair profile
[`glm53-flash-nvfp4-spark-tp2`](../glm53-flash-nvfp4-spark-tp2/profile.json)
with vLLM's decode-context parallelism 2 (DCP2): the two Sparks of a pair split
the KV cache between them instead of each holding a full copy, so the same
memory holds about twice the context tokens. Use it for long contexts or many
concurrent long requests on one pair; the pair profile at decode-context
parallelism 1 prefills faster.

## What the profile changes

Against the pair profile's `csf` checkpoint entry:

| Setting | This profile | `glm53-flash-nvfp4-spark-tp2` `csf` | Why |
|---|---|---|---|
| `--decode-context-parallel-size` | 2 | 1 | splits the KV cache over the pair |
| `--cp-kv-cache-interleave-size` | 4 | unset | GLM-5.3-Flash's attention under decode-context parallelism needs KV blocks of a multiple of 4 tokens; vLLM does not start with the default of 1 |
| `VLLM_GLM53_MHC_PREFILL_SHARD` | 0 | 1 | the vLLM of the CSF-reading images starts mHC prefill row ownership only at TP2/DCP1 and TP4/DCP1, 2 or 4 (`configure` in its `vllm/models/glm5next/nvidia/mhc_prefill_sharding.py`; `sparkring_sircl.vllm.pins.MHC_ADMITS_DEFAULT`), so prefill runs full rows |
| `--kv-cache-memory-bytes` | 15,032,385,536 (14 GiB) per Spark | 10 GiB | see [Memory](#memory) |
| served model name | `GLM-5.3-Flash-CSF-DCP2-TP2` | `GLM-5.3-Flash-CSF-TP2` | clients can tell the two deployments apart |
| `cache_namespace` | `glm53-flash-csf-dcp2` | `glm53-flash-nvfp4-spark` | compile caches stay apart from the DCP1 deployment's |

Every other setting is the pair profile's `csf` setting: MTP3 with the CSF
draft settings (marlin draft experts, draft tensor parallelism 2,
probabilistic draft sampling, standard rejection, BF16 draft KV),
`--async-scheduling`, the B12X attention, MoE, linear and KDA-prefill
backends, KDA prefill coalescing, W4A16 decode experts
(`VLLM_B12X_MOE_FP4_FORCE_A16=1`) with 4-bit-activation prefill
(`B12X_W4A16_A4_PREFILL_MIN_TOKENS=1536`) and small-M occupancy 2, FP8 KV,
the 2,048-token split pages, the 1,048,576-token context window, eight
sequences and 8,192-token prefill chunks.

On SIRCL, the installer gives the decode-context-parallel group a SIRCL
session of its own (`SIRCL_GROUPS=tp,dcp`); on a pair that group holds the
same two ranks as the tensor-parallel group.

## Memory

A Spark's CPU and GPU share one 128 GB memory. The CSF checkpoint's weights
take 84.5 GiB per Spark. The profile keeps the pair profile's GPU memory
fraction (0.87) and context window and sets 14 GiB of KV cache per Spark.

## Install

No image lock of this package lists the profile, so
`sudo sparkring install --profile glm53-flash-nvfp4-spark-tp2-dcp2` is
refused on every image. It installs only with a development image lock
(`--image-lock PATH`) that lists it and whose vLLM reads CSF: the lock of
release 2026.10.2 (`runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/installer-image.json`)
with this ID added to `profiles`. The profile has no Compose exports.

SIRCL's serve launcher
([runbook](../../spark_transport/sircl/sparkring_sircl/vllm/RUNBOOK.md#decode-context-parallelism-in-profile-serving))
does not read this profile: it serves only profiles that an image lock lists
and whose recipe sets decode-context parallelism 1. It serves the same
settings from the pair profile with
`--dcp-size 2 --mhc-prefill-shard off` and the `csf` entry's options.

## Evidence

No hardware run of this profile's settings is recorded yet.
