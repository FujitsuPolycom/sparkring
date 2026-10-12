# GLM-5.3-Flash CSF on four Sparks with SparkCache (research-only)

This profile runs
GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD, created by Local Inference Lab, Inc., a
non-profit organization, available at
https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD.
GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD is licensed under the Local Inference Lab
License, Version 1.0.

Status: **research-only**. This variant of the installer profile
[`glm53-flash-nvfp4-spark-tp4`](../glm53-flash-nvfp4-spark-tp4/profile.json) adds
SparkRing's persistent context cache, SparkCache, to the CSF checkpoint of
GLM-5.3-Flash on a four-Spark ring. The checkpoint publisher is Local Inference Lab
([`local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)
at `dec48abd33ef`). Stage 1 of its [hardware test plan](../../performance/sparkcache-flash-csf-hardware-test-plan.md) ran on four
Sparks of a ring of eight ([record](../../performance/records/images/dev-20261010-sparkcache-gab86b70-plugins-status036-glm53-flash-nvfp4-sparkcache-tp4-path4-20261011.md)): steps 1-7 and 9 passed on every
rank, and step 8 and the two-start A/B did not run. The SparkCache model
profile `glm53-flash-hybrid` was qualified on shared release 2026.09.3 on
other checkpoints and another vLLM line.

## What the variant changes

The profile keeps every speed setting of the cache-off profile except its
4-bit-activation prefill switch (`B12X_W4A16_A4_PREFILL_MIN_TOKENS=1536`, which
in the image's vLLM puts every uncaptured prefill row on 4-bit activations,
whatever its length). No start of this variant has run with it, so its prefill
MoE calls run W4A16: MTP3 with the
CSF checkpoint's draft settings (marlin draft experts, draft tensor
parallelism, probabilistic draft sampling, standard rejection),
`--async-scheduling`, the B12X attention, linear and KDA-prefill backends with
KDA prefill coalescing, W4A16 small-M occupancy 2, the 1,024-token split pages
and the ring's 37 GiB of KV cache and 1,048,576-token context in 1,024-token pages. It serves the
CSF checkpoint only (`--quantization nvfp4_csf --load-format nvfp4_csf`).

On top it enables the SparkCache KV connector:

- model profile `glm53-flash-hybrid`, publication schema `tail-cow-v2`;
- asynchronous page capture in `connector-jobs` lease mode against the
  contract `sparkcache/runtime_patches/vllm-connector-jobs-gab86b70.json` of
  the [derived image layer](../../runtime/images/compositions/installer-sparkcache-gab86b70/README.md),
  whose eleven fingerprints bind the lease semantics to the release image's
  actual vLLM bytes;
- CUDA page restore with the placement library
  `/opt/sparkring/sparkcache/lib/libspark_cache_placement.so` and capture
  through `/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so`, both with
  their pinned SHA-256;
- 64K-token spans (`spark_cache_min_span_tokens` 4096,
  `spark_cache_max_span_tokens` 65536), 8 GiB of cache bytes with a 6 GiB
  low watermark, and two capture slots of 8 GiB;
- the checkpoint identity
  `6af0343976eefbfa0fb59d87befcf6d10f02fef90ba0e7881dd1201fe0ecd76e`, which
  `deploy/glm53_flash/profile.py` in the external SparkCache repository derives
  from the checkpoint's pin manifest; stored entries of the CSF checkpoint
  never restore into another checkpoint, and no other checkpoint's entries
  restore into this one;
- the rank-local root `SPARK_CONTEXT_CACHE_ROOT=/cache/glm53-flash-sparkcache-ab53c097d7ca-dec48abd33ef`,
  which `sparkring storage` classifies as `cache` and which the deployment
  names, so automatic release sees it referenced.

The checkpoint digest, the KV layout identity and the cache root all come from
this profile's own settings; a different checkpoint, a different split-page
size or a different vLLM composition is a different cache namespace.

## Installation

`sparkring install --profile glm53-flash-nvfp4-sparkcache-tp4` installs the
variant only on an image whose lock lists it. No lock in this package does: the
only one is the lock that
[runtime/releases/dev-20261010-sparkcache-gab86b70-plugins-status036/](../../runtime/releases/dev-20261010-sparkcache-gab86b70-plugins-status036/README.md)
records when a Spark builds the derived layer. On every other image, including
the default image of release 2026.10.2, the installer refuses the profile. The
variant has no Compose exports: a Compose deployment runs the default image,
whose lock does not list the profile.

`sudo sparkring check` reports the connector's state for the running
deployment.

## Evidence

Stage 1 of the hardware test plan, on four Sparks of a ring of eight: steps 1-7
and 9 passed on every rank; step 8 and the two-start A/B did not run
([record](../../performance/records/images/dev-20261010-sparkcache-gab86b70-plugins-status036-glm53-flash-nvfp4-sparkcache-tp4-path4-20261011.md)).
The hardware test plan is in the
[gap analysis section 7.2](../../docs/operations/install-reference.md). CPU
conformance of the lease contract against the release image's vLLM is in the
external SparkCache repository
(`sparkcache/runtime_patches/test_gab86b70_contract.py`).
