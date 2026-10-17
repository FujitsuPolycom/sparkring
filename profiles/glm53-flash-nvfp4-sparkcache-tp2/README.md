# GLM-5.3-Flash CSF on two Sparks with SparkCache (research-only)

This profile runs
[GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD),
created by Local Inference Lab, Inc. and licensed under the Local Inference Lab
License, Version 1.0.

Status: **research-only**. This variant of the installer profile
[`glm53-flash-nvfp4-spark-tp2`](../glm53-flash-nvfp4-spark-tp2/profile.json) adds
SparkRing's persistent context cache, SparkCache, to the CSF checkpoint of
GLM-5.3-Flash on a pair. The checkpoint publisher is Local Inference Lab
([`local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)
at `dec48abd33ef`). Its first start, with two 8 GiB capture slots, stopped
both Sparks of a pair during kernel compilation
([record](../../performance/records/images/dev-20261010-sparkcache-gab86b70-plugins-status036-glm53-flash-nvfp4-sparkcache-tp4-path4-20261011.md#two-sparks-with-8-gib-capture-slots)).
With the profile's two 512 MiB slots, a private record reports one start on a
pair of a ring of eight that passed steps 1-7 and 9 of the
[hardware test plan](../../performance/sparkcache-flash-csf-hardware-test-plan.md)
on both ranks ([catalog](../../performance/enhancements.md#sparkcache-persistent-prefix)).
No public record holds that run, and step 8 and the two-start A/B did not run,
so the variant stays research-only. The SparkCache model profile
`glm53-flash-hybrid` was qualified on shared release 2026.09.3 on other
checkpoints and another vLLM line.

## What the variant changes

The profile keeps every speed setting of the cache-off profile except its
4-bit-activation prefill switch (`B12X_W4A16_A4_PREFILL_MIN_TOKENS=1536`, which
in the image's vLLM puts every uncaptured prefill row on 4-bit activations,
whatever its length). No start of this variant has run with it, so its prefill
MoE calls run W4A16: MTP3 with the
CSF checkpoint's draft settings (marlin draft experts, draft tensor
parallelism, probabilistic draft sampling, standard rejection),
`--async-scheduling`, the B12X attention, linear and KDA-prefill backends with
KDA prefill coalescing, W4A16 small-M occupancy 2, the 2,048-token split pages
and the pair's 10 GiB of KV cache and 1,048,576-token context. It serves the
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
  `spark_cache_max_span_tokens` 65536), 32 GiB of cache bytes with a 28 GiB
  low watermark, and two capture slots of 512 MiB, the value of the qualified
  shared-2026.09.3 profile `glm53-flash-spark-tp2-dcp1-sparkcache`. Beside the
  base profile's 84.5 GiB of weights and 10 GiB of KV cache per Spark, two
  8 GiB slots would leave about 0.7 GiB for compilation and the API server
  ([record](../../performance/records/images/dev-20261010-sparkcache-gab86b70-plugins-status036-glm53-flash-nvfp4-sparkcache-tp4-path4-20261011.md#two-sparks-with-8-gib-capture-slots));
- the checkpoint identity
  `6af0343976eefbfa0fb59d87befcf6d10f02fef90ba0e7881dd1201fe0ecd76e`,
  a recorded value whose original derivation is not available. The external
  SparkCache helper `deploy/glm53_flash/profile.py` validates a supplied digest;
  it does not derive one from the checkpoint pin manifest. The recorded value
  remains pinned;
- the rank-local root `SPARK_CONTEXT_CACHE_ROOT=/cache/glm53-flash-sparkcache-ab53c097d7ca-dec48abd33ef`,
  which `sparkring storage` classifies as `cache` and which the deployment
  names, so automatic release sees it referenced.

The checkpoint digest, the KV layout identity and the cache root all come from
this profile's own settings; a different checkpoint, a different split-page
size or a different vLLM composition is a different cache namespace.

## Installation

`sparkring install --profile glm53-flash-nvfp4-sparkcache-tp2` installs the
variant only on an image whose lock lists it. No lock in this package lists
it; the lock that
[runtime/releases/dev-20261010-sparkcache-gab86b70-plugins-status036/](../../runtime/releases/dev-20261010-sparkcache-gab86b70-plugins-status036/README.md)
records when a Spark builds the derived layer does. On every other image, including
the default image of release 2026.10.5, the installer refuses the profile. The
variant has no Compose exports: a Compose deployment runs the default image,
whose lock does not list the profile.

`sudo sparkring check` reports the connector's state for the running
deployment.

## Evidence

- Two 8 GiB capture slots: the first start stopped both Sparks during kernel
  compilation; memory exhaustion is inferred, not measured
  ([record](../../performance/records/images/dev-20261010-sparkcache-gab86b70-plugins-status036-glm53-flash-nvfp4-sparkcache-tp4-path4-20261011.md#two-sparks-with-8-gib-capture-slots)).
- Two 512 MiB capture slots, the profile's value: one start, reported by a
  private record (2026-10-11: SparkCache TP2 hardware test of GLM-5.3-Flash
  CSF) and summarized in the
  [catalog](../../performance/enhancements.md#sparkcache-persistent-prefix).
  Steps 1-7 and 9 passed on both ranks; after a planned restart and after
  SIGKILL each rank restored a 32,768-token prefix; the lowest `MemAvailable`
  was 9.52 GiB. Step 8 and the two-start A/B did not run. No public record
  holds the run.

The hardware test plan is in
[sparkcache-flash-csf-hardware-test-plan.md](../../performance/sparkcache-flash-csf-hardware-test-plan.md). CPU
conformance of the lease contract against the release image's vLLM is in the
external SparkCache repository
(`sparkcache/runtime_patches/test_gab86b70_contract.py`).

## Request defaults

When a request chooses no effort, this profile always thinks at `high`.
An install option such as `--reasoning-effort max` replaces that deployment
default; a request can still choose `low`, `high` or `max`. GLM has no
thinking-off switch.

`--generation-config auto` loads the pinned checkpoint's sampling defaults:
`temperature: 1.0` and `top_p: 0.95`. The generation-config hashes and values
are recorded in the [thinking records](../thinking.json) and
[GLM-5.3 record](../research-thinking.json); no sampling override is needed.

Recorded measurements that omitted an effort choice used the chat template's
`max` default. They do not measure this profile's `high` deployment default.
Measurements with explicit request settings retain those stated conditions.

The [CRC32 variant](../glm53-flash-nvfp4-sparkcache-crc32-tp2/README.md) uses a
separate profile ID and the SparkCache source layer over release 2026.10.4.
This profile retains its earlier image composition and evidence.
