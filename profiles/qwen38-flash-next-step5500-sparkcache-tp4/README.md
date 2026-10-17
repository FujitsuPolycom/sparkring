# Qwen3.8-Flash-Next step5500 with SparkCache on 4 Sparks

This profile runs [Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/60215d26cf5e42c2db6128774032d57fc62678da),
created by Local Inference Lab, Inc. and licensed under the Local Inference Lab
License, Version 1.0. The checkpoint is fixed at `60215d26cf5e42c2db6128774032d57fc62678da`.

Status: **Experimental** (`research-only`). The default image, release
2026.10.5's, admits this profile
([release record](../../runtime/releases/dev-20261016-sparkcache-4a17f1f5198e-status038/README.md)).
On that image it installed on four Sparks and served requests; on the image
composition's recheck build, with `--sparkcache-integrity sha256`, it restored
17,280 of 18,533 tokens of a saved session after a corrupted rank's cache was
dropped ([record](../../performance/records/images/dev-20261016-sparkcache-4a17f1f5198e-status038-sparkcache-recovery-ring8-20261016.md)).
No performance record exists. The
[hardware acceptance plan](../../docs/research/qwen38-sparkcache.md) states the
remaining gates.

Serving settings match [qwen38-flash-next-qad-tp4](../qwen38-flash-next-qad-tp4/README.md)
with its `qad-step5500-ple1000` checkpoint selected. The connector uses
`qwen38-flash-next-hybrid`, CUDA restore, asynchronous connector-job capture,
two 512 MiB slots and CRC32 `tail-cow-v2`. The cache budget is 32 GiB per rank
with a 28 GiB low watermark. Minimum span is 4,096 tokens; the configured maximum
is 61,920, aligned to 1,440-token manager pages.

The [checkpoint identity receipt](checkpoint-identity.json) binds the complete
pin manifest, configuration, loader and embedded MTP settings. Target and draft
share `759a6ce3138e95cddaa4dc8e53480274d2a3e75154cf1893ff275237b1ded2b6`. No recorded identity is replaced.
The rank-local root is `/cache/qwen38-flash-next-step5500-sparkcache-tp4-ab53c097d7ca-60215d26cf5e`.
Change this root if HC prefill, MXFP8 projection, draft backend or kernel-build
settings change. Do not share it with step 4000 or another checkpoint.

The research inventory contains five groups: three 12-layer GDN groups, one
PLE group and one attention group with 12 QSA layers plus BF16 MTP. The
corrected composition requires hardware qualification of incremental saves
and restored outputs. Prefix-cache retention remains at vLLM's default zero.

The [source composition](../../runtime/images/compositions/installer-sparkcache-3daf882e358b/README.md)
pins SparkCache `3daf882e358b0c0846f14b44122204a8774f2f66`. Checkpoint and native-library pins
are unchanged. The normalized worker identity misses entries written by
SparkCache a136fc5 with wrapper metadata. Build and qualify the image before selecting its local lock.
