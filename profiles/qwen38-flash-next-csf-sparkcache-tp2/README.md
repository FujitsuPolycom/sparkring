# Qwen3.8-Flash-Next csf with SparkCache on 2 Sparks

This profile runs [Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD/tree/dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2),
created by Local Inference Lab, Inc. and licensed under the Local Inference Lab
License, Version 1.0. The checkpoint is fixed at `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2`.

Status: **Experimental** (`research-only`). No serving, CUDA restore or performance
qualification record exists for the corrected composition. Follow the
[build and hardware acceptance plan](../../docs/research/qwen38-sparkcache.md).
An image lock must explicitly admit `qwen38-flash-next-csf-sparkcache-tp2`; published locks do not.

Serving settings match [qwen38-flash-next-tp2](../qwen38-flash-next-tp2/README.md)
with its `csf` checkpoint selected. The connector uses
`qwen38-flash-next-hybrid`, CUDA restore, asynchronous connector-job capture,
two 512 MiB slots and CRC32 `tail-cow-v2`. The cache budget is 8 GiB per rank
with a 6 GiB low watermark. Minimum span is 4,096 tokens; the configured maximum
is 49,152, aligned to 2,848-token manager pages.

The [checkpoint identity receipt](checkpoint-identity.json) binds the complete
pin manifest, configuration, loader and embedded MTP settings. Target and draft
share `7906c93c59bf153168170b601cf36ae7cdd31927797dc4608320fe1ba8ec2c78`. No recorded identity is replaced.
The rank-local root is `/cache/qwen38-flash-next-csf-sparkcache-tp2-ab53c097d7ca-dc5b8ab2fa14`.
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
