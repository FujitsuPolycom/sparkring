# Jovian Judgement R33 integration for Spark

Status: research-only; image construction and qualification are incomplete.

This integration reproduces the pinned R33 software composition for ARM64 and
adds SparkRing transport, startup, profile and optional SparkCache integration.
The published R33 image is AMD64 and cannot supply native binaries to DGX Spark.
Architecture-specific artifacts require matching source and ABI validation.

The upstream image is
`localinferencelab/vllm@sha256:3ae04d964f8e7e936ef4b34dd75406b9169fead8062fb8b4b1634f3bbf512827`.
Its release source lock has SHA256
`c4b1029eb355736f94b02efa19b10efe4d9680cac41ae38b952b05de9e4381c4`.

| Component | R33 source |
|---|---|
| vLLM | `voipmonitor/vllm` at `ae89131442359dc332d9c46009be3c1f8cdee0b4` |
| B12X | `FujitsuPolycom/b12x` at `1b8d84a374bd48a770a02d19f850c3983c7f53ad` |
| LMCache | `local-inference-lab/LMCache` at `29bc5a2efde737c436b04499eb62cd1776cebeec` |
| FlashInfer | `803c4664f4771ddc418f20a57f752469a237a825` |
| FlashKDA | `3b225bf26bb8e218928a1fe14751cb48cf31d11b` plus the release patch |
| Build recipe | `local-inference-lab/blackwell-llm-docker` at `11c5c7fc7fc8fcad33994d6608885f646e50e4f2` |

Installed-component identities take precedence over inherited container labels.
The ARM64 foundation image
`sha256:6704db5df61d1110afaba538554abb024c85edb8f37a98e4e631166a10af3217`
contains CUDA 13.3, PyTorch source
`cf30153c4c131c8164ee7798e5022d810682e2cb`, and canonical NCCL 2.31.2 source
`fb6f40999a2a9e63104d4ae4a84118bce61528f8`. The routed NCCL candidate is
compiled from patched tree `aa7028b2b2a55af4817f8d742e17717dd4509ee7`
with library SHA-256
`84a4b8d83fb5fa1f0d640d311ad38b45140672dae9889775fe1e4a3990479e47`.
The native artifacts and vLLM ARM64 wheel are built; distributed GPU, model,
TP2/TP4, and cache-recovery qualification remains pending.

## Integration rules

- Preserve R33 behavior before applying missing SparkRing changes. Classify
  existing patches as present, superseded or still required against exact source.
- Keep hardware-forwarded ring/mesh routing, both host PCIe domains, and TP2
  communication assets source-bound. Test actual backend activation.
- Retain the upstream cache components while making the selected external
  connector explicit per profile. LMCache qualification does not qualify SparkCache.
- Preserve safe asynchronous capture, block ownership and hybrid restore recovery.
- Package one image with model/topology profiles; do not claim every model is
  qualified merely because the generic image contains its dependencies.
- Update the published image, source lock, verification receipts and quickstarts
  together only after qualification. Existing published inputs remain unchanged.

## Qualification

Start with CPU/source/native compatibility checks, then test TP2 and TP4 model
loading, sampler warmup, exact responses, prefix reuse, capture/restore and failure
recovery. Compare bounded matched prefill/decode runs with confirmed feature
activation. Test switch users through the appropriate profile and retain its
explicit untested-hardware status unless switched hardware is available.

Record the exact image and configuration used for every result. Keep rollback
inputs and restore verified serving after disruptive tests.
