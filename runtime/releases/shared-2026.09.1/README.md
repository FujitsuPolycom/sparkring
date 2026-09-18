# SparkRing 2026.09.1

Shared ARM64 serving image for NVIDIA GB10 clusters, with vLLM, SparkCache and
an isolated SGLang runtime. CUDA 13.3; PyTorch 2.13.0. Model weights are separate.

Status: **implemented**. Image/source verification passed; the
[qualification record](qualification.json) identifies serving checks and limits.
Presence of a model integration does not qualify its deployment.

## Image and profiles

```bash
docker pull ghcr.io/fujitsupolycom/sparkring@sha256:3a8cdcf34ac51fdeb5a433107710e7787d52834281f377a7432b7df737035bbb
```

The versioned tag is `ghcr.io/fujitsupolycom/sparkring:shared-2026.09.1`.
Use a profile's complete quickstart, not an image substitution in unrelated flags.

| Profile | Guide |
|---|---|
| Qwen3.8-Flash-Next QAD, TP2 | [Two-node quickstart](../../../profiles/qwen38-flash-next-tp2/README.md) |
| Qwen3.8-Flash-Next QAD, TP2 + SparkCache | [Persistent-cache selection](../../../profiles/qwen38-flash-next-tp2-sparkcache/README.md) |
| Qwen3.8-Flash-Next QAD, TP4 | [Four-node quickstart](../../../profiles/qwen38-flash-next-qad-tp4/README.md) |
| Qwen3.8-Flash-Next QAD, TP4 + SparkCache | [Persistent-cache selection](../../../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md) |

Other profiles retain their own image pins. GLM and DeepSeek runtime components
are included but have no hardware qualification on this version.

## Behavior

Sparse-attention metadata clears unused CUDA-graph request IDs, preventing
shorter Qwen requests from inheriting padding metadata from longer requests.
Graphs, MTP, coalescing and custom communication remain available. The
[source correction](../../../integrations/vllm/patches/qwen-qsa-padding.md)
documents its invariant and regression checks.

The runtime includes B12X #394 sparse-attention selection, multimodal
hyperconnection forwarding, and a configuration audit before API readiness.
Feature selection and qualification remain profile-specific. This guide makes
no throughput claim.

## Sources and rollback

[Publication identity](publication.json), [source inputs](sources/README.md) and
[component licenses](components.md) identify the payload. GitHub Releases holds
source attachments and release notes; GHCR Packages holds container layers.
Components retain their own terms. Keep the prior image, deployment settings and
cache directory for rollback. Each SparkCache profile selects a source-bound
compatibility contract and a release-specific namespace.
