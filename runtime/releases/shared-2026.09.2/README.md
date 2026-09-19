# SparkRing 2026.09.2

Shared ARM64 serving image for NVIDIA GB10 clusters, with vLLM, SparkCache and
an isolated SGLang runtime. CUDA 13.3; PyTorch 2.13.0. Model weights are separate.

Status: **implemented; full-model qualification pending**. Source checks,
installed-file verification, an anonymous image pull and a bounded CUDA startup
reproducer passed. Those checks do not qualify every serving profile. The
[qualification record](qualification.json) states the evidence and limits.

## Image and profiles

```bash
docker pull ghcr.io/fujitsupolycom/sparkring@sha256:b58746401f0d51874696eb7fe37f0cffa5bbd1a1aed1dce26ef7be322e8fe123
```

The versioned tag is `ghcr.io/fujitsupolycom/sparkring:shared-2026.09.2`.
Use a profile's complete quickstart, not an image substitution in unrelated flags.

| Profile | Status | Guide |
|---|---|---|
| Qwen3.8-Flash-Next QAD, TP2 | Implemented; qualification pending | [Two-node quickstart](../../../profiles/qwen38-flash-next-tp2/README.md) |
| Qwen3.8-Flash-Next QAD, TP2 + SparkCache | Implemented; qualification pending | [Persistent-cache selection](../../../profiles/qwen38-flash-next-tp2-sparkcache/README.md) |
| Qwen3.8-Flash-Next QAD, TP4 | Implemented; qualification pending | [Four-node quickstart](../../../profiles/qwen38-flash-next-qad-tp4/README.md) |
| Qwen3.8-Flash-Next QAD, TP4 + SparkCache | Implemented; qualification pending | [Persistent-cache selection](../../../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md) |

GLM and DeepSeek components are included but untested on this version. Their
profiles retain separate image pins; inclusion is not a compatibility guarantee.

## Runtime behavior

During B12X kernel timing, automatic Python cyclic cleanup is deferred until
the host releases queued GPU work. This prevents a reproduced startup deadlock
in which CUDA library unloading waits for that same work. Cleanup resumes after
the timing gate opens; libraries are not permanently retained. Explicit cleanup
calls, reference-count destruction and unrelated driver stalls are not covered.
The [correction and tests](../../../integrations/b12x/patches/stream-gate-gc.md)
describe this boundary.

Qwen sparse-attention metadata also clears unused request IDs before CUDA-graph
execution. The runtime retains B12X #394, multimodal hyperconnection forwarding,
checkpoint coalescing, custom communication and a configuration audit before
API readiness. Each profile selects its applicable features. No benchmark or
throughput claim is made here.

## Sources and rollback

[Publication identity](publication.json), [source inputs](sources/README.md) and
[component licenses](components.md) identify the payload. GitHub Releases holds
source attachments and release notes; GHCR Packages holds container layers.
Keep the preceding image, deployment settings and cache directory for rollback.
SparkCache selections use a dedicated `shared-2026092` namespace and retain the
source-bound vLLM connector contract. This namespace choice does not change the
checkpoint format or assert compatibility with earlier disk entries.
