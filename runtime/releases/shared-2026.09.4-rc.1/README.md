# SparkRing 2026.09.4-rc.1

Status: **research-only**. This is an opt-in ARM64/GB10 prerelease candidate.
Full-model qualification is pending. Stable quickstarts retain their existing
image selections; this candidate is not a replacement for their tested defaults.

Pull by version or immutable digest:

```bash
docker pull ghcr.io/fujitsupolycom/sparkring:shared-2026.09.4-rc.1
docker pull ghcr.io/fujitsupolycom/sparkring@sha256:19ab50bdb7689f2a5cd3167a8c97af66a1333c5bc01ca929881a573cc0fa3e10
```

## Included changes

- LIL vLLM `af9e4dca109e0348323c0182e98a3aaf7282bfc3` and B12X
  `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`, with SparkRing integrations retained.
- Separate scratch storage for concurrent GLM projection branches, avoiding
  overlapping temporary buffers.
- MTP output-buffer sizing for each captured batch size.
- Shorter lifetimes for temporary startup and kernel-tuning allocations.
- Agreement between ranks before reusing distributed tuning-cache selections.
- Native NVFP4 weight kernels with BF16 activations, route-packing updates, and
  tuning inputs that represent speculative verification workloads.
- CUDA Python 13.3 packages matched to the CUDA 13.3 foundation.
- QSA raw-ring bounds checks and prepared shared-pool validation programs.
- SparkCache request-scope isolation; incompatible or unverified cache entries
  remain misses rather than usable state.

## Qualification scope

GPU component evidence covers KDA projection behavior, stable selection,
raw-ring memory safety, shared-pool replay, large offsets, chunked selection,
route packing, native MoE, dense A16 and preparation behavior. Some evidence
comes from byte-equivalent runtime predecessors; publication records must retain
the exact image and external-test identities. Component success is not model
serving, persistent-cache recovery, media or performance qualification.

| Profile | Qualification on this prerelease |
|---|---|
| [GLM Flash SparkCache TP2](../../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) | Pending; Spark and QAD checkpoints require separate checks |
| [GLM Flash SparkCache TP4](../../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) | Pending; Spark and QAD checkpoints require separate checks |
| [Qwen Flash-Next QAD TP2](../../../profiles/qwen38-flash-next-tp2/README.md) | Pending |
| [Qwen Flash-Next QAD TP2 + SparkCache](../../../profiles/qwen38-flash-next-tp2-sparkcache/README.md) | Pending |
| [Qwen Flash-Next QAD TP4](../../../profiles/qwen38-flash-next-qad-tp4/README.md) | Pending |
| [Qwen Flash-Next QAD TP4 + SparkCache](../../../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md) | Pending |
| [DeepSeek SGLang](../../../profiles/deepseek-v41-flash-sglang-cycle/README.md) | Runtime inventory preserved; startup and generation pending |

Qwen Flash-Next QAD TP2/TP4, their SparkCache configurations, and the isolated
SGLang runtime also require exact-image model checks. Success on older releases
does not qualify these configurations on this prerelease.

Verifier-tuning validation is incomplete. An unconditional bitwise graph/eager
comparison failed on a nondeterministic reduction path; repeated eager execution
showed the same scale of variation. The corrected test retains its numerical
reference limits, but its complete M9/M16 run has not passed. Experimental
multi-sequence MTP compaction is excluded.

Issue #278 and separate long-duration collective timeouts remain unresolved.
No performance improvement or general image/video accuracy is claimed.

## Publication

The [publication record](publication.json) binds the immutable registry digest.
The GitHub prerelease attaches reconstructed source archives, complete patches,
the source manifest and component notices. No mutable stable tag or profile
default changes.
