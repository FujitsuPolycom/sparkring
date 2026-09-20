# SparkRing 2026.09.4-rc.3

Status: **research-only prerelease**. This opt-in ARM64/GB10 image is pullable by
Docker and has bounded Qwen3.8-Flash-Next NVFP4 QAD serving evidence. Stable
profile defaults remain unchanged.

```bash
docker pull ghcr.io/fujitsupolycom/sparkring:shared-2026.09.4-rc.3
docker pull ghcr.io/fujitsupolycom/sparkring@sha256:9450b6964aa87cdb44e3b190df9b6db12289a8850ee3513e5278241e70245a64
```

The image has two root filesystem layers. An anonymous pull into an empty Docker
27 daemon completed, and the pulled image verified all 235,690 files in its
installed-runtime receipt. The earlier `shared-2026.09.4-rc.1` and temporary
`shared-2026.09.4-rc.2` registry images exceed Docker's layer-depth limit and
must not be used.

## Included changes

- LIL vLLM `af9e4dca109e0348323c0182e98a3aaf7282bfc3` and B12X
  `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`, with SparkRing integrations retained.
- Separate scratch storage for concurrent GLM projection branches.
- MTP output-buffer sizing for each captured batch size.
- Shorter lifetimes for temporary startup and kernel-tuning allocations.
- Agreement between ranks before distributed tuning-cache reuse.
- Native NVFP4/BF16 kernels, route-packing updates and speculative-verification
  tuning inputs.
- CUDA Python 13.3 packages matched to the CUDA 13.3 foundation.
- QSA raw-ring bounds checks and prepared shared-pool validation programs.
- SparkCache request-scope isolation; unverified or incompatible entries remain
  cache misses.
- A prepared RoCEnante manifest bound to the installed Kraken B12X source tree.
- A flattened runtime filesystem that avoids Docker's registry-import depth limit.

## Profile evidence

| Profile | Evidence on this image filesystem |
|---|---|
| [Qwen Flash-Next QAD TP2 + SparkCache](../../../profiles/qwen38-flash-next-tp2/README.md) | **Bounded qualified:** startup, `READY` generation, 33K-token chunked prefill, solid-color image/video check, RoCEnante and healthy SparkCache on two ranks |
| [Qwen Flash-Next QAD TP4 + SparkCache](../../../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md) | **Bounded qualified:** startup, `READY` generation, 33K-token chunked prefill, solid-color image/video check, HC prefill/decode, RoCEnante and healthy SparkCache on four ranks |
| [Qwen Flash-Next QAD TP2 without SparkCache](../../../profiles/qwen38-flash-next-tp2/README.md) | Expected compatible; not checked on this prerelease |
| [Qwen Flash-Next QAD TP4 without SparkCache](../../../profiles/qwen38-flash-next-qad-tp4/README.md) | Expected compatible; not checked on this prerelease |
| [GLM Flash SparkCache TP2](../../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) | Pending; Spark and QAD checkpoints require separate checks |
| [GLM Flash SparkCache TP4](../../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) | Pending; Spark and QAD checkpoints require separate checks |
| [DeepSeek SGLang](../../../profiles/deepseek-v41-flash-sglang-cycle/README.md) | Runtime inventory preserved; startup and generation pending |

The Qwen checks used 262,144-token context, 16 sequences, batch 8192, 24 GiB
FP8 KV per rank, MTP3, three-image/one-video limits and fresh SparkCache/compiler
namespaces. TP2 exposed 2,877,721 KV tokens; TP4 exposed 3,131,214. The final
published image differs from the tested runtime only in OCI labels; its root
filesystem layers and runtime configuration are otherwise identical.

The media fixture contained three 64×64 red/green/blue images and one one-second
224×224 red video. Both deployments returned the four colors in order. This is
bounded input/interpretation evidence, not arbitrary media accuracy.

The chunked-prefill fixture contained 33,042 prompt tokens and required the
model to retrieve one deterministic code. Both deployments returned the exact
code. The configured scheduling batch is 8,192 tokens; this check establishes
correct chunked execution for one bounded prompt, not throughput or full-context
stability.

Both deployments also returned `SCORE_OK` with finite selected-token and
top-token log probabilities. This is a bounded numerical sanity check, not a
model-quality evaluation.

Full-context pressure, retained restart/physical restore, arbitrary media,
matched performance, GLM model serving and isolated SGLang generation remain
unqualified. Experimental multi-sequence MTP compaction is excluded. Issue
`#278` and separate long-duration collective failures remain unresolved.

## Publication

The [publication record](publication.json), [qualification record](qualification.json)
and [clean-pull verification](verification.json) bind the immutable image and
the stated evidence. Source archives and patches are byte-identical to the
attachments published with `shared-2026.09.4-rc.1`; their hashes are retained in
[source-bundle.json](source-bundle.json).
