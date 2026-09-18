# Shared serving candidate: 2026.09.0-rc.1

Status: **built; bounded Qwen TP2/TP4 text and SparkCache checks passed;
unpublished, other profile checks pending**.
`shared-2026.09.0-rc.1` is a proposed publication identifier, not an available
image tag or a recommendation. No profile selects this candidate by default.

## Purpose and source identity

The ARM64/GB10 composition reconciles Local Inference Lab's vLLM and B12X with
SparkRing transports and SparkCache, retaining an isolated SGLang environment.
Profiles select one engine and their applicable optimizations. Sharing an image
does not run two servers on one GPU or enable every optimization for every model.

The [component and provenance index](components.md) identifies licenses, retained
attribution, source inputs and the pending publication payload.

| Component | Frozen input |
|---|---|
| vLLM, `dev/karmic-kraken` | `35bab057b1751a6076a457803bcc4b78809689cf` |
| B12X | `a83336581a3a907076e60797df69ab66df5a2ff1` |
| Combined comparison image, Docker configuration ID | `sha256:b03062b032bb147f5255463d4f068bfc476bf966c78c82ddeb66df4fb1b0b37d` |
| Isolated SGLang source | `e087e662ba1ac4ef7747537e2a9141085efd4561` |
| Combined-image Mia adapter | `e59e6eb67479aa68f6fa700c600dc90a0729b5ec` |

The [upgrade runner](../../images/upgrades/README.md) owns reconciliation,
construction and verification. Its finalized receipt must also identify patch
dispositions, SparkCache source and cache lease, transports, native wheels and
dependency overrides. Source pins alone do not identify the resulting image.

The [comparison image record](../../deepseek-v41-sglang/combined-image/local-build.json)
and [runtime isolation guide](../../deepseek-v41-sglang/combined-image/README.md)
retain their own evidence; those measurements do not qualify this candidate.
Standalone Mia adapter updates are not automatically part of the isolated
adapter revision listed above.

## Publication record

| Field | State |
|---|---|
| Intended registry package | `ghcr.io/fujitsupolycom/sparkring` |
| Proposed image tag and Git tag | `shared-2026.09.0-rc.1` |
| Platform | `linux/arm64`; NVIDIA GB10/SM121 target |
| Runtime-build Docker configuration ID | `sha256:768620d56aaed88528941bf97b1d1d8a7f4aec0918a4c2d681689372f3838b32` |
| Publication candidate Docker configuration ID | `sha256:22da81cae0572ae2985a5c34a125db4f3cc58e871fa7e6256dffe59828f1ae5d` |
| Registry manifest digest | Not published; no pull command available |
| Source/adoption commit | Pending |
| Native installed-payload receipt SHA256 | `3e60f6ef14760d7f4a41847d17ab80ffe88d4cf324675600f2d7402d81869f1c` |
| Installed-payload and cache-source binding | Passed; 46 source files and installed SparkCache consumer verifier checked |
| Runtime checks | 10 passed; imports, package constraints and CLI only |
| Two/four-rank transport component evidence | Same manifest tested on `a75bd02ffc1d`; dedicated probe not rerun on `22da81cae057` |
| Anonymous digest-pull verification | Pending publication |

Docker configuration IDs and registry manifest digests are different identities.
Do not substitute the comparison image ID for a candidate pull digest. Add a
machine-readable `release.json` only after its image and immutable input hashes
exist; the release-selection schema is not a progress log.

The [build/runtime validation record](../../../performance/records/images/shared-22da81cae057-validation-20260918.md)
binds these results to the named images. Native artifacts were reused
(`native_rebuilt=false`) and verified. Its metadata-equivalence proof covers all
123 filesystem layers and identical configuration except labels. The
[transport record](../../../performance/records/transport/rocenante-prepared-a75bd02f-20260918.md)
retains its separate image identity. Qwen's bounded model/cache results follow
below; no public registry digest exists.

## Profile validation

These links identify existing recipes, not launch instructions for this
unpublished image. Their image pins remain unchanged.

| Configuration | Candidate validation |
|---|---|
| [Qwen3.8-Flash-Next QAD, TP4 + SparkCache](../../../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md) | [Bounded text and restart/restore passed](../../../performance/records/qwen38-flash-next/shared-22da81ca-tp2-tp4-cache-20260918.md); 7,200 cached tokens per fixture, all four ranks |
| [GLM-5.3-Flash NVFP4-Spark, TP4/DCP1 + SparkCache](../../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) | Pending |
| [Qwen3.8-Flash-Next QAD, TP2 + SparkCache](../../../profiles/qwen38-flash-next-tp2-sparkcache/README.md) | [Bounded text and restart/restore passed](../../../performance/records/qwen38-flash-next/shared-22da81ca-tp2-tp4-cache-20260918.md); 5,696 cached tokens per fixture, both ranks |
| [GLM-5.3-Flash NVFP4-Spark, TP2/DCP1 + SparkCache](../../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) | Pending |
| [DeepSeek vLLM](../../../profiles/deepseek-v41-flash-cycle/README.md) / [SGLang](../../../profiles/deepseek-v41-flash-sglang-cycle/README.md) | Separate engine/profile checks pending |

Each result must identify the exact image, checkpoint, TP/DCP, speculation,
cache mode and workload. Launch, correct generation and bounded cache
capture/restore are initial serving checks, not full-context, media,
concurrent-cache, performance or soak qualification. Untested profiles retain
their established image selections after candidate publication.

## Retained limitations

- Retained SGLang containers on the comparison image failed to restart on two
  ranks; one CUDA-initialization probe reproduced SIGBUS. Fresh containers
  recovered, but the cause is unresolved. Rebuilding is not a demonstrated fix.
- Qwen TP2 completion rates on the comparison image were variable/lower than
  its parent. Parent-image throughput is not candidate performance evidence.
- Qwen HC row ownership remains TP4-only; TP2 must disable it. GLM and Qwen
  feature selections are separate.
- Source changes require a matching SparkCache lease and isolated trial cache
  namespace. An inherited lease or successful import is not restore proof.

## Version and publication procedure

Use `shared-YYYY.MM.PATCH-rc.N` for a candidate and `shared-YYYY.MM.PATCH` for an
approved serving release. The month identifies the release line, `PATCH`
advances for distinct releases within it, and `N` advances when candidate bytes
change. These are SparkRing versions, not LIL branch names or builder revision
numbers. Published tags and evidence must never be reassigned to different bytes.

1. Finalize and verify the build, installed payload and profile-scoped evidence.
2. Publish the immutable candidate tag; record its registry digest and verify an
   anonymous digest pull.
3. Create the matching GitHub prerelease with its pull command, pinned source,
   receipts, licenses, test table and limitations. Releases is the version
   landing page; GHCR Packages stores the image layers.
4. Promote individual canonical profiles only after qualification and explicit
   approval. Regenerate their exports and preserve rollback selections.

Publication can expose a candidate with profile validation pending; it does not
make that image every quickstart's default. Do not use floating `latest` to
bypass qualification or delete existing images or host-tool assets.
