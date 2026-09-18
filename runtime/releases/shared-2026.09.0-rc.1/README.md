# Shared serving candidate: 2026.09.0-rc.1

Availability: **published opt-in candidate**.
The [GitHub prerelease](https://github.com/FujitsuPolycom/sparkring/releases/tag/shared-2026.09.0-rc.1)
provides checksummed sources, evidence and the publication receipt.
Bounded Qwen TP2/TP4 text/cache checks and short DeepSeek vLLM TP2 and SGLang TP4
text checks passed. GLM has bounded restore results with unresolved limitations;
other profile configurations remain unqualified.
`shared-2026.09.0-rc.1` is an opt-in candidate; no profile selects it by default.

```bash
docker pull ghcr.io/fujitsupolycom/sparkring@sha256:9439d39ad4b104c06cad3416a927f0f90852b456f0b06aa6a1dcf9475bf91a83
```

## Purpose and source identity

The ARM64/GB10 composition reconciles Local Inference Lab's vLLM and B12X with
SparkRing transports and SparkCache, retaining an isolated SGLang environment.
Profiles select one engine and their applicable optimizations. Sharing an image
does not run two servers on one GPU or enable every optimization for every model.

The [component and provenance index](components.md) identifies licenses, retained
attribution, source inputs and source-attachment requirements.

| Component | Frozen input |
|---|---|
| vLLM, `dev/karmic-kraken` | `35bab057b1751a6076a457803bcc4b78809689cf` |
| B12X | `a83336581a3a907076e60797df69ab66df5a2ff1` |
| Combined comparison image, Docker configuration ID | `sha256:b03062b032bb147f5255463d4f068bfc476bf966c78c82ddeb66df4fb1b0b37d` |
| Isolated SGLang source | `e087e662ba1ac4ef7747537e2a9141085efd4561` |
| Combined-image Mia adapter | `e59e6eb67479aa68f6fa700c600dc90a0729b5ec` |

The [upgrade runner](../../images/upgrades/README.md) owns reconciliation,
construction and verification. The installed
`/opt/sparkring/receipts/native-installed.json` records source, native-wheel,
feature/cache bindings and isolated-runtime inventories. The validation record
below binds its hash to this image. Source pins alone do not identify the image.

The [comparison image record](../../deepseek-v41-sglang/combined-image/local-build.json)
and [runtime isolation guide](../../deepseek-v41-sglang/combined-image/README.md)
retain their own evidence; those measurements do not qualify this candidate.
Standalone Mia adapter updates are not automatically part of the isolated
adapter revision listed above.

## Publication record

| Field | State |
|---|---|
| Registry package | `ghcr.io/fujitsupolycom/sparkring` |
| Published image tag | `shared-2026.09.0-rc.1` |
| Matching GitHub prerelease / Git tag | `shared-2026.09.0-rc.1` |
| Platform | `linux/arm64`; NVIDIA GB10/SM121 target |
| Runtime-build Docker configuration ID | `sha256:768620d56aaed88528941bf97b1d1d8a7f4aec0918a4c2d681689372f3838b32` |
| Publication candidate Docker configuration ID | `sha256:22da81cae0572ae2985a5c34a125db4f3cc58e871fa7e6256dffe59828f1ae5d` |
| Registry manifest digest | `sha256:9439d39ad4b104c06cad3416a927f0f90852b456f0b06aa6a1dcf9475bf91a83` |
| Runtime/tooling adoption commit | `3d85d78b16e917dafca915c223f8067594cfcc81`; not the earlier build-checkout identity |
| Native installed-payload receipt SHA256 | `3e60f6ef14760d7f4a41847d17ab80ffe88d4cf324675600f2d7402d81869f1c` |
| Installed-payload and cache-source binding | Passed; 46 source files and installed SparkCache consumer verifier checked |
| Runtime checks | 10 passed; imports, package constraints and CLI only |
| Two/four-rank transport component evidence | Same manifest tested on `a75bd02ffc1d`; dedicated probe not rerun on `22da81cae057` |
| Anonymous digest-pull verification | Passed; exact Linux ARM64 configuration `22da81cae057` |

Docker configuration IDs and registry manifest digests are different identities.
The [publication receipt](publication.json) binds the verified public digest to
the configuration ID; [release.json](release.json) pins that selection and its
inputs. The [source-bundle record](source-bundle.json) identifies the frozen
source archive attached to the GitHub prerelease. Do not substitute a
configuration ID for a registry manifest digest.

The [build/runtime validation record](../../../performance/records/images/shared-22da81cae057-validation-20260918.md)
binds these results to the named images. Native artifacts were reused
(`native_rebuilt=false`) and verified. Its metadata-equivalence proof covers all
123 filesystem layers and identical configuration except labels. The
[transport record](../../../performance/records/transport/rocenante-prepared-a75bd02f-20260918.md)
retains its separate image identity. Qwen's bounded model/cache results follow
below; those checks do not promote a profile's image selection.

## Profile validation

These links identify existing recipes, not image overrides for this candidate.
Their image pins remain unchanged.

| Configuration | Candidate validation |
|---|---|
| [Qwen3.8-Flash-Next QAD, TP4 + SparkCache](../../../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md) | [Bounded text and restart/restore passed](../../../performance/records/qwen38-flash-next/shared-22da81ca-tp2-tp4-cache-20260918.md); 7,200 cached tokens per fixture, all four ranks |
| [GLM-5.3-Flash NVFP4-Spark, TP4/DCP1 + SparkCache](../../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) | [Repeated restart restored both fixtures on all ranks](../../../performance/records/glm53-flash/shared-22da81ca-tp2-tp4-cache-20260918.md); an earlier exact-answer mismatch remains unresolved. Research-only |
| [Qwen3.8-Flash-Next QAD, TP2 + SparkCache](../../../profiles/qwen38-flash-next-tp2-sparkcache/README.md) | [Bounded text and restart/restore passed](../../../performance/records/qwen38-flash-next/shared-22da81ca-tp2-tp4-cache-20260918.md); 5,696 cached tokens per fixture, both ranks |
| [GLM-5.3-Flash NVFP4-Spark, TP2/DCP1 + SparkCache](../../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) | [64K/C4/2GiB resource trial restored both fixtures](../../../performance/records/glm53-flash/shared-22da81ca-tp2-tp4-cache-20260918.md). Larger settings hit memory pressure; public high-capacity profile not qualified |
| [DeepSeek-V4-Flash-0731, vLLM TP2](../../../profiles/deepseek-v4-flash-0731-pair/README.md) | [Startup and one C1 text request passed](../../../performance/records/deepseek-v4-flash/shared-22da81ca-tp2-smoke-20260918.md): 18 prompt / 2 completion tokens; 131K context, C8 capacity, 4,096 batch, 8GiB KV/rank, no SparkCache. The public 1M/C32/16GiB defaults are not qualified |
| [DeepSeek-V4.1 isolated SGLang TP4](../../../profiles/deepseek-v41-flash-sglang-cycle/README.md) | [Fresh startup and one authenticated C1 text request passed](../../../performance/records/deepseek-v41-flash/shared-22da81ca-sglang-tp4-smoke-20260918.md); no SparkCache or restart qualification |
| [DeepSeek-V4.1 vLLM](../../../profiles/deepseek-v41-flash-cycle/README.md) and other configurations | Not qualified |

Each result must identify the exact image, checkpoint, TP/DCP, speculation,
cache mode and workload. Launch, correct generation and bounded cache
capture/restore are initial serving checks, not full-context, media,
concurrent-cache, performance or soak qualification. Untested profiles retain
their established image selections after candidate publication.

## Retained limitations

- OpenAI `cache_salt` does not partition this image's SparkCache disk keys.
  A salted request is not a cold external-cache control or a multi-tenant
  isolation boundary. Use separate deployment/cache namespaces or disable
  external caching when isolation is required.
- SGLang restart reliability is unqualified. Comparison image `b03062b032bb`
  exhibited restart failures and a CUDA-initialization SIGBUS; the cause remains
  unresolved. This candidate is not a demonstrated fix.
- Candidate performance is unqualified. Measurements from other image IDs do
  not establish this image's throughput.
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
