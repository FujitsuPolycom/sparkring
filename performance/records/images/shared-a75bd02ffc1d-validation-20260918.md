# Shared serving candidate: bounded image validation

Status: **qualified for the checks below; model serving pending; unpublished**.
The [sanitized evidence](shared-a75bd02ffc1d-validation-20260918.json) binds
runtime-build configuration ID
`sha256:cc4f6c8f7665a45307f846e7332953ed0d1aae0ebd6842c94109e743ae12b8aa`
to metadata-only candidate ID
`sha256:a75bd02ffc1dd29713f97f599ccb10a88ba76bb5cd9d3554a98c875e96c0b942`.
Neither ID is a public registry pull digest.

| Check | Result and boundary |
|---|---|
| Installed runtime payload | 235,822 files verified; native artifacts reused (`native_rebuilt=false`) |
| Isolated SGLang payload | 177,742 files verified within its own inventory; counts are not additive |
| SparkCache source contract | All 46 source hashes verified; installed `verify_contract` accepts schema and required members |
| Runtime checks on the candidate | 10 passed, exit 0, no timeout: GB10 detection; native vLLM; Qwen/GLM model and MTP imports; B12X; FlashInfer; SparkCache; serving CLI; package constraints; isolated SGLang |
| Metadata equivalence | All 123 ordered filesystem layers and raw/runtime configuration except labels are identical |
| [Two/four-rank transport](../transport/rocenante-prepared-a75bd02f-20260918.md) | All ranks passed 15 cases, including four frozen-kernel CUDA graph replays |

Native receipt SHA256:
`165f5841b3b11a74209996ce5ee26d64d2f7062fb5a78a9443b9650cd764cdef`.
Build-input SHA256:
`84c5ba8cd9c02e230a5c2ed85421fb55400a9a59fe4130b0179a574de8bcca53`.
The evidence records the complete source snapshot hashes, consumer verifier
hash, source-report hashes and metadata-equivalence proof digest.

The [executed CPU/image-gate archive](../../../runtime/releases/shared-2026.09.0-rc.1/qualification/execution-oracles-c7847ecb.tar.gz)
preserves exact hash-bound inputs, including their line endings. Its SHA256 is
`4ed1dadd7d7d687d2942e163eef8a2549e41dde46a08003753703dc8b46db5db`.
The installed-contract gate invokes the image's own SparkCache verifier before
model startup; it does not substitute source hashes for consumer validation.

No model generation, persistence restore, throughput, media or soak result is
asserted. CacheIdentity wire values, digest salts and geometry are unchanged by
the preparation fixes; the source migration still requires a fresh candidate
persistence namespace. The image-specific records for `35cf12b2`, `98d5a51e`
and `5252600f` retain their own evidence and do not describe this execution.
