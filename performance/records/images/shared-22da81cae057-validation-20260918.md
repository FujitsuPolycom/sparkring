# Shared serving candidate: image validation

Status: **qualified for bounded image checks and separately recorded Qwen
text/cache checks; unpublished**. The [sanitized record](shared-22da81cae057-validation-20260918.json)
binds runtime-build ID
`sha256:768620d56aaed88528941bf97b1d1d8a7f4aec0918a4c2d681689372f3838b32`
to metadata-only candidate ID
`sha256:22da81cae0572ae2985a5c34a125db4f3cc58e871fa7e6256dffe59828f1ae5d`.
Neither configuration ID is a public registry digest.

- Installed-payload verification passed for 235,822 files. Native artifacts were
  reused (`native_rebuilt=false`), not compiled anew in this build.
- The active SparkCache lease binds 46 source files and passes the installed
  consumer's schema/hash/member verification.
- All 10 runtime import, GPU-detection, dependency and CLI checks passed on the
  candidate with exit 0 and no timeout.
- Metadata equivalence preserves all 123 ordered filesystem layers and every
  raw/runtime configuration field except labels.
- [Qwen TP2/TP4 text and restart/restore checks](../qwen38-flash-next/shared-22da81ca-tp2-tp4-cache-20260918.md)
  passed under their bounded conditions. GLM model checks remain pending.

Native receipt SHA256:
`3e60f6ef14760d7f4a41847d17ab80ffe88d4cf324675600f2d7402d81869f1c`.
Source snapshot hashes are vLLM
`0481eb71311f399d8550ad2cdd30a5843fb8263c8a9521b7926ea9ab76550aac`
and B12X
`771fa73ec51687e36eb6ecf8aa5d8af39302f59dce6bd82d69165ba4c6230452`.
The JSON record retains the build-input and original report digests.

The [transport component evidence](../transport/rocenante-prepared-a75bd02f-20260918.md)
uses image `a75bd02ffc1d` with the same manifest
`7d8beed57e541c756995b73b0909a826642cef7c00f8147316cc8cd97a2f905c`.
That dedicated TP2/TP4 probe was **not rerun on this image**. Its original image
and harness identities remain unchanged; successful Qwen serving is separate
evidence, not a relabeling of that component run.
