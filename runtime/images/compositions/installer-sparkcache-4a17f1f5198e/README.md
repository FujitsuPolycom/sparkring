# Combined SparkCache source layer

From the SparkRing repository root on an authorized ARM64 build host with the
pinned parent image available:

```bash
bash runtime/images/compositions/installer-sparkcache-4a17f1f5198e/build.sh
```

Status: **implemented**. The build recorded in [`built-image.json`](built-image.json)
(image `sha256:f625053225d4b57bfd7d0bb2e8348fea38e69472b10cef87c4701928abc7c4a8`)
is the image of release 2026.10.5
([release record](../../../releases/dev-20261016-sparkcache-4a17f1f5198e-status038/README.md)).
The recipe writes that file beside these inputs only after image admission. It
does not publish a release or push an image.

[sources.json](sources.json) pins SparkCache `4a17f1f5198eb2402bcd6669425c62857937d39f` for
GLM-5.3-Flash, Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash, including quorum
recovery after rank-local corruption, idle publication draining, schema-mismatch
miss reasons and change-only counters. Native libraries and the eleven-file
lease contract retain their pinned bytes. Integrity mode determines the
publication schema and cache identity; changing modes produces a clean miss.

The preparer authenticates parent receipts, checks all source bytes, and
rejects unreviewed inherited runtime modules. It verifies the parent vLLM
metrics logger against its receipt before applying the connector formatter.
The resolved descriptor and installed receipts pin both inherited and resulting
hashes, including the metrics logger. Unknown logger implementations fail closed.

INFO contains startup, restore, save and eviction summaries. Detailed timing,
JSON attribution and topology are DEBUG records. Counter logs contain changed
values only; idle reports are silent and `/metrics` remains complete.
Use `sparkring install --profile PROFILE --sparkcache-debug` to collect
detailed SparkCache logs and reuse attribution while keeping vLLM at INFO.
The deployment retains its logging configuration through restart and recovery.

[build-inputs.json](build-inputs.json) admits twenty profiles: the parent's
fifteen, GLM CRC32 TP2/TP4, Qwen SparkCache TP2/TP4 and DeepSeek SparkCache TP4.
The published GLM `8abbe9da67b8` and Qwen `3daf882e358b` compositions and
vendor directories retain their exact bytes and recorded builds.

Follow the [DeepSeek hardware gates](../../../../profiles/deepseek-v41-flash-sparkcache-tp4/HARDWARE.md)
and each profile's validation requirements before claiming serving qualification.
