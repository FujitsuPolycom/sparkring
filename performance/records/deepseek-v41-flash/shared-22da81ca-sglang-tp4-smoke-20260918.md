# DeepSeek-V4.1 SGLang TP4 bounded serving check

Status: **qualified for fresh startup and one authenticated short text request**.
The shared image with Docker configuration ID
`sha256:22da81cae0572ae2985a5c34a125db4f3cc58e871fa7e6256dffe59828f1ae5d`
started the isolated SGLang runtime on four NVIDIA GB10 nodes and returned `42`
with a normal stop after two completion tokens.
[Machine-readable evidence](shared-22da81ca-sglang-tp4-smoke-20260918.json)
records checkpoint metadata, settings and receipt hashes.

| Setting | Tested value |
|---|---|
| Model / parallelism | DeepSeek-V4.1-Flash / TP4, EP4 |
| Context / request capacity | 655,360 / 8 |
| Prefill chunk / admission slots | 4,096 / 1 |
| Token budget | 1,500,000 requested; 1,499,904 reported |
| Static memory fraction / speculation | 0.90 / DSpark5 |
| SparkCache | Disabled |
| Request | C1; 18 prompt tokens; temperature 0; thinking disabled |

The test reused existing read-only checkpoint and packed NVMe Engram tables.
It did not download, copy or repack weights. Authentication used the saved
secret-file reference; no key material is included in this record.

First-use kernels compiled during readiness warmup. This fresh-container result
does not resolve the retained-container restart failure observed on the separate
`b03062b` baseline image. No performance, full-context, concurrency-pressure,
SparkCache, media or soak claim follows from the short request. Existing profile
image selections remain unchanged.
