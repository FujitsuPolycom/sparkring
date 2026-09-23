# Qwen3.8-Flash-Next with SparkCache on four Sparks

Status: **qualified for bounded correctness and restart checks**.
Use the [Qwen QAD TP4 quickstart](../qwen38-flash-next-qad-tp4/README.md) and select
`PROFILE=qwen38-flash-next-qad-tp4-sparkcache` before image selection and retain
that value through rendering.
Image/model/fabric preparation and start/stop/restart instructions are shared.
Both variants pull [shared-2026.09.3](../../runtime/releases/shared-2026.09.3/README.md);
no additional cache image or source overlay is required.

The [configuration](../qwen38-flash-next-qad-tp4/sparkcache.json) retains TP4/DCP1,
MTP3, 262K context, 16 sequences, batch8192, 24 GiB FP8 KV per rank and media
limits of three images/one video. HC row sharding, fusion, checkpoint coalescing,
compact MTP and projection overlap are enabled. The startup audit reports
configuration/source checks before API readiness; real HC execution is separate.

Persistence uses aligned checkpoints and 32-token requested attention blocks.
The per-rank disk limit is 4 GiB, with two 512 MiB capture slots and a 256 MiB
restore budget. A dedicated release-specific namespace prevents accidental reuse
of a previous deployment's entries. Keep the previous namespace for rollback.

The [qualification record](../../runtime/releases/shared-2026.09.3/qualification.json)
records bounded text/media correctness, concurrent-request
checks, retained restarts and physical cache restore on all four ranks. It does
not claim performance, full-context/C16-pressure stability or arbitrary video
accuracy. Results apply to the identified image and configuration only.

The API has no configured key: restrict it to trusted clients or an authenticated
gateway. Request `cache_salt` does not isolate persistent entries in this image.
Use separate deployments/cache namespaces where tenant isolation is required.
