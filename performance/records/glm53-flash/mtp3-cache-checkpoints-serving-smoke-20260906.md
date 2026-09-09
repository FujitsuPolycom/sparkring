# Native-MTP3 cache/checkpoint image: bounded serving smoke

Status: **qualified** for the eight correctness checks below. The deployment
profile remains **research-only**.

## Conditions

The test used four NVIDIA GB10 Sparks in the managed hardware-forwarded mesh,
TP4/DCP4, native MTP depth three, and two recurrent prefill checkpoints. The
model was `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` at revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`.

The exact serving image was
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:11a556a54041fd823d152a7f051ac4f7c617dc539030df26e93008392fee0746`,
config ID
`sha256:6921a6c163ea40b603e19a0332330efe3dbccbf4dce9f6cbbf6b756c9231835a`.
It includes SparkCache revision
`48bbd2be4a7b972e56632a2d7b934bac5460f272`.

Each rank had 24 GiB KV capacity, 16 sequence slots, and an 8,192-token batch
limit. Persistent-cache limits were **40 GiB maximum and 32 GiB low watermark
per rank**, with periodic full captures disabled. The cache was warm and not
filled to capacity; one aggregate log sample reported 6.0 GiB used across
160.0 GiB configured capacity.

The client sent one sequence of eight requests at concurrency one, beginning
September 6, 2026 at 22:04:39 UTC. Requests used temperature zero, a 512-token
output limit, and low reasoning effort, except the deliberate unsupported
reasoning request. The conversation contained a repeated garden passage and
an access word, followed by a request to repeat that word. Image requests
asked for the color of a red or blue square, in red/blue/red order.

## Measurement

The one-off Python HTTP client's source SHA-256 and complete API observations
are retained in the [machine-readable record](mtp3-cache-checkpoints-serving-smoke-20260906.json).
Elapsed times use `time.perf_counter` around request construction, response
reading, and result checks. Start timestamps use `time.time`. Each duration
is a single observation; no variance estimate or matched throughput comparison
is available.

All four ranks passed the 5,472-file image verifier before launch. The host
memory and mesh startup gates passed, and request-shape and sampling warmup
completed before the smoke requests. Rank logs supplied commit times and
container health observations. The JSON retains those extracted measurements
without host addresses or unrelated client log entries.

## Result

All eight checks passed; their elapsed times sum to 8.013 seconds.

| Case | Result | Elapsed time |
|---|---|---:|
| Arithmetic text | Correct answer, normal stop | 0.255 s |
| 5,628-token conversation start | Correct access word | 3.100 s |
| 5,650-token continuation | Correct retained word; API reported 4,608 cached tokens | 1.657 s |
| Streamed arithmetic | Correct answer, normal stop, SSE completion marker | 0.249 s |
| Unsupported reasoning suppression | Expected HTTP 400 | 0.018 s |
| Red image | Correct color | 0.979 s |
| Blue image | Correct color | 0.869 s |
| Repeated red image | Correct color | 0.886 s |

All ranks committed the same 4,096-token context, whose digest prefix was
`5a8faef36b84`. Commit times were 164.0, 178.0, 184.9, and 182.9 ms for ranks
zero through three. No explicit persistent-restore event was observed during
the eight checks. All four containers were running and healthy afterward.

## Conclusion

The immutable image passed bounded text, growing-conversation, streaming,
reasoning-rejection, and image-response correctness checks. It also published
a persistent context on all four ranks. These observations establish serving
behavior for the exact image, beyond its separate
[source-equivalence checks](mtp3-integrated-image-source-equivalence.md).

## Limitations

The cache was configured for 40 GiB per rank but was not filled. This is not
a near-capacity eviction test, sustained-publication test, isolated benchmark,
long-duration soak, or unattended-availability qualification. It is not a
matched reproduction of [SparkCache #60](https://github.com/FujitsuPolycom/sparkcache/issues/60)
or [#61](https://github.com/FujitsuPolycom/sparkcache/issues/61).

API cached-token accounting does not distinguish local prefix retention from
persistent restoration. The image prompts contained 223 tokens, including
196 image tokens, below the persistent cache's minimum span. They establish
image-response correctness, not multimodal persistent-restore qualification.
The one-off harness is identified by hash but is not packaged as a reusable
public harness.

Public requests could occupy scheduler slots before readiness warmup finished
in this image. External traffic contended with warmup; after its queue drained,
warmup completed without a model restart. The startup-admission fix in merged
[SparkRing #237](https://github.com/FujitsuPolycom/sparkring/pull/237) is
**implemented in repository source but absent from this published image**.
Its HTTP 503 admission gate requires a rebuilt image and separate startup
verification. Merging source does not change the running image.
