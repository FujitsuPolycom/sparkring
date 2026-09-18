# GLM text and persistent-cache checks on the shared ARM64 image

Status: **research-only; bounded results with unresolved limitations**.
The tested Docker configuration is
`sha256:22da81cae0572ae2985a5c34a125db4f3cc58e871fa7e6256dffe59828f1ae5d`.
[Machine-readable evidence](shared-22da81ca-tp2-tp4-cache-20260918.json) binds
the test settings and private receipt hashes. The local NVFP4-Spark checkpoint
revision is `df116c4fb16b1d37ae43d2cfd624de26ffbc832e`, not the public profile's
`ec0c3c` selection. No profile defaults were changed.

| Setting | TP4 | TP2 resource trial |
|---|---|---|
| Context limit / KV per node | 1,048,576 / 24 GiB | 65,536 / 2 GiB |
| Sequences / batch | 16 / 8,192 | 4 / 4,096 |
| Target / recurrent page size | 512 / 512 | 512 / 512 |
| Prefill coalescing | Enabled | Disabled |
| Loader / speculation | B12X / MTP3 | B12X / MTP3; Humming draft MoE |

Both tests use C1 requests, aligned recurrent checkpoints, low reasoning effort
and synthetic verification keys in 7,870- and 8,194-token prompts. Their purpose
is to test correct key retrieval and all-rank external-cache restoration after
worker process restart, not throughput or maximum capacity.

## TP4 result and unresolved failure

The cold requests returned exact keys and published 7,680 and 8,192 tokens on
all four ranks. After the first worker restart, the 7,870-token request restored
7,680 tokens correctly; the 8,194-token request failed exact-answer validation.
That failed response was not retained, so its cause is unresolved.

Repeating a full worker restart with unchanged image, settings and fixtures
passed both exact-key checks, restoring 7,680 and 8,192 tokens on every rank.
This successful repeat is evidence of bounded capability, not proof that the
earlier failure was fixed. The private harness now preserves raw responses before
validation.

## TP2 resource and alignment boundaries

The 7.5 GiB KV/1M-context and 4 GiB KV/262K-context trials encountered severe
host memory pressure and were stopped. Exit code 137 followed operator
termination; it is not evidence of a kernel OOM kill.

Mixed 256-token recurrent and 2,048-token attention pages also prevented
publication at 7,680 tokens. The resource trial uses uniform 512-token pages,
smaller capture buffers and the settings above. Coalescing is disabled because
its admission condition requires an 8,192-token batch.

Both cold resource-trial requests returned the correct key and published cache
entries. The 8,194-token response wrapped the key in Markdown bold, failing the
strict formatting check. A separately recorded policy accepts only the literal
key or that exact key wrapped in one `**` pair; wrong keys, added prose and
truncated replies still fail. After both workers restarted, both fixtures
returned exact bare keys and restored 7,680 and 8,192 tokens on each rank.
The machine-readable record preserves both validation policies. This is not a high-capacity profile
qualification or a performance recommendation.

## Exclusions

No full-context, concurrent-load, media, corruption, eviction or soak test is
covered. OpenAI `cache_salt` does not partition SparkCache disk entries; use
separate deployments/cache namespaces when isolation is required. Adding a salt
is not a valid cold external-cache control for this composition.
