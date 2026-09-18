# Qwen QAD: bounded TP2/TP4 SparkCache qualification

Status: **qualified for sequential text generation and cache restart/restore**.
The [sanitized receipts](shared-22da81ca-tp2-tp4-cache-20260918.json) identify image
`sha256:22da81cae0572ae2985a5c34a125db4f3cc58e871fa7e6256dffe59828f1ae5d`
and Qwen3.8-Flash-Next NVFP4-QAD checkpoint revision
`629bc3218833a38b475b719f34aa571666f4a03e`.

| Configuration | KV pin per rank | Initial text checks | Two cache fixtures after restart |
|---|---:|---|---|
| TP2/DCP1 | 33 GiB | 2 correct answers | 5,696 cached tokens per fixture; both ranks restored |
| TP4/DCP1 | 40 GiB | 2 correct answers | 7,200 cached tokens per fixture; all four ranks restored |

Both deployments use a configured 262,144-token context limit, 16 maximum
sequences, 8,192 batched tokens, MTP3 with one independent MTP module, aligned
recurrent checkpoints, and default prefix-cache retention interval **0**.
SparkCache is read-write with a 4 GiB persistent-store limit per rank. Image/video
limits are configured as 3/1; media inputs were not exercised.

Two separate prompts contained 7,870 and 8,194 tokens. Each seed request had
zero cached tokens, returned the expected verification key, and produced a
committed publication on every rank. All worker processes were restarted before
replaying the same requests. Container start timestamps changed while image,
configuration and mount hashes remained identical. Both replay answers matched,
the API reported the cached-token counts above, and every physical rank logged
the matching restore. Requests were sequential (C1), temperature 0, seed 779386,
maximum 128 output tokens and thinking disabled.

This qualifies neither the configured context/concurrency limits nor throughput,
media correctness, corruption recovery, eviction pressure or soak. Multi-module
MTP cache behavior is outside this result. Public quickstart defaults and image
availability are unchanged.
