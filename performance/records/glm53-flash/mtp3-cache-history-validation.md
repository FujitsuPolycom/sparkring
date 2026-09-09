# Persistent-cache pressure with native MTP3 decoding

Status: **research-only**. This record qualifies only the functional and
measurement conditions below; it is not a production throughput guarantee.

## Conditions

GLM-5.3-Flash-NVFP4-Spark revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e` ran on four NVIDIA GB10 hosts with
TP4/DCP4, native MTP depth three, 24 GiB KV capacity per rank, and dual-rail
hardware-forwarded RoCE transport. The serving image was
`sha256:75050f7b4dd7287f1ecb3e7e34226d24aa6c5b3e012a8bd6c8399e411cfbd908`,
containing SparkCache `607ccef061d0f511f45a2a8a93f74514c955d3a3` and the source
transforms in `runtime/glm53-spark-mtp3-mesh/experiments/cache-reuse/`.

The transport library SHA-256 was
`056243fad27d224b82e437925ffa2aed42037e6bd29f239f56076a832f6ca5cb`;
the SparkCache placement library SHA-256 was
`2657cdd2e54a097c9544e4c79ae62c0646db6db123ff24e4f0c384238c3a1e8d`.
The persisted cache used a 2 GiB maximum and 1.5 GiB low watermark per rank,
eight restore lanes, and a 16,384-token full-capture interval. Periodic full
capture is opt-in and increases write traffic.

Four text-only conversations started around 100K tokens and grew by about
2K tokens per turn, with a 512-token response limit and rotation at 160K.
Admission lasted 3,600 seconds; a 100M prompt-token ceiling was not reached.
Thinking was enabled. Three approximately 5.3K-token, 300-output-token probes
ran before and after load. Startup and warmup completed before measurement.

## Measurement

The harness `performance/harnesses/validation/conversation_soak.py` records
client monotonic-clock TTFT and stream-delta offsets. Decode rate is estimated
as `(completion_tokens - 1) / (last_delta - first_delta)`; it is not exact
token-level inter-token latency. Reported cached tokens come from API usage.
They do not independently distinguish local hits from external restores.

The numeric observations are in
[`mtp3-cache-history-observations.json.gz`](mtp3-cache-history-observations.json.gz).
The artifact identifies the raw receipt and harness by SHA-256 and preserves
all 551 response timings and stream offsets. Endpoint addresses, request IDs,
and prompt/generated text are omitted; numeric observations are unchanged.
The artifact verifier recomputes response counts and probe medians.

## Result

The run completed 551 responses without request errors: 545 conversation
requests and six probes. It admitted 70,074,391 prompt tokens, including
cached tokens, in 60m58.46s including probes and drain. All 525 continuations
reported at least half their prompt tokens cached.

| Probe median | Before load | After load |
|---|---:|---:|
| Client TTFT | 2.9436 s | 2.9798 s |
| Estimated decode rate | 51.20 tokens/s | 47.80 tokens/s |

The decode estimate decreased 6.65%; individual probe ranges overlap. Three
samples per side do not support a precise confidence interval or a claim of
zero performance drift.

## Conclusion

This four-rank composition completed the specified sustained cache workload
without request errors or a population of low-reported-cache continuations.
The probes did not reproduce the approximately 35% post-load slowdown
described in SparkCache issue #60 under these conditions.

## Limitations

The image combines SparkCache and runtime changes, so this is not an isolated
comparison of any individual patch. No matched original-image or restore-only
control was completed. The 2 GiB test policy is not the original issues'
40 GiB cache policy; multimodal traffic, C8/C16, and near-1M serving are not
qualified. Some resumed schedules still miss safely. One startup attempt
exited for an undetermined reason; the same image passed startup on retry.
This record does not qualify unattended startup availability or close
SparkCache issues #60 or #61.
