# GLM-5.3 adaptive-MTP and DFlash7 decode-log diagnostic

Status: **research-only**. This record preserves bounded server-log evidence.
It is not a benchmark or a controlled A/B comparison because the prompt,
generated output, and client timing receipt were not retained.
It does not qualify either runtime, any image, or any SparkCache behavior.

## Conditions

Both observations used four NVIDIA DGX Spark systems at TP4/DCP1/PP1 and the
target checkpoint
`local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`.

The adaptive-MTP service used image ID
`sha256:d209cf986c1f14f320e53d8b425c5e3a255eef9320f25b632af50f5b5c977314`.
Its immutable composition identified:

- retained compiled vLLM commit `da4d7be6c97434f6942292ed8abbf4b32dc44355`;
- Python vLLM commit `0b67266a0f37d6146a8403fb8482403c62f412d5`;
- B12X commit `b1d541f9e71a35f030d45fae437630fff7507c2a`;
- SparkCache commit `20838ace3ebda570ca039cb7f1976c29da554b39`;
- SparkRing revision `914c94d084d6881e90660305dedaa410ef02b167`;
  and
- embedded-MTP draft identity
  `2e06d909ce5bb71c0c0e3e8be74a70e3b41d92ba4c30196cfb0957fb812acef6`.

Adaptive MTP used maximum depth five, initial depth three, and a 32-step
observation window. The observed rank-0 log interval was
`2026-08-30T03:23:25Z` through `2026-08-30T03:25:15Z`.

The retained DFlash7 service used vLLM commit
`da4d7be6c97434f6942292ed8abbf4b32dc44355`, B12X commit
`2fcf23a0ce269be27b2e03fece73d46e90e6aeea`, and SparkCache commit
`2b86fb9d02fa3595cca5caa864b81aedce44b8bb`. Its external BF16 drafter was
`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`
at fixed depth seven. Per-rank image IDs are preserved in the linked receipt.
The observed rank-0 log interval was `2026-08-29T21:13:57Z` through
`2026-08-29T21:15:07Z`.

After the DFlash7 service was restored, a second rank-0 interval from
`2026-08-30T04:02:49Z` through `2026-08-30T04:04:09Z` recorded the same
fixed-depth-seven runtime under operator traffic. The prompt and output were
again not retained, so this interval is diagnostic evidence rather than a
controlled replay.

The two compositions differ in vLLM Python, B12X, SparkCache, model loading,
and draft implementation. The observations therefore do not isolate one
changed variable.

## Measurement

The source is vLLM's rank-0 ten-second `Avg generation throughput` gauge and
the speculative-decoding metrics emitted at the same timestamps. The
machine-readable fixture preserves all 12 adaptive-MTP observations, the
eight earlier DFlash7 throughput observations, and nine post-switch DFlash7
observations used here.

Neither interval contains a `spark-context-cache` store, restore, or
maintenance event. SparkCache was enabled, but it was idle during the measured
decode windows. The cumulative cache-hit percentages printed beside the gauge
do not identify work performed during an individual interval.

No client request shape, prompt digest, output-token sequence, or quality
result was retained. The displayed ranges are minima and maxima, not means or
confidence intervals.

## Result

| Runtime observation | Depth seen | Draft acceptance | Generation throughput |
|---|---:|---:|---:|
| Adaptive embedded MTP | 2–4 | 48.9–81.2% | 24.5–38.7 tok/s |
| Retained DFlash7, earlier interval | fixed 7 | not summarized for this record | 40.6–136.0 tok/s |
| Retained DFlash7, post-switch interval | fixed 7 | 47.6–67.0% | 48.8–74.5 tok/s |

The operator also reported a matched coding-peak observation of approximately
70 tok/s for DFlash7 and 33.5 tok/s for adaptive MTP. No prompt or output
receipt accompanies those two values, so they are recorded as user-reported
context rather than measured evidence.

## Conclusion

The adaptive runtime changed draft depth between two and four while serving,
which confirms that acceptance-based control executed. During the recorded
windows, its server generation gauge remained below both retained DFlash7
intervals. The post-switch interval independently reproduced the operator's
approximately 70 tok/s DFlash7 performance class. The evidence does not
establish a general DFlash7 advantage because the requests and outputs were
not preserved and the runtime compositions were not otherwise matched.

SparkCache performed no logged work in either decode interval. These numbers
do not measure or implicate SparkCache restore, placement, publication, or
maintenance.

## Limitations

- The prompt, output token IDs, response text, sampling settings, and client
  timing receipt are absent.
- Server gauges are periodic windows, not request-level benchmark results.
- The two runtime compositions differ in more than draft implementation.
- The observations have no repetitions or uncertainty calculation.
- High DFlash7 throughput could reflect an easy or repetitive output; output
  quality cannot be reconstructed from throughput logs.
- The user-reported coding-peak pair is not independently auditable.

## Provenance

The [machine-readable diagnostic fixture](../../receipts/glm53-flash/adaptive-mtp-vs-dflash7-20260829/diagnostic.json)
contains exact identities, intervals, and transcribed observations. It marks
the missing prompt and output receipts explicitly.
