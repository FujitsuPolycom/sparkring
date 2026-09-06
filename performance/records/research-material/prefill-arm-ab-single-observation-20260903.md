# Prefill Arm A/B single-observation comparison

**Status: research-only.** This record contains one manually transcribed
observation per arm and context size. The semantic identities of Arm A and Arm
B are not recorded, so the data does not establish a transport-performance
result.

![Prefill Arm A and Arm B comparison](prefill-arm-ab-single-observation-20260903.svg)

## Conditions

The machine-readable dataset defines `arm_a` and `arm_b` for a prefill workload
at concurrency one. Each cell has one observation. The model,
checkpoint, serving image, source revisions, hardware topology, parallel
configuration, cache state, and warm-up state are not recorded.

## Measurement

TTFT and token-throughput values are stored at the precision available in the
source benchmark summary. TTFT delta is `(Arm B / Arm A - 1) * 100`; a
negative value is better. Throughput delta uses the same formula; a positive
value is better. The machine-readable dataset is
[`prefill-arm-ab-single-observation-20260903.json`](../../receipts/research-material/prefill-arm-ab-single-observation-20260903.json).

## Result

| Context | Arm A TTFT | Arm B TTFT | TTFT delta | Arm A throughput | Arm B throughput | Throughput delta |
|---:|---:|---:|---:|---:|---:|---:|
| 8k | 3.45 s | 3.33 s | -3.5% | 2,372 tok/s | 2,464 tok/s | +3.9% |
| 16k | 6.62 s | 6.25 s | -5.6% | 2,477 tok/s | 2,597 tok/s | +4.8% |
| 32k | 12.75 s | 11.69 s | -8.3% | 2,572 tok/s | 2,765 tok/s | +7.5% |
| 64k | 25.02 s | 23.52 s | -6.0% | 2,620 tok/s | 2,743 tok/s | +4.7% |
| 128k | 50.44 s | 46.64 s | -7.5% | 2,599 tok/s | 2,763 tok/s | +6.3% |

## Conclusion

Arm B has lower TTFT and higher throughput at all five recorded
context sizes. The TTFT reduction ranges from 3.5% to 8.3%; the throughput
increase ranges from 3.9% to 7.5%. The data justifies a controlled repeated A/B
with recorded arm identities; it does not establish a cause.

## Limitations

- Every displayed cell contains one observation, so no variability or
  uncertainty can be calculated.
- The source benchmark summary is not retained; values are manually
  transcribed and deltas inherit its displayed precision.
- The two arms lack immutable artifact identities and recorded commands.
- Arm order, cache state, warm-up policy, and changed variables are unknown.
- The semantic identity of each arm is unknown.
- Transport counters do not identify which collective implementation ran.
- The record cannot be generalized beyond the five recorded context sizes.
