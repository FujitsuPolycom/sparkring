# SGLang decoder-tail replay on a four-Spark cycle

Status: **implemented**. This controlled comparison attributes the measured
prefill improvement to decoder-tail replay. It does not qualify a larger
context setting or unattended production deployment.

## Conditions

Four GB10 DGX Sparks, directly cabled as a four-node cycle, TP4/EP4, local NVMe
checkpoint `deepseek-ai/DeepSeek-V4.1-Flash` at
`dba1be0a40aa45a94ad051997016db3960a90277`, SparkRing patched NCCL 2.30.7 mounted
in place over SGLang's pip NCCL. Both arms used the same locally built image,
Mia commit `e59e6eb67479aa68f6fa700c600dc90a0729b5ec`, image ID
`sha256:4252984d1cd642a51bd0cde06665a4f6a73726141dec760392a73cf968bce744`.
The source/base identity is recorded in the [runtime pins](../../../runtime/deepseek-v41-sglang/pins.json).

The four ranks were rebooted before each arm. Context 262144, prefill chunk
4096, maximum eight requests, memory fraction 0.90, requested total-token pool
1500000, DSpark block five without SPS/STS tables, b12x MXFP8 backend, MoE fused
finalize disabled, expandable segments disabled, Engram NVMe packed layout,
96 I/O threads and zero GiB row cache. Only the decoder replay flag differed.

## Measurement

Client wall-clock prompt throughput is server-reported prompt tokens divided
by request wall time for a one-token response. Target prompt sizes are labels;
actual token counts are recorded in the raw samples. Decode uses fixed-length
256-token generation at C1 and C4. The prompt distribution and decode timing
are separate from the multi-category vLLM benchmark; do not compare their
absolute rates as though the workloads were identical.

There was one observation per size/concurrency in each arm. A warm-up request
preceded the ladder, but the first nominal 16K prefill still included additional
initialization and is retained without being represented as steady-state
performance. There is no repeated-run confidence interval. Quality capture
used 20 fixed prompts, including four roughly 12–15K prompts, first-token top-20
log probabilities, and 48-token greedy continuations. Truncated-distribution
KL is a diagnostic, not a full-vocabulary divergence measure.

Raw observations: [replay-off prefill](sglang-decoder-replay-20260911/prefill-sgl-replay0.json),
[replay-on prefill](sglang-decoder-replay-20260911/prefill-sgl-replay1.json),
[replay-off decode](sglang-decoder-replay-20260911/decode-sgl-replay0.json),
[replay-on decode](sglang-decoder-replay-20260911/decode-sgl-replay1.json),
[off quality](sglang-decoder-replay-20260911/sgl-replay0.json),
[on quality](sglang-decoder-replay-20260911/sgl-replay1.json),
[off needle](sglang-decoder-replay-20260911/needle-sgl-replay0.json), and
[on needle](sglang-decoder-replay-20260911/needle-sgl-replay1.json).
Endpoint addresses are sanitized; numerical observations are retained.

## Result

| Measurement | Replay off | Replay on |
|---|---:|---:|
| Nominal 64K prefill, tok/s | 2006 | 3144 |
| Roughly 131K needle prefill, tok/s | 1861 | 2934 |
| Needle TTFT, seconds | 70.0 | 44.4 |
| C1 decode aggregate, tok/s | 35.4 | 37.4 |
| C4 decode aggregate, tok/s | 80.5 | 76.3 |

Both needle requests returned the exact passphrase. All 20 first-token choices
and all 20 greedy continuations matched. Top-20 diagnostic KL mean was 0.006,
maximum 0.12. These are the handoff's rounded summaries; raw quality distributions
are retained for independent recomputation.

## Conclusion

Enabling decoder-tail replay increased nominal 64K prefill throughput by about
57% in this comparison. With replay disabled, SGLang prefill was near the
separately measured vLLM implementation's approximately 2K tok/s. The matched
SGLang comparison isolates the replay flag; it does not attribute this gain to
two-batch overlap or generic engine scheduling.

The optimization retains all rows through layer 20, then executes layers 21–39
on the final 128 extend tokens per request. For a full 4096-token chunk, the
layer-token row count falls from 163840 to 88448, approximately 54% of the full
path. This is a work-count illustration, not a FLOP or measured communication
ratio. Smaller late-layer tensors reduce computation and collective payload.

## Limitations

The shortcut retains global KV/indexer memory but changes late-layer local SWA
visibility. A finite matching prompt set does not prove mathematical equivalence
to full prefill. Prompt-token log probabilities and arbitrary full-prompt hidden
state capture are not supported by the shortcut. The A/B does not cover mixed
batches, exact tail/chunk-boundary placement, sampled decoding, vision/tools,
400K retrieval, or long soaks. Those gates belong to the ongoing deployment
qualification and are not implied by this result. No communication profile was
captured in this experiment. Absolute results apply to this image, hardware,
transport, and workload only.
