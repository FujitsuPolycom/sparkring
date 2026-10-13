# Bounded pair comparison: GLM-5.3-Flash TP2/DCP2

Status: **Plan, not executed**. Pair serving correctness and performance are
**unknown**. Maximum reservation: 60 minutes, positions 6-7 only. Build and
inspect the image and prepare client files on the workstation before the
reservation. Do not do uncounted GPU warm-up or compilation on the pair.

Use the same derived image, CSF checkpoint `dec48abd33ef`, TP2/DCP2,
interleave 4, 14 GiB KV per rank, 8192-token ceiling, MTP3, transport, clock
policy and profile arguments for both arms. The only differences are:

| Arm | KDA coalescing | mHC row ownership |
|---|---:|---:|
| A | 0 | 0 |
| B | 1 | 1 |

This comparison measures the combined effect. It does not attribute a gain
to either feature alone. Alternate arm order in a separately authorized
repeat if promotion is proposed; no repeat is hidden in this reservation.

## Clock and stop conditions

| Elapsed minutes | Work |
|---|---|
| 0-15 | Confirm exclusive availability of positions 6-7; start A and wait for readiness |
| 15-18 | A correctness screen and diagnostics |
| 18-29 | A benchmark, hard timeout 660 seconds |
| 29-30 | Stop A, preserve logs; start B |
| 30-45 | B readiness, including any compilation |
| 45-48 | B correctness screen and diagnostics |
| 48-59 | B benchmark, hard timeout 660 seconds |
| 59-60 | Stop only this comparison's containers, collect receipts and release the pair |

An independent owner-side deadline must stop the comparison's containers
by minute 60 even if a client hangs. No automatic recovery or retries. If
readiness misses its deadline, stop that arm and report the run incomplete.
Abort on nonfinite output, failed functional check, repeated degenerate
responses, collective error, rank disagreement, GPU OOM or host memory
below the owner's operational floor. Leave other ring positions untouched.

**Measured prior startup:** the both-off DCP2 arm took about 880 seconds
on image d52737a109e0. **Inferred:** the 15-minute startup budget is tight.
Twenty 20-second benchmark cells need at least 400 seconds before prefill,
warm-up and coding requests. **Unknown:** whether each complete matrix fits
660 seconds. Keep every partial cell and its timeout status; do not shorten
the requested matrix silently, extrapolate missing cells or call a truncated
run a successful full comparison. If it cannot finish, report the timing
constraint and seek a later reservation instead of overrunning this one.

## Benchmark and correctness

Run the owner's installed **llm-inference-bench 0.7.6** from its repository,
with the same endpoint/model settings per arm and a distinct output path:

```bash
LLM_BENCH_NO_UPDATE_CHECK=1 timeout --signal=TERM --kill-after=5s 660s \
  ~/venvs/lmcache-lil/bin/python llm_decode_bench.py \
  --host "$API_HOST" --port "$API_PORT" --model GLM-5.3-Flash-CSF-DCP2-TP2 \
  --dcp-size 2 --temperature 1 --display-mode plain --no-hw-monitor --no-resume \
  --concurrency 1,2,4,8 --contexts 8k,16k,32k,64k,128k \
  --duration 20 --max-tokens 2048 --coding-peak --output "$ARM_OUTPUT/bench.json"
```

Keep raw logs as well as JSON. Use the benchmark's cold scouts for prefill
8K-128K, recording actual prompt tokens, TTFT and prompt tok/s. Confirm
prefix-cache hits are absent in cold scouts; preserve hit counts. Report
decode output tok/s, engine steps/s, MTP acceptance length, cell duration
and coding median. Do not compare cached TTFT to uncached prefill.

Within each three-minute correctness slot run the repository acceptance
harness with `--skip-install --steps readiness,functional,stress`, this
profile and its API endpoint, wrapped in a timeout. Preserve the seven
functional checks and 256-request greedy screen results and completion
count. Reserve a small part of that slot for identical cold and warm
prompts of 8192+, 16384+ and 32768+ tokens, greedy decoding with fixed input
tokens, 32 output tokens, and logprobs where available. Compare A/B output
token IDs, first divergence and logprob differences; repeat prompts to
exercise prefix reuse and use unique prefixes for cold requests. A missing
or timed-out check is unknown, not a pass. Sampling at temperature 1 is a
performance condition, not a token-equality test.

Collect both ranks' startup argv/environment, source hashes/build pins, KV
capacity, transport receipts, diagnostics, memory and error logs. Enable
the same bounded diagnostic logging on both arms. B must show actual KDA
coalescing and `GLM_MHC_PREFILL` rows=8192/owner_rows=4096; otherwise a
fallback cannot count as a feature-on measurement. For N base layers,
expect 2N reduce-scatters and 2N plus auxiliary gathers per admitted forward.
SIRCL TP and DCP sessions must both be present, with the agreed NCCL policy.

The correctness screen detects gross failures; even identical greedy tokens
do not establish full-logit equality. Preserve these limits in the report.
