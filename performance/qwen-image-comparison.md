# Bounded Qwen image comparison

Compare images with the same checkpoint, profile settings, hosts and workload.
Keep unrelated traffic and image transfers idle while measuring. Record image
IDs, source revisions, effective arguments and feature activation separately.
These tests measure serving behavior; they do not establish long-duration stability.

## Prefill

Use a fresh fixture identifier and output paths for each independent comparison.
The first arm creates exact token fixtures; reuse that fixture file and identifier
for the other arm after restarting the model. Each size has three disjoint
prefixes and generates one output token.

```bash
python performance/harnesses/qwen_prefill.py \
  --base-url http://SERVER:8015/v1 --model Qwen3.8-Flash-Next-NVFP4-QAD \
  --fixtures /tmp/qwen-prefill-fixtures.json --fixture-id shared-image-comparison \
  --output /tmp/qwen-prefill-baseline.json
```

The default sizes are 16K and 32K. Output records actual prompt/output accounting,
elapsed request time, token throughput, and fixture/harness hashes. The elapsed
time includes the single decode token and API overhead. Inspect any reported
cached-token count; do not relabel a repeated prefix as a cold measurement.

## Decode

```bash
python performance/harnesses/qwen_collectives.py \
  --base-url http://SERVER:8015/v1 --model Qwen3.8-Flash-Next-NVFP4-QAD \
  --label baseline --output /tmp/qwen-decode-comparison --trials 2 \
  --fixture-id shared-image-comparison
```

Use a different label but the same fresh fixture identifier for the other image.
The harness runs a cold pass and then a warm pass at C1 and C8. It reports wall
throughput separately from aggregate decode-window throughput. The decode
window starts at the first content chunk across requests and ends with the
last completed stream; its numerator subtracts one initial token per request.
This streaming metric is not isolated GPU decode time or a scheduler token-step rate.

Compare medians across repeated trials, retaining individual results and method
definitions. Pair performance results with a semantic response check and actual
feature evidence on every rank. An image's CPU inventory verification does not
replace serving acceptance.
