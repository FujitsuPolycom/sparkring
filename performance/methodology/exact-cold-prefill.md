# Exact-token cold-prefill measurement

The [portable HTTP probe](../harnesses/validation/exact_cold_prefill.py) measures
client-observed first-token latency for Qwen chat requests. It reproduces the
request and timing method used by the private `qwen_four_profile_checks.py`
prefill arm and its `model_cache_probe.py` fixture builder. Historical records
retain their actual executed harness identities; this portable extraction was
not the script executed for those measurements.

Install `httpx`, then run against an otherwise idle, already warmed server:

```bash
python performance/harnesses/validation/exact_cold_prefill.py \
  --base-url http://SERVER:PORT --model SERVED_MODEL_NAME \
  --output cold-prefill.json --run
```

The URL is the server root, without `/v1`. `OPENAI_API_KEY` supplies optional
authentication and is not written into the output. The output must not exist.
The probe performs no SSH, deployment, restart, cache deletion or image changes.

## Conditions and metric

- Nine serial inference requests: 8,192, 65,536 and 131,072 prompt tokens,
  repeated in that order three times; concurrency one.
- Every fixture starts with a fresh random nonce before the verification key
  and repeated archive filler. `/tokenize` must return exact nonnegative integer
  token IDs with `add_generation_prompt=true`. Up to 16 adjustment attempts
  target the exact count; approximation is not accepted.
- Qwen thinking is disabled, temperature is 0 and seed is 779386. Fixture
  preparation uses a 128-token output budget; the measured request overrides
  that to **one streamed token** and requests final usage.
- The timer starts immediately before the streamed completion request and ends
  at the first nonempty `content` or `reasoning_content` delta. Throughput is
  exact prompt tokens divided by that client-observed latency. Network/API
  overhead and initial decode are included; this is not GPU-only prefill time.
- A successful sample requires the exact prompt count, a positive completion
  count and explicit **zero `cached_tokens`**. Missing cache credit is not zero.
  Request `cache_salt` is deliberately not used to assert coldness.

The output retains synthetic requests, tokenization attempts and raw responses,
token/request hashes, raw stream lines, parsed events, usage and timing. Partial
evidence is saved on HTTP, parsing or validation failure. Failed samples must
not be averaged into successful throughput. Synthetic outputs can be several
megabytes because raw tokenizer responses are retained.

This is a bounded throughput probe, not cache-restore, response-correctness,
full-context, media or concurrency qualification. The one-token response is not
checked for the full verification key. Record model revision, image digest,
TP/DCP, KV allocation, SparkCache configuration, warmup and stable worker
identities separately when publishing a comparison. This standalone program
does not inspect worker lifetimes or prove the server is idle.

Run GPU-free HTTP-mock checks with:

```bash
python -m pytest performance/harnesses/validation/test_exact_cold_prefill.py -q
```
