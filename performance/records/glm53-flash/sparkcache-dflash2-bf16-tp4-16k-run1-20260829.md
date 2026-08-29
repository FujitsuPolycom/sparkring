# GLM-5.3 Flash SparkCache 16K throughput observation

Status: **research-only**. This record reports one observation of the
SparkCache-enabled GLM-5.3 Flash service. It has no repetitions, uncertainty
estimate, or cache-disabled A/B baseline.

## Conditions

- Four NVIDIA DGX Spark systems formed a direct RoCE cycle with TP4, DCP1, and
  PP1.
- Every rank ran
  `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943`
  with image ID
  `sha256:7c007cf673c35f5818da7fea8faa343304baed00f489efdcbd027d6616b8a290`.
- The target was
  `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`.
  The BF16 seven-token draft was
  `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`.
- The runtime used vLLM commit
  `da4d7be6c97434f6942292ed8abbf4b32dc44355` and B12X commit
  `2fcf23a0ce269be27b2e03fece73d46e90e6aeea`.
- SparkCache was enabled with a 48 GiB maximum and 40 GiB low watermark per
  rank. The requests used a 16,384-token context, 512 maximum output tokens,
  temperature 1.0, fully unique contexts, and a declared 549,950-token KV
  budget.
- `llm_decode_bench.py` reported version `0.4.31`. The measured working file
  has SHA-256
  `07aad353cd9c894e14e9d1392c8509d3af8999c4022d3d22b29423a4572f5851`.

## Measurement

The prefill value is one integrated decode-scout observation: prompt tokens
divided by client time to first token. The C1 decode value is completion-token
throughput from OpenAI continuous usage over a 29.953-second covered
measurement window. The client counted 1,080 output tokens, and the server
counter agreed exactly.

The sanitized command was:

```powershell
python llm_decode_bench.py --host http://<rank-0-address> --port 8015 --model glm-5.3-flash-nvfp4-dflash7-bf16-tp4 --concurrency 1,4,8 --contexts 16k --prefill-contexts 16k --prefill-duration 10 --max-tokens 512 --duration 30 --decode-warmup-seconds 3 --cell-warmup-timeout-seconds 180 --temperature 1.0 --token-targeting exact --display-mode plain --no-hw-monitor --isolated-server --unique-context-percent 100 --chat-template-kwargs '{"enable_thinking":false}' --kv-budget 549950 --output ./benchmark-16k-run1.json
```

The command records the argument that requested disabled thinking. The retained
benchmark receipt did not serialize that argument, so this record does not
claim that the server applied it.

## Result

| Cell | Result | Disposition |
|---|---:|---|
| 16K integrated-scout prefill | 2,371 tok/s | valid single observation |
| 16K sustained C1 decode | 36.05648849867254 tok/s | valid single observation |
| 16K sustained C4 decode | — | invalid/excluded: capacity-limited |
| 16K sustained C8 decode | — | invalid/excluded: capacity-limited; readiness warmup reached its timeout |

The excluded cells have no published throughput value and do not contribute to
the README summary.

## Conclusion

Under the conditions above, the SparkCache-enabled service produced the two
valid throughput observations shown in the table. With no A/B baseline, these
values do not measure a SparkCache speedup or slowdown.

## Limitations

- This is a single observation. It does not provide run-to-run variability or
  an uncertainty estimate.
- C4 and C8 were capacity-limited, so their throughput is invalid and excluded.
- The run did not include a cache-disabled baseline and cannot isolate the
  effect of SparkCache.
- The exact benchmark source snapshot is not published. The working-file hash
  binds the measured script, but reproduction is therefore limited.
- The result applies only to the image, model revisions, topology, request
  shape, and settings recorded here.

## Provenance

The [sanitized machine-readable receipt](../../receipts/glm53-flash/sparkcache-dflash2-bf16-tp4-20260829/benchmark-16k-run1.json)
was derived from a retained input with SHA-256
`0b5ef2215eeb3dac942fea715b3c9336fbce1fac6d3e3581ac6d77eab46fca92`.
The retained input is not published because it contains private site
identifiers. Runtime and model identities are independently recorded in
[`runtime/glm53-flash/pins.json`](../../../runtime/glm53-flash/pins.json).
