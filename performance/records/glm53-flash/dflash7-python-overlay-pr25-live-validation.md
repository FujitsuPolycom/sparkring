# GLM-5.3 DFlash7 Python-overlay bounded live validation

Status: **qualified** only for the startup, health, semantic smoke, and exact
SparkCache restore cases described below. Page-delta restore performance is
**research-only**. DFlash response quality is **unsupported** by this record.

## Conditions

- Four NVIDIA DGX Spark systems; TP4, DCP1, and PP1.
- Local image ID
  `sha256:9faa36a9f37aee16d97ab9214ef3153b4d200121126e6b2dee5ebb63109fea18`
  on all four ranks. Its `org.opencontainers.image.revision` label and the
  SparkRing source commit are
  `e2d92fdc7d0306d664d6fd9f296dc2adcaf0fe05`.
- vLLM native extensions from
  `da4d7be6c97434f6942292ed8abbf4b32dc44355`, vLLM Python source from
  `0b67266a0f37d6146a8403fb8482403c62f412d5`, and B12X from
  `b1d541f9e71a35f030d45fae437630fff7507c2a`.
- [SparkCache pull request 25](https://github.com/FujitsuPolycom/sparkcache/pull/25)
  commit
  `5d571018de5b63a9a90e5c11e6d6e86bbff4a957`, Git tree
  `e864ed9ad64f771188fdb59aa9738e348134d636`, and clean deployable-source
  SHA-256
  `f7c0565521fddeff7085e4cc08043cb8d1e2bde33abc67f83b8608a162d05b88`.
- GLM-5.3 NVFP4 target loaded with fastsafetensors. The external BF16 DFlash
  checkpoint loaded with safetensors through the exact draft-loader patch.
  Serving used seven speculative tokens, draft TP4, FP8 KV, 32 sequences,
  and 256-token vLLM blocks.
- SparkCache publication schema `tail-cow-v1`, resolved internally to
  `page-tail-cow-v1` for opaque GLM pages.
- The corrected launch translated the canonical CUDA restore settings to the
  names accepted by the recorded SparkCache commit. The exact mapping is
  documented in the
  [DFlash7 Python-overlay quickstart](../../../docs/GLM53_DFLASH7_PYTHON_OVERLAY_SPARKCACHE_TP4_QUICKSTART.md#launch-the-recorded-image-with-sparkcache-pull-request-25-keys).

The machine-readable identities, conditions, measurements, and scoped status
are retained in
[`validation.json`](../../receipts/glm53-flash/dflash7-python-overlay-pr25/validation.json).

## Measurement

Startup timings came from loader and readiness logs. Health gates checked all
four containers for HTTP health, restart count, and the runtime OOM-killed
flag. Two raw-completion requests checked that generation continued; they were
not scored for response quality.

Restore timings came from SparkCache logs for one flat restore, one
reconstructed page-delta restore, and one persistent 128K-class restore. The
persistent case first primed the cache, issued an unrelated sentinel request,
and then restored 813,068,464 bytes per rank. One retained-prefix wave was
observed at each of concurrency 2, 8, and 16. Each wave recorded one restore
event followed by retained-prefix followers; it does not assert one restore
per request. Wall-clock and component timing values are single observations,
so no variability estimate is available.

## Result

| Gate or measurement | Observed result |
|---|---|
| Target fastsafetensors load | 69.52 seconds |
| Draft safetensors load | 4.18 seconds |
| Total model load | 79.27 seconds |
| Warm readiness | approximately 190 seconds |
| Four-rank health | 4 healthy; restart counts `0,0,0,0`; OOM-killed flags `false,false,false,false` |
| Raw semantic smoke | completion tokens `2,2` |
| Flat restore | 11,520 tokens; rank 0; 29.258 ms |
| Reconstructed page-delta restore | 17,152 tokens; 703.826 ms total; 579.259 ms read; 67 chunks; no `TypeError` |
| 128K-class prime | 55.522 seconds; completion token 13; snapshot 2,440.7 ms; commit 1,687.7 ms |
| Unrelated sentinel | completion token 271 |
| Persistent restore | 813,068,464 bytes per rank; rank range 123.690-153.253 ms |
| Concurrency 2 retained-prefix wave | 0.808 seconds; 2 HTTP 200 responses; every completion token 13; one restore observed |
| Concurrency 8 retained-prefix wave | 1.506 seconds; 8 HTTP 200 responses; every completion token 13; one restore observed |
| Concurrency 16 retained-prefix wave | 1.781 seconds; 16 HTTP 200 responses; every completion token 13; one restore observed |

## Conclusion

The exact four-rank image completed target fastsafetensors loading and
separate DFlash safetensors loading, stayed healthy, produced continued raw
completions, and restored the exact flat, reconstructed page-delta, persistent
128K-class, and retained-prefix cases recorded above. These observations
qualify those bounded cases only. The 703.826 ms page-delta observation shows
functional reconstruction without the former `TypeError`; it does not support
a page-delta performance claim.

## Limitations

- A 7,168-token base followed by a 12,032-token request can fail page
  reconstruction because the base geometry is incompatible. The correction
  exists only in
  [draft SparkCache pull request 28](https://github.com/FujitsuPolycom/sparkcache/pull/28)
  and is absent from this image.
- Null-block publication failures at 6,912 tokens remain under investigation.
- Page-delta read and reassembly are too slow for a performance qualification.
- No DFlash quality benchmark was run on this artifact. The two-token raw
  completions prove continued generation, not answer quality.
- One wave at each concurrency does not establish latency variability,
  throughput, soak behavior, or behavior beyond 16 concurrent requests.
- The retained evidence does not include a complete command transcript,
  collection timestamp, or independent clock audit.
