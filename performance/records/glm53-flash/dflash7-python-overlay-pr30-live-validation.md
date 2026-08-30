# GLM-5.3 DFlash7 with SparkCache bounded validation

Status: **qualified** only for the startup, health, semantic generation, and
exact SparkCache restore cases described below. The 262,144-token restore
latency is **research-only**. Shared-prefix concurrency and DFlash response
quality are **unsupported** by this record.

## Conditions

- Four NVIDIA DGX Spark systems at TP4/DCP1/PP1.
- Local image ID
  `sha256:eef863d8bc578815a80b0e2d9f0d745102b6363415225101fd92171a2e5a55cb`
  on every rank. The image has no published OCI digest.
- SparkRing image revision
  `b7a7265c62c1df05b17d1221d8fd0c97c54240d1`; retained vLLM native commit
  `da4d7be6c97434f6942292ed8abbf4b32dc44355`; vLLM Python commit
  `0b67266a0f37d6146a8403fb8482403c62f412d5`; B12X commit
  `b1d541f9e71a35f030d45fae437630fff7507c2a`.
- SparkCache commit `5ec6a9953ad5d39120298bbfc26e95a6fa4b1dc3`, Git tree
  `94c236b9dfbf5f70075eb47877fd9caaa5d8c249`, and deployable-source SHA-256
  `bc238f96e550c7ec27d4081dd1f2e741d404aaf5c8572d89ccc5e76812be4d63`.
- GLM-5.3 NVFP4 target loaded with fastsafetensors queue size one. The external
  BF16 DFlash checkpoint loaded with safetensors. Serving used seven draft
  tokens, draft TP4, FP8 KV, 32 sequences, and 256-token vLLM blocks.
- SparkCache used `tail-cow-v1`, resolved to `page-tail-cow-v1` for opaque GLM
  pages, with the canonical CUDA configuration keys, eight storage-read
  workers, and two request-placement lanes.

The machine-readable artifact identities, observations, and limitations are
in
[`validation.json`](../../receipts/glm53-flash/dflash7-python-overlay-pr30/validation.json).

## Measurement

Startup values came from the target-loader, draft-loader, model-runner, and API
readiness logs. Container inspection checked the same complete image ID on all
four ranks, running state, restart count, and the runtime OOM flag.

Restore values came from each rank's SparkCache completion log. The client
observed the same completion token after the 131,072-token prime and replay,
and the same completion token after the 262,144-token prime and committed
replay. The first 262,144-token replay began before all-rank manifest
visibility and recomputed the unavailable tail; it is not counted as a cache
restore. Each reported shape has one retained observation and no variability
estimate.

## Result

| Gate or measurement | Observed result |
|---|---|
| Target fastsafetensors load | 53.93 seconds |
| Draft safetensors load | 4.85 seconds |
| Total model load | 64.375 seconds |
| API readiness | approximately 146 seconds after container start |
| Four-rank health | four running containers; restart counts `0,0,0,0`; OOM flags `false,false,false,false` |
| One-shot cache reset | scheduler and worker reported that the configured operation token had already completed; no repeated removal occurred |
| Arbitrary page boundary | 7,168-token base followed by verified 12,032-token restore; rank-zero total 487.657 ms; 47 objects |
| Persistent 131,072-token restore | prime and replay completion token 13; rank range 123.6-151.4 ms |
| Persistent 262,144-token restore | prime and replay completion token 916; 7.835-second request; rank range 5,842.1-7,217.9 ms |
| Rank-zero 262,144-token phases | 1,024 objects; 4,437.355 ms read; 961.720 ms CUDA submission; 313.987 ms CUDA synchronization |

## Conclusion

The exact four-rank image loaded the target and external draft, remained
healthy, continued generation, and restored the authenticated arbitrary-page,
131,072-token, and 262,144-token cases without changing their observed
completion tokens. Those bounded cases are qualified for correctness. The
large restore is too slow to support a performance claim.

## Limitations

- This SparkCache revision has no retained C2, C8, or C16 shared-prefix wave on
  the exact image. Concurrency observations from another image do not qualify
  this artifact.
- The 262,144-token result is one prime and one committed replay.
- The 1,024-object physical layout dominates large page-delta restore. A
  macro-object layout requires separate implementation and qualification.
- No scored DFlash response-quality benchmark was run.
- Sanitized raw HTTP bodies and a complete command transcript were not
  retained with this record.
- The image exists only on the observed hosts and cannot be pulled by digest.
- `--prefill-schedule-interval 8` was not part of the recorded profile and
  requires an independent research-only mixed-traffic test.
