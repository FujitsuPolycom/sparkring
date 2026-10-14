# GLM-5.3 NVFP4-CSF on six Sparks: one installation on release 2026.10.4's image

Status: **research-only**; one start. The model served. The installer did not
record the deployment as complete (see Result).

## Conditions

- Profile [`glm53-nvfp4-csf-tp6`](../../../profiles/glm53-nvfp4-csf-tp6/README.md)
  with its own settings: decode-context parallelism 2, a two-token MTP draft
  at TP6, 8 sequences, 4,096 batched tokens, 32 GiB of KV cache per rank, a
  135,168-token context and the `high` thinking default.
- Command: `sudo sparkring install --profile glm53-nvfp4-csf-tp6 --on 0-5 --no-auto-recover --yes`,
  2026-10-14, package `0.1.0~dev.1791978968+git7f1aecd0d69b` on every Spark.
- Installer image `dev-20261013-dcp2prefill-status038-sircl-tp6`
  (configuration ID `3cc9a81f`), the default image of release 2026.10.4.
- Checkpoint `local-inference-lab/GLM-5.3-NVFP4-CSF` @ `a4d40693`, served
  through the derived checkpoint `csf-serving` @ `bbd8f71d`.
- Positions 0-5 of a ring of eight DGX Sparks (GB10). SIRCL carried every
  collective with NCCL off and `SIRCL_ROUTE_SCOPE=fabric`. Positions 6 and 7
  kept serving another model and carried the relayed route 5-6-7-0.

## Measurement

Rank startup logs, the API, and the installer's log and deployment state.
The profile's four start checks were judged with
`runtime.host.start_checks.judge` on the inputs the installer gathers, after
the installer had stopped.

## Result

- All six ranks started and the API served requests.
- `GPU KV cache size: 1,255,552 tokens` on rank 0, with 32.0 GiB reserved per
  rank through `kv_cache_memory_bytes`.
- The four start checks (`csf-w4a16-a4-prefill`, `csf-scale-prefetch`,
  `w4a16-host-barrier-reset` and `glm53-tp6-resident-quantization`) took effect.
- Host memory after the start: 113-116 GiB used of 121 GiB per Spark.
- The installer stopped at its smoke phase. `scripts/installer_host.py`
  `start_check_names` refuses a check that `runtime.host.start_checks.CHECKS`
  does not list, and `CHECKS` does not list `glm53-tp6-resident-quantization`.
  The deployment's state was incomplete, and `sparkring status` showed no group
  for positions 0-5. Packages from commit `12192bdc` validate start checks
  against `start_checks.KNOWN`, which lists this check.

## Conclusion

The profile loads and serves on release 2026.10.4's image with a KV pool of
1,255,552 tokens. An installation completes only after `CHECKS` lists the
profile's resident-quantization check.

## Limitations

- One start, on positions 0-5 of a ring of eight beside a serving model.
- No correctness, long-context or performance measurement.
- The KV figure is the engine's capacity. It does not show that concurrent
  full-context requests fit.
- The full log, receipts and start-check inputs are in the operator's
  bring-up record of 2026-10-14, which is not in this repository.
