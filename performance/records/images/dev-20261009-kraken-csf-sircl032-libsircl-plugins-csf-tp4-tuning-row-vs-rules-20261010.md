# SIRCL's measured cycle-4 table against its own rules: GLM-5.3-Flash CSF TP4

Status: **research-only; serving A/B runner, one warm-up and one measured start per arm; not serving-qualified**.

## Conditions

- Ring A of the two rings of four (fabric `097062777e17…`), image lock `dev-20261009-kraken-csf-sircl032-libsircl-plugins` (`1a8c10354eb0`, SIRCL 0.3.2), profile `glm53-flash-nvfp4-spark-tp4` on its `csf` checkpoint with the runner's settings, SIRCL ring sessions with NCCL off, the runner's site on each Spark's LAN address; GPU SM clocks locked to 2,418 MHz.
- Arm "rules" passes no SIRCL tuning table, which is what an installation takes from the default table's `cycle` row (source `rules`). Arm "measured" passes `runtime/common/sircl-tuning/cycle-4.json` (sha256 `3b8f02ba…`, table `df333d97acb1a4b2`: 8 link slots of 1 MiB and its measured algorithm choices), which the default table's measured `cycle-4` row gives an installation. The rank environments differ only in `SIRCL_TUNING_TABLE` (`plan-facts.json` in each run directory).
- `phase1-16k` requests; each arm's arm checks passed and its 4 output fingerprints were identical. Run directories: [dev-20261009-kraken-csf-sircl032-libsircl-plugins-csf-tp4-tuning-row-vs-rules-20261010](dev-20261009-kraken-csf-sircl032-libsircl-plugins-csf-tp4-tuning-row-vs-rules-20261010/).

## Result

Output tokens per second / engine steps per second / accepted tokens per step, and time to first token.

| Arm | 0K, 1 stream | 0K, 8 streams | 32K, 1 stream | 32K, 8 streams | First token, 8K / 16K / 32K | Ready |
|---|---|---|---|---|---|---|
| rules | 65.2 / 24.1 / 2.71 | 209.6 / 77.6 / 2.70 | 61.4 / 22.7 / 2.71 | 209.8 / 76.5 / 2.74 | 2.88 / 5.74 / 11.66 s | 436 s |
| measured cycle-4 table | 65.2 / 24.0 / 2.72 | 221.9 / 81.5 / 2.72 | 62.2 / 22.4 / 2.77 | 214.3 / 78.9 / 2.72 | 2.61 / 5.31 / 10.66 s | 409 s |

Measured against rules over 12 cells (0K, 16K and 32K at 1, 2, 4 and 8 streams): engine steps per second -0.5 % in the median (-2.8 % to +5.0 %), output tokens per second +1.7 % (-0.9 % to +7.4 %), accepted tokens per step within 0.27; time to first token -9.3 % / -7.6 % / -8.6 % at 8K / 16K / 32K.

## Conclusion

The measured cycle-4 table shortens prefill: the first token of an 8K-32K prompt comes 7.6-9.3 % sooner than under SIRCL's own rules. Decode is unchanged within the start-to-start spread of about 2 % in the median ([second-start record](dev-20261009-kraken-csf-sircl032-libsircl-plugins-tp4-starts-qwen-control-20261010.md)): engine steps per second move -0.5 %, and the +1.7 % in output tokens per second follows the accepted length, which differs by cell (by up to 0.27); a different all-reduce algorithm can change the summation order and with it the greedy draft acceptance. One start per arm.
