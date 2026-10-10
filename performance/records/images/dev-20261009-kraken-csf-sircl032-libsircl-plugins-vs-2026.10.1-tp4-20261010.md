# Image 1a8c1035 against 2026.10.1: TP4 on a ring of four

Status: **research-only; `sparkring install` deployments measured through their API; one measured start per row; not serving-qualified**.

## Conditions

- Ring A of the two rings of four (fabric `097062777e17…`), one package (source revision `6555b89b`), GPU SM clocks locked to 2,418 MHz (2,398-2,411 MHz read at each `sparkring check`).
- 2026.10.1: `sparkring install --image 2026.10.1 --transport prepared`, image `dev-20261004-kraken-cuda1342-nccl2323-status034` (`aba309e4610c`), each profile's default checkpoint on that image. 2026.10.2: image lock `dev-20261009-kraken-csf-sircl032-libsircl-plugins` (`1a8c10354eb0`), SIRCL ring sessions with NCCL off on the measured `cycle-4` tuning row.
- Each row is a fresh installation, measured with the serving A/B runner's `phase1-16k` requests after its warm-up requests, then removed; every one passed the known-answer requests (3 of 3) and `sparkring check` and returned identical output fingerprints. Both 2026.10.1 installations planned no download (hub bytes 0). Run directories: [this record's](dev-20261009-kraken-csf-sircl032-libsircl-plugins-vs-2026.10.1-tp4-20261010/), the [second-start record's](dev-20261009-kraken-csf-sircl032-libsircl-plugins-tp4-starts-qwen-control-20261010/) and the [ring-of-four TP4 record's](dev-20261009-kraken-csf-sircl032-libsircl-plugins-tp4-cycle4-20261010/).

## Result

Output tokens per second / accepted tokens per step; the last column is the median (range) over 12 cells (0K, 16K and 32K at 1, 2, 4 and 8 streams) against the same model's 2026.10.1 row.

| Model | Release | Transport | Checkpoint | Ring | 0K, 1 stream | 0K, 8 streams | 32K, 1 stream | 32K, 8 streams | 32K time to first token | Against 2026.10.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash | 2026.10.1 | prepared (NCCL) | `nvfp4-spark` | A | 63.2 / 2.68 | 192.9 / 2.70 | 60.6 / 2.65 | 192.2 / 2.67 | 9.53 s | - |
| GLM-5.3-Flash | 2026.10.2 | SIRCL | `nvfp4-spark` | A | 65.7 / 2.76 | 202.3 / 2.70 | 60.7 / 2.63 | 198.8 / 2.73 | 9.72 s | +3.7 % (-3.1 % to +8.2 %) |
| GLM-5.3-Flash | 2026.10.2 | SIRCL | `csf` (default) | A | 67.5 / 2.75 | 225.9 / 2.75 | 64.2 / 2.74 | 218.7 / 2.73 | 10.48 s | +8.8 % (-1.0 % to +17.1 %) |
| GLM-5.3-Flash | 2026.10.2 | SIRCL | `csf` (default) | A | 66.8 / 2.71 | 215.7 / 2.69 | 61.6 / 2.67 | 213.4 / 2.72 | 10.66 s | +7.8 % (+1.1 % to +14.9 %) |
| Qwen3.8-Flash-Next | 2026.10.1 | prepared (NCCL) | `qad-step5500-ple1000` | A | 76.5 / 2.43 | 269.2 / 2.44 | 60.2 / 1.93 | 226.6 / 2.02 | 7.07 s | - |
| Qwen3.8-Flash-Next | 2026.10.2 | SIRCL | `qad-step5500-ple1000` | A | 79.0 / 2.49 | 283.4 / 2.45 | 60.9 / 1.98 | 237.9 / 2.04 | 7.13 s | +2.1 % (-2.0 % to +5.3 %) |
| Qwen3.8-Flash-Next | 2026.10.2 | SIRCL | `qad-step5500-ple1000` | B | 81.3 / 2.50 | 292.7 / 2.47 | 62.1 / 2.02 | 234.0 / 2.07 | 7.02 s | +3.3 % (-7.6 % to +8.7 %) |
| Qwen3.8-Flash-Next | 2026.10.2 | SIRCL | `qad-step5500-mxfp8-attention` | A | 85.3 / 2.47 | 297.7 / 2.46 | 67.5 / 2.05 | 235.5 / 2.01 | 7.03 s | +10.5 % (+2.8 % to +12.5 %) |

## Conclusion

On the same checkpoint, 2026.10.2 is not slower than 2026.10.1: GLM-5.3-Flash NVFP4-Spark decodes +3.7 % and Qwen3.8-Flash-Next `qad-step5500-ple1000` +2.1 % and +3.3 % in the median, and the time to the first token of a 32K prompt stays within 2 %. With one start per row and a start-to-start spread of up to 2.2 % in the median ([second-start record](dev-20261009-kraken-csf-sircl032-libsircl-plugins-tp4-starts-qwen-control-20261010.md)), the Qwen difference is at the edge of that spread. The checkpoints 2026.10.2 installs add to that: GLM-5.3-Flash on its default `csf` checkpoint decodes +7.8 % and +8.8 % against 2026.10.1's NVFP4-Spark, and takes 10.48-10.66 s instead of 9.53 s to the first token of a 32K prompt; Qwen3.8-Flash-Next on `qad-step5500-mxfp8-attention` decodes +10.5 %. Release 2026.10.2's image, `d52737a109e0`, has a byte-identical SIRCL layer, and its two GLM-5.3-Flash CSF starts measured -2.2 % in the median against this image's, within the start-to-start spread ([requalification](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-requal-20261010.md)).

`sparkring install --image 2026.10.1` reinstalled both profiles from local files on the same package and ring, so the release's documented rollback path works for these two profiles.
