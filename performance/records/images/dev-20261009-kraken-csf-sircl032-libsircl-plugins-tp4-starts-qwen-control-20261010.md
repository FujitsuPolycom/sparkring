# TP4 second starts and the Qwen checkpoint control on a ring of four

Status: **research-only; `sparkring install` deployments measured through their API; not serving-qualified**.

## Conditions

- Image `sha256:1a8c10354eb0bbaad898ebcfc1b31d5c5cfc40daab2f9b8eaaeb074524aea952`, image lock `dev-20261009-kraken-csf-sircl032-libsircl-plugins`; installs from source revision `6555b89b`; SIRCL ring sessions with NCCL off on the default tuning table's measured `cycle-4` row (`runtime/common/sircl-tuning/cycle-4.json`).
- Ring A (fabric `097062777e17…`) of the two rings of four; its first CSF start and ring B's first Qwen start are the [ring-of-four TP4 record](dev-20261009-kraken-csf-sircl032-libsircl-plugins-tp4-cycle4-20261010.md)'s. GPU SM clocks locked to 2,418 MHz (2,398-2,411 MHz read after each install and measurement, no active clock-event reasons).
- Every start is a fresh installation, measured with the serving A/B runner's `phase1-16k` requests after its warm-up requests, then removed. Each passed the known-answer requests (3 of 3) and `sparkring check`, and returned identical output fingerprints for its 4 requests. Each run directory holds `tables.txt`, `summary.json`, `source.json` and the deployment lock's identifying fields, including the selected checkpoint (`deployment.json`) ([directory](dev-20261009-kraken-csf-sircl032-libsircl-plugins-tp4-starts-qwen-control-20261010/)).
- The path of four is four consecutive Sparks of the ring of eight ([record](dev-20261008-kraken-csf-sircl-libsircl-tp2-tp4-matrix-20261009.md): image `816c6d6a7e96`, SIRCL 0.3.0, the ring-schedule settings, two measured starts each).

## Result

Output tokens per second / accepted tokens per step, and the median over the 8 cells both setups measured (0K and 32K at 1, 2, 4 and 8 streams) against the path of four on the same checkpoint.

| Install | Checkpoint | Ring | Start | 0K, 1 stream | 0K, 8 streams | 32K, 1 stream | 32K, 8 streams | Against the path of four |
|---|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash CSF | `csf` | A | 1 | 67.5 / 2.75 | 225.9 / 2.75 | 64.2 / 2.74 | 218.7 / 2.73 | -0.8 % (-3.6 % to +0.7 %) |
| GLM-5.3-Flash CSF | `csf` | A | 2 | 66.8 / 2.71 | 215.7 / 2.69 | 61.6 / 2.67 | 213.4 / 2.72 | -3.6 % (-6.1 % to +1.6 %) |
| GLM-5.3-Flash CSF | `csf` | path of four | - | 68.3 / 2.73 | 224.3 / 2.72 | 64.6 / 2.77 | 220.8 / 2.74 | - |
| Qwen3.8-Flash-Next | `qad-step5500-ple1000` | B | 1 | 81.3 / 2.50 | 292.7 / 2.47 | 62.1 / 2.02 | 234.0 / 2.07 | -8.9 % (-16.1 % to -1.4 %) |
| Qwen3.8-Flash-Next | `qad-step5500-ple1000` | A | 2 | 79.0 / 2.49 | 283.4 / 2.45 | 60.9 / 1.98 | 237.9 / 2.04 | -9.2 % (-13.2 % to +0.3 %) |
| Qwen3.8-Flash-Next | `qad-step5500-mxfp8-attention` | A | 1 | 88.6 / 2.56 | 302.6 / 2.47 | 64.2 / 1.95 | 232.9 / 2.02 | -2.6 % (-7.1 % to -1.7 %) |
| Qwen3.8-Flash-Next | `qad-step5500-mxfp8-attention` | A | 2 | 85.3 / 2.47 | 297.7 / 2.46 | 67.5 / 2.05 | 235.5 / 2.01 | -2.3 % (-6.2 % to +0.2 %) |
| Qwen3.8-Flash-Next | `qad-step5500-mxfp8-attention` | path of four | - | 91.0 / 2.50 | 310.6 / 2.47 | 69.1 / 2.00 | 237.2 / 2.04 | - |

## Conclusion

Start to start, a fresh installation repeats within -2.2 % (CSF) and +0.7 % (Qwen, `qad-step5500-mxfp8-attention`) in the median over 12 cells, with single cells from -6.1 % to +5.0 %; the default Qwen checkpoint on ring A is -0.5 % against ring B.

The Qwen3.8-Flash-Next gap against the path of four is the checkpoint: on the same ring, `qad-step5500-mxfp8-attention` decodes +6.3 % and +7.8 % in the median against the default `qad-step5500-ple1000`, and on the path of four's checkpoint the ring of four is -2.6 % and -2.3 %, within 3 % of it. On the path of four's checkpoints, every start of both profiles is 0.8-3.6 % below the path of four in the median; that comparison also changes the image, the SIRCL build and the transport settings, so this residual is not attributed to the topology alone.
