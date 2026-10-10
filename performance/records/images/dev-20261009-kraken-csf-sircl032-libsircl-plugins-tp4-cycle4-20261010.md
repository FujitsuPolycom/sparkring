# TP4 installs on a ring of four Sparks

Status: **research-only; `sparkring install` deployments on image `1a8c10354eb0`, measured through their API; one measured start per install; not serving-qualified**.

## Conditions

- Image `sha256:1a8c10354eb0bbaad898ebcfc1b31d5c5cfc40daab2f9b8eaaeb074524aea952`, image lock `dev-20261009-kraken-csf-sircl032-libsircl-plugins` (v3; SIRCL 0.3.2, libsircl 0.6.0, the three GLM-5.3 plugins); installs from source revision `6555b89b`.
- Eight DGX Sparks recabled as two separate rings of four (cycles of four; fabrics `097062777e17…` and `f1cb76795938…`); GPU SM clocks locked to 2,418 MHz.
- SIRCL ring sessions with NCCL off, group `cycle-4`; every install chose the default tuning table's measured `cycle-4` row, whose session takes table `runtime/common/sircl-tuning/cycle-4.json` (8 link slots of 1 MiB; [tune record](../transport/sircl-cycle4-tune-two-rings-20261009.md)).
- Each deployment measured right after its installer scenario passed, before removal, with the [serving A/B runner](../../harnesses/serving_ab/README.md)'s `phase1-16k` requests after its warm-up requests; no container was started for the measurement. Each run directory holds `tables.txt`, `summary.json`, `source.json` and the deployment lock's identifying fields (`deployment.json`) ([directory](dev-20261009-kraken-csf-sircl032-libsircl-plugins-tp4-cycle4-20261010/)).
- Compared with the same profiles on four consecutive Sparks of the ring of eight ([path-of-four record](dev-20261008-kraken-csf-sircl-libsircl-tp2-tp4-matrix-20261009.md): image `816c6d6a7e96`, SIRCL 0.3.0, the ring-schedule settings, two measured starts each).

## Result

Output tokens per second / accepted tokens per step; the path of four has no 16K cells.

**GLM-5.3-Flash CSF** (profile `glm53-flash-nvfp4-spark-tp4`, checkpoint `csf`; path of four: `mx-csf-tp4`)

| Run | Context | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---|---|---|---|---|
| ring of four | 0K | 67.5 / 2.75 | 104.3 / 2.74 | 152.4 / 2.71 | 225.9 / 2.75 |
| ring of four | 16K | 65.6 / 2.73 | 93.1 / 2.64 | 128.7 / 2.61 | 214.5 / 2.74 |
| ring of four | 32K | 64.2 / 2.74 | 94.4 / 2.66 | 126.6 / 2.69 | 218.7 / 2.73 |
| path of four | 0K | 68.3 / 2.73 | 106.5 / 2.81 | 152.4 / 2.74 | 224.3 / 2.72 |
| path of four | 32K | 64.6 / 2.77 | 97.8 / 2.74 | 127.4 / 2.68 | 220.8 / 2.74 |

Time to first token, 8K / 16K / 32K prompt: ring of four 2.60 / 5.10 / 10.48 s; path of four 2.56 / - / 10.49 s.

**DeepSeek-V4.1-Flash** (profile `deepseek-v41-flash-tp4`; path of four: `mx-dsv41-tp4`)

| Run | Context | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---|---|---|---|---|
| ring of four | 0K | 63.6 / 2.51 | 95.6 / 2.39 | 132.0 / 2.47 | 190.6 / 2.43 |
| ring of four | 16K | 61.9 / 2.54 | 92.8 / 2.48 | 130.4 / 2.46 | 181.9 / 2.47 |
| ring of four | 32K | 63.8 / 2.64 | 93.6 / 2.49 | 127.5 / 2.50 | 182.8 / 2.58 |
| path of four | 0K | 61.9 / 2.57 | 94.3 / 2.43 | 138.8 / 2.47 | 181.9 / 2.44 |
| path of four | 32K | 61.5 / 2.58 | 93.2 / 2.46 | 132.5 / 2.57 | 182.0 / 2.59 |

Time to first token, 8K / 16K / 32K prompt: ring of four 1.85 / 3.76 / 7.57 s; path of four 1.88 / - / 7.58 s.

**Qwen3.8-Flash-Next** (profile `qwen38-flash-next-qad-tp4`, default checkpoint `qad-step5500-ple1000`; path of four: `mx-qwen-tp4`)

| Run | Context | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---|---|---|---|---|
| ring of four | 0K | 81.3 / 2.50 | 120.5 / 2.58 | 201.8 / 2.40 | 292.7 / 2.47 |
| ring of four | 16K | 61.1 / 1.95 | 105.7 / 2.06 | 155.3 / 2.08 | 238.6 / 2.06 |
| ring of four | 32K | 62.1 / 2.02 | 97.1 / 1.98 | 149.4 / 2.06 | 234.0 / 2.07 |
| path of four | 0K | 91.0 / 2.50 | 143.6 / 2.48 | 218.5 / 2.49 | 310.6 / 2.47 |
| path of four | 32K | 69.1 / 2.00 | 112.9 / 2.04 | 159.7 / 2.03 | 237.2 / 2.04 |

Time to first token, 8K / 16K / 32K prompt: ring of four 1.75 / 3.41 / 7.02 s; path of four 1.73 / - / 6.84 s.

## Comparison

Ring of four against path of four, on the cells both measured (0K and 32K):

| Install | Output tokens/s, median (range) | Accepted tokens per step, largest difference |
|---|---|---|
| GLM-5.3-Flash CSF | -0.8 % (-3.6 % to +0.7 %) | 0.08 |
| DeepSeek-V4.1-Flash | +0.9 % (-4.9 % to +4.8 %) | 0.07 |
| Qwen3.8-Flash-Next | -8.9 % (-16.1 % to -1.4 %) | 0.10 |

## Conclusion

GLM-5.3-Flash CSF and DeepSeek-V4.1-Flash decode on a ring of four within a few percent of the path of four. The installed Qwen3.8-Flash-Next profile on its default checkpoint is slower than the path-of-four run of `qad-step5500-mxfp8-attention`; this pair does not separate the checkpoint from the topology.

Each comparison also changes the image and SIRCL build, the transport settings and, for GLM-5.3-Flash CSF, the serving settings (the path-of-four run used the runner's CSF overrides; the install used the profile's `csf` entry, asynchronous scheduling and 16 sequences). One measured start per install.
