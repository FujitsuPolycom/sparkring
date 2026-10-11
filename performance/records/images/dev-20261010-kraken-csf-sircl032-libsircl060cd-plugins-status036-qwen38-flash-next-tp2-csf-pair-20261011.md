# Qwen3.8-Flash-Next CSF checkpoint on two Sparks of the ring of eight

Lane: **public-functional**. Status: **research-only**; maturity
**live-validated** for one start. Evidence scope: **one `sparkring install`
deployment of `qwen38-flash-next-tp2 --checkpoint csf` on positions 6 and 7 of
an eight-Spark ring with installer image `d52737a109e0`, measured once through
its API with GPU clocks not locked; not serving-qualified.**

## Conditions

- **Profile and checkpoint.** `qwen38-flash-next-tp2 --checkpoint csf`:
  `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` at revision
  `dc5b8ab2fa14`, loaded with `--quantization nvfp4_csf --load-format
  nvfp4_csf`; every other setting is the profile's (MTP, three tokens, a
  24 GiB KV cache per rank, 262,144-token context). Served model name
  `Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD-TP2`, port 8000.
- **Image.** Installer image
  `sha256:d52737a109e083d3eef05c0fc0db4d09fc1bf34485a13467382130963ecfe66b`,
  release 2026.10.2's
  (`dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`), whose
  vLLM is the pinned build that reads the nvfp4_csf format.
- **Package.** SparkRing revision `46eeb18b` (branch
  `claude/ring-test-20261010`), which carries this checkpoint entry; installed
  2026-10-11 03:20 UTC.
- **Hardware.** Eight DGX Sparks (GB10) cabled as one ConnectX-7 RoCE ring of
  eight. The deployment ran on positions 6 and 7, a cabled pair, on SIRCL ring
  sessions with NCCL absent. GPU clocks were not locked. Other deployments
  served on positions 4 and 5, and an installation and its tests ran on
  positions 0 to 3 during the measurement
  ([SparkCache record](dev-20261010-sparkcache-gab86b70-plugins-status036-glm53-flash-nvfp4-sparkcache-tp4-path4-20261011.md)).
- **Harness.** The [serving A/B runner](../../harnesses/serving_ab/README.md)'s
  `phase1-16k` metrics through the installed API, from a workstation on the
  owner's network, 2026-10-11 05:00-05:15 UTC; one warm-up start, then one
  measured start; the runner started nothing.

## Measurement

- **Decode:** aggregate output tok/s, engine steps/s and MTP accept length per
  cell; temperature 0; 30 s cells of at most 1,024 output tokens; contexts 0,
  16K and 32K; 1, 2, 4 and 8 streams; one measured start, so each cell is one
  observation.
- **Time to first token:** cold (`cached_tokens` 0), the median of the
  samples at 8K, 16K and 32K prompts.
- **Outputs:** the runner's four temperature-0 fingerprint prompts.
- **Raw files:** [tables.txt, summary.json and source.json](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-qwen38-flash-next-tp2-csf-pair-20261011/).

## Result

Decode aggregate output tok/s / engine steps/s / accept length:

| Context | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---|---|---|---|
| 0K | 62.2 / 24.68 / 2.52 | 101.6 / 40.37 / 2.52 | 152.7 / 61.11 / 2.50 | 221.1 / 90.27 / 2.45 |
| 16K | 47.6 / 23.65 / 2.01 | 73.1 / 35.51 / 2.06 | 119.3 / 57.80 / 2.06 | 168.9 / 84.24 / 2.00 |
| 32K | 48.3 / 23.48 / 2.06 | 68.3 / 35.70 / 1.91 | 119.6 / 56.82 / 2.11 | 171.5 / 82.84 / 2.07 |

Time to first token, cold: 2.31 s at 8K, 4.39 s at 16K and 9.07 s at 32K
(3,614 prompt tok/s at 32K).

The fingerprint prompts answered correctly: `391` for 17 × 23, counting from
1, and an `is_prime` function.

Against the profile's other two checkpoints, from other records, images and
starts (indicative only):

| Checkpoint | 32K decode tok/s, 1 / 4 / 8 streams | 32K prefill tok/s | Source |
|---|---|---|---|
| `csf` | 48.3 / 119.6 / 171.5 | 3,614 | this record |
| default, `qad-step5500-ple1000` | 42.9 / 109.9 / 162.9 | 3,649 | [ring of four checks](dev-20261009-kraken-csf-sircl032-libsircl-plugins-ring-b-checks-20261010.md), image `1a8c10354eb0`, one start on a ring of four |
| `qad-step5500-mxfp8-attention` | 51.1 / 117.1 / 179.0 | 3,917 | [TP2/TP4 matrix](dev-20261008-kraken-csf-sircl-libsircl-tp2-tp4-matrix-20261009.md) (`mx-qwen-tp2`), image `816c6d6a7e96`, mean of two starts on a pair of the ring of eight |

## Conclusion

The CSF checkpoint installs, loads and serves on two Sparks with release
2026.10.2's image. In this one start it decoded 5.3-12.6 % faster at 32K
than the default checkpoint's recorded start, with prefill 1.0 % slower, and
within -5.5 % to +2.1 % of the MXFP8-attention checkpoint's two starts, with
prefill 7.7 % slower.

## Limitations

- One measured start, GPU clocks not locked, and other workloads on the ring
  during the window; the comparisons cross images, rings and harness runs.
- The installation's own functional checks and `sparkring check` output are
  not part of this record.
- No paired A/B on one pair with two starts per checkpoint, which a change of
  the profile's default checkpoint needs.
- No correctness screen, soak or long-context run beyond 32K.
