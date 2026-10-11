# GLM-5.3 NVFP4-CSF on the ring of eight: one start

Lane: **public-functional**. Status: **research-only**; maturity
**live-validated** for one start. Evidence scope: **one `sparkring install`
deployment of `glm53-nvfp4-csf-tp8` on all eight Sparks of a ring of eight
with installer image `d52737a109e0` and the lock of package `46eeb18b`, which
lists the profile: its start checks, rank 0's weight memory, and one measured start
through its API with GPU clocks not locked; not serving-qualified.** The
owner has parked the profile: no published lock admits it, and its next
measurement waits for the owner.

## Conditions

- **Profile and checkpoint.** `glm53-nvfp4-csf-tp8`, every setting of
  `glm53-nvfp4-tp8` (TP8, DCP4, MTP with two tokens, MXFP8 dense linears on
  the B12X linear backend, W4A16 experts, the three GLM-5.3 vLLM plugins,
  SIRCL ring sessions with NCCL off) serving
  `local-inference-lab/GLM-5.3-NVFP4-CSF` at revision `a4d406932f2d` through
  its derived serving directory, with `--quantization nvfp4_csf --load-format
  nvfp4_csf` and no `--hf-overrides`
  ([profile guide](../../../profiles/glm53-nvfp4-csf-tp8/README.md)). Served
  model name `GLM-5.3-NVFP4-CSF-TP8`, port 8015.
- **Reasoning effort.** The deployment ran with `--reasoning-effort high`.
  The reference start below ran with the profile's default, `max`.
- **Image.** Installer image
  `sha256:d52737a109e083d3eef05c0fc0db4d09fc1bf34485a13467382130963ecfe66b`,
  release 2026.10.2's, with the lock that package `46eeb18b` carries for it,
  which adds this profile to the 13 that release 2026.10.2's lock lists.
- **Package.** SparkRing revision `46eeb18b` (branch
  `claude/ring-test-20261010`).
- **Hardware.** Eight DGX Sparks (GB10), one ConnectX-7 RoCE ring of eight,
  rank `r` on position `r`. GPU clocks not locked.
- **Harness.** The [serving A/B runner](../../harnesses/serving_ab/README.md)'s
  `phase1-16k` metrics through the installed API, 2026-10-11 02:29 UTC; one
  measured start; the runner started nothing.
- **Reference.** `glm53-nvfp4-tp8` on the same image and ring, one measured
  start at 2026-10-10 12:55 UTC with the same metrics and GPU clocks not
  locked, from the
  [requalification record](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-requal-20261010.md#glm-53-tp8-on-the-ring-of-eight).

## Measurement

- **Start checks:** the profile's start-check document, judged from the rank
  containers' logs, as the installer of package `46eeb18b` writes it; the
  column-split check judged again from rank 0's saved log excerpt with the
  rule of commit `3e991b93`
  (`runtime/host/fixtures/start_checks/glm53-nvfp4-csf-tp8-rank0.log`).
- **Weight memory:** vLLM's `Model loading took ... GiB` line in rank 0's log.
- **Decode:** aggregate output tok/s, engine steps/s and MTP accept length;
  temperature 0; 30 s cells of at most 1,024 output tokens; contexts 0, 16K
  and 32K; 1, 2, 4 and 8 streams; one observation per cell.
- **Time to first token:** cold, median of the valid samples.
- **Raw files:** [tables.txt, summary.json and source.json](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-glm53-nvfp4-csf-tp8-ring8-20261010/).

## Result

### Start checks and weights

| Check | Verdict, package `46eeb18b` | Verdict, rule of `3e991b93` | Evidence |
|---|---|---|---|
| Column-split loading (`glm53full_speedups`) | did not take effect | took effect | rank 0 logged both `patched for` lines (`GLM53FULL_LATENT_SHARD`, `GLM53FULL_EH_PROJ_TP`), `fused_qkv_a_proj is column-parallel over TP8: rank 0 computes fused rows [0, 328) of 2624` and `eh_proj is RowParallelLinear(12288 -> 6144) over TP8`, and no `PatchRefused` or replicated fallback. The older rule matched `glm_dcp_decode_comm`'s own `nothing patched` line, which that plugin prints when its items are off |
| Draft quantization | took effect | — | rank 0's weights fit quantized draft experts |

The profile's other start checks are not part of this record.

| Rank | Position | Weights, GiB |
|---|---|---|
| 0 | 0 | 49.63 |
| 1-7 | 1-7 | not read |

Rank 0's 49.63 GiB is the standard profile's 51.69 GiB with the same splits
(16 starts of `glm53-nvfp4-tp8`, runner logs, not a repository record) less
2.06 GiB that the compressed expert scales free; without the splits a rank
would hold about 1.2 GiB more.

### Decode

Aggregate output tok/s / engine steps/s / accept length:

| Context | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---|---|---|---|
| 0K | 44.9 / 18.76 / 2.40 | 62.8 / 27.55 / 2.28 | 86.4 / 37.66 / 2.30 | 130.1 / 53.49 / 2.43 |
| 16K | 36.4 / 18.12 / 2.01 | 55.4 / 25.45 / 2.18 | 82.2 / 38.32 / 2.14 | 116.9 / 52.98 / 2.21 |
| 32K | 39.2 / 17.72 / 2.21 | 60.4 / 27.23 / 2.22 | 85.9 / 37.57 / 2.29 | 123.1 / 54.07 / 2.28 |

Time to first token, cold: 8.79 s at 8K and 12.82 s at 16K. Every 32K request
and the third repeat at each length returned no first-token event or usage,
so no 32K value is reported.

Against the reference start, tok/s / steps/s / accept length:

| Context | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---|---|---|---|
| 0K | -8.4 % / -3.0 % / -0.14 | -12.2 % / -4.0 % / -0.21 | -15.5 % / -9.1 % / -0.17 | -9.0 % / -8.1 % / -0.03 |
| 16K | -17.6 % / -3.1 % / -0.36 | -10.9 % / -4.8 % / -0.15 | -8.3 % / -1.2 % / -0.17 | -9.9 % / -6.1 % / -0.09 |
| 32K | -12.5 % / -1.9 % / -0.27 | -8.2 % / +1.0 % / -0.22 | -10.1 % / -3.6 % / -0.16 | -11.7 % / -5.1 % / -0.17 |

Medians: -10.5 % tok/s, -3.8 % engine steps/s, -0.17 accept length.

### Outputs

The fingerprint prompts gave `391` for 17 × 23, counted to 30 and wrote an
`is_prime` function, in a terser style than the reference start's (the count
reply opens `Simple task.`, where `glm53-nvfp4-tp8`'s open `The user wants`).
The prompt log-probabilities of a fixed 1,071-token text were bit-identical to
two `glm53-nvfp4-tp8` starts on image `af06e272` that loaded the same split
through `--load-format b12x` (runner logs, not a repository record).

## Conclusion

The profile installs and serves on eight Sparks with release 2026.10.2's
image, and `glm53full_speedups`' column-split loading takes effect through the
nvfp4_csf loader; the start check of package `46eeb18b` misjudged it, and the
rule of `3e991b93` judges it correctly. The comparison with
`glm53-nvfp4-tp8` is confounded: this start ran with `--reasoning-effort high`
and the reference with the default `max`, which changes the generated text and
with it how often the draft is accepted. The accept-length and tok/s
differences above therefore do not measure the checkpoint. The engine-step
difference (median -3.8 %, larger at 8 streams) fits the CSF experts' scale
reads in CUDA-graph decode, with one start per profile.

## Limitations

- One start per profile, GPU clocks not locked, and different reasoning
  efforts; a comparison needs both profiles at the same effort and at least
  two starts each.
- Ranks 1-7's weights and the profile's other start-check verdicts were not
  read.
- No 32K time to first token, no correctness screen, no soak.
- The deployment's lock is the test package's; no published lock admits the
  profile.
