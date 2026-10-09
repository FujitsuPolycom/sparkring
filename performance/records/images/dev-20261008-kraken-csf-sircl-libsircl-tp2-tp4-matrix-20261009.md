# Two- and four-Spark serving configurations on SIRCL with ring schedules

Status: **research-only; serving A/B runner measurements of one configuration per campaign on image `816c6d6a7e96`; two measured starts per campaign; not an installer run; not serving-qualified**.

Each campaign served one installer profile, with the deviations listed below, through the
[serving A/B runner](../../harnesses/serving_ab/README.md) on SIRCL ring sessions with NCCL off. Every
campaign set the same SIRCL session settings, called the ring-schedule settings below:

| Variable | Value |
|---|---|
| `SIRCL_LARGE_SCHEDULE`, `SIRCL_GATHER_SCHEDULE`, `SIRCL_SCATTER_SCHEDULE` | `ring` |
| `SIRCL_ONESHOT_MAX_BYTES` | `65536` |
| `SIRCL_LINK_SLOT_BYTES` | `1048576` |
| `SIRCL_GATHER_LINK_CHUNK_BYTES`, `SIRCL_REDUCE_LINK_CHUNK_BYTES`, `SIRCL_SCATTER_LINK_CHUNK_BYTES` | `1048576` |

These are SIRCL session settings: an installer deployment takes them only from SparkRing's SIRCL tuning
table ([runtime/common/sircl-tuning-defaults.json](../../../runtime/common/sircl-tuning-defaults.json)),
never from a profile. The other settings each campaign adds are folded into its profile as listed under
Conclusion.

## Conditions

- **Image:** `sha256:816c6d6a7e96f226475c2873cd180ec0d95d1ecb79e8322451816d2cc63d6747`, image lock
  `dev-20261008-kraken-csf-sircl-libsircl-cu1342-nccl2323-status034` (v3; SIRCL 0.3.0 with ABI 9; SIRCL's
  pinned vLLM build `sparkring-kraken-beta-20261007-bc9ea774`). The lock is not in the repository; each
  campaign's `plan.json` records its name and SHA-256. Image `af06e272` adds the GLM-5.3 plugin layer on
  top of this image; none of these profiles loads those plugins.
- **Hardware:** eight DGX Sparks (GB10) cabled as a ring of eight with ConnectX-7 RoCE. A two-Spark
  campaign ran on two cabled neighbours (a pair group); a four-Spark campaign ran on four consecutive
  Sparks (a path of four, whose end ranks reach each other through relays). Positions are in each
  campaign's `plan.json`.
- **Host:** GPU SM clocks locked to 2,418 MHz on every Spark (`nvidia-smi -lgc 2418,2418`), which the
  installer does not set. Before every start the runner wrote back dirty pages, dropped the page cache and
  waited for stable available memory.
- **Transport:** arm `S` (SIRCL ring sessions on every collective, NCCL off) or arm `S+` (`S` with
  `SIRCL_FUSED_NORM=1`); no SIRCL tuning table; the ring-schedule settings above at the container level.
- **Starts:** one warm-up start (`W-…`, not measured into the tables) and two measured starts (`…1`,
  `…2`) per campaign.

| Campaign | Profile | Arm | Sparks | Deviations from the profile (besides the ring-schedule settings) |
|---|---|---|---|---|
| `mx-csf-tp2-a` | `glm53-flash-nvfp4-spark-tp2` | S | pair | the CSF checkpoint `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd33ef` with `--quantization nvfp4_csf`, `--load-format nvfp4_csf`, the draft's experts on `marlin`, `VLLM_B12X_MOE_FP4_FORCE_A16=1`, `B12X_W4A16_SMALL_M_OCCUPANCY=2` |
| `mx-csf-tp2-b` | `glm53-flash-nvfp4-spark-tp2` | S | pair | as `mx-csf-tp2-a`, plus `--async-scheduling`, `VLLM_B12X_KDA_PREFILL_COALESCING=1` and the four-Spark profile's draft settings (`draft_tensor_parallel_size` 2, probabilistic draft sampling, standard rejection sampling) |
| `mx-csf-tp4` | `glm53-flash-nvfp4-spark-tp4` | S | path of four | the CSF checkpoint with the same quantization, loader, draft experts on `marlin`, `VLLM_B12X_MOE_FP4_FORCE_A16=1` and `B12X_W4A16_SMALL_M_OCCUPANCY=2` |
| `mx-qwen-tp2` | `qwen38-flash-next-tp2` | S | pair | checkpoint `qad-step5500-mxfp8-attention` (the installer-derived `sparkring-derived/Qwen3.8-Flash-Next-NVFP4-QAD5500-MXFP8-Attention@648b194a96e5`) |
| `mx-qwen-tp2-s4000` | `qwen38-flash-next-tp2` | S | another pair | checkpoint `qad-step-4000` |
| `mx-qwen-tp4` | `qwen38-flash-next-qad-tp4` | S | path of four | checkpoint `qad-step5500-mxfp8-attention` |
| `mx-dsv41-tp4` | `deepseek-v41-flash-tp4` | S+ | path of four | `SIRCL_FUSED_NORM=1` (the arm) |

## Measurement

The runner's `phase1` metrics: decode engine steps per second from vLLM's speculative-decoding counters,
in 30 s cells at temperature 0 with at most 1,024 output tokens, at 0 and 32K tokens of context and 1, 2,
4 and 8 concurrent streams; the aggregate output rate and accepted tokens per step of each cell; the time
to first token of a cold 8K and 32K prompt; four fixed prompts' output fingerprints and prompt
log-probabilities compared between the two measured starts. Each campaign directory holds `tables.txt`
and `summary.json` as the runner wrote them and the identifying fields of its `plan.json`
([directory](dev-20261008-kraken-csf-sircl-libsircl-tp2-tp4-matrix-20261009/)). Steps per second are
the median of the two starts with their range.

## Result

Decode engine steps per second, median of two starts, at 0K / 32K context:

| Campaign | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---|---|---|---|
| `mx-csf-tp2-a` | 15.30 / 14.17 | 22.26 / 22.62 | 33.74 / 32.23 | 49.60 / 48.21 |
| `mx-csf-tp2-b` | 15.29 / 14.24 | 22.31 / 22.44 | 32.71 / 32.26 | 49.70 / 37.14 (26.63-47.64) |
| `mx-csf-tp4` | 25.00 / 23.33 | 37.91 / 35.74 | 55.63 / 47.62 | 82.51 / 80.66 |
| `mx-qwen-tp2` | 25.60 / 24.39 | 41.35 / 38.06 | 60.98 / 59.40 | 89.83 / 87.97 |
| `mx-qwen-tp2-s4000` | 26.97 / 25.67 | 43.68 / 39.42 | 64.65 / 61.19 | 92.39 / 90.76 |
| `mx-qwen-tp4` | 36.45 / 34.50 | 57.83 / 55.25 | 87.94 / 78.89 | 125.71 / 116.08 |
| `mx-dsv41-tp4` | 24.13 / 23.84 | 38.94 / 38.01 | 56.50 / 51.73 | 74.79 / 70.47 |

Aggregate output tokens per second at 0K context, 1 / 8 streams, each start: `mx-csf-tp2-a` 42.2 and
43.3 / 134.0 and 137.5; `mx-csf-tp2-b` 41.7 and 42.2 / 138.9 and 128.1; `mx-csf-tp4` 69.2 and 67.5 /
225.9 and 222.7; `mx-qwen-tp2` 60.9 and 65.0 / 216.3 and 217.7; `mx-qwen-tp2-s4000` 62.9 and 62.4 / 227.2 and
219.6; `mx-qwen-tp4` 91.9 and 90.1 / 312.9 and 308.4; `mx-dsv41-tp4` 60.0 and 63.8 / 176.3 and 187.5.

Time to first token, median of the two starts, 8K / 32K prompt: `mx-csf-tp2-a` 4.410 / 15.901 s;
`mx-csf-tp2-b` 3.802 / 15.326 s; `mx-csf-tp4` 2.560 / 10.491 s; `mx-qwen-tp2` 2.124 / 8.365 s;
`mx-qwen-tp2-s4000` 2.125 / 8.366 s; `mx-qwen-tp4` 1.728 / 6.842 s; `mx-dsv41-tp4` 1.882 / 7.583 s.

Every start passed the runner's arm checks. Output fingerprints identical between the two starts: 4 of 4
(`mx-csf-tp2-a`), 2 of 4 (`mx-csf-tp2-b`), 1 of 4 (`mx-csf-tp4`), 3 of 4 (`mx-qwen-tp2`, `mx-qwen-tp2-s4000`,
`mx-qwen-tp4`, `mx-dsv41-tp4`); prompt log-probability rank-1 agreement 0.977-0.994.

## Conclusion

On this image and hardware, each listed configuration served and measured as above with the ring-schedule
settings, which are a SIRCL tuning table's to set for pair and path-of-four groups. The profiles take the
other settings as follows:

- `glm53-flash-nvfp4-spark-tp4`: `B12X_W4A16_SMALL_M_OCCUPANCY=2` in the `csf` checkpoint's settings (the other checkpoints keep the image default, 1). The rest of
  `mx-csf-tp4` is the `csf` checkpoint's existing settings, which an image whose lock lists the pinned
  vLLM build installs without `--checkpoint`.
- `mx-csf-tp2-b` against `mx-csf-tp2-a`, the same image in the same session: time to first token 14 %
  lower at 8K and 4 % lower at 32K, decode steps within 3 % in every cell but one, where one start of
  `mx-csf-tp2-b` fell to 26.63 steps/s at 32K context and 8 streams; the campaign's operator attributes
  that cell to an underfilled measurement. `glm53-flash-nvfp4-spark-tp2` therefore takes
  `mx-csf-tp2-b`: `--async-scheduling` for every checkpoint (a flag a
  checkpoint entry cannot add), and in the `csf` checkpoint's settings `B12X_W4A16_SMALL_M_OCCUPANCY=2`,
  `VLLM_B12X_KDA_PREFILL_COALESCING=1` (`0`, the image default, for the other checkpoints) and the draft
  keys `draft_tensor_parallel_size` 2, `kv_cache_dtype` auto, `draft_sample_method` probabilistic and
  `rejection_sample_method` standard.
- `qwen38-flash-next-tp2` and `qwen38-flash-next-qad-tp4`: nothing besides the ring-schedule settings. On a pair the
  `qad-step-4000` and `qad-step5500-mxfp8-attention` checkpoints measured within 1-6 % of each other,
  on two different pairs. Both profiles keep their default checkpoint, `qad-step5500-ple1000`, which no
  campaign ran; both measured checkpoints stay selectable with `--checkpoint`, and the installer-derived
  `qad-step5500-mxfp8-attention` cannot be a checkpoint table's default.
- `deepseek-v41-flash-tp4`: `SIRCL_FUSED_NORM=1`, as the `S+` arm ran; a profile may pin that switch.

## Limitations

- One configuration per campaign: no campaign compares the ring-schedule settings, the W4A16 occupancy or
  `SIRCL_FUSED_NORM` with their defaults on this image. Their separate gains, where measured, are earlier
  measurements listed in [performance/enhancements.json](../../enhancements.json).
- GPU clocks were locked; an installer deployment runs at the Sparks' default clocks unless the operator
  locks them.
- Two-Spark settings were measured on a pair group of a ring of eight and four-Spark settings on a path
  of four. A four-Spark cycle (a four-Spark cluster) runs them unmeasured.
- The serving A/B runner ran the installer's container settings without the installer: no `sparkring
  install` run, functional check or correctness screen is part of this record.
- The GLM-5.3-Flash pair's checkpoints other than `csf` run `--async-scheduling` unmeasured.
- `mx-qwen-tp2` and `mx-qwen-tp2-s4000` ran on different pairs; their difference includes the
  pair-to-pair spread.
- Temperature 0, 30 s cells, two measured starts; output fingerprints differ between starts, as greedy
  decoding with speculative verification can on these images.
