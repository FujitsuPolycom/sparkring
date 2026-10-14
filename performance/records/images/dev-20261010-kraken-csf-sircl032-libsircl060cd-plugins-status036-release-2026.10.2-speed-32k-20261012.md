# Sources of the README speed table: release 2026.10.2's image at 32K context

Lane: **public-functional**. Status: **research-only**; maturity **live-validated** for one start per
row. Evidence scope: **the nine rows of the README's Speed table, each traced to one llm-inference-bench
0.7.6 result file from a plain `sparkring install` deployment on release 2026.10.2's installer image,
measured 2026-10-12 and 2026-10-13. Every cell is one sample. Not serving-qualified.**

This record holds no new measurement. It names, for every cell of the README's
[Speed](../../../README.md#speed) table, the raw benchmark file and field that produced it. The same
runs are described, with their wider context and settings comparisons, in the
[ring-of-eight record](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring8-20261012.md),
the [ring-of-four record](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring4-20261012.md)
and the [four-Spark record](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-tp4-20261013.md).

## Conditions

- **Hardware.** DGX Sparks (GB10) joined by ConnectX-7 RoCE cables with no switch, in two cablings:
  - one cycle of eight, positions 0-7;
  - two independent rings of four, ring A and ring B, whose members the ring-of-four record lists.

  A *path of four* is four consecutive positions of the cycle of eight; a *pair* is two adjacent
  positions. Every run receipt records GPU application clocks locked at 2,418 MHz.
- **Image.** Release 2026.10.2's installer image `dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`,
  `sha256:d52737a109e083d3eef05c0fc0db4d09fc1bf34485a13467382130963ecfe66b`, read from the `image_id` and
  `image_release` fields of each installation's result.
- **Package.** SparkRing built from revision `48e4ffb1eb94124f610dc8ac0c5d856ca562a5b7` (the
  `source_revision` field of each installation's result), package
  `0.1.0~dev.1791783260+git48e4ffb1eb94`, on every Spark.
- **Installations.** `sudo sparkring install` with the arguments in the table below and no research
  option. Each run receipt is identical before and after its benchmark in every setting; only GPU
  temperature and power differ.
- **Benchmark.** llm-inference-bench 0.7.6 from a workstation on the owner's network, temperature 1.0,
  `--concurrency 1,2,4,8 --contexts 8k,16k,32k,64k,128k --duration 20 --max-tokens 2048 --coding-peak`;
  DeepSeek runs add 16 streams. The README uses three of its outputs:
  - **Decode, 1 / 4 / 8 streams.** Aggregate output tok/s of the sustained-decode cell at the 32K context
    and that many concurrent streams, 20 s per cell after warmup (`summary_table.32768.<streams>` in the
    result JSON; the last "Aggregate decode tok/s" table of the console output).
  - **Prefill, 32K prompt.** Prompt tokens divided by time to first token for one request of about
    32,200 tokens, measured from the client (`prefill.32768.client_tok_per_sec`; the "Prefill tok/s" table).
  - **Coding peak.** Median generation tok/s of five single-stream coding requests
    (`coding_peak.summary.median_generation_tok_s`; the "Coding Peak" table).

### Runs

UTC times come from each run's timeline file.

| Run directory | `--profile` in the README | Install arguments | Sparks | Benchmark (UTC) | Receipt digest | Activity on other Sparks during the benchmark |
|---|---|---|---|---|---|---|
| `deepseek-v41-flash-tp8-cycle8` | `deepseek-v41-flash-tp8` | `--profile deepseek-v41-flash-tp8 --no-auto-recover` | cycle of eight, 0-7 | 2026-10-12 08:06:25-08:20:23 | `sha256:20674b66...` | none; the run used all eight |
| `glm53-nvfp4-tp8-cycle8` | `glm53-nvfp4-tp8` | `--profile glm53-nvfp4-tp8 --no-auto-recover` | cycle of eight, 0-7 | 2026-10-12 08:28:17-08:46:52 | `sha256:d3d1ad56...` | none; the run used all eight |
| `deepseek-v41-flash-tp4-path4-cycle8` | `deepseek-v41-flash-tp4` | `--profile deepseek-v41-flash-tp4 --on 0-3` | cycle of eight, path 0-3 | 2026-10-12 09:12:05-09:26:42 | `sha256:94a1a64c...` | GLM-5.3-Flash TP2 on 4-5 and Qwen TP2 on 6-7 benchmarked at the same time |
| `qwen38-flash-next-tp2-pair-cycle8` | `qwen38-flash-next-tp2` | `--profile qwen38-flash-next-tp2 --checkpoint csf --on 6,7` | cycle of eight, pair 6-7 | 2026-10-12 09:12:05-09:25:54 | `sha256:f7ddf670...` | DeepSeek TP4 on 0-3 and GLM-5.3-Flash TP2 on 4-5 benchmarked at the same time |
| `deepseek-v41-flash-tp4-cycle4-ring-a` | `deepseek-v41-flash-tp4` | `--profile deepseek-v41-flash-tp4` | ring A, 0-3 | 2026-10-12 19:16:48-19:30:48 | `sha256:738cfb1b...` | ring B, which has no cable to ring A, installed its two pairs |
| `glm53-flash-tp2-pair-ring-b` | `glm53-flash-tp2` | `--profile glm53-flash-nvfp4-spark-tp2 --checkpoint csf --on 0,1` | ring B, pair 0-1 | 2026-10-12 19:34:24-19:51:49 | `sha256:e672894a...` | Qwen TP2 on ring B's other pair finished installing at 19:34:17 and was idle |
| `qwen38-flash-next-tp2-pair-ring-b-quiet` | `qwen38-flash-next-tp2` | `--profile qwen38-flash-next-tp2 --checkpoint csf --on 2,3`, installed 19:15:56-19:34:17 (timeline in `qwen38-flash-next-tp2-pair-ring-b`) | ring B, pair 2-3 | 2026-10-12 20:58:14-21:12:34 | `sha256:6cb7f4e4...` | ring B otherwise idle; this is the second benchmark of the same start |
| `qwen38-flash-next-tp4-cycle4-ring-a` | `qwen38-flash-next-tp4` | `--profile qwen38-flash-next-qad-tp4 --no-auto-recover` | ring A, 0-3 | 2026-10-12 23:19:32-23:31:54 | `sha256:b50f422e...` | ring B, which has no cable to ring A, installed its own Qwen TP4 start (23:21-23:33) |
| `glm53-flash-tp4-path4-cycle8` | `glm53-flash-tp4` | `--profile glm53-flash-nvfp4-spark-tp4 --checkpoint csf --on 0-3 --no-auto-recover` | cycle of eight, path 0-3 | 2026-10-13 00:15:48-00:29:54 | `sha256:5fb6009d...` | Qwen TP4 loaded on 4-7, then benchmarked from 00:27:35 |
| `qwen38-flash-next-tp4-path4-cycle8` | `qwen38-flash-next-tp4` | `--profile qwen38-flash-next-qad-tp4 --on 4-7 --no-auto-recover` | cycle of eight, path 4-7 | 2026-10-13 00:27:35-00:39:38 | `sha256:c8bc0298...` | GLM-5.3-Flash TP4 benchmark on 0-3 until 00:29:54 |

`glm53-flash-tp2`, `glm53-flash-tp4` and `qwen38-flash-next-tp4` are aliases of the profile IDs
`glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4` and `qwen38-flash-next-qad-tp4`
([alias table](../../../profiles/README.md)). The two TP8 runs, DeepSeek TP4's path 0-3 and Qwen TP2's
pair 6-7 used the cycle of eight as cabled before the rings of four. The GLM-5.3-Flash TP4 and Qwen TP4
paths used the cycle of eight re-cabled and set up again after them, 2026-10-12 23:50-23:59 UTC.

## Results

Output tok/s as llm-inference-bench printed them, before the README rounds them to whole numbers. Every
cell has a raw source.

| README row | Decode, 1 / 4 / 8 streams | Prefill, 32K prompt | Coding peak | Source of decode and prefill | Source of coding peak |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash, 8, ring of eight | 73.7 / 174.2 / 258.9 | 4,851 | 133.8 | `deepseek-v41-flash-tp8-cycle8` | same |
| DeepSeek-V4.1-Flash, 4, ring of four | 70.6 / 136.7 / 194.3 | 4,388 | 102.4 | `deepseek-v41-flash-tp4-cycle4-ring-a` | same |
| DeepSeek-V4.1-Flash, 4, path of four | 70.4 / 139.1 / 198.0 | 4,426 | 101.7 | `deepseek-v41-flash-tp4-path4-cycle8` | same |
| GLM-5.3, 8, ring of eight | 45.8 / 93.7 / 129.1 | 1,387 | 51.4 | `glm53-nvfp4-tp8-cycle8` | same |
| GLM-5.3-Flash CSF, 4, path of four | 68.1 / 135.3 / 203.5 | 3,165 | 79.7 | `glm53-flash-tp4-path4-cycle8` | same |
| GLM-5.3-Flash CSF, 2, pair | 37.8 / 83.1 / 112.9 | 2,420 | 49.0 | `glm53-flash-tp2-pair-ring-b` | same |
| Qwen3.8-Flash-Next QAD step 5500, 4, ring of four | 73.0 / 203.6 / 288.1 | 4,553 | 111.5 | `qwen38-flash-next-tp4-cycle4-ring-a` | same |
| Qwen3.8-Flash-Next QAD step 5500, 4, path of four | 77.9 / 209.5 / 274.3 | 4,377 | 111.3 | `qwen38-flash-next-tp4-path4-cycle8` | same |
| Qwen3.8-Flash-Next CSF, 2, pair | 68.1 / 149.7 / 219.0 | 3,716 | 90.7 | `qwen38-flash-next-tp2-pair-ring-b-quiet` | same |

Every row is one benchmark of one start. The Qwen3.8-Flash-Next TP2 row is the ring-B pair's second
benchmark. Another start of the same profile and checkpoint, on the pair 6-7 of the cycle of eight, measured
64.7 / 154.7 / 225.0 decode, 3,716 prefill and a coding peak of 94.8; the ring-of-four record compares the two
starts' receipts, which differ only in layout and SIRCL fabric fields.

### Raw files

The [raw directory](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.2-speed-32k-20261012/)
holds one subdirectory per run directory named above, each with:

- `bench-result.json`: llm-inference-bench's result file, including per-request samples;
- `bench-output.txt`: its console output;
- `run-timeline.txt`: the install, receipt and benchmark times of the run.

[`cells.json`](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.2-speed-32k-20261012/cells.json)
lists every README cell at full precision with its file and field.
[`sources.json`](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.2-speed-32k-20261012/sources.json)
gives each file's path and SHA-256 in the operator's research archive, which is not part of this
repository. The copies replace three LAN addresses, the workstation's host name and the archive's local
path with the placeholders `sources.json` lists; no measured value is changed.

## Conclusion

Every cell of the README's Speed table matches a value in a committed llm-inference-bench 0.7.6 result
file from release 2026.10.2's image at temperature 1. Eight rows each come from one benchmark of one
start. The Qwen3.8-Flash-Next TP2 row takes its coding peak from a different start of the same profile
than its decode and prefill cells.
