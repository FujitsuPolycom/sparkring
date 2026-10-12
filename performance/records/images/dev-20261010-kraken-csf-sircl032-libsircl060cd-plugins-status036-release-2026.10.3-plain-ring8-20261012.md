# Release 2026.10.3's changed profiles installed without research options on the ring of eight

Lane: **public-functional**. Status: **research-only**; maturity **live-validated** for one start per
profile. Evidence scope: **plain `sparkring install` deployments of `deepseek-v41-flash-tp4`,
`glm53-flash-nvfp4-spark-tp2 --checkpoint csf` and `qwen38-flash-next-tp2 --checkpoint csf` on an
eight-Spark ring with installer image `d52737a109e0`; run receipts of each; one temperature-1 measurement
per profile; an installation on part of the ring beside serving deployments; an automatic-recovery restart
of a four-rank group. Not a ring-of-four result and not serving-qualified.**

## Conditions

- **Hardware.** Eight DGX Sparks (GB10) cabled as one ConnectX-7 RoCE cycle of eight, positions 0-7.
  DeepSeek ran on positions 0-3 (a path of four inside the cycle), GLM on the pair 4-5, Qwen on the pair
  6-7, each on SIRCL ring sessions with NCCL absent. GPU application clocks were locked at 2,418 MHz
  (`nvidia-smi -lgc`), set before these runs and left unchanged.
- **Package.** SparkRing `0.1.0~dev.1791783260+git48e4ffb1eb94` (SHA-256 `d9b8a2701efb132c...`) on all
  eight Sparks. Its revision `48e4ffb1` is this release's candidate `f143cc2e` plus code for research install
  options (`--research-env`, `--research-vllm-arg`) that these installations do not use. Between
  `f143cc2e` and this release's head the three profiles change only in README text; their
  configurations are identical.
- **Image.** Release 2026.10.2's installer image
  `sha256:d52737a109e083d3eef05c0fc0db4d09fc1bf34485a13467382130963ecfe66b`, unchanged.
- **Installations.** `sudo sparkring install --profile <profile> [--checkpoint csf] --on <positions> --yes`
  with no other option, 2026-10-12 08:47-09:12 UTC; automatic recovery on afterwards.
- **Receipts.** The run-receipts tool (`performance/harnesses/serving_ab/receipt`, branch
  `claude/run-receipts-20261011`) read each deployment's resolved vLLM arguments, engine configuration,
  serving environment, SIRCL receipts, image and clocks, before and after each measurement.
- **Measurement.** llm-inference-bench 0.7.6 from a workstation on the owner's network, temperature 1,
  `--concurrency 1,2,4,8 --contexts 8k,16k,32k,64k,128k --duration 20 --max-tokens 2048 --coding-peak`
  (DeepSeek adds 16 streams). The three deployments were measured at the same time, 2026-10-12
  09:12-09:30 UTC, each on its own Sparks and cables; one start each, so every cell is one sample.

## Results

### Resolved settings

| Deployment | Receipt digest | Against the research-option start it replaces |
|---|---|---|
| `deepseek-v41-flash-tp4`, positions 0-3 | `sha256:94a1a64c...` | identical digest: the profile's defaults reproduce the start installed with `--async-scheduling` and `--engram-config={..."projection_tp":true}` as research options |
| `glm53-flash-nvfp4-spark-tp2 --checkpoint csf`, positions 4-5 | `sha256:ec2e5c19...` | one setting differs: the earlier start was installed with `--reasoning-effort high`; this one uses the model's default |
| `qwen38-flash-next-tp2 --checkpoint csf`, positions 6-7 | `sha256:f7ddf670...` | no setting differs |

Receipts before and after each measurement show no setting change.

### Throughput (temperature 1, one start)

| Deployment | Decode tok/s, 1 stream, 8K-128K | Decode tok/s at the highest concurrency | Prefill tok/s at 8K / 128K | Coding peak, median of 5 |
|---|---|---|---|---|
| DeepSeek TP4 | 62.7-70.4 | 272.9-280.7 (16 streams) | 4,491 / 4,075 | 101.7 |
| GLM-5.3-Flash CSF TP2 | 37.2-43.0 | 109.1-119.6 (8 streams) | 2,489 / 2,411 | 50.9 |
| Qwen3.8-Flash-Next CSF TP2 | 64.3-82.2 | 225.0-229.0 (8 streams) | 3,809 / 3,427 | 94.8 |

### Installation on part of the ring

`qwen38-flash-next-tp2 --checkpoint csf --on 6,7` ran 2026-10-12 16:07-16:14 UTC while GLM served on
positions 4-5 and DeepSeek on 0-3. The installation planned images and checkpoints for its two Sparks only
(`nodes: 2`, placement `[4, 5]` and `[6, 7]` in the installers' results). GLM's containers, started at
15:59 UTC, kept running with no restart through it. No Spark holds an image created during these
installations.

### Automatic recovery of a four-rank group

With `deepseek-v41-flash-tp4` serving on positions 0-3 and automatic recovery on, rank 3's container
(position 3) was stopped with `docker stop` at 2026-10-12 16:52:47 UTC. `sparkring-recover.timer` runs
the recovery check about once a minute:

| UTC | Recovery |
|---|---|
| 16:53:16 | Rank 3 stopped; waits for a confirming check |
| 16:54:22 | Confirmed; stops the model on all four ranks and starts it |
| 16:59:34 | "The model started again"; every rank runs and the API answers |

The deployment served again 6 min 47 s after the stop. A completion request returns the expected answer.
GLM on positions 4-5 and Qwen on 6-7 kept running with no restart (container start times 15:59 and 16:08
UTC unchanged).

## Conclusion

The release's plain profiles resolve to the measured fastest settings on these layouts, install on part
of a ring of eight without disturbing the deployments beside them, and recover a stopped rank. The
DeepSeek TP4 result is for a path of four inside a cycle of eight; no start of these settings has run on
a ring of four.

## Not adopted: Engram projection sharding at TP8

`deepseek-v41-flash-tp8` with `--engram-config={"table_memory":"disk","disk_resident_scales":true,"projection_tp":true}`
as a research option, against the profile's start without it, on the cycle of eight, same package, image
and method, one start each (receipts differ only in `engram.projection_tp`): engine steps/s equal at one
stream (30.8-32.3 both), about 6 % lower at eight (102-110 against 110-117), 2-5 % higher at sixteen
(142-159 against 135-155); prefill 1-3 % lower; coding-peak median 124.0 against 133.8; KV cache 33.2M
against 32.7M tokens. No consistent gain, so `deepseek-v41-flash-tp8` keeps its replicated projections.
