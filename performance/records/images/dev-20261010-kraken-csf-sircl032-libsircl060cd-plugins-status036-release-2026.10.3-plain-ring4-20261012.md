# Release 2026.10.3's changed profiles installed without research options on two rings of four

Lane: **public-functional**. Status: **research-only**; maturity **live-validated** for one start per
profile. Evidence scope: **two independent rings of four DGX Sparks, each set up with
`sparkring setup --re-form` like a new four-Spark installation; plain `sparkring install` deployments of
`deepseek-v41-flash-tp4` on one ring and of `glm53-flash-nvfp4-spark-tp2 --checkpoint csf` and
`qwen38-flash-next-tp2 --checkpoint csf` on the two pairs of the other; run receipts of each; one
temperature-1 measurement per profile. Not serving-qualified.**

## Conditions

- **Hardware.** Eight DGX Sparks (GB10) recabled into two independent ConnectX-7 RoCE rings of four with no
  cable between them:
  - ring A: spark-edfd, spark-ebb8, spark-ebee, spark-4a87;
  - ring B: spark-aa42, spark-3286, spark-0a0f, spark-931e.
- **Setup.** Each ring was set up from its own Node A, spark-edfd for ring A and spark-aa42 for ring B, with
  `sparkring setup --re-form --ssh-user sparkops --no-share-internet`, 2026-10-12 18:42-19:00 UTC. Each setup
  reported:

  ```
  Fabric verified: 4 cables on 4 Sparks (cycle-4), relays 16 rules and 16 routes, reboot-persistent.
  ```

  Every cable measured 212.7-213.7 Gb/s in both directions at once, and no ConnectX driver restarted.
  GPU application clocks stayed locked at 2,418 MHz.
- **Package and image.** SparkRing `0.1.0~dev.1791783260+git48e4ffb1eb94` on all eight Sparks, the package of
  the [ring-of-eight record](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring8-20261012.md);
  release 2026.10.2's installer image `d52737a109e0`, unchanged.
- **Installations.** `sudo sparkring install --profile <profile> [--checkpoint csf] [--on <positions>] --yes`
  with no research option, 2026-10-12 18:54-19:34 UTC; automatic recovery on.
- **Measurement.** llm-inference-bench 0.7.6 from a workstation on the owner's network, temperature 1:

  ```
  --concurrency 1,2,4,8 --contexts 8k,16k,32k,64k,128k --duration 20 --max-tokens 2048 --coding-peak
  ```

  DeepSeek adds 16 streams. There is one client per ring at a time and one start per profile, so every cell
  is one sample. While Qwen was measured, ring B's other pair was copying a 25 GB container image over the
  LAN and building at low priority.

## Results

### Resolved settings

Each deployment's run receipt, compared with the same profile's plain installation on the ring of eight,
differs only in layout and SIRCL transport tuning. Model, speculative-decoding, KV-cache and scheduler
settings are identical, including DeepSeek's Engram projection sharding.

| Deployment | Layout | Differences from the ring-of-eight receipt |
|---|---|---|
| `deepseek-v41-flash-tp4`, ring A | cycle of four | The group is a whole cycle of four instead of a path of four inside the cycle of eight. SIRCL takes the installer's measured cycle-of-four tuning table instead of the path-of-four row: large `auto`, gather `auto`, scatter `pieces`, 512 KiB link pieces, an 80 KiB one-shot limit. |
| `glm53-flash-tp2`, ring B positions 0-1 | pair | `SIRCL_FABRIC` `ring:4`, positions, fabric name and digest |
| `qwen38-flash-next-tp2`, ring B positions 2-3 | pair | the same layout fields |

### Throughput (temperature 1, one start)

| Deployment | Decode tok/s, 1 stream, 8K-128K | Decode tok/s at the highest concurrency | Prefill tok/s at 8K / 128K | Coding peak, median of 5 |
|---|---|---|---|---|
| DeepSeek TP4, ring of four | 60.2-70.6 | 261.4-280.8 (16 streams) | 4,384 / 3,978 | 102.4 |
| DeepSeek TP4, path of four in the cycle of eight | 62.7-70.4 | 272.9-280.7 | 4,491 / 4,075 | 101.7 |
| GLM-5.3-Flash CSF TP2, pair of ring B | 36.2-38.3 | 107.3-117.7 (8 streams) | 2,421 / 2,384 | 49.0 |
| GLM-5.3-Flash CSF TP2, pair of the cycle of eight | 37.2-43.0 | 109.1-119.6 | 2,489 / 2,411 | 50.9 |
| Qwen3.8-Flash-Next CSF TP2, pair of ring B | 58.7-66.1 | 211.7-228.6 (8 streams) | 3,284 / 3,276 | 87.2 |
| Qwen3.8-Flash-Next CSF TP2, pair of the cycle of eight | 64.3-82.2 | 225.0-229.0 | 3,809 / 3,427 | 94.8 |

The ring-of-eight rows are from the ring-of-eight record, measured with the three deployments' clients
running at the same time.

## Conclusion

The release's plain profiles install on a new ring of four and on its pairs, and resolve the same model
settings as on the ring of eight. SIRCL selects the ring-of-four tuning without any option. DeepSeek TP4
measures the same on a whole ring of four as on a path of four, within one start's variation: its decode
all-reduces are small and latency-bound, so the relayed closing link of a path costs little.

The two pairs measured lower than on the ring of eight: GLM by up to about 10 % at one stream, and Qwen by
5-10 % with an 8K prefill 14 % lower. Their settings are identical, and a pair is the same single cable in
either ring. Qwen's run overlapped the image copy described above and used a different second Spark. One
start each cannot separate these causes from start-to-start variation, so no layout effect is claimed.
