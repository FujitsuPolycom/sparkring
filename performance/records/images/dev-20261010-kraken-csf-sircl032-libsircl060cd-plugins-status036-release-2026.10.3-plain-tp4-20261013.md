# Release 2026.10.3's four-Spark GLM-5.3-Flash and Qwen3.8-Flash-Next profiles installed without research options

Lane: **public-functional**. Status: **research-only**; maturity **live-validated** for one start per
deployment. Evidence scope: **plain `sparkring install` deployments of `qwen38-flash-next-tp4` on two
independent rings of four DGX Sparks, then of `glm53-flash-tp4` and `qwen38-flash-next-tp4` side by side as
two paths of four inside one cycle of eight; run receipts of each; one temperature-1 measurement per
deployment. Not serving-qualified.**

## Conditions

- **Hardware.** Eight DGX Sparks (GB10) with ConnectX-7 RoCE cables and no switch, cabled two ways in turn:
  - two independent rings of four, as in the
    [ring-of-four record](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring4-20261012.md):
    ring A spark-edfd, spark-ebb8, spark-ebee, spark-4a87; ring B spark-aa42, spark-3286, spark-0a0f,
    spark-931e;
  - one cycle of eight, positions 0-7: spark-edfd, spark-ebb8, spark-ebee, spark-931e, spark-aa42,
    spark-3286, spark-0a0f, spark-4a87. It was set up from spark-edfd with
    `sparkring setup --re-form --ssh-user sparkops --no-share-internet`, 2026-10-12 23:50-23:59 UTC, and
    reported:

    ```
    Fabric verified: 8 cables on 8 Sparks (cycle-8), relays 96 rules and 96 routes, reboot-persistent.
    ```

    Every PCI function measured 212.8-214.0 Gb/s, and no ConnectX driver restarted.
  GPU application clocks stayed locked at 2,418 MHz in every run receipt.
- **Package and image.** SparkRing `0.1.0~dev.1791783260+git48e4ffb1eb94` on all eight Sparks; release
  2026.10.2's installer image `d52737a109e0`, unchanged.
- **Installations.** `sudo sparkring install --profile <profile> [--checkpoint csf] [--on <positions>]
  --no-auto-recover --yes`, with no research option. Without `--checkpoint`, `qwen38-flash-next-tp4`
  installs its default checkpoint, `qad-step5500-ple1000`. `glm53-flash-tp4` was installed with
  `--checkpoint csf`, the checkpoint this image's lock prefers for it.
- **Measurement.** llm-inference-bench 0.7.6 from a workstation on the owner's network, temperature 1:

  ```
  --concurrency 1,2,4,8 --contexts 8k,16k,32k,64k,128k --duration 20 --max-tokens 2048 --coding-peak
  ```

  One start per deployment, so every cell is one sample. Node A admits one installation at a time, so the
  two paths of the cycle of eight ran in sequence with an overlap (UTC, 2026-10-13):

  | Deployment | Install | Benchmark |
  |---|---|---|
  | `glm53-flash-tp4`, positions 0-3 | 00:00-00:15 | 00:15:48-00:29:54 |
  | `qwen38-flash-next-tp4`, positions 4-7 | 00:16-00:27 | 00:27:35-00:39:38 |

  GLM's benchmark ran while Qwen loaded and compiled on the other path, and for its last two minutes beside
  Qwen's benchmark. The rings of four ran on 2026-10-12: ring A's benchmark 23:19-23:31, ring B's
  23:33-23:45.

## Results

### Resolved settings

Each run receipt is the same before and after its benchmark apart from GPU temperature and power. The two
rings of four's Qwen receipts differ only in the fabric digest.

| Deployment | Model settings from the receipt | SIRCL layout |
|---|---|---|
| `qwen38-flash-next-tp4`, ring A and ring B | checkpoint `qad-step5500-ple1000`; FP8 target KV cache in 32-token blocks; MTP draft of 3 tokens with its KV cache in the model's dtype (BF16); MXFP8 LM head | `ring:4`, one-shot limit 80 KiB |
| `glm53-flash-tp4`, positions 0-3 of the cycle of eight | checkpoint `csf`; `B12X_W4A16_A4_PREFILL_MIN_TOKENS=1536`; MTP draft of 3 tokens | `ring:8`, a path of four, one-shot limit 64 KiB |
| `qwen38-flash-next-tp4`, positions 4-7 of the cycle of eight | as on the rings of four, including the KV cache size | `ring:8`, a path of four, one-shot limit 64 KiB; compared with ring A's receipt, only the layout and SIRCL's tuning-table fields differ |

The KV cache held 2,831,661 tokens for Qwen on each ring of four and 5,668,802 tokens for GLM on the path.

### Throughput (temperature 1, one start)

Output tok/s. Decode is at 32K context; prefill is a single prompt.

| Deployment | Decode, 1 / 4 / 8 streams at 32K | Decode, 1 stream, 8K-128K | Decode, 8 streams, 8K-128K | Prefill at 8K / 32K / 128K | Coding peak, median of 5 |
|---|---|---|---|---|---|
| Qwen TP4, ring A | 73.0 / 203.6 / 288.1 | 73.0-87.3 | 278.0-291.1 | 4,573 / 4,553 / 3,916 | 111.5 |
| Qwen TP4, ring B | 78.4 / 207.6 / 280.0 | 77.6-86.5 | 273.2-292.8 | 4,172 / 4,569 / 3,926 | 110.4 |
| Qwen TP4, path of four in the cycle of eight | 77.9 / 209.5 / 274.3 | 73.6-78.3 | 271.2-290.9 | 4,353 / 4,377 / 3,844 | 111.3 |
| GLM-5.3-Flash CSF TP4, path of four in the cycle of eight | 68.1 / 135.3 / 203.5 | 61.6-68.1 | 196.5-211.1 | 3,399 / 3,165 / 3,200 | 79.7 |

## Conclusion

`qwen38-flash-next-tp4` runs both of the release's Qwen settings together on four Sparks, the MXFP8 LM
head and the draft's BF16 KV cache, with its default checkpoint. Its three starts, on two rings of four and
on a path of four inside the cycle of eight, measured 73.0-78.4 tok/s at 32K and one stream, 274.3-288.1 at
eight streams, a 4,377-4,569 tok/s 32K prefill and a 110.4-111.5 tok/s coding peak. The path of four lies
within the spread of the two rings, so no layout effect is claimed. Against the two-Spark profile with the
CSF checkpoint (68.1 / 149.7 / 219.0 tok/s, 3,716 prefill, 90.7 coding peak in the ring-of-eight record),
four Sparks decode 1.25-1.4 times as fast at four and eight streams and reach a 22 % higher coding peak; the
checkpoints differ, so this is a profile comparison, not a scaling measurement.

`glm53-flash-tp4` installs as a path of four inside a cycle of eight and resolves the release's settings,
including 4-bit-activation prefill. It decodes 1.6-1.8 times the two-Spark profile's rate at 32K
(37.8 / 83.1 / 112.9 tok/s in the
[ring-of-eight record](dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring8-20261012.md))
and prefills a 32K prompt 31 % faster. No start of it on a whole ring of four was measured.
