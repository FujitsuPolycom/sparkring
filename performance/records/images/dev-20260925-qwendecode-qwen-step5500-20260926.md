# Qwen3.8-Flash-Next step 5500 on the Qwen installer profiles

Status: **implemented; functional checks passed; measured on one pair and one
ring; not serving-qualified**.

`sudo sparkring install` installed `qwen38-flash-next-tp2` (one directly
cabled Spark pair) and `qwen38-flash-next-qad-tp4` (one four-Spark ring) with
checkpoint revision `60215d26cf5e42c2db6128774032d57fc62678da` of
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (branch
`qad-step5500-ple1000`) on image
`dev-20260925-qwendecode-cuda1342-nccl2323-status031`. The installed package
was `sparkring_0.1.0~dev.1790439220+git47ca98112842`, built from the profile
configuration this record accompanies.

The checkpoint's MTP routed experts are MXFP8, which the B12X MoE backend does
not implement, so the draft runs on the `humming` MoE backend. The profiles keep
the BF16 target LM head (`VLLM_MXFP8_LM_HEAD=0`). All other settings match the
step-4000 configuration measured in the
[six-profile record](dev-20260925-qwendecode-installer-profiles-20260926.md),
including MTP3 with probabilistic drafting.

Both installations found the checkpoint already on every Spark, hard-linked
its 41 weight files, copied its 12 other files and downloaded nothing; linking
and verifying took 17–20 s per Spark. The first start compiled kernels for the
revision: API readiness took 571 s on the pair and 492 s on the ring. Both
passed the installer's response check and counting, arithmetic and code
checks.

## Method

- Probe: [`measure.sh`](../qwen38-flash-next/installer-tuning-20260925/programs/measure.sh),
  as in the six-profile record: 512-token single-stream decode by prompt type
  at temperature 0 (two runs) and 1.0 (three runs), prefill as the time to the
  first token of one cold prompt of about 4K, 16K and 64K tokens (second of two
  runs), and three greedy 512-token decodes.
- Matrix: [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
  `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting,
  1, 8 and 16 concurrent streams, 0/8K/16K/32K/64K context, 8K–64K prefill, up
  to 2,048 output tokens and 17 s per cell after a 5 s warm-up. Decode is the
  aggregate output rate across streams; steps per second and tokens per step
  come from vLLM's speculative-decoding counters.

## Probe results

| Cluster | Decode prose / code / JSON, temperature 0 (tok/s) | Temperature 1.0 | Prefill 4K / 16K / 64K (tok/s) | Greedy median |
|---|---|---|---|---|
| Pair (TP2) | 49.3 / 77.4 / 84.8 | 51.0 / 75.5 / 81.5 | 4,038 / 4,077 / 3,762 | 49.2 |
| Ring (TP4) | 70.7 / 108.4 / 118.1 | 67.6 / 109.8 / 118.2 | 4,740 / 5,024 / 4,664 | 70.8 |

On the same probe and image, step 4000 decoded 59.5 / 88.4 / 101.0 (pair) and
87.9 / 126.8 / 142.2 (ring) tokens per second at temperature 0, and prefilled
4,259 and 5,003 tokens per second at 16K.

## Matrix results

Aggregate decode in tokens per second; prefill in tokens per second.

| Pair (TP2) | Prefill | 1 stream | 8 streams | 16 streams |
|---|---:|---:|---:|---:|
| 0 context | — | 56.2 | 196.6 | 283.3 |
| 8K | 3,692 | 44.3 | 163.8 | 232.6 |
| 16K | 3,883 | 47.2 | 172.2 | 238.9 |
| 32K | 3,820 | 49.3 | 160.9 | 237.9 |
| 64K | 3,666 | 43.3 | 162.4 | 233.1 |

| Ring (TP4) | Prefill | 1 stream | 8 streams | 16 streams |
|---|---:|---:|---:|---:|
| 0 context | — | 76.8 | 289.3 | 415.6 |
| 8K | 4,768 | 58.8 | 232.9 | 337.7 |
| 16K | 4,810 | 62.4 | 227.0 | 342.5 |
| 32K | 4,777 | 64.8 | 225.4 | 325.9 |
| 64K | 4,559 | 63.1 | 232.5 | 331.2 |

The repository owner's matrices of the same checkpoint and settings, run in
serving containers derived from the installer deployments before installation,
are kept beside these results. Prefill agreed within 2% and verification steps
per second within 3% in every cell. Decode rates at 8K–64K context on the ring
were up to 20% higher in the owner's run because its drafts averaged 2.2–2.4
tokens per verification step against 1.8–2.0 here: at temperature 1.0,
acceptance follows the generated text, not the deployment.

## Files

The [measurement directory](dev-20260925-qwendecode-qwen-step5500-20260926/)
holds each probe's output (`*-probe.txt`, `*-probe.json`,
`*-probe-temperature1.txt`) and each matrix (`*-matrix-installer.json` for the
installer deployments, `*-matrix-test-containers.json` for the owner's runs).
Matrix files keep the benchmark settings, prefill, per-cell results and summary
tables; the benchmark client's host diagnostics and the server address are
removed.
