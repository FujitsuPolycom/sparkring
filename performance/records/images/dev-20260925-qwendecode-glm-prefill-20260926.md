# GLM-5.3-Flash prefill settings on the installer profiles

Status: **implemented; measured on one pair and one four-Spark ring; not
serving-qualified**.

The `glm53-flash-nvfp4-spark-tp4` profile sets five prefill settings that the
installer image implements:

| Setting | Effect |
|---|---|
| `NCCL_IB_EXTENDED_IPV4_GIDS=1` | NCCL 2.32.3 publishes all four ring NIC functions for prefill collectives |
| `VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=512`, `VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=512` | The target MLA cache and the recurrent-state cache use separate 512-token pages, so vLLM neither pads the recurrent page nor raises the attention block to 2,560 tokens as it does for one shared page layout |
| `VLLM_B12X_KDA_PREFILL_COALESCING=1` | KDA prefill checkpoints are coalesced |
| `VLLM_B12X_MLA_CKV_GATHER=0` | MLA compressed-KV gathering is off |

The `--block-size` and `--mamba-block-size` arguments are 512 to match. The
tables below call the configuration without these five settings the
**shared-page configuration**: 256-token shared pages, extended GIDs off, no
KDA coalescing and compressed-KV gather `auto`. The pair profile,
`glm53-flash-nvfp4-spark-tp2`, keeps its settings: none of the candidates
below raised its prefill.

Both profiles serve `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision
`a608241037e4` with MTP3.

## Method

- **Settings search:** serving containers derived from the installer
  deployment on image `dev-20260925-qwendecode-cuda1342-nccl2323-status031`,
  one group of settings added per variant.
  [`prefill_probe.py`](dev-20260925-qwendecode-glm-prefill-20260926/programs/prefill_probe.py)
  measures one cold prompt of about 8K, 32K and 128K tokens (one output token,
  time to first token) and repeats 8K and 32K;
  [`decode_probe.py`](dev-20260925-qwendecode-glm-prefill-20260926/programs/decode_probe.py)
  measures 512-token greedy decode by prompt type, two runs.
- **Matrices:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
  `llm_decode_bench.py` 0.6.2 at temperature 1.0 with exact token targeting,
  20 s per cell after a 5 s warm-up, up to 2,048 output tokens. Prefill is one
  cold prompt's length divided by its time to first token. Decode cells give
  aggregate output tokens/s, then verification steps/s × tokens per step.

## Four Sparks: settings search

Probe, tokens/s. Prefill shows the repeat run for 8K and 32K.

| Variant | Prefill 8K | 32K | 128K | Decode prose / code / JSON |
|---|---:|---:|---:|---|
| Shared-page configuration | 2,870 | 2,924 | 2,793 | 57.2 / 68.2 / 78.8 |
| + extended GIDs | 3,142 | 3,217 | 3,078 | 59.3 / 73.8 / 77.0 |
| + extended GIDs, KDA coalescing | 3,243 | 3,220 | 3,088 | 60.4 / 71.5 / 78.6 |
| + SIRCL fused dual-rail prefill | 3,304 | 3,185 | 3,033 | 56.9 / 68.0 / 76.4 |
| + split pages 512/512 | 3,167 | 3,548 | 3,487 | 62.2 / 68.4 / 78.6 |
| + MLA compressed-KV gather off | 3,272 | 3,229 | 3,098 | 61.9 / 71.9 / 82.3 |
| **+ split pages 512/512, gather off (profile)** | **3,181** | **3,589** | **3,480** | **60.1 / 67.6 / 79.6** |

Rows below the second add their settings to extended GIDs and KDA coalescing.
The SIRCL row loads the SIRCL transport and vLLM adapter from image
`shared-2026.09.3` into this image; its fused dual-rail prefill session opened
on all four ranks and added no prefill over NCCL with extended GIDs, so the
profile does not use it.

## Four Sparks: matrices

| Prefill (tokens/s) | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Shared-page configuration | 2,702 | 2,827 | 2,873 | 2,849 | 2,766 |
| Profile settings | 3,800 | 3,758 | 3,706 | 3,644 | 3,519 |
| Profile settings, second run | 3,773 | 3,759 | 3,712 | 3,591 | 3,491 |

Decode with the profile settings (second run):

| Context | 1 stream | 2 streams | 4 streams | 8 streams | 16 streams |
|---|---:|---:|---:|---:|---:|
| 8K | 58.6 (25.5 × 2.30) | 96.2 (37.4 × 2.57) | 135.9 (54.4 × 2.50) | 187.2 (73.9 × 2.53) | 275.2 (105.3 × 2.61) |
| 64K | 60.1 (24.4 × 2.46) | 91.1 (36.5 × 2.50) | 127.9 (51.2 × 2.50) | 181.5 (69.9 × 2.60) | 272.9 (101.7 × 2.68) |
| 128K | 65.5 (25.1 × 2.61) | 89.4 (35.7 × 2.50) | 127.0 (50.9 × 2.50) | 183.4 (73.3 × 2.50) | 257.1 (98.5 × 2.61) |
| 256K | 62.4 (24.5 × 2.54) | 86.9 (35.5 × 2.45) | 132.0 (50.0 × 2.64) | 183.3 (69.4 × 2.64) | — |

Verification steps per second match the shared-page configuration (25.3 at one stream
and 102.5 at 16 streams, no context): the settings change prefill only.

For comparison, the owner's matrix of the `GLM-5.3-Flash-NVFP4` QAD checkpoint
on image `shared-2026.09.3` (SIRCL transport) prefilled 3,746 / 3,692 / 3,666 /
3,610 / 3,501 tokens/s at 8K-128K and decoded 43.8 tokens/s at 16.2 steps/s at
one stream; its draft experts are MXFP8 and run on the `humming` MoE backend.

## Pair

Probe on the installer deployment, then added settings; repeat runs for 8K and
32K.

| Variant | Prefill 8K | 32K | 128K | Decode prose / code / JSON |
|---|---:|---:|---:|---|
| Profile settings | 2,470 | 2,476 | 2,401 | 35.1 / 40.9 / 44.7 |
| + extended GIDs, KDA coalescing | 2,475 | 2,459 | 2,387 | 36.3 / 39.5 / 45.4 |
| + split pages 2,048/256, 2 OpenMP threads, 32 CUDA connections | 2,100 | 2,404 | 2,484 | 34.6 / 39.0 / 42.4 |
| + MXFP8 target and NVFP4 MTP LM heads | 2,111 | 2,246 | 2,362 | 35.1 / 40.8 / 45.3 |

Matrix with the profile settings: prefill 2,310 / 2,320 / 2,372 / 2,375 / 2,337
tokens/s at 8K-128K; decode 40.2 tokens/s at one stream (15.0 steps/s × 2.68)
and 104.4 at eight streams, no context.

## Installed profile

`sparkring install --profile glm53-flash-nvfp4-spark-tp4` on image
`dev-20260927-h2dstaging-cuda1342-nccl2323-status031` (the settings-search
image plus vLLM's host-to-device staging fix) installed from the published
one-line installer in 475 s, including the first kernel compile. The log shows
split pages of 512 tokens for both caches and no attention block-size
adjustment; the KV pool is 2,173,412 tokens. With no other traffic, the probe
measured prefill 3,374-3,399 tokens/s at 16K and decode 60.6 / 64.7 / 77.5
tokens/s (prose / code / JSON); the issue #294 concurrency probe returned no
degenerate response in 1,024 requests.

## Files

The [measurement directory](dev-20260925-qwendecode-glm-prefill-20260926/)
holds each matrix (`tp4-matrix-shared-pages.json`,
`tp4-matrix-profile-settings.json`,
`tp4-matrix-profile-settings-long-context.json`, `tp2-matrix-installer.json`
and the QAD reference `tp4-matrix-native-qad-reference.json`, with the
benchmark client's host diagnostics and the server address removed), the probe
output of both settings searches (`tp4-probes.txt`, `tp2-probes.txt`) and the
probe programs.
