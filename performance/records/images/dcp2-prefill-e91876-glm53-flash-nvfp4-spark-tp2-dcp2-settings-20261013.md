# GLM-5.3-Flash CSF on a pair at DCP2: request count, prefill features and KV interleave

Lane: **public-functional**. Status: **research-only**. Evidence scope: **single runs (two for the
combination) of hand-made containers on four cabled pairs of an eight-Spark ring, plus one installed start
of the resulting settings on positions 6 and 7, measured on 2026-10-13 through the API with GPU clocks not
locked; not serving-qualified.**

## Conditions

- **Checkpoint.** `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` at `dec48abd33ef`.
- **Images.**
  - Installer image `sha256:d52737a109e0`.
  - The source-derived image `sha256:e91876e4997c`, built with
    `runtime/images/derive_dcp2_prefill.py` from that image. It admits KDA prefill coalescing and mHC prefill
    row ownership at TP2/DCP2 (`runtime/images/compositions/dcp2-prefill/`). Local tag only; not published.
- **Hardware.** Eight DGX Sparks (GB10) in one ConnectX-7 ring. Each setting ran on its own cabled pair
  (positions 0-1, 2-3, 4-5 or 6-7) on SIRCL ring sessions. Up to four pairs served and were benchmarked at
  the same time.
- **Base settings.** Every setting of this profile before these changes: TP2, DCP2, interleave 4, MTP3,
  8,192-token prefill budget, one parallel prefill, 14 GiB of KV cache per Spark and eight requests.
  Hand-made containers carried the rendered profile command with one or more of these differences.
- **Benchmark.** llm-inference-bench 0.7.6 at temperature 1:
  `--contexts 16k,64k,128k --concurrency 1,4,8` (eight-request settings) or `1,4,8,16` (16-request
  settings), `--duration 20 --max-tokens 2048 --coding-peak`.
  - The hand-made comparisons ran the client under WSL. The installed run ran it under native Windows
    Python with exact `/tokenize` prompt targeting.
  - The two clients differ: the same container measured 5-6 % lower prefill and about 6 % lower single-stream
    steps/s from WSL. So hand-made settings are compared with a hand-made copy of the base settings measured
    the same way.
- **Correctness.** A seven-prompt screen per setting: every setting answered 7 of 7 correctly, with no
  degenerate response.

## Results

Prefill is the client's prompt tokens divided by time to first token. Decode is aggregate output tok/s.

**Hand-made settings, WSL client:**

| Setting | Prefill 16K / 64K / 128K | Decode c8 at 16K / 64K / 128K | Decode c16 | KV tokens |
|---|---|---|---|---|
| Base settings | 1,878 / 1,942 / 1,934 | 109 / 112 / 109 | not admitted (8 requests) | 4,126,165 |
| Interleave 16 | 1,775 / 1,894 / 1,894 | 110 / 99 / 105 | not admitted | not recorded |
| Interleave 64 | 1,845 / 1,911 / 1,917 | 115 / 105 / 111 | not admitted | not recorded |
| 16 requests, capture to 64 tokens | 1,829-1,869 / 1,901-1,932 / 1,905-1,922 (2 runs) | 104-105 / 107-108 / 105-110 | 142-150 | not recorded |
| 16 requests + two parallel prefills | 1,849 / 1,898 / 1,902 | 105 / 105 / 104 | 142-148 | not recorded |
| 16 requests + mHC row ownership | 1,885 / 1,976 / 1,973 | 115 / 108 / 107 | 142-145 | 4,126,165 |
| 16 requests + mHC + KDA coalescing | 1,864-2,010 / 1,967-1,992 / 1,967-1,984 (2 runs) | 106-114 / 107-111 / 102-104 | 140-146 | 3,835,395 |

**The installed settings,** native Windows client: `sparkring install` of this profile with mHC and KDA
coalescing on, 16 requests and `--reasoning-effort high`.

| Context | Prefill tok/s | Decode tok/s c1 / c4 / c8 / c16 | Steps/s c1 / c4 / c8 / c16 |
|---|---|---|---|
| 16K | 2,217 | 39 / 73 / 111 / 151 | 15.4 / 30.1 / 43.0 / 58.8 |
| 64K | 2,174 | 37 / 74 / 113 / 157 | 15.4 / 30.2 / 43.6 / 61.1 |
| 128K | 2,151 | 38 / 76 / 110 / 152 | 15.4 / 30.7 / 43.0 / 60.1 |

The installed base settings measured earlier the same day with the same client, at the template's `max`
thinking default:
- prefill 1,997 / 2,057 / 2,040 tok/s;
- steps/s at c1 / c4 / c8 of 15.4 / 31.4 / 42.8 at 16K, 15.5 / 31.1 / 44.7 at 64K, and 15.1 / 30.9 / 43.8 at
  128K.

The coding peak was 49 tok/s on the installed settings, against 50.

## Conclusions

- **16 requests:** raising the request limit from 8 to 16, with CUDA graphs captured for every multiple of
  4 tokens up to 64, adds a c16 level of 140-157 tok/s aggregate. c1-c8 are unchanged within run-to-run
  variation, which reached 8 % in single c4-c8 cells and 16K prefill samples.
- **mHC row ownership:** about +2 % prefill at 64K and 128K against the 16-request settings without it.
- **KDA prefill coalescing:** on top of mHC, prefill changed by -1 % to +1 % (within variation), while the KV
  cache fell by 7 %, from 4,126,165 to 3,835,395 tokens. The profile therefore leaves it off.
- **Interleave 16 or 64, two parallel prefills:** no gain. Two parallel prefills raised mean time to first
  token at c16 from 19-23 s to 27-33 s.
- **The installed settings against the installed base settings:**
  - prefill is 5-11 % higher at 16K-128K;
  - engine steps/s at c1-c8 agree within 1-3 %;
  - decode tok/s at c4 is 7-14 % lower, because the accepted tokens per step fell with the `high` thinking
    default (2.4-2.6 against 2.6-2.75 at `max`), not because steps were slower.
- These results come from settings with KDA coalescing on. With it off, the profile's settings were measured
  only as the hand-made "16 requests + mHC" row.
