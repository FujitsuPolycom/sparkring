# NF3 DCP2 live configuration variant

## Scope

This document records a **public-functional-lane, live-validated configuration
variant** exercised on four directly cabled DGX Sparks on 2026-08-01. It is an
exact clone of the live NF3 configuration in
[`NF3_LIVE_CONFIGURATION_20260731.md`](NF3_LIVE_CONFIGURATION_20260731.md),
apart from the enumerated DCP/capacity changes below and removal of one invalid
inherited environment value before vLLM starts.

The machine-readable delta and observed benchmark evidence are
[`docs/configurations/glm52-nf3-live-dcp2-20260801.json`](configurations/glm52-nf3-live-dcp2-20260801.json).
This is a dated configuration snapshot, not the repository's reference
profile.

## Exact changes from the DCP4 snapshot

| Setting | DCP4 source | DCP2 variant |
|---|---:|---:|
| `--decode-context-parallel-size` | 4 | **2** |
| `--max-model-len` | 1,048,576 | **524,288** |
| `VLLM_SPARK_DCP_SIZE` | 4 | **2** |
| `VLLM_SPARK_MAX_MODEL_LEN` | 1,048,576 | **524,288** |
| `VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS` | 1,048,576 | **524,288** |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | present with an invalid empty value | **removed before vLLM import** |

The explicit KV allocation remained **9,000,000,000 bytes per rank**. It was
not cut to 4.5 GB: DCP2 itself halves aggregate capacity because each context
token is represented on two DCP owners instead of four.

Everything else remained pinned to the DCP4 snapshot, including the image ID,
model and MTP draft, TP4, maximum sequences 8, maximum batched tokens 4096,
adaptive MTP2/4, NVFP4-latent plus FP8-RoPE KV, fastsafetensors, prefix
caching, CUDA-graph buckets, SparkRing/SIRCL custom TP/all-gather/vocabulary
paths, and patched NCCL fallback.

## Observed live result

| Check | Observed result |
|---|---|
| rank-to-DCP mapping | TP0/DCP0, TP1/DCP1, TP2/DCP0, TP3/DCP1 |
| model endpoint | `GLM-5.2-NF3` |
| endpoint maximum model length | **524,288** |
| KV reservation | **8.38 GiB/rank** |
| reported KV capacity | **562,816 tokens** |
| maximum 524,288-token request concurrency | **1.07x** |
| piecewise graph capture | **15/15 completed** |
| full graph capture | **16/16 completed** |
| API health | HTTP 200 |
| correctness smoke | exact response `DCP2_OK` |
| all-rank startup state | running, no OOM |

The observed capacity is exactly half of the DCP4 snapshot's 1,125,632-token
capacity, as predicted from the unchanged per-rank KV allocation.

The first failed launch was not a DCP2 failure. The DCP4 container definition
carried `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=`. The deployed vLLM attempts to parse
any present value as an integer, so the empty value failed during worker
initialization. The accepted launch removes the variable with
`env -u VLLM_PREFIX_CACHE_RETENTION_INTERVAL` before importing vLLM. The
original DCP4 containers were retained, stopped, as a rollback path.

## Operator benchmark snapshot

After the accepted DCP2 startup, the operator ran the normal decode benchmark
against `GLM-5.2-NF3` and supplied the following rendered result. These are
the actual measured values from that run, preserved here as a known snapshot.

| Measurement | Context / concurrency | Result |
|---|---|---:|
| uncached prefill | 8,192 tokens | **630 tok/s** (13.02 s TTFT) |
| uncached prefill | 16,255 tokens | **652 tok/s** (24.92 s TTFT) |
| uncached prefill | 64,512 tokens | **654 tok/s** (98.69 s TTFT) |
| uncached prefill | 128,886 tokens | **644 tok/s** (200.16 s TTFT) |
| aggregate decode | 16K, C1 | **19.0 tok/s** |
| aggregate decode | 16K, C2 | **29.9 tok/s** |
| aggregate decode | 16K, C4 | **44.5 tok/s** |
| aggregate decode | 16K, C8 | **57.7 tok/s** |
| sequential coding peak | C1, three runs | **23.0 tok/s median**, 23.2 mean, 24.4 max, 0 CJK runs |

The benchmark screenshot does not include the complete command line, prompt
corpus, temperatures, or warmup policy. The figures are therefore an
operator-measured **DCP2 snapshot**, useful for comparison and regression
tracking but not a generalized published claim.

## Evidence boundary

A direct API request completed correctly. Its ten-second server window showed
healthy speculative acceptance, but that window is deliberately not reported
as decode performance. The benchmark snapshot above is the only performance
evidence recorded for this variant; a future fully declared matrix should be
used before making a DCP2-versus-DCP4 generalized claim.
