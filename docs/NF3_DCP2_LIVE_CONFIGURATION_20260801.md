# NF3 DCP2 live configuration variant

## Scope

This document records a **public-functional-lane, live-validated configuration
variant** exercised on four directly cabled DGX Sparks on 2026-08-01. It is an
exact clone of the live NF3 configuration in
[`NF3_LIVE_CONFIGURATION_20260731.md`](NF3_LIVE_CONFIGURATION_20260731.md),
apart from the enumerated DCP/capacity changes below and removal of one invalid
inherited environment value before vLLM starts.

The machine-readable delta is
[`docs/configurations/glm52-nf3-live-dcp2-20260801.json`](configurations/glm52-nf3-live-dcp2-20260801.json).
This is a configuration and correctness result, not a performance benchmark.

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

The first failed launch was not a DCP2 failure. The old container definition
carried `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=`. Current vLLM attempts to parse
any present value as an integer, so the empty value failed during worker
initialization. The accepted launch removes the variable with
`env -u VLLM_PREFIX_CACHE_RETENTION_INTERVAL` before importing vLLM. The
original DCP4 containers were retained, stopped, as a rollback path.

## Evidence boundary

A direct API request completed correctly. Its ten-second server window showed
healthy speculative acceptance, but that window is deliberately not reported
as decode performance. A compact external harness run was stopped after its
full-stream completion policy made it unsuitable as a quick smoke test. DCP2
throughput therefore remains to be measured with a bounded, declared benchmark
matrix before any speed comparison with DCP4 is published.
