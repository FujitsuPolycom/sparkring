# Historical Aiden MXFP4/GPTQ lane

This page preserves the original public SparkRing reference lane:

- model:
  `aidendle94/GLM-5.2-MXFP4-Experts-GPTQ`
  at `46537e0e16fcd156627800139b41b9c497fc7ee2`;
- ARM64 base image:
  `aidendle94/sparkrun-vllm-ds4-gb10`
  at
  `sha256:93824a946f1f0ad0867132a2c3809e0e7d8bec6ab38e7d0ef9fc3046e11bc8c7`;
- topology: four directly cabled DGX Sparks, TP4/DCP4;
- KV format: `nvfp4_ds_mla` with per-token outer scaling;
- adaptive MTP depths 2/4, window 32.

It is retained for historical reproduction and comparison. It is no longer
the default deployment target. The current deployment lane is the
madeby561 NF3 hybrid documented in [the NF3 quickstart](../NF3_QUICKSTART.md).

## Coherent reference matrix

| Context | Cold prefill | C1 decode | C2 aggregate | C4 aggregate | C8 aggregate |
|---:|---:|---:|---:|---:|---:|
| 8K | 844 tok/s | 20.3 | 27.1 | 40.5 | 49.2 |
| 16K | 876 tok/s | 19.0 | 26.4 | 37.9 | 53.3 |
| 32K | 830 tok/s | 20.3 | 27.6 | 38.6 | 51.9 |
| 64K | 832 tok/s | 20.3 | 27.0 | 39.4 | 50.9 |
| 128K | 796 tok/s | 19.7 | 26.3 | 37.2 | 47.7 |

These results used the complete serving configuration described in
[RESULTS.md](../RESULTS.md). Concurrency numbers are aggregate throughput.

## Independent bring-up findings

An external user reproduced the eager TP4/DCP4 lane and reported
18.3 tok/s median on 500-token code completions after correcting six setup
problems. The findings are preserved because they remain useful regression
tests:

1. Pass the exact 78-layer `index_topk_pattern`; startup must report 57
   skipped sparse-indexer layers per rank.
2. Order transport peers by the native XOR schedule:
   `[rank ^ 1, rank ^ 3]`.
3. Set `VLLM_NVFP4_MLA_PER_TOKEN_SCALE=1`.
4. Remove the base image's inherited
   `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`; an empty string is not equivalent.
5. Override the base image's root-owned/offline Hugging Face settings when
   downloading as the host user.
6. Enable the GLM reasoning and tool-call parsers.

The public launcher retains fail-closed checks for these invariants even
though this model lane is historical.

## CUDA-graph finding

The independent reproduction generated correct output in eager mode but
returned a one-token `lock` response with the historical CUDA-graph profile.
DCP1 did not isolate the failure because TP4 and vocabulary graph paths
remained active.

Do not treat successful capture or `/health` as correctness. Any attempt to
revive this lane's graph mode must pass
[the CUDA-graph correctness gate](../CUDAGRAPH_CORRECTNESS_GATE.md).

## Why deployment moved to NF3

The NF3 lane has:

- a smaller checkpoint;
- faster measured reloads;
- a 511,488-token validated KV pool;
- dedicated SM121 NF3 kernels;
- bounded startup profiling;
- a pointer-stable 768 MiB workspace reserved before graph capture; and
- successful live overlap of a 512-token decode with an 18,562-token prefill.

The historical lane remains valuable evidence, but maintaining two default
deployment matrices made setup and validation unnecessarily fragile.
