# DeepSeek-V4.1-Flash runtime for GB10 (image builder)

Status: **research-only builder; live-benchmarked on one private four-Spark cycle, not qualified.**
This directory builds one ARM64 image that serves `deepseek-ai/DeepSeek-V4.1-Flash` on
GB10 with the vLLM `dsv41-feat` branch and FlashInfer 0.7.0rc1. It contains no model
weights and publishes no image: build it yourself with `build-image.sh` and compare the
layer identities against [`image-receipt.json`](image-receipt.json).

The `gb10-vllm-serving` lineage used by the other DeepSeek profiles cannot serve this
model: it has no `deepseek_v41` model, and B12X has no CSA2, Engram, mHC or block-5
DSpark kernels. This is the first SparkRing profile built on a stock upstream vLLM image.

## Image chain

| layer | content | why |
|---|---|---|
| base | `vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c` (multi-arch, the merge-base of `dsv41-feat`) | binary-compatible with the branch's Python tree |
| `fi3` | FlashInfer `07869c61ba581e6d6b8ad8d142f4a6c89b707cc1` (v0.7.0rc1) built from source with pinned CUTLASS/CCCL/spdlog; the 0.6.18 jit-cache/cubin wheels removed | 0.6.18 has no SM120 sparse-MLA decode configuration for V4.1's top-k 1152 |
| `fi4` | `mxfp8_gemm_cutlass_sm120` JIT module prebuilt for `sm_121a` | its runtime compile (seven CUTLASS translation units) exhausted host memory on a serving fleet |
| `fi5` | `sparse_mla_sm120` prebuilt under the exact runtime environment (`tools/prewarm5.py`) | cache keys include the nvcc flags; a debug prewarm is a runtime miss |
| final | `vllm-project/vllm` `dsv41-feat` @ `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba` Python tree copied over site-packages, plus `_C_stable_libtorch` rebuilt for GB10 from that tree (stable-only CMake build, CUTLASS v4.7.1) | the branch's kernel changes live in that one extension |

`tools/verify5.py` reports the mxfp8 module as a "MISS" because `try_load()` consults only
the AOT directory; the JIT path loads both modules in well under a second with ninja
finding nothing to build. `build-image.sh` does not run it by default.

## Building on a node that is still serving

Every compile runs inside a cgroup (`--memory`, `--memory-swap`, `--cpus`) and the script
waits for host `MemAvailable` headroom before each step, retries at one job, and kills a
step whose cgroup sits at its memory ceiling for five minutes (reclaim thrash, not an OOM
kill). Measured on GB10, 2026-09-10:

- FlashInfer 0.7.0rc1 from source: 26 min under a 7 GiB cgroup.
- `mxfp8_gemm_cutlass_sm120` prebuild: OOM-killed at two jobs under 7 GiB, succeeded at one job in 4 min.
- `sparse_mla_sm120` prebuild: 34 s.
- `_C_stable_libtorch`: one CUTLASS translation unit drives a single `cicc` past 7 GiB and
  the step stalls; with the node idle it compiles in 7.5 min at six jobs inside a 40 GiB cgroup.

When no serving container is running the script sizes itself to 40 GiB / 16 CPUs / 6 jobs;
otherwise 7 GiB / 6 CPUs / 2 jobs, which builds the FlashInfer layers but not the extension.
`FI_ONLY=1` stops after the FlashInfer layers so the extension can be built later.

```bash
# on an idle GB10 (or a serving one for the FlashInfer layers only)
FI_ONLY=1 bash runtime/deepseek-v41-gb10/build-image.sh
bash runtime/deepseek-v41-gb10/build-image.sh          # extension + final layer
docker save local/sparkring-deepseek-v41:overlay5 | ssh <rank> docker load
```

The receipt in `$WORK/receipt.txt` records base identity, source commits, layer identities,
the extension's SHA-256 prefix and the minimum host `MemAvailable` seen during the build.

## Patches (bind-mounted at launch, nothing baked)

[`patches/`](patches/) holds the seven files the launcher mounts over the image, byte-identical
to boot 9 of https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark (MIT; Kai
authored the SM12x page-size fixes), with their md5s in `patches/MD5SUMS` and the mount
manifest in `patches/mounts.txt`:

| file | mounted over `vllm/…` | what it fixes |
|---|---|---|
| `engram.py`, `weight_utils.py` | `models/deepseek_v4_1/common/engram.py`, `model_executor/model_loader/weight_utils.py` | the two Engram tables stay in the safetensors shards; rows are `preadv`'d on demand and dequantized on the CPU; includes the rank-offset fix |
| `model_state.py` | `models/deepseek_v4_1/nvidia/model_state.py` | Engram rows staged in `prepare_inputs` so the decode step is CUDA-graph capturable |
| `attention.py`, `flashinfer_sparse.py`, `sparse_swa.py` | `models/deepseek_v4_1/…`, `v1/attention/backends/mla/sparse_swa.py` | SM12x page sizes: 64 compressed states per page, 64-token SWA backend, indexer cache 64 states per page (DeepGEMM paged MQA logits accepts 32 or 64) |
| `sparse_attn_indexer.py` | `model_executor/layers/sparse_attn_indexer.py` | decode top-k via `top_k_per_row_decode`; `persistent_topk` oversubscribes GB10's 48 SMs and needs 128 KB of shared memory |

Why the Engram tables cannot stay in memory on GB10: vLLM's default `cpu_offload=True`
pins them in host memory, and on a unified-memory device that is the same pool the GPU
allocates from. Row-sharded, the two FP8 tables are 47 GiB per rank on top of ~71 GiB of
other weights against 121.7 GiB visible. With the tables on NVMe a rank loads 78.8 GiB
(text-only) or 81.6 GiB (with the DSpark draft layers and vision encoder).

## Launch

The image's own entrypoint is `vllm serve`. The profile launcher
[`scripts/deepseek_v41_cycle_serve.sh`](../../scripts/deepseek_v41_cycle_serve.sh) bind-mounts
the patches and SparkRing's patched NCCL, and passes the cycle transport environment. See
[the quickstart](../../docs/DEEPSEEK_V41_FLASH_QUICKSTART.md).
