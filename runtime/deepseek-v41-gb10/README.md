# DeepSeek-V4.1-Flash runtime for GB10 (image builder)

Status: **implemented builder**. The recorded local image has bounded serving
measurements on one four-Spark cycle; rebuilt images are not qualified.
This directory builds one ARM64 image that serves `deepseek-ai/DeepSeek-V4.1-Flash` on
GB10 with the vLLM `dsv41-feat` branch and FlashInfer 0.7.0rc1. It contains no model
weights and publishes no image: build it yourself with `build-image.sh` and compare the
layer identities against [`image-receipt.json`](image-receipt.json).

This build selects the vLLM model implementation and FlashInfer dependencies
recorded below. Images selected by the DeepSeek-V4-Flash-0731 profile are a
separate runtime contract and are not interchangeable with this image.

## Image chain

| Build role and local tag suffix | Content | Constraint |
|---|---|---|
| vLLM parent | `vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c` (multi-arch, the merge-base of `dsv41-feat`) | binary-compatible with the branch's Python tree |
| FlashInfer source build (`fi3`) | FlashInfer `07869c61ba581e6d6b8ad8d142f4a6c89b707cc1` (v0.7.0rc1) built from source with pinned CUTLASS/CCCL/spdlog; the 0.6.18 jit-cache/cubin wheels removed | 0.6.18 has no SM120 sparse-MLA decode configuration for V4.1's top-k 1152 |
| MXFP8 GEMM precompile (`fi4`) | `mxfp8_gemm_cutlass_sm120` JIT module prebuilt for `sm_121a` | precompilation avoids seven CUTLASS translation units during serving startup |
| Sparse MLA precompile (`fi5`) | `sparse_mla_sm120` prebuilt under the exact runtime environment (`tools/prewarm5.py`) | cache keys include the nvcc flags; a debug prewarm is a runtime miss |
| Serving image (`overlay5`) | `vllm-project/vllm` `dsv41-feat` @ `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba` Python tree copied over site-packages, plus `_C_stable_libtorch` rebuilt for GB10 from that tree (stable-only CMake build, CUTLASS v4.7.1) | the branch's kernel changes live in that one extension |

`tools/verify5.py` reports the mxfp8 module as a "MISS" because `try_load()` consults only
the AOT directory; the JIT path loads both modules in well under a second with ninja
finding nothing to build. `build-image.sh` does not run it by default. The prewarm stage fails if the
sparse module raises during compilation or loading, or if the MXFP8 GEMM cache
is absent. A load exception is not accepted as proof of successful compilation.

## Build resources and reuse

Use a dedicated build workspace and stop model workloads before the full
build. The extension compile needs substantially more memory than the
FlashInfer stages. `WORK` defaults to `$HOME/deepseek-v41-build`; the script
manages its source checkout and intermediate files there.

The script selects 7 GiB, six CPUs and two jobs when a running container name
matches `SERVING_CONTAINER_RE` (default `^glm53-mtp3|^vllm_`); otherwise it selects
40 GiB, 16 CPUs and six jobs. This name check does not detect every GPU workload.
Set `MEM`, `CPUS`, `JOBS` and `MIN_AVAIL_GIB` explicitly when the defaults do not
fit the host. Memory headroom is checked before each stage.

FlashInfer stages use a watchdog for sustained cgroup memory saturation.
The extension stage has no equivalent watchdog. Selected failed stages retry
with one job; watchdog-terminated FlashInfer stages do not retry automatically.
`FI_ONLY=1` stops before the extension and serving-image stages.

```bash
FI_ONLY=1 bash runtime/deepseek-v41-gb10/build-image.sh
# Run the full build with model workloads stopped.
bash runtime/deepseek-v41-gb10/build-image.sh
docker save local/sparkring-deepseek-v41:overlay5 | ssh <rank> docker load
```

Existing local stage tags are reused based on their names. The build does not
revalidate every cached stage against a receipt. Use a dedicated `TAG` namespace
and retain the resulting image IDs; a tag alone does not attest its contents.
The [recorded build receipt](image-receipt.json) retains resource observations
and image identities for comparison, not a promise of identical rebuild IDs.

The receipt in `$WORK/receipt.txt` records base identity, source commits, layer identities,
the extension's SHA-256 prefix and the minimum host `MemAvailable` seen during the build.

## Patches (bind-mounted at launch, nothing baked)

The launcher mounts seven runtime patch files from [`patches/`](patches/).
The [MD5 manifest](patches/MD5SUMS) identifies their bytes, and the
[mount manifest](patches/mounts.txt) identifies each destination below `vllm/`.
The patch source is the MIT-licensed
[DeepSeek-V4.1 GB10 integration](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark);
Kai authored the SM12x page-size fixes. `engram.py` additionally implements
configurable balanced hash-column assignment and packed single-read shards.
These files are mounted at launch, not baked into the serving image.

| file | mounted over `vllm/…` | what it fixes |
|---|---|---|
| `engram.py`, `weight_utils.py` | `models/deepseek_v4_1/common/engram.py`, `model_executor/model_loader/weight_utils.py` | the two Engram tables stay in the safetensors shards; rows are `preadv`'d on demand and dequantized on the CPU; includes the rank-offset fix |
| `engram.py` (additions) | same | `DSV41_ENGRAM_BALANCED=1`: rank r owns hash columns c with c % tp == r (two heads of each n-gram order) instead of six columns of one order, all-gather permuted back — the stock split made rank 3 read 5× the rows of rank 0 per prefill chunk and stall the ring; `DSV41_ENGRAM_PACKED_DIR`: sparse per-layer shard with weight+scale adjacent (264 B/row, `tools/pack_engram_rows.py`), one `preadv` per row; manifest-checked, falls back with a warning |
| `model_state.py` | `models/deepseek_v4_1/nvidia/model_state.py` | Engram rows staged in `prepare_inputs` so the decode step is CUDA-graph capturable |
| `attention.py`, `flashinfer_sparse.py`, `sparse_swa.py` | `models/deepseek_v4_1/…`, `v1/attention/backends/mla/sparse_swa.py` | SM12x page sizes: 64 compressed states per page, 64-token SWA backend, indexer cache 64 states per page (DeepGEMM paged MQA logits accepts 32 or 64) |
| `sparse_attn_indexer.py` | `model_executor/layers/sparse_attn_indexer.py` | decode top-k via `top_k_per_row_decode`; `persistent_topk` oversubscribes GB10's 48 SMs and needs 128 KB of shared memory |

Why the Engram tables cannot stay in memory on GB10: vLLM's default `cpu_offload=True`
pins them in host memory, and on a unified-memory device that is the same pool the GPU
allocates from. Row-sharded, the two FP8 tables are 47 GiB per rank on top of ~71 GiB of
other weights against 121.7 GiB visible. With the tables on NVMe a rank loads 78.8 GiB
(text-only) or 81.6 GiB (with the DSpark draft layers and vision encoder).

## Tools

- `tools/prewarm5.py`, `tools/verify5.py` — FlashInfer prewarm/verify (tonyd2wild).
- `tools/pack_engram_rows.py --model-dir … --out-dir /cache/engram-packed --tp 4 --rank <r> --balanced` — builds this
  rank's packed Engram shards (run once per rank inside the image, CPU only, ~9 min; 48 GB allocated per rank as sparse
  101 GB files). `--contiguous` matches the stock head split.
- `tools/test_engram_packed.py` — CPU equality check: packed rows == two-read rows for random rows in the rank's ranges.

Packed output directories must be absent before packing. The helper refuses
an existing directory, including partial output, because the manifest does not
bind rows to a checkpoint revision or content hash. Preserve existing output
and select a fresh per-checkpoint directory; ordinary serving restarts do not
invoke the packer and can retain their existing directory.

The pinned loader verifies geometry and coverage, not checkpoint identity.
Select packed data only for the checkpoint used to produce it. Enforcing that
identity inside the loader requires a separately versioned patch and manifest
contract; this builder does not claim that migration is implemented.

## Launch

The deployment profile configures a 1,048,576-token request limit. Its recorded
capacity and request measurements used limits through 430,080 tokens; the
configured default is not evidence of a completed 1M-token request.


The image's own entrypoint is `vllm serve`. The profile launcher
[`scripts/deepseek_v41_cycle_serve.sh`](../../scripts/deepseek_v41_cycle_serve.sh) bind-mounts
the patches and SparkRing's patched NCCL, and passes the cycle transport environment. See
[the quickstart](../../docs/DEEPSEEK_V41_FLASH_QUICKSTART.md).
