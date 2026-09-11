# DeepSeek-V4.1-Flash four-Spark cycle profile

## Status

**Implemented and live-benchmarked on one private four-Spark cycle; not qualified.** The image
is a local build from pinned sources with no published digest, and seven upstream-pending
patches are bind-mounted over it. Deploy with the [quickstart](../DEEPSEEK_V41_FLASH_QUICKSTART.md).

## Serving contract

| Setting | Value |
|---|---|
| Image | built by `runtime/deepseek-v41-gb10/build-image.sh`; recorded build `sha256:af86a3d2bb0d267faa7f31777cdbe855addc1348f0b9f8323016ebf17d3dae3c` ([receipt](../../runtime/deepseek-v41-gb10/image-receipt.json)) |
| vLLM / FlashInfer | `vllm-project/vllm` `dsv41-feat` @ `e47aa780…` on nightly `8a728663…`; FlashInfer `07869c61…` (0.7.0rc1) |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40aa45a94ad051997016db3960a90277`, stock, on every rank's NVMe |
| Parallelism | TP4 across a four-Spark cycle, `--nnodes 4`, `mp` executor |
| Loader | `--load-format safetensors` |
| Request limit / sequences / scheduler | 1,048,576 tokens / 8 / 8,192 (16 sequences probed as an admission option) |
| Memory | `--gpu-memory-utilization 0.83`, `--block-size 128`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`; Engram tables on disk (`--engram-config '{"cpu_offload": false}'`, `DSV41_ENGRAM_DISK=1`, 64 reader threads, `DSV41_ENGRAM_BALANCED=1`, packed single-read shards) |
| Speculation | DSpark, 5 tokens, greedy draft, block rejection, adaptive verification off |
| Graphs | `FULL_AND_PIECEWISE`, exact capture sizes, `VLLM_USE_BREAKABLE_CUDAGRAPH=1` |
| Parsers | `--tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41`; thinking off by default |
| Multimodal | 4 images per prompt, 1 GiB processor cache |
| Transport | SparkRing patched NCCL 2.30.7 preloaded; two RoCE devices, subnet-aware routing, `NCCL_SWITCHLESS_RING_ONLY=1`, 4 channels |
| API model name | `deepseek-v4.1-flash` |
| Auth | optional `API_KEY_FILE` (one bearer key per line → `--api-key K1 K2 …`); `/health` keyless |

## Evidence boundary

The [benchmark record](../../performance/records/deepseek-v41-flash/cycle-tp4-dspark5-graphs-20260910.md)
describes one four-Spark cycle measured on 2026-09-10/11, with all ranks
rebooted before each configuration:

| Measured configuration | Memory and KV capacity | Bounded results |
|---|---|---|
| Text-only eager, 131K request limit, speculation disabled | 78.79 GiB per rank; 1,687,422 KV tokens | Correct greedy output; 14.5–14.8 tok/s |
| 300K request limit, memory utilization 0.80, probabilistic draft, 32 Engram threads | 85.71 GiB consumed and 15–16 GiB available per rank; 1,171,588 KV tokens | 131K/262K retrieval, vision and tool checks; 20-minute C8 soak, 648 requests, zero failures or hangs |
| 430,080-token request limit, settings pinned in the benchmark record | 13–15 GiB available per rank; 2,182,642 KV tokens, or 5.07 full-length requests | 400K retrieval pass; decode within the 300K configuration's ±5% run-to-run band; six-hour C8 soak, 1,417 waves and 11,336 requests, zero failures or hangs, stable memory |

These results are evidence for the recorded image identity and checkpoint revision on that
cycle. They do not qualify a different build, revision, topology or request shape, and they do
not evaluate output quality beyond needle recall and the end-to-end checks. The default context is 1,048,576 tokens; the linked record measures limits through
430,080 tokens, including a 400K needle pass.
