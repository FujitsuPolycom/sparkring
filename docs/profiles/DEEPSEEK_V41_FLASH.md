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
| Request limit / sequences / scheduler | 300,000 tokens / 8 / 8,192 |
| Memory | `--gpu-memory-utilization 0.80`, `--block-size 128`; Engram tables on disk (`--engram-config '{"cpu_offload": false}'`, `DSV41_ENGRAM_DISK=1`) |
| Speculation | DSpark, 5 tokens, probabilistic draft, block rejection, adaptive verification off |
| Graphs | `FULL_AND_PIECEWISE`, exact capture sizes, `VLLM_USE_BREAKABLE_CUDAGRAPH=1` |
| Parsers | `--tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41`; thinking off by default |
| Multimodal | 4 images per prompt, 1 GiB processor cache |
| Transport | SparkRing patched NCCL 2.30.7 preloaded; two RoCE devices, subnet-aware routing, `NCCL_SWITCHLESS_RING_ONLY=1`, 4 channels |
| API model name | `deepseek-v4.1-flash` |

## Evidence boundary

One four-Spark cycle, 2026-09-10, all ranks rebooted before each boot. Text-only eager boot:
78.79 GiB per rank, KV 1,687,422 tokens at 131K, correct greedy output, 14.5–14.8 tok/s without
speculation. Serving-shape boot: 85.71 GiB consumed per rank, KV 1,171,588 tokens at 300K,
15–16 GiB MemAvailable per rank while serving; benchmark, needle (131K, 262K), vision and
tool-calling checks and a 20-minute c=8 soak (648 requests, zero failures or hangs) recorded in
the [benchmark record](../../performance/records/deepseek-v41-flash/cycle-tp4-dspark5-graphs-20260910.md).

These results are evidence for the recorded image identity and checkpoint revision on that
cycle. They do not qualify a different build, revision, topology or request shape, and they do
not evaluate output quality beyond needle recall and the end-to-end checks. 1M context was not
run on this profile.
