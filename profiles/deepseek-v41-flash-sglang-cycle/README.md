# DeepSeek-V4.1-Flash on SGLang

Checkpoint: [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), DeepSeek's published weights.

Run DeepSeek-V4.1-Flash across four DGX Sparks using SGLang and the pinned Mia
adapter. This is a separate runtime from the [vLLM profile](../deepseek-v41-flash-cycle/README.md).
The recipe's fallback reference identifies that manual alternative; the
launcher does not automatically switch runtimes after a failure.

| Setting | Default |
|---|---|
| Parallelism | TP4/EP4 on a direct four-Spark cycle |
| Request context | 262,144 tokens |
| Prefill chunk / concurrent requests | 4,096 tokens / 8 |
| Prefill admission | One free slot (`MIN_FREE_SLOTS_DELAY=1`) |
| Shared token budget | 1,500,000 requested; allocation is runtime-dependent |
| Speculation | DSpark block five, verify-all without SPS/STS tables |
| Authentication | Required private file with distinct bearer keys |
| Image | Local [standalone build](../../runtime/deepseek-v41-sglang/pins.json) or [shared SparkRing composition](../../runtime/deepseek-v41-sglang/README.md#shared-sparkring-image) |

## Build, prepare and launch

Follow the [SGLang runtime guide](../../runtime/deepseek-v41-sglang/README.md#build-and-prepare)
to build one ARM64 image, distribute its exact ID, and prepare a private rank
environment. The SGLang Engram layout is incompatible with the vLLM packed files;
use separate output directories. The guide also identifies the required patched
NCCL library: standalone mode mounts it over the SGLang library, while the
shared composition bundles it. Both modes select SGLang's own library; do not
substitute the vLLM preload procedure.

The launcher passes `--min-free-slots-delay 1`, allowing a waiting prefill to
enter when one request slot is available. This disables SGLang's DFlash-family
admission batching, which otherwise waits for two free slots at eight concurrent
requests. Existing rank files inherit this default; set `MIN_FREE_SLOTS_DELAY`
explicitly to a larger positive integer to batch admissions. SGLang caps the
threshold at the maximum running-request count. This does not enable mixed
prefill/decode execution or change context and KV limits.
Set `MIN_FREE_SLOTS_DELAY=0` to omit the flag and use SGLang's automatic policy.

The [single-slot admission test](../../performance/records/deepseek-v41-flash/sglang-single-slot-admission.md)
records a bounded C8 latency/throughput comparison on a separate combined image.
It is not performance qualification of every image built from the public pins.

From the repository root, inspect the selected configuration and offline plan:

```bash
python scripts/profiles.py resolve deepseek-v41-flash-sglang-cycle
python scripts/launch.py deepseek-v41-flash-sglang-cycle -- --check /private/rank-0.env
```

The shared launcher prints the adapter command; arguments after `--` belong
to the SGLang adapter. Check each rank's environment file before startup. Follow
[preparation and startup](../../runtime/deepseek-v41-sglang/README.md#serve-and-validate)
for the explicit `--prepare`, `--pack`, and `--run` actions. Start workers 3, 2,
1, then rank 0. Each adapter invocation checks only the host on which it runs.

## Evidence and limits

Status: **Development**. The [controlled prefill record](../../performance/records/deepseek-v41-flash/sglang-decoder-replay-20260911.md)
uses the recipe's 262K context setting. The [six-hour streaming record](../../performance/records/deepseek-v41-flash/sglang-soak-20260912.md)
used a 430,080-token context setting and an actual shared pool of 1,499,904
tokens; it does not change these defaults or qualify an arbitrary rebuilt image.

Decoder-tail replay changes late-layer local attention visibility and is not
proven equivalent to full prefill. Report its quality and workload limits
alongside performance measurements. This profile does not include SparkCache or an unattended
recovery service. The source contribution and its qualification records originate
from [PR #267](https://github.com/FujitsuPolycom/sparkring/pull/267).

The [2026-09-17 record](../../performance/records/deepseek-v41-flash/sglang-79f656a6-nvfp4-20260917.md)
covers the adapter commit `79f656a6` (completion cap, loop abort), the admission-delayer and
token-pool settings, a rejected 1024-token prefill chunk, and the `nvidia/DeepSeek-V4.1-Flash-NVFP4`
checkpoint variant with its distribution comparison; the [runtime guide](../../runtime/deepseek-v41-sglang/README.md#serving-controls)
lists the corresponding optional settings. The recipe defaults are unchanged.

The contributor's [655360-context report](https://github.com/FujitsuPolycom/sparkring/pull/267#issuecomment-5653279829)
uses additional SGLang memory and execution overlays from #39187 and #39068.
The [shared composition](../../runtime/deepseek-v41-sglang/README.md#shared-sparkring-image)
includes those execution changes and the bounded-memory backport. Its launcher
admits the exact composition before running the documented 655360-context
alternative. The standalone builder does not include them. The recipe retains
262K. The [combined-image record](../../performance/records/deepseek-v41-flash/sglang-shared-bounded-prefill.json)
covers 11 successful API checks, including exact retrieval from 640K prompt
tokens; it does not qualify vision, concurrent long prompts or unattended recovery.
[Dual-domain NCCL results](../../performance/records/transport/nccl-dual-domain-deepseek.md)
also describe a separate transport configuration, not this adapter's defaults.
