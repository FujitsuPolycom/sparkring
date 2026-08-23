# DeepSeek-V4-Flash-0731 Spark quickstart

Deploy `deepseek-ai/DeepSeek-V4-Flash-0731` across directly cabled DGX Sparks
in either of two topologies: **two tensor-parallel ranks on a cabled pair**, or
**four on a cycle**. Both setups have **candidate** status. The pair has been
benchmarked; the cycle has not yet completed the same tests.

The checkpoint revision is not pinned. Record the revision and file hashes
used for each deployment.

The machine-readable serving contracts are
[`recipes/deepseek-v4-flash-0731-pair.json`](../recipes/deepseek-v4-flash-0731-pair.json)
for the pair and
[`recipes/deepseek-v4-flash-0731.json`](../recipes/deepseek-v4-flash-0731.json)
for the cycle.

Both topologies serve the checkpoint's full 1,048,576-token request length with
identical scheduler settings. They differ in the parallelism flags, the
key-value reservation each node can afford, and the transport half of the
environment template. Image, entrypoint, checkpoint, speculation, parsers and
verification are shared.

| | Two-Spark pair | Four-Spark cycle |
|---|---|---|
| Ranks | 0-1 | 0-3 |
| Cabling | one DAC, cage 0 to cage 0 | four DACs as `0-1-2-3-0` |
| Weights resident per rank | 80.97 GiB | 40.82 GiB |
| `--tensor-parallel-size` / `--nnodes` | 2 | 4 |
| `--kv-cache-memory-bytes` | 17179869184 (16 GiB) | 17179869184 (16 GiB) |
| 1M pool / concurrency | 2,198,756 tokens / 2.10x observed | not measured yet |
| Context / Seq / Batch | 1M, 32, 4096 | 1M, 32, 4096 |
| Free memory per node | 4.9–6.0 GiB observed, with swap in use | older setup measured ~50 GB; new setup not measured |
| Environment template | `deepseek-v4-flash-0731-pair.env.example` | `deepseek-v4-flash-0731.env.example` |

`--max-model-len 1048576`, `--max-num-seqs 32`, `--max-num-batched-tokens
4096`, `--block-size 256`, and the 16 GiB reservation are the settings used by
both base topologies.

Both commands explicitly enable asynchronous scheduling and retain complete
input-length reservation before admission:

```text
--async-scheduling --scheduler-reserve-full-isl
```

The running engine previously selected those values automatically. Spelling
them out prevents a future vLLM default from silently changing the profile.
Disabling full-input reservation is a separate research experiment because it
can improve admission latency while increasing later queueing, preemption, or
KV pressure under chunked prefill.

## 1. Prepare the ranks

Complete [the prerequisites](PREREQUISITES.md) for the topology you are
deploying, including the fabric routing and forwarding checks. On a cycle, a
rendezvous reaches ranks that are not directly cabled to each other only if
every node relays for its neighbours, and Docker's default `FORWARD` policy
blocks that relay.

Download the official FP8 checkpoint once, distribute identical bytes to every
rank, and choose the same container model mount on each host. The model
contains 48 safetensors shards totaling about 167 GB.

Pull the immutable ARM64 runtime image on every rank before launching any rank:

```bash
docker pull ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028
```

This is the `serving_image.manifest_digest` pinned by
[`runtime/faststart-lock.json`](../runtime/faststart-lock.json). The image
registers `DeepseekV4ForCausalLM`; override its GLM-specific entrypoint as shown
below.

Copy the environment template for your topology to one local file per rank.

For a pair, replace `<FABRIC_IFNAME>`, `<RANK_FABRIC_IP>`, and `<GID_INDEX>`.
Both ranks name the same interface when the pair is cabled cage 0 to cage 0:

```bash
cp scripts/config/deepseek-v4-flash-0731-pair.env.example /path/to/rank-0.env
cp scripts/config/deepseek-v4-flash-0731-pair.env.example /path/to/rank-1.env
```

For a cycle, replace `<NCCL_SOCKET_IFNAME>` and `<RANK_FABRIC_IP>`. Even ranks
use cage 0 (`enp1s0f0np0`) and odd ranks use cage 1 (`enp1s0f1np1`):

```bash
cp scripts/config/deepseek-v4-flash-0731.env.example /path/to/rank-0.env
cp scripts/config/deepseek-v4-flash-0731.env.example /path/to/rank-1.env
cp scripts/config/deepseek-v4-flash-0731.env.example /path/to/rank-2.env
cp scripts/config/deepseek-v4-flash-0731.env.example /path/to/rank-3.env
```

The templates' `LD_PRELOAD`, `VLLM_NCCL_SO_PATH`, and `NCCL_*` values are
required by the published image. Do not use a GLM profile environment file, and
do not use the cycle template on a pair: its transport settings name two host
channel adapters, enable subnet-aware routing, and skip tree connect, none of
which applies when the two ranks are directly adjacent.

The `<GID_INDEX>` placeholder in the pair template is the RoCE GID index for
that interface's IPv4 address. `show_gids` lists them. Do not copy a value from
another deployment; the correct index depends on the RoCE version and the
address configured on the host.

## 2. Launch one rank per host

Both launches mount a writable `/cache` directory. The environment sets
`XDG_CACHE_HOME=/cache/jit`, so without that mount every replaced container
recompiles its just-in-time kernels from scratch.

### Two-Spark pair

Set `RANK` to `0` or `1` on the corresponding host, and `RANK0_FABRIC_ADDR` to
rank 0's fabric address. Rank 0 serves the API; rank 1 must use `--headless`.

```bash
docker run -d --name deepseek-v4-flash-r"$RANK" \
  --network host --ipc host --shm-size 16g --gpus all \
  --ulimit memlock=-1:-1 --device /dev/infiniband \
  -v /path/to/deepseek-v4-flash-0731:/models/deepseek-v4-flash-0731:ro \
  -v /path/to/jit-cache:/cache \
  --env-file /path/to/rank-"$RANK".env \
  --entrypoint /opt/venv/bin/vllm \
  ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028 \
  serve /models/deepseek-v4-flash-0731 \
  --tensor-parallel-size 2 --nnodes 2 --node-rank "$RANK" \
  --master-addr "$RANK0_FABRIC_ADDR" --master-port 29500 \
  --distributed-executor-backend mp \
  --dtype bfloat16 \
  --max-model-len 1048576 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --async-scheduling --scheduler-reserve-full-isl \
  --gpu-memory-utilization 0.70 \
  --kv-cache-memory-bytes 17179869184 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --tokenizer-mode deepseek_v4 \
  --kernel-config '{"enable_cutedsl_warmup": false}' \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --speculative-config '{"method": "dspark",
    "num_speculative_tokens": 5, "moe_backend": "b12x"}' \
  --served-model-name deepseek-v4-flash-0731 \
  $([ "$RANK" -eq 0 ] && echo "--host 0.0.0.0 --port 8000" || echo "--headless")
```

### Four-Spark cycle

Set `RANK` to `0`, `1`, `2`, or `3` on the corresponding host. Rank 0 serves the
API; the other ranks must use `--headless`. The command is identical to the
pair's except for the parallelism flags and the key-value reservation:

```bash
docker run -d --name deepseek-v4-flash-r"$RANK" \
  --network host --ipc host --shm-size 16g --gpus all \
  --ulimit memlock=-1:-1 --device /dev/infiniband \
  -v /path/to/deepseek-v4-flash-0731:/models/deepseek-v4-flash-0731:ro \
  -v /path/to/jit-cache:/cache \
  --env-file /path/to/rank-"$RANK".env \
  --entrypoint /opt/venv/bin/vllm \
  ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028 \
  serve /models/deepseek-v4-flash-0731 \
  --tensor-parallel-size 4 --nnodes 4 --node-rank "$RANK" \
  --master-addr "$RANK0_FABRIC_ADDR" --master-port 29500 \
  --distributed-executor-backend mp \
  --dtype bfloat16 \
  --max-model-len 1048576 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --async-scheduling --scheduler-reserve-full-isl \
  --gpu-memory-utilization 0.70 \
  --kv-cache-memory-bytes 17179869184 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --tokenizer-mode deepseek_v4 \
  --kernel-config '{"enable_cutedsl_warmup": false}' \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --speculative-config '{"method": "dspark",
    "num_speculative_tokens": 5, "moe_backend": "b12x"}' \
  --served-model-name deepseek-v4-flash-0731 \
  $([ "$RANK" -eq 0 ] && echo "--host 0.0.0.0 --port 8000" || echo "--headless")
```

`--kv-cache-dtype fp8_ds_mla` is required in both topologies: it declares the
model's MLA key-value geometry and matches the allocated layout. Do not
substitute generic `fp8`.

## 3. Sizing: four coupled parameters, and one inactive guard

`--max-model-len`, `--kv-cache-memory-bytes`, `--max-num-seqs` and
`--max-num-batched-tokens` are not independent. Changing one alone will
usually fail or waste memory.

**Set `--max-num-batched-tokens` explicitly.** The normalized comparison target
uses 4096 in every DeepSeek base and SparkCache profile. A historical pair run
with 8192 measured higher C32 throughput, but it is not the default because it
would make the base and cache comparison change two variables. The 4096 target
requires a fresh C1/C2/C4/C8/C16/C32 capacity and throughput sweep.

**A longer request limit is close to free; a larger pool is not.** This model
has bounded cache groups whose cost per sequence is fixed, so per-token pool
cost falls as the request limit rises — about 18.5 KB per token at a
131,072-token limit and about 6 KB at 1,048,576. Raising `--max-model-len` from
131,072 to 1,048,576 on a pair, changing nothing else, cost no additional
memory.

**Sequence count and speculation depth inflate the per-request floor.** The
engine refuses to start unless the pool can hold one full-length request. The
32-sequence target retains the bounded sequence count while the pair reservation
increases to 16 GiB. The normalized pair reported a 2,198,756-token pool and
2.10x full-length capacity. A separate cycle gate must record its pool and free
memory before a concurrency campaign begins.

**`--kv-cache-memory-bytes` disables the memory-utilization guard.** The engine
states this explicitly: it "skipped memory profiling. This does not respect the
`gpu_memory_utilization` config." `--gpu-memory-utilization 0.70` is therefore
**not** a safety ceiling in either launch above — it is inert whenever an
explicit byte count is given. Nothing will stop an oversized reservation from
exhausting the node.

Check free memory after a launch rather than trusting the flag. The normalized
pair had only 4.9–6.0 GiB available per node and used swap. The roughly 50 GB
cycle observation came from a historical configuration, not the normalized
block-256, batch-4096 target.
A node driven to 2-3 GB free with swap in use is one long prefill away from
having its engine core killed, and on this platform severe memory exhaustion
can leave a host answering ICMP while refusing to complete any new connection,
recoverable only by a power cycle.

## 4. Verify rank 0

Wait for the API health endpoint, then issue a deterministic chat request.

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731",
       "messages":[{"role":"user","content":"What is 17 * 23?"}],
       "max_tokens":16,"temperature":1.0,"top_p":1.0}'
```

Check rank logs for a successful rendezvous across every rank and for DSpark
metrics. `Mean acceptance length` above one confirms speculative decoding is
active, and `GPU KV cache size` reports the token capacity the reservation
bought.

## Measured

The TP2/DCP1 setup above was benchmarked on two directly cabled DGX Sparks
without SparkCache. It remains **candidate** because the checkpoint revision is
not pinned and the full test set is not complete.

The serving profile was measured at temperature 1.0 and top-p 1.0.

| Context | Prefill tok/s | C1 | C2 | C4 | C8 | C16 | C32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 1,822 | 67.62 | 82.76 | 107.43 | 142.53 | — | — |
| 8K | 1,921 | 34.97 | 111.43 | 103.63 | 151.06 | — | — |
| 16K | 2,005 | 58.36* | 61.12 | 114.37 | 164.93† | 202.71 | 295.34 |
| 32K | 1,999 | 51.59‡ | 78.18 | 104.70 | 160.54 | — | — |
| 64K | 1,938 | 32.59 | 90.27 | 97.66 | — | — | — |
| 128K | 1,808 | 59.10 | 77.59 | — | — | — | — |

Decode values are aggregate generated tok/s. `*` and `‡` are N=5 means;
`†` is an N=3 mean. The
[full TP2 benchmark record](../performance/records/deepseek-v4-flash/normalized-tp2-base-20260822.md)
lists variability, exclusions, and the machine-readable summary. Synthetic
sustained text changes DSpark acceptance, so single observations should not be
read as transport or thermal verdicts.

The profile remains collective-latency-sensitive: the model issues two
all-reduces per layer across 43 layers, and each token waits on synchronous
round-trips that cannot fully overlap, because
decode is autoregressive: the input to step `N+1` includes the token produced
at step `N`.

Speculation amortises those round-trips. At a mean acceptance length of 3.06 on
a cycle, one forward pass yields about three tokens, so the collectives cost
roughly a third as much per output token. Raising acceptance is equivalent to
making collectives cheaper.

**Do not expect a faster all-reduce backend on this hardware.** The engine
reports selecting `PYNCCL` from the potential set `NCCL_SYMM_MEM`,
`QUICK_REDUCE`, `FLASHINFER`, `AITER_CUSTOM`, `CUSTOM`, `SYMM_MEM`, `PYNCCL`.
Every faster entry requires peer-accessible GPU memory on one host — NVLink or
PCIe peer-to-peer — or is ROCm-only. With one GPU per node communicating over
RoCE, `PYNCCL` is the correct selection rather than a fallback. Setting
`VLLM_ENABLE_PCIE_ALLREDUCE`, `VLLM_PCIE_ALLREDUCE_BACKEND=cpp` and
`VLLM_CPP_AR_1STAGE_NCCL_CUTOFF` on a cycle changed nothing: 56.0 against 56.5
tok/s at one request, 174.2 against 177.5 at eight. Those variables belong to
single-host multi-GPU profiles and are not worth carrying here.

## What these results cover

Each topology has separate results recorded in its serving contract.

The two-Spark launch was exercised on 2026-08-21 on two directly cabled Sparks:
both ranks rendezvoused, a deterministic completion, an emitted tool call and a
34-tool chat request returned coherent output with no leaked markers, and DSpark
speculation ran at depth 5 with a 2.76 mean acceptance length. It mounted only
the checkpoint and a cache directory, so the published image needs no source
overlay.

The four-Spark launch was exercised on 2026-08-21 against the same pinned
image: all four ranks rendezvoused, each loading 40.82 GiB of weights, a
deterministic completion and an emitted tool call were correct, and DSpark
speculation ran at depth 5 with a 3.06 mean acceptance length.

The historical throughput figures are client-observed on the stated prompt and
generation lengths. The normalized TP2 table is isolated-server sustained
decode and standalone prefill. Neither dataset measures output quality or
qualifies the setup for general use. The normal launch uses
patched NCCL from the environment template; SIRCL width-4096 graph collectives
are research-only and are not part of this quickstart. See
[the profile record](profiles/DEEPSEEK_V4_FLASH_0731.md) and
[results](RESULTS.md).

The offline-only [four-Spark SIRCL A/B plan](DEEPSEEK_V4_FLASH_SIRCL_AB.md)
validates both transport arms against this quickstart's machine-readable
recipe, applies the research experiment's 4,096-token batch budget to both
arms, and preserves its model, memory, scheduler-mode, and five-token DSpark
contract. The plan has no execution mode and does not promote SIRCL into this
quickstart.
