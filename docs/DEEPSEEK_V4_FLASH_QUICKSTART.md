# DeepSeek-V4-Flash-0731 Spark quickstart

Deploy `deepseek-ai/DeepSeek-V4-Flash-0731` across directly cabled DGX Sparks
in either of two topologies: **two tensor-parallel ranks on a cabled pair**, or
**four on a cycle**. Both profiles are **implemented** and neither is
qualified. No checkpoint revision is pinned in this repository; record the
revision and file hashes used by an operator deployment.

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
| `--kv-cache-memory-bytes` | 12884901888 (12 GiB) | 17179869184 (16 GiB) |
| Resulting pool / concurrency at 1M | 1,139,967 tokens, 1.09x | 1,519,925 tokens, 1.45x |
| Context / Seq / Batch | 1M , 32 , 8192 | 1M , 32 , 8192 |
| Free memory per node while serving | 8-10 GB | ~50 GB |
| Environment template | `deepseek-v4-flash-0731-pair.env.example` | `deepseek-v4-flash-0731.env.example` |

`--max-model-len 1048576`, `--max-num-seqs 32` and `--max-num-batched-tokens
8192` are the same in both.

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
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.70 \
  --kv-cache-memory-bytes 12884901888 \
  --kv-cache-dtype fp8_ds_mla \
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
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.70 \
  --kv-cache-memory-bytes 17179869184 \
  --kv-cache-dtype fp8_ds_mla \
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

**Set `--max-num-batched-tokens` explicitly.** Left unset, the engine derives a
scheduled-token budget from the speculation settings and warns that it is
suboptimal: at 32 sequences and speculation depth 5 it derived 1,920 tokens.
Setting 8192 raises the budget to 7,936 and measured **+51% aggregate
throughput at 32 concurrent requests** on a pair, with single-stream and
8-stream throughput unchanged. A small budget only bites once enough sequences
compete for it.

**A longer request limit is close to free; a larger pool is not.** This model
has bounded cache groups whose cost per sequence is fixed, so per-token pool
cost falls as the request limit rises — about 18.5 KB per token at a
131,072-token limit and about 6 KB at 1,048,576. Raising `--max-model-len` from
131,072 to 1,048,576 on a pair, changing nothing else, cost no additional
memory.

**Sequence count and speculation depth inflate the per-request floor.** The
engine refuses to start unless the pool can hold one full-length request. On a
pair at 1,048,576 tokens that floor was about 6.2 GiB at 32 sequences with the
derived budget, and **11.04 GiB at 64 sequences with an 8192 budget** — the same
context, nearly twice the floor. Raising sequences therefore forces a larger
reservation, which buys less pool per byte: 64 sequences at 16 GiB produced a
*smaller* pool (1,519,925 tokens) than 32 sequences at 12 GiB relative to the
memory spent, while measuring within 2% on every concurrency cell. Prefer 32.

**`--kv-cache-memory-bytes` disables the memory-utilization guard.** The engine
states this explicitly: it "skipped memory profiling. This does not respect the
`gpu_memory_utilization` config." `--gpu-memory-utilization 0.70` is therefore
**not** a safety ceiling in either launch above — it is inert whenever an
explicit byte count is given. Nothing will stop an oversized reservation from
exhausting the node.

Check free memory after a launch rather than trusting the flag. A pair serving
this configuration reports 8-10 GB free per node; a cycle reports about 50 GB.
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
       "max_tokens":16,"temperature":0}'
```

Check rank logs for a successful rendezvous across every rank and for DSpark
metrics. `Mean acceptance length` above one confirms speculative decoding is
active, and `GPU KV cache size` reports the token capacity the reservation
bought.

## Measured

Client-observed aggregate throughput, 256-token prompts, 512-token
generations, both topologies on the settings above:

| Concurrent requests | Two-Spark pair | Four-Spark cycle |
|---:|---:|---:|
| 1 | 36.7 tok/s | 56.5 tok/s |
| 8 | 123.8 tok/s | 177.5 tok/s |
| 16 | 173.4 tok/s | 256.1 tok/s |
| 32 | 237.7 tok/s | 347.5 tok/s |

Doubling the node count buys about 1.45x across the whole ladder rather than
2x. Decode is latency-bound on collectives, not bandwidth-bound: the model
issues two all-reduces per layer across 43 layers, about 688 KB per token in
total, which is a fraction of a percent of a 200 Gb/s link, but each token
waits on roughly 86 synchronous round-trips that cannot overlap, because
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

## Evidence boundary

Each topology carries its own evidence, recorded in its serving contract.

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

Throughput figures above are client-observed on the stated prompt and
generation lengths. They are not qualified measurements, and no output-quality
measurement of any kind is recorded for either topology. The normal launch uses
patched NCCL from the environment template; SIRCL width-4096 graph collectives
are research-only and are not part of this quickstart. See
[the profile record](profiles/DEEPSEEK_V4_FLASH_0731.md) and
[results](RESULTS.md).
