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

The two topologies differ in exactly three places: the parallelism flags, the
per-rank key-value reservation and request length that follow from how much of
the model each rank holds, and the transport half of the environment template.
Everything else — image, entrypoint, checkpoint, speculation, parsers,
verification — is identical.

| | Two-Spark pair | Four-Spark cycle |
|---|---|---|
| Ranks | 0-1 | 0-3 |
| Cabling | one DAC, cage 0 to cage 0 | four DACs as `0-1-2-3-0` |
| Weight share per rank | one half | one quarter |
| `--tensor-parallel-size` / `--nnodes` | 2 | 4 |
| `--kv-cache-memory-bytes` | 10737418240 (10 GiB) | 34359738368 (32 GiB) |
| `--max-model-len` | 131072 | 524288 |
| Environment template | `deepseek-v4-flash-0731-pair.env.example` | `deepseek-v4-flash-0731.env.example` |

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
  --max-model-len 131072 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.70 \
  --kv-cache-memory-bytes 10737418240 \
  --kv-cache-dtype fp8_ds_mla \
  --tokenizer-mode deepseek_v4 \
  --kernel-config '{"enable_cutedsl_warmup": false}' \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --speculative-config '{"method": "dspark",
    "num_speculative_tokens": 5, "moe_backend": "b12x"}' \
  --served-model-name deepseek-v4-flash-0731 \
  $([ "$RANK" -eq 0 ] && echo "--host 0.0.0.0 --port 8000" || echo "--headless")
```

Two ranks hold half the weights each rather than a quarter, so less memory
remains for the key-value reservation than on a cycle. The 10 GiB reservation
and 131,072 request length above are the documented pair values. Raising either
without measuring free memory after load risks a rank being killed during
startup; on this platform that failure can leave the host answering ICMP while
refusing to complete any new connection, recoverable only by a power cycle.

### Four-Spark cycle

Set `RANK` to `0`, `1`, `2`, or `3` on the corresponding host. Rank 0 serves the
API; the other ranks must use `--headless`.

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
  --max-model-len 524288 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.70 \
  --kv-cache-memory-bytes 34359738368 \
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
substitute generic `fp8`. `--gpu-memory-utilization 0.70` is the documented
GB10 value for both launches.

## 3. Verify rank 0

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

## Evidence boundary

Each topology carries its own evidence, recorded in its serving contract.

The two-Spark launch was exercised on 2026-08-21 on two directly cabled Sparks:
both ranks rendezvoused, a deterministic completion, an emitted tool call and a
34-tool chat request returned coherent output with no leaked markers, and DSpark
speculation ran at depth 5 with a 2.76 mean acceptance length. It mounted only
the checkpoint and a cache directory, so the published image needs no source
overlay.

The four-Spark launch was exercised on 2026-08-21 against the same pinned
image: all four ranks rendezvoused across the cycle, each loading 40.82 GiB of
weights, the engine reported a 4,382,668-token key-value pool at 8.36x maximum
concurrency, a deterministic completion and an emitted tool call were correct,
and DSpark speculation ran at depth 5 with a 3.06 mean acceptance length. It
mounted only the checkpoint and a cache directory.

Performance observations in either topology are not qualified measurements. The
normal launch uses patched NCCL from the environment template; SIRCL width-4096
graph collectives are research-only and are not part of this quickstart. See
[the profile record](profiles/DEEPSEEK_V4_FLASH_0731.md) and
[results](RESULTS.md).
