# DeepSeek-V4-Flash-0731 four-Spark quickstart

Deploy `deepseek-ai/DeepSeek-V4-Flash-0731` as four tensor-parallel ranks on a
directly cabled DGX Spark cycle. The profile is **implemented** and is not
qualified. No checkpoint revision is pinned in this repository;
record the revision and file hashes used by an operator deployment.
The machine-readable serving contract is
[`recipes/deepseek-v4-flash-0731.json`](../recipes/deepseek-v4-flash-0731.json).

## 1. Prepare the ranks

Complete [the prerequisites](PREREQUISITES.md). Download the official FP8
checkpoint once, distribute identical bytes to all four ranks, and choose the
same container model mount on each host. The model contains 48 safetensors
shards totaling about 167 GB.

Pull the immutable ARM64 runtime image on every rank before launching any rank:

```bash
docker pull ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028
```

This is the `serving_image.manifest_digest` pinned by
[`runtime/faststart-lock.json`](../runtime/faststart-lock.json). The image
registers `DeepseekV4ForCausalLM`; override its GLM-specific entrypoint as shown
below.

Copy the tracked environment template to one local file per rank. Replace only
`<NCCL_SOCKET_IFNAME>` and `<RANK_FABRIC_IP>`. Even ranks use cage 0
(`enp1s0f0np0`) and odd ranks use cage 1 (`enp1s0f1np1`) for the cycle.

```bash
cp scripts/config/deepseek-v4-flash-0731.env.example /path/to/rank-0.env
cp scripts/config/deepseek-v4-flash-0731.env.example /path/to/rank-1.env
cp scripts/config/deepseek-v4-flash-0731.env.example /path/to/rank-2.env
cp scripts/config/deepseek-v4-flash-0731.env.example /path/to/rank-3.env
```

The template's `LD_PRELOAD`, `VLLM_NCCL_SO_PATH`, and `NCCL_*` values are
required by the published image. Do not use a GLM profile environment file.

## 2. Launch one rank per host

Set `RANK` to `0`, `1`, `2`, or `3` on the corresponding host. Set
`RANK0_FABRIC_ADDR` to rank 0's selected fabric address. Rank 0 serves the API;
the other ranks must use `--headless`.

```bash
docker run -d --name deepseek-v4-flash-r"$RANK" \
  --network host --ipc host --shm-size 16g --gpus all \
  --ulimit memlock=-1:-1 --device /dev/infiniband \
  -v /path/to/deepseek-v4-flash-0731:/models/deepseek-v4-flash-0731:ro \
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

`--kv-cache-dtype fp8_ds_mla` is required: it declares the model's MLA
key-value geometry and matches the allocated layout. Do not substitute generic
`fp8`. The 32 GiB reservation and 0.70 memory utilization are the documented
GB10 values for this launch.

## 3. Verify rank 0

Wait for the API health endpoint, then issue a deterministic chat request.

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731",
       "messages":[{"role":"user","content":"What is 17 * 23?"}],
       "max_tokens":16,"temperature":0}'
```

Check rank logs for successful four-rank rendezvous and DSpark metrics. `Mean
acceptance length` above one confirms speculative decoding is active.

## Evidence boundary

The four-Spark implemented launch exercised API health, chat completions, tool
calling, and DSpark speculation. Its performance observations are not qualified
measurements. The normal launch uses patched NCCL from the environment template;
SIRCL width-4096 graph collectives are research-only and are not part of this
quickstart. See [the profile record](profiles/DEEPSEEK_V4_FLASH_0731.md) and
[results](RESULTS.md).
