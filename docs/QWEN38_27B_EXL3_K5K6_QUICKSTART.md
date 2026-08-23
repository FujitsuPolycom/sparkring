# Qwen3.8-27B EXL3 K5/K6 four-Spark quickstart

Serve `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated` as four tensor-parallel
ranks on the direct-cable Spark cycle.

**Status: candidate; implemented and live-benchmarked.** The four-Spark
serving object started and served the pinned checkpoint. A bounded run recorded
startup, memory, key-value capacity, functional checks, and limited
performance. It is not a restart, sustained-load, or complete qualification
result.

The machine-readable settings are in
[`recipes/qwen38-27b-exl3-k5k6.json`](../recipes/qwen38-27b-exl3-k5k6.json).

| Setting | Four-Spark candidate |
|---|---|
| Model | `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated@ab3a91a13813df8096cb4c1d560ed3669035d0cf` |
| Topology | direct cycle `0-1-2-3-0` |
| Parallelism | TP4/DCP1, one process per Spark |
| Executor | multi-node `mp` |
| Request limit | 262,144 tokens |
| Maximum sequences | 64 |
| Scheduler budget | 8,192 tokens |
| Scheduling | chunked prefill, asynchronous, full-input-length reservation |
| Hybrid block geometry | request 16 attention tokens; runtime aligns effective attention and mamba blocks to 1,600 tokens |
| Key-value dtype | FP8 |
| EXL3 prefill | FP8, reconstruction tile 256 |
| Speculation | Qwen MTP, depth 3 |
| Prefix reuse | native prefix caching, mamba alignment |
| External cache | disabled |
| Collective transport | patched NCCL; SIRCL disabled |

The candidate uses the companion Qwen base recipe's cache-free 8,192-token
scheduler budget. The companion LMCache launch's 3,072-token budget is a
transfer constraint and does not apply here. Pass `--block-size 16`; the
pinned runtime then aligns the effective attention and mamba cache blocks to
1,600 tokens. Do not pass `--block-size 1600` directly.

## 1. Prepare the cycle

Complete the [four-Spark prerequisites](PREREQUISITES.md). The four fabric
edges must be cabled as `0-1`, `1-2`, `2-3`, and `3-0`. Every rank must route
to the non-adjacent fabric subnets through a neighbour, forward IPv4 traffic,
and allow the relay through Docker's forwarding chain.

Run the topology checks before preparing a serving process:

```bash
python scripts/ring_doctor.py --help
```

Use the command form appropriate for the local site configuration. Do not
start Qwen until every rank can reach the rendezvous address and every direct
cable has passed the repository's cable checks.

## 2. Prepare one identical runtime per rank

The profile uses the source-built runtime documented by
[`FujitsuPolycom/qwen38-spark-pair`](https://github.com/FujitsuPolycom/qwen38-spark-pair/tree/b9e1031b80b6f3f64bfc75ae3922322f56954fd6)
at commit `b9e1031b80b6f3f64bfc75ae3922322f56954fd6`. The repository name describes
its measured pair deployment; its model loader, kernels, and Python
environment are the inputs reused here.

Follow its container, Python, ExLlamaV3, vLLM, model-download, and patched-NCCL
steps on rank 0. Stop before the LMCache setup. The relevant identities are:

| Input | Identity |
|---|---|
| Base image | `nvcr.io/nvidia/cuda@sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d` |
| vLLM mirror | `FujitsuPolycom/vllm`, tag `qwen38-tested-20260817`, commit `229effc810ee6b8112f661472f6aace4eb8c787d` |
| ExLlamaV3 | `5f3c537ca9d89893d771256f5c43c93656553fbb` plus ARM64 patch SHA-256 `594b01547b0d801cf95926ea973719354150893121019aba2ad8832bc9f17fdb` |
| Torch | `2.12.0+cu132` |
| B12X | `1.2.4` |
| Python dependency reference | companion `requirements-freeze.txt` blob SHA-256 `d773c781bcc1de6cf81a64f9fa6b2ab80535f77eea08c5aeb5b96c2ce4423ba8` |
| Patched NCCL | SHA-256 `e69a8c240f45d10166bcd901d99db78bb63147adda66e586d8dd505c6d608b54` |

Replicate the prepared `/ws/venv`, source trees, model, patched NCCL library,
and chat template to ranks 1-3. Each prepared container must mount its local
work directory at `/ws` and expose `/dev/infiniband`. No public image or source
archive represents the measured runtime, so public reproduction remains
pending.

The pinned CUDA base does not contain Python, the CUDA 13.2 toolkit, or the
userspace verbs provider. Install the companion recipe's build packages plus
`procps`, `libibverbs1`, `ibverbs-providers`, and `ibverbs-utils` in every
prepared container. `/dev/infiniband` without `libibverbs.so.1` is
insufficient: NCCL will discover the device nodes and then fail before model
loading. The launcher uses `pgrep` from `procps` for its duplicate-engine
guard.

Verify the public source base commits, working-tree states, and runtime imports
on every rank:

```bash
/ws/venv/bin/python -c 'import vllm; print(vllm.__version__)'
/ws/venv/bin/python -c 'import torch; from exllamav3_ext import exl3_gemm; print("exl3 ok")'
git -c safe.directory=/ws/src/vllm-gg -C /ws/src/vllm-gg rev-parse HEAD
git -c safe.directory=/ws/src/vllm-gg -C /ws/src/vllm-gg diff --binary | sha256sum
git -c safe.directory=/ws/src/exllamav3 -C /ws/src/exllamav3 rev-parse HEAD
git -c safe.directory=/ws/src/exllamav3 -C /ws/src/exllamav3 diff --binary | sha256sum
sha256sum /ws/nccl-patched/libnccl.so.2
```

The source commands must return vLLM
`229effc810ee6b8112f661472f6aace4eb8c787d` and ExLlamaV3
`5f3c537ca9d89893d771256f5c43c93656553fbb`. The vLLM diff must have the empty
SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
the ExLlamaV3 ARM diff must have SHA-256
`594b01547b0d801cf95926ea973719354150893121019aba2ad8832bc9f17fdb`.
The Python version string alone does not identify the Qwen hybrid-prefix
safety ports. The tracked launcher also rejects any additional vLLM status and
any ExLlamaV3 status that differs from the pinned ARM patch state.

Download the model at the pinned revision. Verify that `SHA256SUMS` itself has
SHA-256 `7626d18481e7f995fd1d9ff211083b7fd57f044daba39e107fb29a48207f24c4`,
then run all 16 entries in that file on every rank. Copy
`scripts/chat_template_agentic.jinja` from the pinned companion checkout to
`/ws/chat_template_agentic.jinja`; its expected SHA-256 is
`4f9201169f5bacd1a494c8824470a1ef899c7024d23a2b166e42493e7efd9ac9`.

After replication, copy SparkRing's rank launcher into the `/ws`-mounted work
directory on every rank:

```bash
cp scripts/qwen38_dgx4_serve.sh /path/to/qwen-runtime/qwen38_dgx4_serve.sh
```

## 3. Configure each rank

Copy the cycle environment template to `/ws/rank.env` inside each prepared
container's work directory:

```bash
cp scripts/config/qwen38-27b-exl3-k5k6.env.example /path/to/qwen-runtime/rank.env
```

Replace `<RENDEZVOUS_IFNAME>`, `<RANK_RENDEZVOUS_IP>`, `<NCCL_IB_HCA>`, and
`<NCCL_IB_GID_INDEX>` separately on every host. The rendezvous interface is
the management NIC and must allow unrestricted mutual TCP because vLLM's
multiprocess executor uses random worker ports in addition to port 29500. List
the two RoCE devices in the local ring order. Use `show_gids` to find the
RoCEv2 entry for each device's fabric IPv4 address; both devices on one rank
must expose that address type at the selected rank-global GID index. Do not
copy index 3 from the recorded run without checking the site. Keep every other
value equal. See [the architecture](ARCHITECTURE.md).

The environment intentionally contains no DeepSeek, GLM, LMCache,
SparkCache, or SIRCL settings. It combines the Qwen EXL3 runtime gates with
the patched-NCCL cycle transport.

## 4. Start the four ranks

Set these host variables on each Spark:

```bash
RANK=REPLACE_WITH_0_TO_3
RANK0_RENDEZVOUS_ADDR=REPLACE_WITH_RANK0_MANAGEMENT_ADDRESS
CONTAINER=REPLACE_WITH_PREPARED_QWEN_CONTAINER
```

This launch stops being safe if another model stack owns the GPU, rank-0 API
port, or rendezvous port. Inspect the running containers and GPU processes on
all four hosts, identify any serving stack, and preserve its launch contract
before stopping it. Do not start Qwen until the selected Qwen containers are
the only containers assigned to the GPUs. The rank launcher separately rejects
an existing `vllm serve` process in its container and bound rank-0 ports; it
cannot see a process isolated in another container.

Start ranks 1-3, then rank 0 without a long delay between them:

```bash
docker exec -d \
  --env RANK="$RANK" \
  --env RANK0_RENDEZVOUS_ADDR="$RANK0_RENDEZVOUS_ADDR" \
  --env QWEN_ENV_FILE=/ws/rank.env \
  "$CONTAINER" \
  bash -lc 'mkdir -p /ws/logs && exec bash /ws/qwen38_dgx4_serve.sh >> "/ws/logs/qwen38-r${RANK}.log" 2>&1'
```

[`scripts/qwen38_dgx4_serve.sh`](../scripts/qwen38_dgx4_serve.sh) carries the
full serving command, including:

```text
--tensor-parallel-size 4
--decode-context-parallel-size 1
--nnodes 4
--distributed-executor-backend mp
--max-model-len 262144
--max-num-seqs 64
--max-num-batched-tokens 8192
--enable-chunked-prefill
--async-scheduling
--scheduler-reserve-full-isl
--block-size 16
--kv-cache-dtype fp8
--enable-prefix-caching
--mamba-cache-mode align
--speculative-config {"method":"qwen3_5_mtp","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}
--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}
```

Rank 0 listens on port 8000. Ranks 1-3 use `--headless`.

## 5. Validate a deployment

Wait for rank 0 to answer:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

Then send a temperature-zero request twice and compare `message.content`,
`reasoning_content`, tool calls, and finish reason. Do not compare dynamic API
IDs or timestamps. Check all four logs for successful rendezvous, the same
model revision, the reported key-value pool, graph capture, and MTP acceptance
counters.

The bounded record below intentionally has partial coverage. Before making a
wider or qualified performance claim, record at least:

- available host memory and swap after model load;
- reported key-value tokens and maximum 262,144-token concurrency;
- arithmetic, instruction-following, tool-call, and vision checks;
- eager-versus-graph output comparison;
- native shared-prefix and divergent-suffix checks;
- cold prefill at 4K, 16K, and 64K;
- sustained decode at C1, C8, and C32; and
- API, rank-process, NCCL, and host health after the workload.

The one- and two-Spark observations in the pinned companion Qwen recipe are
useful controls but are not TP4 results. The bounded TP4 bring-up is recorded in
[`performance/records/qwen38-27b/dgx4-live-20260823.md`](../performance/records/qwen38-27b/dgx4-live-20260823.md).
Future runs should add immutable records under `performance/records/qwen38-27b/`
without changing the candidate settings to match an unrecorded or
configuration-drifted launch.

## Pending integration

**SparkCache: Pending.** No Qwen3.8-27B SparkCache composition recipe or live
cache evidence is published. The four-Spark base profile disables external
key-value caching; cache work is not required for this bring-up.
