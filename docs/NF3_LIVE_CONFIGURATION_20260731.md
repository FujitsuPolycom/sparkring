# NF3 1M live configuration snapshot

## Scope

This document records the configuration that was **actually running** on the
four-Spark operator cluster on 2026-07-31 (2026-08-01 UTC). It was assembled
from read-only `docker inspect`, image labels, the embedded runtime manifest,
all four rank commands and transport environments, rank-0 startup logs, and
the live `/health` and `/v1/models` endpoints.

The machine-readable, line-by-line record is
[`docs/configurations/glm52-nf3-live-1m-20260731.json`](configurations/glm52-nf3-live-1m-20260731.json).
That file removes management addresses and SSH identities but retains the
rank topology, immutable hashes, arguments, runtime-controlled environment,
and observed activation evidence.

This is a **public-functional-lane, live-observed configuration snapshot**.
It is not a benchmark result and is not an acceptance receipt for whatever
commit happens to be checked out later.

## Actual effective serving configuration

| Setting | Observed value |
|---|---|
| target model | `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid@66f3623dd8fefb5ca8046706912d5d31c8d196af` |
| served model name | `GLM-5.2-NF3` |
| MTP draft | `aidendle94/GLM-5.2-MXFP4-Experts-GPTQ@46537e0e16fcd156627800139b41b9c497fc7ee2`, `mtp-draft/` |
| topology | four directly cabled DGX Sparks, one direct-cycle ring |
| TP / PP / DP / DCP | `4 / 1 / 1 / 4` |
| DCP backend | `ag_rs` |
| attention backend | `B12X_MLA_SPARSE` |
| model dtype reported by vLLM | BF16 activations, `modelopt_fp4` quantization |
| routed-expert layout | layers 3-77: 64 NVFP4 + 192 NF3 experts per layer and rank-local `I=512` |
| maximum model length | **1,048,576 tokens** |
| maximum sequences | **8** |
| maximum batched tokens / prefill chunk | **4,096** |
| prefix caching | enabled |
| SparkCache | **disabled** (`SPARK_CONTEXT_CACHE_ENABLE=0`) |
| load format | `fastsafetensors` |
| async scheduling | disabled in the effective engine configuration |
| API | rank 0, port `8000`, OpenAI-compatible |
| reasoning / tool parsers | `glm47 / glm47` |

The live model endpoint returned `GLM-5.2-NF3` and
`max_model_len=1048576`. The container had no Docker healthcheck, but the API
health endpoint returned HTTP 200 and all four containers were still running
with no OOM exit.

## KV configuration

| Setting | Observed value |
|---|---|
| KV dtype | `nvfp4_ds_mla` |
| storage description | NVFP4 compressed latent + FP8 RoPE |
| scaling | per-token (`VLLM_NVFP4_MLA_PER_TOKEN_SCALE=1`) |
| explicit allocation | **9,000,000,000 bytes/rank** (8.38 GiB) |
| reported capacity | **1,125,632 tokens** |
| maximum 1,048,576-token request concurrency | `1.07x` |
| initial free unified memory before KV reservation | 112.44 GiB/rank |
| `--gpu-memory-utilization` | `0.89`, but it does not size KV because explicit KV bytes take precedence |

The one-million-token maximum is a per-request limit, not a promise that eight
one-million-token requests fit simultaneously.

## MTP configuration

The command sets a maximum of four MTP draft tokens. The installed adaptive
controller then selects depths 2 or 4 over a 32-round window:

```text
num_speculative_tokens=4
VLLM_ADAPTIVE_SPEC_DEPTHS=2,4
adaptive_speculative_tokens_window=32
VLLM_SPARK_TRUE_ADAPTIVE_DRAFT=1
SPARK_ADAPTIVE_MTP_CONTROL=1
SPARK_GLM52_MTP_INDEX_REUSE=1
```

Startup logged both `Spark true adaptive drafting installed` and activation
of target sparse-index reuse after step 0.

## Compilation and CUDA graphs

| Setting | Effective value |
|---|---|
| eager mode | no (`enforce_eager=False`) |
| compilation mode | `VLLM_COMPILE` |
| graph mode | `FULL_AND_PIECEWISE` |
| vLLM capture buckets | `1,2,4,8,16,24,32,40` |
| maximum capture width | `40` |
| graph warmups | `1` |
| compile range endpoint | `4096` |
| shared capture stream | enabled |
| NF3 graph workspace reserve | `805,306,368` bytes/rank |
| observed workspace result | pointer-stable, exactly the reserved size |
| observed capture time | 93 seconds |

The SparkRing overlay also receives the wider admission lists
`1,2,3,4,5,6,8,10,12,15,16,20,24,25,30,32,35,40`. Do not confuse those
overlay admission widths with the eight vLLM graph buckets reported by the
engine.

The effective compiler pass configuration enabled `fuse_act_quant` and
requested `fuse_allreduce_rms`; norm quantization, attention quantization,
GEMM/communications fusion, RoPE/KV concatenation, and activation padding were
off. FlashInfer autotuning was disabled.

## Sparse MLA and DCP controls

```text
VLLM_B12X_MLA_CKV_GATHER=1
VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=1048576
VLLM_B12X_MLA_DECODE_GATHER_V2=1
VLLM_B12X_MLA_DECODE_SPARSE_GATHER=0
VLLM_DCP_GLOBAL_TOPK=1
VLLM_DCP_SHARD_DRAFT=1
VLLM_DSV4_INDEXER_SP=1
VLLM_NVFP4_MLA_PER_TOKEN_SCALE=1
```

The required GLM indexer pattern was passed explicitly. Rank 0 logged all 57
expected sparse-indexer skip messages, which proves the pattern was applied
instead of silently running missing indexer weights.

Two requested switches need stronger wording than “enabled”:

- `VLLM_B12X_MLA_DECODE_GATHER_V2=1` is present, but core vLLM reports the
  name as unknown and no dedicated startup arming line was found. The setting
  is recorded; activation is not claimed.
- `VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM=1` is also present without a dedicated
  arming line. It remains unproven in this snapshot.

The custom DCP query/combine plane variable `VLLM_SPARK_TP4_DCP_MODE` is not
present, so this snapshot does not claim that plane is active.

## MoE and kernel controls

The live hybrid loader was armed and installed `HybridNvFp4MoE`. Its important
settings were:

```text
B12X_MOE_FORCE_A16=1
B12X_DENSE_SPLITK_TURBO=1
B12X_NSA_CONTIGUOUS_PREFILL_BLOCK_K=auto
HYBRID_KEPT=b12x_nf3
HYBRID_NF3=b12x_nf3
HYBRID_TIER=both
HYBRID_MXFP8_NATIVE=1
HYBRID_B12X_MAX_TOKENS=4096
VLLM_USE_B12X_MOE=1
VLLM_USE_B12X_FP8_GEMM=1
VLLM_USE_B12X_SPARSE_INDEXER=1
```

`HYBRID_TC_DECODE=0`, so the hybrid fused tensor-core decode path is **off**.
Although `B12X_W4A16_TC_DECODE=1` is inherited, the installed
`hybrid_loader.py` reads `HYBRID_TC_DECODE`; no Python consumer of the former
name was found. The live launch also does not set
`CUDA_DEVICE_MAX_CONNECTIONS`.

## Transport

The running image installed SparkRing custom TP4, all-gather, and vocabulary
adapters. Unsupported operations retain the patched NCCL fallback.

```text
VLLM_SPARK_TP4_MODE=custom
VLLM_SPARK_TP4_ALLGATHER_MODE=custom
VLLM_SPARK_TP4_ALLGATHER_POLICY=spark-custom
VLLM_SPARK_TP4_VOCAB_MODE=custom
VLLM_SPARK_TP4_PREFILL_Q512=1
SPARK_TP4_MAX_INFLIGHT=64
SPARK_TP4_GID0=3
SPARK_TP4_GID1=3
NCCL_NET=IB
NCCL_ALGO=Ring
NCCL_PROTO=<automatic>
NCCL_IB_DISABLE=0
NCCL_IB_GID_INDEX=3
NCCL_IB_MERGE_NICS=0
NCCL_CROSS_NIC=1
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4
NCCL_NET_PLUGIN=none
NCCL_SKIP_TREE_CONNECT=1
```

The fallback NCCL library actually loaded at runtime was `2.30.7+cuda13.0`,
with SHA-256
`ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f`.
The SIRCL transport library SHA-256 was
`d34632178877c1465725479335c8e2f4d057fef0aec2ee46b98275f2bc79d2b3`.
Both hashes matched on all four ranks.

The peer order follows the two-round XOR schedule:

| rank | device 0 | device 1 | peer 0 | peer 1 |
|---:|---|---|---:|---:|
| 0 | `rocep1s0f0` | `rocep1s0f1` | 1 | 3 |
| 1 | `rocep1s0f1` | `rocep1s0f0` | 0 | 2 |
| 2 | `rocep1s0f0` | `rocep1s0f1` | 3 | 1 |
| 3 | `rocep1s0f1` | `rocep1s0f0` | 2 | 0 |

In other words, peer 0 is `rank XOR 1` and peer 1 is `rank XOR 3`.
Management/control traffic uses the configured management interface; bulk
data uses the two directly attached RoCE devices.

## Container contract

All ranks used local image ID
`sha256:ab6bddba38aac663e2427f139aec20e022eced3716d7e204e1e881742f9da11d`.
There is no registry digest for this locally built image. The containers use:

- host networking and host IPC;
- 16 GiB shared memory;
- `CAP_IPC_LOCK` and unlimited memlock;
- `/dev/infiniband` plus the local GPU;
- read-only target and draft model mounts;
- a writable JIT cache;
- a context-cache mount that remains unused because SparkCache is disabled;
- a read-only NF3 startup-profile cap module.

Key installed versions were CUDA 13.2.1, Torch 2.12.0+cu132, B12X 0.23.0,
FlashInfer 0.6.13+cu132, Triton 3.7.0, fastsafetensors 0.3.2, and the patched
vLLM build
`0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea.fi25dd814.cu132.20260626`.

## Effective command

The common command below is sanitized only at the site-specific master
address. Ranks 1-3 append `--headless` and every rank substitutes its own
`--node-rank`.

```text
/opt/venv/bin/vllm serve /models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid
  --tensor-parallel-size 4
  --decode-context-parallel-size 4
  --max-model-len 1048576
  --kv-cache-memory-bytes 9000000000
  --max-num-seqs 8
  --port 8000
  --distributed-executor-backend mp
  --nnodes 4
  --node-rank <0..3>
  --master-addr <RANK0_MANAGEMENT_ADDRESS>
  --master-port 29500
  --dcp-comm-backend ag_rs
  --attention-backend B12X_MLA_SPARSE
  --hf-overrides '{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
  --kv-cache-dtype nvfp4_ds_mla
  --enable-prefix-caching
  --gpu-memory-utilization 0.89
  --max-cudagraph-capture-size 40
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":true}}'
  --kernel-config '{"enable_flashinfer_autotune":false}'
  --enable-auto-tool-choice
  --load-format fastsafetensors
  --served-model-name GLM-5.2-NF3
  --reasoning-parser glm47
  --tool-call-parser glm47
  --host 0.0.0.0
  --speculative-config '{"model":"/mtp-draft","method":"mtp","num_speculative_tokens":4,"draft_attention_backend":"B12X_MLA_SPARSE","adaptive_speculative_tokens_window":32}'
  --max-num-batched-tokens 4096
```

## Misleading inherited variables

The effective Docker command and engine startup record are authoritative.
Copying the raw container environment alone would produce the wrong model:

| inherited variable | inherited value | actual effective value |
|---|---:|---:|
| `MAX_NUM_SEQS` | `128` | `8` |
| `TP_SIZE` | `2` | `4` |
| `SPEC_TOKENS` | `5` | maximum `4`, adaptive depths `2,4` |
| `GRAPH_CAP` | `128` | `40` |
| `GPU_MEMORY_UTILIZATION` | `0.8` | CLI `0.89`, then explicit KV bytes override sizing |
| `NODE_RANK` | `0` on every rank | explicit CLI ranks `0,1,2,3` |
| `MODEL_PATH` | stale DeepSeek path | positional NF3 model path |
| `ASYNC_SCHED` | `1` | async scheduling not enabled in the engine |

These stale variables should be removed from the next clean launcher, but they
do not alter this live process because explicit command arguments or the
engine configuration win.

## Drift from the default NF3 recipe

The existing default recipe remains the smaller public profile. This live
operator profile differs materially:

| Field | default recipe | this live snapshot |
|---|---:|---:|
| maximum model length | 262,144 | **1,048,576** |
| KV bytes/rank | 7,000,000,000 | **9,000,000,000** |
| reported NVFP4+FP8-RoPE capacity | 875,520 | **1,125,632** |
| CKV gather maximum | 458,752 | **1,048,576** |
| Gather V2 requested | `0` | **`1`** |
| custom indexer graph requested | `0` | **`1`** (activation unproven) |
| custom prefill Q512 requested | `0` | **`1`** |
| parser pair | `glm45 / glm47` | **`glm47 / glm47`** |
| served name | `glm-5.2-nf3-hybrid` | **`GLM-5.2-NF3`** |

The immutable image also differs from the older local image ID recorded in
the default recipe. The full snapshot preserves the live ID and all relevant
binary hashes so future comparisons can distinguish configuration drift from
runtime drift.
