# NF3 DCP4 fixed-MTP2 live recipe

## Scope and evidence boundary

This recipe records the configuration **actually running** on the four-Spark
operator cluster when the 2026-08-01 benchmark below was collected. The live
state was re-read on 2026-08-02 at approximately 03:11 UTC from all four PID 1
processes, their Docker definitions, image labels, mounts, startup logs, the
runtime manifest, and the live API.

This is a **public-functional-lane, live-validated configuration variant** on
four directly cabled NVIDIA DGX Sparks / GB10s. It is not an accepted public
bootstrap, not a reference-lane result, and not a result produced from this
repository's checkout. The running image identifies SparkRing source commit
`267289e259a9ecd86c8cdfd8e0ee4a607d37701c`.

The machine-readable recipe and result summary is
[`docs/configurations/glm52-nf3-live-dcp4-fixed-mtp2-20260801.json`](configurations/glm52-nf3-live-dcp4-fixed-mtp2-20260801.json).
It is an exact overlay on the complete live snapshot in
[`NF3_LIVE_CONFIGURATION_20260731.md`](NF3_LIVE_CONFIGURATION_20260731.md).
Management addresses, SSH identities, hostnames, and host paths are sanitized.

## What was actually running

| Setting | Effective value |
|---|---|
| target model | `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid@66f3623dd8fefb5ca8046706912d5d31c8d196af` |
| served name | `GLM-5.2-NF3` |
| MTP draft | `aidendle94/GLM-5.2-MXFP4-Experts-GPTQ@46537e0e16fcd156627800139b41b9c497fc7ee2`, mounted at `/mtp-draft` |
| topology | four directly cabled DGX Sparks, TP4 / PP1 / DP1 / DCP4 |
| MTP mode | **fixed MTP2**; adaptive controller, true-adaptive draft, and target-index reuse off |
| maximum model length | **1,048,576 tokens** |
| maximum sequences | **8** |
| maximum batched tokens | **4,096** |
| KV | `nvfp4_ds_mla`, 9,000,000,000 bytes/rank, 1,125,632 tokens reported |
| attention / DCP backend | `B12X_MLA_SPARSE` / `ag_rs` |
| graph mode | `FULL_AND_PIECEWISE`, capture widths `1,2,4,8,16,24,32,40` |
| transport | SparkRing custom TP4, all-gather, and vocabulary adapters; patched NCCL fallback |
| load / scheduling | `fastsafetensors`, prefix caching and asynchronous scheduling enabled |
| API | rank 0, OpenAI-compatible, port 8000 |

All four containers started at `2026-08-01T20:00:14Z`, were running without an
OOM or restart at collection time, and used the same local ARM64 image ID:

```text
sha256:ab6bddba38aac663e2427f139aec20e022eced3716d7e204e1e881742f9da11d
```

The image has no registry digest. Important identities observed in the running
image were:

| Artifact | SHA-256 / version |
|---|---|
| runtime manifest | `ebf2f8c98c06ef6c398472153efef373a92670fcaa17aa3a59487763c9d7676a` |
| installed receipt | `df3efbdbf520c6ca96d73d4fb4287325008eca1417b8b3962666fb3efc5950f4` |
| patched NCCL | `ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f` |
| SIRCL/Spark transport | `d34632178877c1465725479335c8e2f4d057fef0aec2ee46b98275f2bc79d2b3` |
| startup-profile cap mount | `905e8c188df8557778686246e3a7472b1b4ac8e1ec7b02b7b44dc561e1e63b7c` |
| vLLM | `0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea.fi25dd814.cu132.20260626` |
| Torch / CUDA | `2.12.0+cu132` / `13.2.1` |
| B12X / FlashInfer / Triton | `0.23.0` / `0.6.13+cu132` / `3.7.0` |
| fastsafetensors | `0.3.2` |

## Exact delta from the complete DCP4 snapshot

Start with the Docker and environment contract in
[`glm52-nf3-live-1m-20260731.json`](configurations/glm52-nf3-live-1m-20260731.json),
whose SHA-256 at capture time was
`5f11c143fe35aa89fab981a37b57d791f4b4b30e626fc2a06ec024ffe7b193e0`.
Apply only this common overlay:

```text
SPARK_ADAPTIVE_MTP_CONTROL=0
SPARK_GLM52_MTP_INDEX_REUSE=0
VLLM_SPARK_MTP_ADAPTIVE_WINDOW=0
VLLM_SPARK_MTP_MODE_ID=fixed-mtp2
VLLM_SPARK_MTP_TOKENS=2
VLLM_SPARK_TRUE_ADAPTIVE_DRAFT=0
```

Remove these variables before vLLM imports:

```text
VLLM_ADAPTIVE_SPEC_DEPTHS
VLLM_PREFIX_CACHE_RETENTION_INTERVAL
```

Replace the speculative configuration with:

```json
{"model":"/mtp-draft","method":"mtp","num_speculative_tokens":2,"draft_attention_backend":"B12X_MLA_SPARSE"}
```

Everything else in the base DCP4 snapshot is unchanged, including DCP4,
the one-million-token limit, 9 GB/rank KV allocation, graph widths, transport,
model/image identity, mounts, and rank peer ordering. After excluding only
`NCCL_IB_HCA`, `SPARK_TP4_DEVICE{0,1}`, and `SPARK_TP4_PEER{0,1}`, the sorted
effective PID 1 environment on every rank had the same SHA-256:

```text
8e8acc049b3aacb067a5e6a80f239006d6345eeb9de363b29715c71a01e98543
```

`VLLM_SPARK_NF3_PROFILE=reference-four-spark-adaptive-2-4-c8` is a
stale descriptive label. It does not reactivate adaptive MTP: the process has
no `VLLM_ADAPTIVE_SPEC_DEPTHS`, the three adaptive switches above are zero,
and the engine reports `num_spec_tokens=2`.

## Effective launch command

The command below is complete except for the sanitized master address and the
rank substitution. Ranks 1-3 append `--headless`.

```text
/usr/bin/env
  -u VLLM_ADAPTIVE_SPEC_DEPTHS
  -u VLLM_PREFIX_CACHE_RETENTION_INTERVAL
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
  --speculative-config '{"model":"/mtp-draft","method":"mtp","num_speculative_tokens":2,"draft_attention_backend":"B12X_MLA_SPARSE"}'
  --max-num-batched-tokens 4096
  [--headless on ranks 1-3]
```

The container contract is unchanged from the base snapshot: host network and
IPC, 16 GiB shared memory, `CAP_IPC_LOCK`, unlimited memlock,
`/dev/infiniband`, all GPUs, read-only model/draft/profile mounts, and writable
JIT/context-cache mounts. SparkCache is disabled despite the mounted directory.

## Startup proof

The rank-0 engine log confirmed the effective, not merely requested, state:

- `SpeculativeConfig(... num_spec_tokens=2)` and DCP size 4;
- BF16 activations, `modelopt_fp4`, `nvfp4_ds_mla`, and
  `B12X_MLA_SPARSE`;
- 87.33 GiB model load/rank and 112.25 GiB initial free unified memory;
- an 8.38 GiB explicit KV reservation/rank, 1,125,632-token capacity, and
  1.07x maximum one-million-token request concurrency;
- 20/20 piecewise and 11/11 full graph captures completed in 89 seconds;
- pointer-stable 805,306,368-byte NF3 graph workspace;
- asynchronous scheduling and the `act_quant` plus `allreduce_rms` custom
  fusions enabled;
- custom TP4, all-gather, and vocabulary adapters installed.

At collection time `/health` returned HTTP 200 and `/v1/models` returned
`GLM-5.2-NF3` with `max_model_len=1048576`.

## Benchmark recipe

The recovered source artifact was produced by `llm_decode_bench.py` version
`0.4.31` (script SHA-256
`8de7c32c0abae3c664226fb9c1c197d0752c0a0f3f5a87b3357326f1407f9c07`).
Its raw JSON SHA-256 was
`41bb27c9411f998d77fef08691bf8d4d453f5562b66ba639115b0c11e720dd2f`.
The raw file is intentionally not committed because its diagnostics contain
management and workstation identities.

Equivalent explicit invocation:

```powershell
python llm_decode_bench.py `
  --host <RANK0_MANAGEMENT_ADDRESS> `
  --port 8000 `
  --model GLM-5.2-NF3 `
  --contexts 16k,32k,64k,128k `
  --concurrency 1,2,4,8 `
  --duration 25 `
  --max-tokens 2048 `
  --temperature 0 `
  --prefill-contexts 8k,64k,128k `
  --unique-context-percent 100 `
  --dcp-size 4 `
  --kv-budget 1125632 `
  --decode-warmup-seconds 3 `
  --coding-peak `
  --coding-peak-runs 5 `
  --coding-peak-max-tokens 2000 `
  --hw-ssh-hosts "S0=<RANK0_SSH>,S1=<RANK1_SSH>,S2=<RANK2_SSH>,S3=<RANK3_SSH>" `
  --output <RESULT_PATH>.json
```

This was duration-based sustained decode using OpenAI continuous-usage token
counts, with fully unique per-stream contexts, `ignore_eos=true`, integrated
decode-scout prefill, one hidden C1 warmup at 128K, and exact remote DCP4 KV
capacity. Burst/E2E, P2PMark, and AMD fabric tests were not run.

## Observed benchmark result

### Prefill

The headline is client-side `prompt_tokens / TTFT`; each row has one sample.
The 8K row is scout-only and the others are the decode matrix's integrated
scouts.

| requested context | observed prompt tokens | TTFT | prefill tok/s | Prometheus validation |
|---:|---:|---:|---:|---:|
| 8K | 8,199 | 10.579 s | **775** | unavailable |
| 16K | 16,247 | 20.844 s | **779** | 782 |
| 32K | 32,345 | 42.157 s | **767** | 769 |
| 64K | 64,539 | 85.625 s | **754** | 755 |
| 128K | 128,910 | 176.016 s | **732** | 734 |

### Sustained aggregate decode

| context | C1 | C2 | C4 | C8 |
|---:|---:|---:|---:|---:|
| 16K | **18.7** | **31.1** | **43.9** | **60.4** |
| 32K | **19.2** | **31.1** | **43.5** | **60.4** |
| 64K | **19.9** | **29.2** | **41.9** | invalid: 1 error, only 7/8 effective |
| 128K | **19.2** | **28.5** | **42.0** | invalid: 4 errors, only 4/8 effective |

The invalid C8 cells had raw measured-window values of 58.24 tok/s at 64K and
41.08 tok/s at 128K, but the benchmark correctly suppressed them. Both hit the
readiness timeout and failed to sustain the requested concurrency; they must
not be quoted as C8 throughput.

The sequential coding probe completed 5/5 runs: **21.8 tok/s median**, 21.7
mean, 22.3 maximum, 20.7 minimum, and zero CJK runs. It used the standard
Sieve-of-Eratosthenes prompt, a 2,000-token maximum, server-default temperature,
and measured `completion_tokens / (last_stream_time - first_token_time)`.

Remote hardware telemetry contains **642 samples over 7,182.969 seconds**
across four Sparks. Average GPU utilization was 95.46%,
maximum GPU temperature 89 C, and total power averaged 254.57 W with a 287.23 W
maximum. These are run-wide monitoring summaries, not normalized energy or
per-cell efficiency claims.

## Operational safety

Collecting the state above was read-only. Reusing the launch command is not:
starting replacement containers on these ports can interrupt serving. Follow
the repository's launcher plan and acceptance-gate safety rules, inspect the
generated plan, and obtain explicit authorization for the named hosts before
stopping or replacing any running stack.
