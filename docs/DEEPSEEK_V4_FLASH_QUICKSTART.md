# DeepSeek-V4-Flash-0731 four-Spark quickstart

Status: **functional launch, operator-observed performance.** This
configuration serves on the four-Spark ring and has been exercised
with decode, prefill, tool calling, and speculative decoding. It is
not shadow-qualified: no numerical comparison window has closed for
its collectives, and no performance number here is a qualified
measurement. Evidence scope and limitations:
[profile page](profiles/DEEPSEEK_V4_FLASH_0731.md).

The [validated-profiles registry](profiles/README.md) explains what
the maturity labels mean.

## What you get

Four-rank tensor-parallel serving of the official FP8 checkpoint with
the model's native DSpark speculative decoding, an OpenAI-compatible
endpoint, and tool calling. Operator-observed on the ring: roughly
55-65 tokens per second single-stream on random text, 2,400-2,700
tokens per second prefill from 8K through 128K context, and a
524,288-token request limit. Coding prompts decode faster than random
text because draft acceptance is higher.

## 1. Prerequisites

Complete the [prerequisites checklist](PREREQUISITES.md) first: four
directly cabled DGX Sparks, RoCEv2 on both ports per node, qualified
cables, and a working container runtime. Nothing below substitutes
for that.

You also need a vLLM build that can load `DeepseekV4ForCausalLM` with
the B12X kernel family. One is published:

```bash
docker pull ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028
```

That image registers `DeepseekV4ForCausalLM` and carries B12X, LMCache,
the patched NCCL, and the SparkRing transport library. It also registers
`Glm4MoeForCausalLM`.

What the image removes is the ARM64 CUDA build: the transport library,
its probe binaries, the patched NCCL, and the forked vLLM all arrive
prebuilt. It also carries the `quack-kernels` correction described below,
so no bind mount and no file outside this repository is required to serve
this checkpoint from it.

`runtime/faststart-lock.json` owns this pin. The command above repeats it
rather than establishing a second one.

### What the correction is, and how to rebuild it

`quack-kernels` 0.5.0 reaches the published image as a transitive
dependency, so it escapes the two things `runtime/exl3-r7` applies to a
copy it installs directly: the pinned `apache-tvm-ffi` 0.1.10 in
`runtime/exl3-r7/requirements-tvm-ffi.txt`, and the hash-bound
compatibility edits in `runtime/exl3-r7/bake_runtime_artifacts.py`.
`quack-kernels` 0.5.0 declares `nvidia-cutlass-dsl>=4.5.2` with no upper
bound and caps `apache-tvm-ffi<0.2`; the image resolves cutlass-dsl to
4.6.0 and `apache-tvm-ffi` to 0.1.9.

Two failures follow, both raised on the first forward pass during memory
profiling:

- cutlass-dsl 4.6.0 defines `ThrMma` and `ThrCopy` in `cutlass.cute.atom`,
  while `quack-kernels` 0.5.0 annotates them at `cutlass.cute.core` at
  module scope. Importing quack therefore raises `AttributeError: module
  'cutlass.cute.core' has no attribute 'ThrMma'`. This checkpoint's
  attention path reaches that import from
  `site-packages/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`,
  behind the `has_cutedsl` guard in
  `site-packages/vllm/utils/import_utils.py`, which tests only whether
  `cutlass` imports at all and so never suppresses it. Both paths are in
  the image's own vLLM installation: this repository vendors a different
  vLLM tree and does not contain either file, so they are verifiable only
  against the image.
- cutlass-dsl 4.6.0 calls `make_kwargs_wrapper` with a
  `map_dataclass_to_tuple` argument that `apache-tvm-ffi` 0.1.9 does not
  accept, raising `TypeError`.

The published digest above is that correction already applied. To
reproduce it, or to apply it to a different parent, build from a checkout
of this repository:

```bash
mkdir -p /var/tmp/sparkring-correction
cp runtime/exl3-r7/bake_runtime_artifacts.py /var/tmp/sparkring-correction/
cat > /var/tmp/sparkring-correction/Dockerfile <<'DOCKERFILE'
FROM ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:35b29616dc05677b98f647282e81a99fbca1969791ccbfca711c11a44285385e
COPY bake_runtime_artifacts.py /tmp/bake_runtime_artifacts.py
RUN /opt/venv/bin/pip install --no-cache-dir "apache-tvm-ffi==0.1.10" \
 && /opt/venv/bin/python /tmp/bake_runtime_artifacts.py quack \
      /opt/venv/lib/python3.12/site-packages/quack \
 && rm /tmp/bake_runtime_artifacts.py
DOCKERFILE
docker build -t sparkring/gb10-vllm-serving:corrected /var/tmp/sparkring-correction
```

`bake_runtime_artifacts.py` verifies the SHA-256 of each file it edits
before and after the edit, so the build fails closed rather than
producing a silently different image if the base ever carries other
bytes. A rank without a route to a package index needs the
`apache-tvm-ffi` 0.1.10 aarch64 wheel staged locally and
`--no-index --find-links` pointed at it; the wheel URL and its SHA-256
are in `runtime/exl3-r7/requirements-tvm-ffi.txt`.

A locally built image carries a different identity from the published
digest, so a launch from one is evidence about that build rather than
about the pin.

## 2. Model

`deepseek-ai/DeepSeek-V4-Flash-0731` (official FP8, e4m3 with 128x128
weight blocks and fp4 experts; 48 safetensors shards, about 167 GB).
Download once and distribute identical bytes to all four ranks; the
launch reads a local path on each rank.

This repository does not pin a revision for this model. If you need
reproducibility, record the revision and per-file hashes you actually
downloaded and pin them yourself.

## 3. Launch

Run one container per rank, rank 0 through rank 3, with `--node-rank`
matching. Rank 0 serves the API; ranks 1 through 3 run `--headless` and
join it over the fabric address of rank 0.

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

The environment a rank needs is larger than the five variables in
section 4. A launch also requires `LD_PRELOAD` naming
`/usr/local/cuda/compat/libcuda.so.1` and the patched
`/opt/sparkring/nccl/libnccl.so.2`, and roughly twenty `NCCL_*`
variables that select the ring algorithm, the two host channel adapters,
and GID index 3. Without the compat preload the worker aborts on an
undefined `cuTensorMapEncodeTiled`; without the NCCL configuration it
aborts on an unhandled system error. A deployment carries these in a
per-rank environment file.

Every rank must hold the image before any rank launches. Rendezvous
waits 601 seconds for all four; a rank still pulling when the others
start will miss that window, and a rank without a route to the registry
cannot pull at all. Distribute first, verify every rank reports the same
image ID, then launch.

These flags in that command are easy to omit, and one is commonly
misread as load-bearing:

- `--kernel-config '{"enable_cutedsl_warmup": false}'` gates the CuteDSL
  GEMM autotune section only. It does not gate
  `_warmup_ll_bf16_router_gemm()`, which runs on any capability-90 or higher
  device regardless of the flag, so it does not prevent the abort an
  uncorrected image raises;
  [`runtime/hotfixes/deployed-r34-20260810/README.md`](../runtime/hotfixes/deployed-r34-20260810/README.md)
  records that call site and the version skew behind it. Section 1's
  correction is what prevents the abort. A launch from the corrected image
  that omits the flag reports `Skipping CuTeDSL warmup because no compile
  units were requested` and serves. The flag is retained here because the
  configuration these observations come from carries it, not because it is
  load-bearing.
- `--kv-cache-dtype fp8_ds_mla` declares the layout the engine allocates.
  `fp8` allocates identically but declares a generic geometry; the engine's
  own kernels never read the declaration, so both serve, and only
  `fp8_ds_mla` is correct for an external key-value consumer.
- `--tokenizer-mode deepseek_v4` selects this checkpoint's tokenizer.
- `--headless` on ranks 1 through 3. Without it every rank tries to bind
  the API port.
- `--entrypoint /opt/venv/bin/vllm`. The image's own entrypoint verifies a
  GLM attestation contract and exits 78 before reaching vLLM when a
  variable such as `SPARKRING_MTP_DRAFT_PATH` is absent, which it is for
  this checkpoint.

Environment the B12X kernel family needs on every rank:

```bash
VLLM_USE_B12X_MOE=1
VLLM_USE_B12X_FP8_GEMM=1
VLLM_USE_B12X_SPARSE_INDEXER=1
VLLM_USE_B12X_MHC=1
VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
```

This launch attaches no external key-value cache tier. The engine's
own prefix caching is unaffected and remains active; external reuse is
`unsupported` for this checkpoint, and the
[LMCache record](LMCACHE_DEEPSEEK_20260819.md) states why.

`--kv-cache-dtype fp8` is correct for that reason. The engine gates the
`fp8_ds_mla` layout on an exact string comparison, so `fp8` selects a
generic path that declares a geometry differing from the one it
allocates. Both values allocate identically and the engine's own kernels
never read the declaration, so the difference is invisible here. It is
not invisible to an external key-value consumer, which reads the
declared geometry: any configuration that adds one must use
`--kv-cache-dtype fp8_ds_mla` instead.

Sizing notes. The 32 GiB per-rank key-value reservation and the
0.70 memory utilization together fit the model on GB10's unified
memory with headroom; 0.80 has been observed to OOM a follower rank
during draft-model load on this hardware class. The 524,288-token
limit is a choice, not a ceiling: the checkpoint's native maximum
position is 1,048,576, and the same reservation serves either.

## 4. Optional: SparkRing transport

Without the environment below, collectives use your NCCL build and
everything above still serves. To route the ring's eager collectives
through the SparkRing transport, set per rank:

```bash
VLLM_SPARK_TP4_MODE=custom
VLLM_SPARK_TP4_EAGER_WIDTHS=4096
SPARK_TP4_LIBRARY=/path/to/libspark_transport_capi.so
SPARK_TP4_PEER0=<this rank's port-0 neighbor>
SPARK_TP4_PEER1=<this rank's port-1 neighbor>
SPARK_TP4_DEVICE0=rocep1s0f0
SPARK_TP4_DEVICE1=rocep1s0f1
SPARK_TP4_GID0=3
SPARK_TP4_GID1=3
```

Peers follow the ring's two perfect matchings, and the RDMA device
order inverts on odd ranks; see [Architecture](ARCHITECTURE.md) for
the topology and [SIRCL](SIRCL.md) for the transport's scope.

Capturing width-4096 decode collectives into CUDA graphs is
research-only and additionally requires
`VLLM_SPARK_TP4_GRAPH_WIDTH4096_RESEARCH=1`,
`VLLM_SPARK_SHARED_CAPTURE_STREAM=1`, and distinct
`SPARK_TP4_GRAPH_SUBMIT_CPU` / `SPARK_TP4_GRAPH_PROGRESS_CPU` values.
That path carries no qualification.

## 5. Verify

The engine reaches API health in roughly three to five minutes on a
warm page cache. A four-rank launch following this page, from the
corrected image with no bind mounts, reached `Application startup
complete` in 390 seconds and answered a chat completion; it reported a
key-value pool of 4,382,668 tokens and a maximum concurrency of 8.36
requests at the 524,288-token limit. Those two figures depend on the
per-rank reservation and the context limit, so a configuration that
changes either will report different ones.

Then:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731",
       "messages":[{"role":"user","content":"What is 17 * 23?"}],
       "max_tokens":16,"temperature":0}'
```

Check the engine log for the key-value pool size and speculative
metrics. `Mean acceptance length` well above 1 confirms speculation
is working; on code prompts it should sit near the top of its range,
on random text considerably lower.

## 6. Sampling

The model card recommends temperature 1.0, with `top_p` 0.95 for
agentic use and 1.0 otherwise. The chat template also accepts a
`reasoning_effort` of `low`, `high`, or `max`; the higher levels
expect a much larger output budget than most serving profiles allow.

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `fp8_ds_mla ... got auto` at load | The architecture's key-value layout requires `--kv-cache-dtype fp8` explicitly |
| `KeyError: model.layers.43.mtp_block...` | The checkpoint carries DSpark draft heads, not an MTP block; use `"method": "dspark"`, not `deepseek_mtp` |
| Speculation rejected at k below 5 | DSpark's block size is 5; `num_speculative_tokens` must be at least 5 |
| Draft acceptance roughly halves | `"moe_backend": "b12x"` missing from the speculative config; on code prompts this was the difference between about 42% and 86% acceptance |
| `B12X EP local expert metadata does not match` | `--enable-expert-parallel` is incompatible with this fork lineage |
| `tool_choice: auto` rejected | Add `--enable-auto-tool-choice --tool-call-parser deepseek_v4` |
| Follower rank killed during load | Memory utilization too high for GB10 unified memory; 0.70 is the validated setting |

If you are adapting a launch environment from one of this
repository's GLM profiles, drop its `SPARKRING_MODEL_*` attestation
variables. They describe a GLM checkpoint, have no effect here, and
make the running container misreport what it is serving.

## Unsupported or unqualified

- **External key-value reuse**, which is optional and omitted from
  the launch above. It was run against this checkpoint on the ring and
  is `unsupported`: the installed cache package stores and restores
  this model's state across a full teardown but cannot load it into
  device memory, so every retrieve aborts and the engine recomputes
  the prompt. Serving without it is unaffected and is the path this
  page describes. Conditions for support, and the validation method
  any external tier must satisfy, are in the
  [LMCache record](LMCACHE_DEEPSEEK_20260819.md). The GLM 3.25-bpw
  profile's LMCache settings do not transfer.
- **A pinned model revision** and per-file hashes, as noted above.
- **Shadow qualification** of any width-4096 collective signature.
