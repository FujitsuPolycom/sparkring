# GLM-5.2 3.5bpw fixed-MTP2 weight-scope contract

## Status and evidence boundary

This profile is a **public-functional-lane, live-validated intermediate
candidate** for four NVIDIA DGX Sparks / GB10 GPUs. The stream-corrected
profile completed target, draft-prefill, and draft-decode graph capture,
matched the MTP-disabled control on repeated 128-token and 256-token greedy
outputs, returned finite logprobs, exercised both speculative positions, and
passed its four-rank transport audit. It is not the repository default or an
accepted public-functional matrix. The fixed-MTP4, 9.25 GB KV candidate
is the qualified GLM-5.2 3.5bpw serving profile; see
[GLM52_35BPW_FIXED_MTP4_PROFILE.md](GLM52_35BPW_FIXED_MTP4_PROFILE.md). `R7` is
the durable recipe identifier of this operator-tuned 3.5-bpw EXL3 profile
(recipe `recipes/glm52-exl3-r7-3.5bpw.json`).

The qualified executable MTP2 profile was produced by a maintainer-held copy
of the `prepare_exl3_r7_mtp2.py` utility from the maintainer's stock-DCP4 GLM-5.2 3.5bpw
control; that utility also emitted a byte-identical rollback profile. A public
generator and stock-DCP4 input chain are tracked at
`scripts/prepare_exl3_r7_mtp2.py` and `scripts/generate_exl3_r7_stock_dcp4.py`;
see the input chain in [GLM52_35BPW_QUICKSTART.md](GLM52_35BPW_QUICKSTART.md).

Relative to the stock-DCP4 control, this profile changes speculative decoding
from disabled to a fixed depth of
two. It does not change the target model's weight formats or enable a second
online-quantization pass for the draft model.

## Target and draft weight ownership

| Model role | Routed experts | Eligible BF16 weights | Online K6 scope |
|---|---|---|---|
| target | producer checkpoint EXL3 K3/K4/K5 | target online-quantization configuration may encode eligible weights as K6 | enabled |
| layer-78 MTP draft | producer checkpoint EXL3 | producer BF16 non-expert weights | disabled |

The draft's producer BF16 non-expert tensors include the MTP input and hidden
normalizations, `eh_proj`, and shared-head non-expert state. The routed expert
payloads are serialized in the attested checkpoint index. The draft
speculative configuration must contain `quantization="exl3"` so vLLM selects
the checkpoint EXL3 loader, but it must not contain a `quantization_config` or
another online-quantization override.

The draft also omits `kv_cache_dtype` and inherits the target's top-level
`--kv-cache-dtype fp8_ds_mla` setting. In the pinned vLLM source, explicitly
restating the draft KV dtype reconstructs `CacheConfig`. That reconstruction
turns the default 16-token block into a user-specified 16-token block, while
`B12X_MLA_SPARSE` accepts 64-token multiples. A four-rank startup with the
redundant draft field failed at draft-model construction with
`block_size not supported`; the corrected omission is therefore a required
startup invariant, not an implicit default.

The process-wide `VLLM_EXL3_ONLINE_TRELLIS_BITS=6` setting does not by itself
apply online K6 to the draft. The composed vLLM runtime constructs a distinct
draft `ModelConfig` and passes only the explicit EXL3 quantization method; it
does not copy the target `ModelConfig.quantization_config`. The EXL3 backend's
BF16 online overlay requires that per-model configuration and therefore stays
inactive for the draft. The pinned image and checkpoint index are part of the
profile attestation, so source or payload drift fails before launch.

The runtime profile declares this invariant through these container labels:

```text
org.sparkring.r7.online-k6-scope=target-only
org.sparkring.r7.target-weight-contract=checkpoint-exl3-routed+online-k6-eligible-bf16
org.sparkring.r7.draft-weight-contract=checkpoint-exl3-routed+producer-bf16-nonexpert
org.sparkring.r7.capture-stream-contract=process-device-shared-target+draft
```

## Target and draft graph-capture stream ownership

Spark TP4 graph sessions require one stable caller CUDA stream. The pinned
vLLM graph manager creates separate capture contexts for the target,
draft-prefill, and draft-decode managers. Without the MTP-specific overlay,
each context owns a new stream even when
`VLLM_SPARK_SHARED_CAPTURE_STREAM=1`; the target capture succeeds and the
first eligible draft graph fails closed with `TP4 session requires one stable
caller CUDA stream`.

The public GLM-5.2 3.5bpw builder generates the shared-capture implementation from its
hash-pinned vLLM preimage and installs it at
`/opt/venv/lib/python3.12/site-packages/vllm/distributed/parallel_state.py`.
The generated file's SHA-256 is
`b087e93463e9a2d9bede71d3a6e4d696c8f2657449e8dc1119b38613d5750e4e`.
The overlay retains one dedicated stream per process and CUDA device, creates
a fresh graph-capture context for every manager so channel identities remain
distinct, preserves explicit caller-provided contexts and DCP/B12X capture
contexts, and rejects overlapping use. Manager-owned CUDA graph pools are not
changed. The stock-DCP4 rollback retains the shared-stream gate but has no
speculative draft managers, so it preserves the single target-manager path.

## Exact speculative configuration

```json
{
  "model": "/models/glm52-exl3-r7-3.5bpw",
  "method": "mtp",
  "num_speculative_tokens": 2,
  "draft_tensor_parallel_size": 4,
  "quantization": "exl3",
  "moe_backend": "b12x",
  "attention_backend": "B12X_MLA_SPARSE",
  "use_local_argmax_reduction": false,
  "draft_sample_method": "greedy"
}
```

The preparer rejects any extra speculative field, explicit draft KV dtype,
draft online-quantization
configuration, custom DCP transport, custom indexer path, eager execution,
adaptive speculative control, or index reuse. The generated candidate retains
TP4/DCP4 `ag_rs`, 9 GB of KV cache per rank, full and piecewise CUDA graphs,
the hybrid SIRCL plus patched NCCL-IB transport, and stock indexer collectives.
