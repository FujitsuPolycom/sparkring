# GLM-5.2 EXL3 DCP4 fixed-MTP2 external recipe

## Scope and evidence boundary

This page records a configuration **actually run** on four directly cabled
NVIDIA DGX Sparks / GB10s on 2026-08-02. It uses the unchanged legacy
[`willfalco/GLM-5.2-EXL3-TR3-3.25bpw`](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw)
checkpoint at revision `d7d79c2d14599dfce7a5d12b85f7ad73f40e623d` with the
repository's receipt-gated ARM64 EXL3 runtime and a fixed-MTP2 configuration
overlay.

This is an **external-evidence, live-validated configuration candidate**. It is
not public-bootstrap acceptance, not a reference-lane result, and not a result
produced by an unmodified launcher from the current checkout. The trial used an
ignored operator-local profile and launcher overlay; no tracked validator was
weakened. The canonical fixed-MTP3 recipe and its pending clean-checkout
four-Spark publication gate remain unchanged.

This is also not the proposed EXL3 `shared_h_v1` format. The deployed checkpoint
has no `rotation_layout` field and uses the legacy `per_expert_v1` layout. It was
not rewritten or requantized. The upstream r19 registry image evaluated during
qualification was AMD64-only and therefore was not run on the ARM64 Sparks.

The raw operator evidence remains outside the tracked documentation because it
contains site identities and paths. The values below were transcribed from the
sanitized run summary, four-rank startup logs, model attestation, live-gate JSON,
duration-based benchmark JSON, and post-run health checks.

The sanitized machine-readable companion is
[`docs/configurations/glm52-exl3-live-dcp4-fixed-mtp2-20260802.json`](configurations/glm52-exl3-live-dcp4-fixed-mtp2-20260802.json).

## Effective configuration

| Setting | Effective value |
|---|---|
| hardware | four directly cabled DGX Sparks / GB10s, 200-Gb/s cycle |
| model | `willfalco/GLM-5.2-EXL3-TR3-3.25bpw@d7d79c2d14599dfce7a5d12b85f7ad73f40e623d` |
| quantization | legacy EXL3/Trellis 3.25 bpw; 192 K3 plus 64 K4 experts in each of 75 MoE layers |
| local ARM64 image ID | `sha256:314d75abdcbd65433fa9e4a744caf8fa31bdc108e9292df65603a6fe823766ad` |
| platform | `linux/arm64`, `sm_121` |
| served name | `glm-5.2-exl3-tr3-3.25bpw` |
| parallelism | TP4 / PP1 / DP1 / DCP4 |
| MTP | **fixed MTP2**; adaptive control, true-adaptive draft, and target-index reuse off |
| maximum model length | **1,048,576 tokens** |
| maximum sequences | **8** |
| maximum batched tokens | **4,096** |
| KV | `nvfp4_ds_mla`, 9,000,000,000 bytes/rank, 1,125,632 tokens reported |
| attention / DCP backend | `B12X_MLA_SPARSE` / `ag_rs` |
| CUDA graphs | `FULL_AND_PIECEWISE`, Q32 ceiling |
| loading | `safetensors` |
| scheduling | chunked prefill enabled; asynchronous scheduling disabled |
| prefix cache | enabled |
| SparkCache | disabled |

The four ranks used the same image ID and the same 81-shard model payload. The
model attestation matched the pinned `config.json` and safetensors-index hashes
from [`recipes/glm52-exl3-tr3-3.25bpw.json`](../recipes/glm52-exl3-tr3-3.25bpw.json).

## Exact delta from the canonical EXL3 recipe

Start with the complete contract in
[`recipes/glm52-exl3-tr3-3.25bpw.json`](../recipes/glm52-exl3-tr3-3.25bpw.json).
Change only the fixed speculative depth:

The trial and the current offline recipe plan both report semantic recipe
SHA-256 `38685e40969aaf6a77c67bc425533ce514996889db87709786a81232435ca555`.
This is the planner's canonical JSON hash, not the raw file-byte hash.

```text
VLLM_SPARK_MTP_MODE_ID=fixed-mtp2
VLLM_SPARK_MTP_TOKENS=2
```

Replace the canonical speculative configuration's depth with two:

```json
{
  "method": "mtp",
  "num_speculative_tokens": 2,
  "moe_backend": "triton",
  "draft_sample_method": "greedy"
}
```

The corresponding canonical values are `fixed-mtp3`, `3`, and
`num_speculative_tokens=3`. All other static environment values, command-line
arguments, graph widths, model pins, image identity, topology, mounts, and
rank-peer ordering remain those of the canonical recipe. In particular, keep
these controls disabled:

```text
SPARK_ADAPTIVE_MTP_CONTROL=0
SPARK_GLM52_MTP_INDEX_REUSE=0
VLLM_SPARK_MTP_ADAPTIVE_WINDOW=0
VLLM_SPARK_TRUE_ADAPTIVE_DRAFT=0
```

Do not edit the tracked EXL3 launcher merely to make it accept this alternative
mode. At the time of the trial its policy intentionally accepted the canonical
fixed-MTP3 and adaptive modes, but not fixed-MTP2. A public implementation of
this variant should add a separately reviewed profile or generalize that policy
with tests.

## Effective serving arguments

The invariant vLLM portion of the launch was equivalent to the following.
Site-specific master addresses and rank substitutions are deliberately omitted;
ranks 1-3 also use `--headless`.

```text
serve /models/glm52-exl3-tr3-3.25bpw
  --tensor-parallel-size 4
  --decode-context-parallel-size 4
  --max-model-len 1048576
  --kv-cache-memory-bytes 9000000000
  --max-num-seqs 8
  --distributed-executor-backend mp
  --nnodes 4
  --node-rank <0..3>
  --master-addr <RANK0_MANAGEMENT_ADDRESS>
  --master-port 29500
  --trust-remote-code
  --quantization exl3
  --moe-backend b12x
  --dcp-comm-backend ag_rs
  --dcp-kv-cache-interleave-size 1
  --attention-backend B12X_MLA_SPARSE
  --kv-cache-dtype nvfp4_ds_mla
  --enable-prefix-caching
  --enable-chunked-prefill
  --no-async-scheduling
  --gpu-memory-utilization 0.89
  --max-cudagraph-capture-size 32
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32],"custom_ops":["all"],"pass_config":{"fuse_allreduce_rms":true}}'
  --kernel-config '{"enable_flashinfer_autotune":false}'
  --enable-auto-tool-choice
  --load-format safetensors
  --served-model-name glm-5.2-exl3-tr3-3.25bpw
  --reasoning-parser glm45
  --tool-call-parser glm47
  --host 0.0.0.0
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton","draft_sample_method":"greedy"}'
  --max-num-batched-tokens 4096
  [--headless on ranks 1-3]
```

The complete canonical environment, including the exact sparse index pattern,
is in the machine-readable recipe; the abbreviated command above is not a
replacement for that source of truth.

## Older-image explicit-unset compatibility

The local ARM64 image used by this external run predates the current public
entrypoint's complete explicit-unset handling. It contains
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` in its image environment even though
the recipe declares that variable absent. If it reaches vLLM, startup fails
because this model has no sliding-window or Mamba KV-cache group.

Passing an empty Docker environment value is not equivalent to unsetting it:
vLLM attempts to parse the empty string as an integer and fails. The successful
trial replaced the Docker entrypoint with `/usr/bin/env`, supplied one `-u`
argument for every name in `SPARKRING_EXPLICITLY_UNSET`, and then invoked the
image's original `/opt/sparkring-exl3/private-entrypoint.sh`. This removed, among
other inherited values:

```text
VLLM_ADAPTIVE_SPEC_DEPTHS
VLLM_PREFIX_CACHE_RETENTION_INTERVAL
```

An implementation must perform those unsets before the original entrypoint can
import vLLM. Prefer a newly built receipt-gated image whose published entrypoint
implements the recipe's explicit-unset contract. Treat the wrapper as a
compatibility measure for this exact older local image, not as a portable public
launch recipe.

The same image also lacked the newer
`/opt/sparkring-exl3/verify_exl3_model.py` path expected by the current launcher.
For this trial the checkpoint was attested separately before cutover, while the
image's own entrypoint retained its hash checks. A public reproduction must not
silently skip model attestation.

## Startup proof

The effective four-rank logs and API state showed:

- 84.47 GiB of model data loaded per rank;
- all 81 target shards and the MTP draft loaded;
- all 75 MoE layers, layers 3-77, detected as 192 K3 plus 64 K4 experts;
- legacy `per_expert_v1` rotation layout;
- `num_spec_tokens=2`, DCP size 4, and `ag_rs` active;
- 8.38 GiB of explicit KV reservation per rank, 1,125,632-token capacity,
  and 1.07x reported concurrency for a one-million-token request;
- 16/16 piecewise and 12/12 full decode captures, plus 16/16 piecewise and
  12/12 full prefill captures; and
- HTTP 200 from `/health` and the expected served name and 1,048,576-token
  limit from `/v1/models`.

The configuration requested `fuse_allreduce_rms=true`, and startup initially
listed `allreduce_rms` among enabled custom fusions. The compilation pass later
reported that FlashInfer allreduce fusion was unsupported for world size four
or lacked a maximum size, then explicitly disabled the AllReduce fusion pass.
The result therefore does not claim that the requested AllReduce/RMS fusion was
active at runtime.

All four trial containers remained running with zero restarts after the live
gate.

## Historical bounded live gate

The repository's EXL3 gate was run after a warmup request. It required three
byte-identical greedy fixed-seed canaries, throughput floors at C1/C2/C8, and
healthy API checks before and after the matrix.

| Gate | Observed result |
|---|---:|
| deterministic canary | pass; 128 tokens, three identical outputs |
| C1 aggregate decode | **21.48 tok/s** |
| C2 aggregate decode | **29.19 tok/s** |
| C8 short-gate aggregate decode | **38.22 tok/s**; one sample, EOS allowed, 944/1,024 possible completion tokens |
| non-streaming full-request smoke | **20,802 prompt tokens / 48.042 s wall time = 433.0 prompt-token/s**; includes a 16-token completion |
| post-run service | four containers running, zero restarts, API HTTP 200 |
| post-run GPU temperature | 57-63 C |

The full-request smoke included a unique nonce to avoid a prefix-cache hit and
exceeded the required 8,823-token prompt size. It was non-streaming, and its
48.042 seconds include generation of a 16-token completion. That value is full
request wall time, **not TTFT**; 433.0 is merely prompt tokens divided by that
wall time. The initial canary before warmup was not deterministic; the three
post-warmup outputs were byte-identical.

These are short candidate-gate measurements, not the duration-based decode
matrix used for the repository's headline results. They establish that this
configuration worked and cleared regression floors of 15, 24, and 35 tok/s at
C1, C2, and C8 respectively. In particular, C8 is one short sample with EOS
allowed: its eight requests returned 944 of 1,024 possible completion tokens.
It is not sustained decode throughput. These observations do not establish a
general performance band, streaming prefill rate, long-context scaling curve,
energy result, or advantage over fixed-MTP3.

### Later correctness audit supersedes the early canary

A later stock `scripts/exl3_live_gate.py` audit did **not** pass at 128 output
tokens. Two greedy fixed-seed outputs first diverged at token index 124.

A six-run localization produced two hashes with a 4/2 frequency split at the
same index. A ten-run logprob sample selected ` if` five times and ` in` five
times, with exact ties observed and non-tied margins of only 0.125-0.25.

The same stock gate passed when bounded to 120 tokens, before that branch. Its
deterministic output SHA-256 was
`6ebd2106f275ecccceeb8ecb01d538c404727eab9df009f2a4815dee246e2849`.

Therefore the earlier three-identical-output observation remains historical
evidence only. The 128-token deterministic release gate is **failed**, and this
configuration is not correctness-accepted.

## Standard `llm_decode_bench` results

### Initial default-timeout attempt

The first v0.4.31 attempt used the harness's default 60-second 16K readiness
deadline. Its uncommitted artifact SHA-256 was
`d3025f37e8fdd2060184ba4c3b85012f5547c9d7c6e304f0a8b442d550b0d8e2`.
Prefill scouts measured 555/545/536 tok/s at 16K/32K/64K, and C1 was valid at
16.62 tok/s. C2, C4, and C8 exceeded the readiness deadline with zero request
errors; C4 and C8 were also underfilled. Their partial rates are invalid and
must not be quoted. This attempt established the need for the explicit
300-second cell-warmup timeout used below.

### Earlier all-valid 16K run with prefill scouts

An earlier `llm_decode_bench.py` v0.4.31 run used the same fixed-MTP2/Q4096
stack. Its raw uncommitted artifact SHA-256 was
`f02b659f798ce2b2a2afb8103008e85ea5185d49bb1c341d583ddf9a13abd69d`.

It measured 16K sustained decode at C1/C2/C4/C8 with every cell valid. It also
recorded integrated or scout-only streaming TTFT at 16K, 32K, and 64K.

| requested context | observed prompt tokens | TTFT | prefill tok/s |
|---:|---:|---:|---:|
| 16K | 16,249 | 28.969 s | **561** |
| 32K | 32,321 | 58.750 s | **550** |
| 64K | 64,513 | 119.016 s | **542** |

| 16K concurrency | C1 | C2 | C4 | C8 |
|---:|---:|---:|---:|---:|
| aggregate decode tok/s | **15.92** | **28.88** | **43.12** | **62.06** |

The run took 808.141 seconds. It used 25-second windows, temperature zero,
fully unique contexts, a three-second decode warmup, and the same 1,125,632
token manual KV budget.

### Later 16K-128K decode extension

The restored stack later ran the same harness across 16K-128K. The script
SHA-256 was
`8de7c32c0abae3c664226fb9c1c197d0752c0a0f3f5a87b3357326f1407f9c07`.

The run used fully unique contexts, OpenAI continuous-usage token accounting,
`ignore_eos=true`, temperature zero, a three-second decode warmup, and
25-second measured windows. Standalone prefill and burst/E2E were not run.

Equivalent sanitized invocation:

```powershell
python llm_decode_bench.py `
  --host <RANK0_MANAGEMENT_ADDRESS> `
  --port 8000 `
  --model glm-5.2-exl3-tr3-3.25bpw `
  --contexts 16k,32k,64k,128k `
  --concurrency 1,2,4,8 `
  --duration 25 `
  --max-tokens 2048 `
  --temperature 0 `
  --unique-context-percent 100 `
  --dcp-size 4 `
  --kv-budget 1125632 `
  --decode-warmup-seconds 3 `
  --cell-warmup-timeout-seconds 300 `
  --skip-prefill `
  --output <RESULT_PATH>.json
```

The requested contexts calibrated to approximately 16,383, 32,767, 65,535,
and 131,071 prompt tokens. The run took 4,202.984 seconds, approximately 70
minutes, and completed without a server request error or abort.

### Sustained aggregate decode

| requested context | C1 | C2 | C4 | C8 |
|---:|---:|---:|---:|---:|
| 16K | **16.25** | **29.20** | **43.52** | **61.26** |
| 32K | **18.14** | **26.61** | **40.61** | invalid: 6/8 effective |
| 64K | **17.59** | **25.66** | invalid: 3/4 effective | invalid: 2/8 effective |
| 128K | **15.33** | invalid: 1/2 effective | invalid: 2/4 effective | invalid: 1/8 effective |

Valid cells report aggregate completion tokens per measured second. Every
valid cell reached the requested concurrency, had no request errors, and was
neither underfilled nor warmup-timed-out.

The six invalid cells hit the readiness timeout before all unique prompts were
admitted. Their partial rates were suppressed and must not be quoted as C2,
C4, or C8 throughput.

The 32K/C8 cell failed admission at only about 262K prompt tokens, far below
the reported 1,125,632-token KV capacity. This points to unique-prompt prefill
and admission latency, not KV capacity alone, as the immediate limit.

Post-run `/health` returned HTTP 200. The server was idle, and its cumulative
abort and error counters remained zero. The fixed-MTP2 stack was left serving.

The uncommitted raw JSON SHA-256 is
`9d524c68ec61c1a71613bfc4dde29a04dd73d54bf89252873e9df448ee95eb55`.
It is omitted because startup diagnostics contain private endpoint and
workstation details.

This result is a **public-functional-lane, external-evidence, live-validated**
measurement on four Sparks. It is not a reference result, clean-checkout public
acceptance, normalized energy evidence, or a correctness release gate.

The broader matched MTP and Q4096/Q8192 campaign, including the initial
default-timeout attempt, earlier standard run, and standard quick B0/M3/adaptive
comparison, is documented separately in
[`EXL3_AB_CAMPAIGN_20260802.md`](EXL3_AB_CAMPAIGN_20260802.md). Its custom
harness numbers use different prompts and are not interchangeable with either
standard matrix.

## Reproduction and release gates

This documentation does not make the ignored operator overlay a supported
public launch path. Promotion requires all of the following:

1. Represent fixed-MTP2 in a reviewed tracked profile and launcher contract,
   with offline validation and tests rather than a copied private launcher.
2. Build the pinned ARM64 runtime from a clean checkout and retain its receipt.
3. Re-attest all 81 model shards and distribute one identical image ID to all
   four ranks.
4. Inspect the dry-run plan and verify that required variables are truly absent
   from PID 1, not merely set to empty strings.
5. Repeat fabric preflight, startup, graph capture, deterministic correctness,
   a nonce-isolated full-request smoke, C1/C2/C8, and post-run health gates.
6. Repeat the duration-based matrix through the tracked clean-checkout public
   path, and compare fixed-MTP2 with fixed-MTP3 under the same image, prompts,
   warmup, and telemetry before making a general performance claim.

The canonical public EXL3 clean-checkout four-Spark gate remains pending and is
not satisfied by this external run.

## Rollback and operational safety

Planning and inspecting this recipe are offline or read-only operations.
Starting it is **STOPS SERVING** because it replaces the stack bound to the live
API and distributed rendezvous ports. Obtain explicit authorization for the
named four hosts and the interruption before executing a cutover.

Before stopping the current stack, record on every rank:

- exact container name, image ID, state, and restart count;
- PID 1 command, sorted environment, mounts, labels, and network mode;
- rank-peer ordering and site/profile hashes; and
- `/health` and `/v1/models` responses.

Retain the prior stack's filled ignored site and launch files. On any model,
image, startup, graph, API, correctness, throughput, temperature, OOM, or
restart failure, remove only the four explicitly named trial containers and
restart the captured baseline through its launcher. Re-check all four container
states and both API endpoints after rollback. Never embed management addresses,
SSH identities, NIC assignments, host model paths, or credentials in a tracked
recipe or evidence file.
