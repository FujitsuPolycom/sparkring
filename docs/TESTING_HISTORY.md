# SparkRing testing history

This document records experiments, regressions, resolved failures, superseded
configurations, and acceptance gaps.

The project README contains current facts and supported entry points. This
document explains how those facts were established and what remains under
validation.

## Entry format

Each entry should state:

- date and configuration;
- question being tested;
- decisive measurement;
- pass, fail, or superseded status;
- evidence location;
- effect on the current release path.

Peak observations must not replace sustained measurements. Shared-prefix,
unique-context, warm, cold, C1, and aggregate-concurrency results must remain
separately labeled.

## 2026-07-30: NF3-hybrid live candidate

### Configuration

| Field | Value |
|---|---|
| Topology | Four DGX Sparks, TP4/DCP4, direct 200 Gb/s ring |
| Served model ID | `glm-5.2-nf3-hybrid` |
| Quantization lane | MXFP8/NVFP4/NF3 hybrid |
| Speculation | True adaptive MTP2/4, 32-round window |
| Execution | CUDA graphs, Q4096 scheduling |
| KV allocation | 7,000,000,000 bytes/rank |
| Reported KV capacity | 511,488 tokens |
| Workspace reserve | 805,306,368 bytes/rank before graph capture |

### Resolved workspace failure

The previous full-Q4096 overlap reached an indexer operation requiring
575.31 MiB after vLLM had locked a 544 MiB workspace.

Growing that workspace after graph capture was unsafe because captured CUDA
graphs could retain the old pointer.

The accepted fix reserves 768 MiB before `GPUModelRunner.capture_model()`.
After capture, it verifies that vLLM locked the same workspace object.

The hook is fail-closed. It is limited to attested NF3 reference profiles and
the exact expected vLLM source and ABI.

### Live acceptance

- All four ranks built 76/76 NF3 plans.
- Model load completed in 368.04-368.10 seconds per rank.
- Each rank reported 87.22 GiB of model memory.
- Piecewise graph capture completed 13/13.
- Full graph capture completed 8/8.
- Every rank reported a pointer-stable 805,306,368-byte workspace.
- The API became healthy and exposed `glm-5.2-nf3-hybrid`.

The overlap gate ran an 18,562-token prefill beside a live 512-token decode.
Both requests returned HTTP 200, and the API remained healthy.

No workspace, CUDA, timeout, traceback, or fatal error appeared on any rank
after API startup.

### Warm sanity performance

| Cell | Result |
|---|---:|
| C1 coding sanity (earlier run) | 20.93 tok/s |
| C2 coding sanity | 33.38 tok/s aggregate |

These are short warm sanity runs, not a replacement for the complete
context-by-concurrency benchmark matrix.

### Publication status

The live candidate was staged from the active development tree. Its profile,
workspace adapter, launch plumbing, and evidence still require consolidation
into the public quickstart before it becomes the default `main` recipe.

## 2026-07-30: first independent manual public-lane bring-up

An external operator reported completing a four-Spark TP4/DCP4 eager-mode
bring-up of the pinned `GLM-5.2-MXFP4-Experts-GPTQ` profile. This is valuable
independent evidence, but it is not yet a clean-checkout acceptance result:
six setup defects required manual correction.

The report identified:

1. `launch.example.json` omitted the checkpoint's 78-layer
   `index_topk_pattern`. Without it, 57 layers attempted sparse-indexer work
   without corresponding indexer weights. The expected startup check is 57
   `skip sparse MLA indexer computation` messages per rank.
2. The public launcher sorted transport peers numerically, while the native
   TP4 schedule assigns slot 0 to `rank ^ 1` and slot 1 to `rank ^ 3`. The two
   orders differ on ranks 2 and 3 and can make both endpoints choose the same
   control-channel role.
3. The example omitted `VLLM_NVFP4_MLA_PER_TOKEN_SCALE=1`, leaving the packed
   low-bit KV path on the legacy scale contract.
4. The inherited image environment included
   `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`; the pinned GLM-5.2 runtime rejects
   that variable for this KV-group layout, so the public launch must remove it
   rather than assign it another value.
5. `download-glm52.sh` inherited a root-owned Hugging Face home and the image's
   offline-mode setting while running as the invoking non-root user.
6. The example omitted automatic tool choice and the GLM reasoning/tool
   parser arguments.

After those manual corrections, the operator reported clean eager-mode
generation at a 13K context and an **18.3 tok/s median** over 500-token code
completions. They also reported mean acceptance improving from 3.14 to 4.17
tokens and target-step time falling from 354 ms to 260 ms after correcting the
indexer configuration. These are externally reported diagnostic measurements,
not repository acceptance evidence or a controlled attribution of every gain
to one variable.

CUDA graphs did **not** pass the external correctness check. With the documented
graph environment, both DCP4 and a DCP1 arm returned the single token `lock` for
the 13K prompt instead of a valid completion. Eager mode remained correct.
DCP1 removes DCP attention collectives but retains TP4 all-reduce, vocabulary,
target, and draft graph paths, so it does not isolate the cause to workspace or
clear the native transport. Until the failure is reproduced, minimized, and
cleared by [CUDAGRAPH_CORRECTNESS_GATE.md](CUDAGRAPH_CORRECTNESS_GATE.md), the
public quickstart's graph path is not accepted.

## 2026-07-29: public faststart validation

The public ARM64 faststart builder completed a native one-Spark build. The
result then passed identical four-rank distribution, startup preflight, model
and MTP loading, B12X prewarm, and KV allocation.

That sequence did not complete a fresh-clone public-lane API acceptance run.
The exact boundary is maintained in
[FASTSTART_VALIDATION.md](FASTSTART_VALIDATION.md).

## 2026-07-28: DCP4 custom-collective window

The v40 window captured the custom DCP query and fused combine paths inside
FULL CUDA graphs.

The sealed C1 sustained-decode cell measured:

| Context | Decode p50 |
|---:|---:|
| 8K | 20.83 tok/s |
| 16K | 19.28 tok/s |
| 32K | 21.43 tok/s |

The window retained 360 stock owner-top-k captures per rank. Other measured
DCP query and combine captures used the custom path.

The 32K result was about 12% above an earlier stock-trio window. That was not
a sealed same-window A/B and remains labeled as indicative.

## 2026-07-27: concurrency and capacity windows

The shared-prefix DCP1 concurrency baseline reached 63.60 tok/s aggregate at
C8. This is a throughput-scaling result, not a unique-context capacity result.

The DCP4 capacity window reported a 375,040-token pool at 3 GB/rank and
56.70 tok/s aggregate C8 decode.

The current NF3 candidate supersedes that capacity figure with a
511,488-token reported pool at 7 GB/rank.

## Superseded or negative paths

Historical negative results belong here rather than in the README.

When adding one:

1. Preserve the exact configuration and evidence.
2. State whether the cause was isolated.
3. Name the configuration that superseded it.
4. Do not present a failed path as a current setup requirement.

Detailed measurements and claim boundaries remain in
[RESULTS.md](RESULTS.md). Runtime reproduction gaps remain in
[RUNTIME_GAPS.md](RUNTIME_GAPS.md).
