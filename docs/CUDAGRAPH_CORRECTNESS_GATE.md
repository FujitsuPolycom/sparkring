# CUDA-graph correctness gate

## Status and scope

CUDA-graph serving is **experimental in the public-functional lane**. A
container reaching `/health` or `/v1/models`, completing graph capture, or
reporting a positive capture-node census proves startup and graph definition.
It does not prove that a real replay consumes fresh model inputs or produces
correct output.

The public first-launch template deliberately uses `--enforce-eager`.
`SETUP.md` section 8.4 reconstructs a historical reference launch; it is not a
shortcut around the graph gates below.

This distinction matters because an external four-Spark reproduction reported
that the full historical graph configuration booted and served, but a roughly
13K-token prompt returned only the token `lock`. The same prompt was correct
in eager mode. A DCP1 graph run had the same symptom.

## What the DCP1 result does and does not eliminate

DCP1 removes DCP attention collectives. It does **not** remove four-rank tensor
parallelism, the graph-native TP all-reduce, the graph-native vocabulary
all-gather, the target FULL graph, or the DSpark draft graph. The DCP1 result
therefore lowers the probability of a DCP query/combine bug, but it does not
clear SparkRing's graph transport.

The repository's own graph documents stop short of that claim:

- `spark_transport/GRAPH_NATIVE_TP4_Q1.md` says the changing-input mixed-Q
  native gate and four-rank vLLM capture/replay are pending;
- `spark_transport/integrations/vllm/TP4_CUDAGRAPH_READINESS.md` says capture
  count alone is not replay evidence;
- `docs/STATUS.json` records no accepted public-lane CUDA-graph result.

Until those gates close, wrong output is a correctness failure, not a
performance regression and not an accepted fallback.

## Ranked hypotheses

1. **Graph-native TP/vocabulary replay is consuming stale or incorrectly
   ordered data.** DCP1 retains TP4, and the public native work has standalone
   replay evidence but no accepted end-to-end vLLM graph result. Prediction:
   keep vLLM CUDA graphs enabled but set
   `VLLM_SPARK_TP4_GRAPH_Q1=0`; the prompt becomes correct using the original
   captured collectives.
2. **The target FULL graph replays stale sparse-MLA/indexer metadata or a
   graph-owned workspace view.** Prediction: original collectives still fail,
   but PIECEWISE-only execution succeeds.
3. **The separately captured DSpark draft graph has a stale input/output
   binding.** Prediction: disabling only
   `VLLM_DSPARK_DISABLE_FORWARD_CUDAGRAPH` fixes the graph candidate while the
   target graph remains enabled.
4. **A workspace was not sized on the path later selected by the 13K
   request.** The NF3 lane exposed a related 544 MiB versus 575.31 MiB
   late request, but there is no evidence yet that the MXFP4 report is the
   same defect. Prediction: `VLLM_DEBUG_WORKSPACE=1` shows either a late
   workspace request or a different buffer/pointer at replay.
5. **The long prompt merely exposes another graph-stable metadata lifetime
   bug.** Prediction: the failure has a sharp context/token boundary even
   with stock captured collectives, MTP off, and PIECEWISE-only execution.

These are hypotheses, not findings. No CPU-only test can execute a CUDA graph,
the B12X kernels, or the device-published RDMA command ring. The offline loop
can make captured live evidence deterministic; it cannot manufacture that
evidence.

## Minimal live A/B

Use one fixed prompt that tokenizes to the failing roughly 13K input. Record
the prompt SHA-256 and use `temperature=0`, a fixed seed, `max_tokens=64`, and
**MTP off** for the first isolation. Disable prefix-cache reuse or make the
prompt unique for every run. Do not change model revision, KV format, top-k
pattern, per-token scaling, TP size, or DCP size between paired runs.

Run in this order:

| Arm | CUDA execution | Spark native graph | DSpark graph | Decisive result |
|---|---|---|---|---|
| A | eager | none | none | establish the token-id oracle |
| B | FULL_AND_PIECEWISE | off: `VLLM_SPARK_TP4_GRAPH_Q1=0` | off | if correct, vLLM/B12X target graphs work with stock collectives |
| C | FULL_AND_PIECEWISE | on | off | if B passes and C fails, TP/vocabulary native graph replay is isolated |
| D | FULL_AND_PIECEWISE | same as last passing arm | on | if only D fails, isolate the DSpark graph |
| E | PIECEWISE only | off | off | if B fails and E passes, isolate the target FULL graph |

Do not use DCP1 as the first separator: it changes attention semantics while
leaving the leading TP/vocabulary hypothesis enabled.

For every graph arm, capture all four rank snapshots immediately before and
after the request. Keep the two snapshots in the same artifact; a single
positive post-request sequence value is not evidence because it may be stale
from an earlier replay. Native graph promotion additionally requires:

- identical positive `captured_nodes` on all ranks;
- each rank is caught up before the request;
- `published_sequence` increases during this request by the same positive
  amount on all ranks;
- post-request
  `published_sequence == consumed_sequence == completed_sequence`;
- `overflow_sequence == 0` both before and after the request;
- verified submission and progress CPU affinity; and
- zero cold fallback, stock-signature drop, CUDA, workspace, or asynchronous
  transport errors.

## Deterministic offline comparison

Save one small JSON artifact for the eager oracle and one for each graph arm:

```json
{
  "schema": "sparkring-graph-canary/v2",
  "mode": "graph",
  "model_identity_sha256": "<sha256 of immutable model identity record>",
  "prompt_sha256": "<sha256 of exact prompt bytes>",
  "request": {
    "temperature": 0,
    "seed": 20260730,
    "max_tokens": 64,
    "mtp_enabled": false
  },
  "response": {
    "http_status": 200,
    "finish_reason": "length",
    "output_token_ids": [1, 2, 3],
    "text_preview": "optional, sanitized"
  },
  "expect_native_graph": false,
  "graph_ranks": [],
  "note": "optional"
}
```

When native graph transport is expected, set `expect_native_graph` to `true`
and provide exactly four entries:

```json
{
  "rank": 0,
  "captured_nodes": 128,
  "before": {
    "published_sequence": 224,
    "consumed_sequence": 224,
    "completed_sequence": 224,
    "overflow_sequence": 0
  },
  "after": {
    "published_sequence": 256,
    "consumed_sequence": 256,
    "completed_sequence": 256,
    "overflow_sequence": 0
  },
  "submit_affinity_verified": true,
  "progress_affinity_verified": true
}
```

Then compare without GPU or cluster access:

```bash
python scripts/graph_canary_gate.py \
  --eager eager-canary.json \
  --graph graph-canary.json \
  --minimum-output-tokens 16
```

The first isolation gate intentionally requires MTP off and exact output token
IDs. It rejects malformed/unknown fields, mismatched prompt/model/request
identity, short output such as the one-token `lock` symptom, token divergence,
finish-reason drift or omission, rank-asymmetric capture, stale replay
counters, request-local advancement that is not rank-synchronous, counters
that did not catch up, overflow before or after the request, and unverified
CPU affinity.

After target-graph correctness passes, repeat with MTP enabled as a separate
acceptance experiment. Do not weaken the first gate to accommodate speculative
nondeterminism; that would make the transport/target diagnosis ambiguous.

## Evidence needed for a source-level fix

The smallest useful report is:

1. exact image ID and SparkRing commit;
2. sanitized `/proc/1/cmdline` and `/proc/1/environ` from every rank;
3. exact prompt SHA-256 and token count;
4. eager and graph output token IDs, HTTP status, and finish reason;
5. graph status/census on every rank immediately before and after the request;
6. first error from all four logs, with `VLLM_DEBUG_WORKSPACE=1`; and
7. which arm in the table first changes failure to pass.

Without those artifacts, changing workspace size or graph transport code would
be guesswork. Keep eager mode as the supported public serving path.
