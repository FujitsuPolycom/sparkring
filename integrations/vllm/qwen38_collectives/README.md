# Qwen collective selection

Status: **implemented**, with bounded DGX4 serving evidence. This optional
selector supports Qwen3.8-Flash-Next on the pinned SparkRing R37 vLLM sources
at TP2 or TP4 when explicitly activated. The measurements below cover the QAD
checkpoint at TP4; they do not establish the best TP2 cutoff. Profiles retain
their own defaults. The selector chooses a collective backend before an operation
is submitted or captured into a CUDA graph.

## Selection contract

The measured configuration uses these settings on every rank:

```text
QWEN_DISPATCH_MODE=both
QWEN_DISPATCH_AR_BYTES=20480
QWEN_DISPATCH_TRACE=0
```

Eligible all-reduces up to 20 KiB use RoCEnante; larger reductions fall through
to NCCL. For the observed BF16 activation width of 2560, the cutoff includes
four rows and excludes eight rows. It is a byte limit, not a concurrency limit.
RoCEnante all-gather eligibility remains bounded by the adapter's independent
16 MiB shard limit. Its original dtype, layout and alignment requirements still
apply. Unsupported operations retain the existing fallback path.

`reduce` mode disables RoCEnante all-gather independently. `nccl` mode rejects
both operations in this selector; disable `VLLM_ENABLE_ROCE_ALLREDUCE` as well
when measuring NCCL without allocating a RoCEnante runtime. The adapter's own
allocation limit remains `VLLM_ROCE_ALLREDUCE_MAX_SIZE=2097152`; the selector's
smaller cutoff controls dispatch without changing that runtime allocation.

Every rank votes on mode and cutoff at initialization. Disagreement is fatal.
Selection depends on shared tensor metadata and immutable configuration, not
rank-local queue depth, pointers or timing. Captured graphs retain their
selected operations; changing policy requires restarting all ranks and
recapturing graphs. A failure after collective submission remains fatal rather
than retrying through another backend.

The import hook checks exact SHA-256 identities for the vLLM adapter and, when
tracing, its CUDA-graph wrapper. Missing or changed sources stop startup.
Tracing records distinct collective signatures and graph descriptors; it does
not count CUDA-graph replay operations. Trace-disabled serving does not install
the graph observer or inspect CUDA capture state for logging.

## Local packaging

From the repository root:

```bash
python integrations/vllm/qwen38_collectives/package.py --destination /tmp/qwen38-policy
python -m pytest integrations/vllm/qwen38_collectives -q
```

The destination must not exist. The package contains the selector, a Python
startup hook and a manifest binding their bytes. Mount the package read-only at
`/opt/sparkring/qwen38-collectives` and its `qwen38_collective_policy.pth` at
`/opt/venv/lib/python3.12/site-packages/qwen38_collective_policy.pth` in every
serving container. Set the same policy environment on all four ranks.

This selector requires the separately verified
[`tp2-rocenante-adaptive` communication bundle](../../../runtime/transport_profiles/README.md),
including its startup hook, four-rank peer map and working mesh paths. The
bundle's name is its artifact identity; it also contains four-rank map support.
Do not copy a TP2 peer map onto TP4. The published R37 image contains this bundle
but requires its transport-selection hook to activate it in spawned workers.
This package neither installs host network rules nor manages their lifetime.

## Evidence and limits

The [comparison record](../../../performance/records/qwen38-flash-next/qad-tp4-collective-dispatch.json) identifies the checkpoint, image,
settings and measured samples. The workload uses 6,537 prompt tokens, 256 output
tokens, disjoint cold prefixes and a repeated warm-prefix pass at C1 and C8.
Two trials per arm compare the same prompts with temperature zero and MTP3.
With an idle API and a fresh server process for each arm, run:

```bash
python performance/harnesses/qwen_collectives.py \
  --base-url http://SERVER:8015/v1 --model Qwen3.8-Flash-Next-NVFP4-QAD \
  --label bounded --output /tmp/qwen38-results --trials 2
```

The command submits inference requests and saves generated text and timing
records. Restart between arms to clear prefix-cache state. Avoid unrelated API
traffic during measurement. Repeating the same fixtures on an existing server
can turn a nominal cold pass into a cache hit.

Warm aggregate decode rate measures output tokens over the interval from the
first content chunk to the last completed stream; it is not cold request
throughput. No broad model-quality, media, long-context or soak claim follows.

Broad RoCEnante dispatch measured approximately 77 tok/s at C1 and 199 tok/s
at C8. NCCL-only measured 67 and 251. The 20 KiB cutoff measured 80 and 248;
the 60 KiB cutoff measured 79 and 250. These samples support a bounded hybrid
policy but do not establish a precise optimal threshold. All-reduce-only
RoCEnante measured 77 and 197, so disabling all-gather alone did not resolve the
C8 difference in this workload. Cold C1 TTFT stayed near two seconds; this is
a decode improvement, not a prefill optimization.

The source-pinned packaged selector was then checked in a separate process
with tracing disabled: two warm trials measured 74/79 tok/s at C1 and 245/244
at C8, approximately 76/245 median. One single-request and eight concurrent
arithmetic checks all returned the expected answers. The selector source hash
matched all four running containers. These final measurements are separate
from the cutoff exploration; do not report the best exploratory sample as a
guaranteed serving rate.
