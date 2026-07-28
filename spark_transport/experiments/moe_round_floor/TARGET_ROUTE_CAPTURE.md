# GLM-5.2 target-route capture

This component is a bounded evidence probe, not a production routing change.
It captures 100--500 Q5/Q6 target verification rounds into an arena allocated
before serving:

```text
routes    int16 [500, 75, 6, 8]   7.2 MB
metadata  int64 [500, 5]
masks     int64 [500, 2]
rejected  int32 [500]          # -1 until exact sampler association
counters  int64 [10]
```

Each routed layer launches one 64-thread CUDA kernel. Layer zero claims a
round slot with a device atomic counter. Layers 1--74 use the active slot for
that CUDA stream, and layer 74 publishes completion only after the exact
75-layer mask exists. Thus CUDA graph replay appends a new round instead of
overwriting the slot used during graph capture.

The hot path has no tensor allocation, `.item()`, synchronization,
device-to-host copy, or file I/O. Overflow, wrong model phase, missing layer
zero, duplicate layers, incomplete rounds, and invalid expert IDs are
separate device counters. Any nonzero fatal counter makes the drain fail
closed. `read_counters(timed_execution_complete=True)` exposes the same
overflow/drop inventory without writing an artifact.

The source-pinned V2 `GPUModelRunner.sample()` wrapper associates only an
armed, single-request, four/five-draft verification result. Its CUDA op reads
the latest completed route slot, verifies request identity and ordering,
checks `num_sampled + num_rejected == width`, then atomically stores
`num_rejected[0]` in the fixed sidecar. Q1 sampling is intentionally ignored.
Batching, duplicate association, missing routes, and impossible counts fail
closed; there is no `.item()`, D2H transfer, or synchronization in this path.

## Exact hook breadcrumb

No new router interception is needed. The deployed vLLM `BaseRouter` already
calls its optional callback immediately after:

```python
topk_weights, topk_ids = FusedMoERouter.select_experts(...)
capture_fn(topk_ids)
```

and before EPLB remaps expert IDs. The relevant source is:

```text
/opt/venv/lib/python3.12/site-packages/vllm/
  model_executor/layers/fused_moe/router/base_router.py  # callback near line 269
```

The legacy V1 `GPUModelRunner._bind_routed_experts_capturer` is useful only as
a design breadcrumb. The deployed worker selects the V2
`vllm.v1.worker.gpu.model_runner.GPUModelRunner`, which has no such method.
Bind directly to its validated `static_forward_context`; do not enable the
legacy scheduler-wide CPU slot store. Build one fixed callback per routed
layer:

```python
callback = capture.make_base_router_callback(
    routed_layer_index=moe_runner.layer_id - first_routed_layer,
    model_role="target",
    stream_slot=0,
)
moe_runner.router.capture_fn = callback
```

Bind these closures only on the target `GPUModelRunner`. Never walk or bind
the MTP/draft model. The factory rejects any role except `target`,
`begin_request(model_role="draft")` independently refuses draft context, and
the CUDA op rejects a non-target device control word.

`begin_request()` belongs at the request/target-round controller boundary,
outside model timing. It maps an opaque salted request key to a fixed request
slot and arms the device node. The per-request round number is incremented
on-device at routed layer zero, so graph replay does not depend on a captured
Python integer. `disarm()` makes already-captured graph nodes no-ops before
ordinary serving resumes.

## Build and drain boundary

Build `target_route_capture_cuda.cu` into the immutable experiment image
before serving. Allocate one `TargetRouteCapture` after CUDA device selection
and before graph capture. The custom op has no outputs or internal allocation,
so its kernel node is captured and replayed with the graph.

After the requested prompt strata finish:

1. stop issuing inference work;
2. call `drain_jsonl(..., timed_execution_complete=True)`;
3. let it synchronize and copy the fixed arena to the host;
4. validate every counter, metadata row, layer mask, width, expert range,
   request/round identity, and rejection association;
5. write canonical `glm52-target-expert-routes/v1` JSONL atomically.

The output carries request key, round, Q5/Q6 width, exact rejected-token count,
`accepted_prefix_tokens = width - 1 - rejected_tokens`, target phase, rank,
image, checkpoint, config hash, and source hashes. It is directly consumable by
`route_reuse.py`; extra provenance fields are intentionally ignored by that
analyzer. Production drain refuses to write fewer than 100 complete rounds.

This probe assumes one sequential target graph per `stream_slot`. Concurrent
execution needs distinct preallocated stream slots. Reusing a request slot
before drain/reset is rejected because it would make the JSONL identity
ambiguous.

## Opt-in live installer

`target_route_capture_live.py` is intentionally inert on import. Its
`install_opt_in()` entrypoint source-pins all three deployed lifecycle seams:

```text
Worker.initialize_from_config
  196bbe8208eb5ba56f0e2eb97c0d8922351f1963ac6dbd3466eae94378864ad9
V2 GPUModelRunner.initialize_kv_cache
  c606851a60fef594fb231c7c68e695d3a1d52396d2e12a0304819bef8c21e808
V2 GPUModelRunner.sample
  4d5ce613197dfa32ab5cce9472ef966ce4bca45f8a41edc87b79527908e9b07d
```

After stock KV initialization, but before graph warmup, it validates the
target runner's `static_forward_context`, loads the CUDA extension, allocates
the arena, and binds exactly layer IDs 3--77 to capture indices 0--74. It
refuses missing, duplicate, extra, non-`BaseRouter`, pre-bound, second-runner,
or built-in scheduler-capturer configurations before exposing callbacks.
The DSpark speculator is never traversed.

The launcher still requires this exact value as
`Q2R_RUNNER_SAMPLE_SHA256`; the explicit environment value becomes part of
the launch evidence and the installer independently verifies it before
mutating either live method.

The process-local low-rate surface is `arm_capture_salted()`,
`disarm_capture()`, `capture_counters()`, and `drain_capture()`. These are
controller/RPC operations outside inference timing, not model-layer calls.
The opt-in overlay supplies the deployed classes explicitly:

```python
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

install_opt_in(
    worker_type=Worker,
    runner_type=GPUModelRunner,
    dependencies=LiveDependencies(
        torch_module=torch,
        moe_runner_type=MoERunner,
        base_router_type=BaseRouter,
    ),
    config=LiveInstallConfig(
        runner_sample_sha256=exact_sample_source_sha256,
    ),
)
```

Call this before any worker executes `initialize_from_config`; importing the
module alone has no effect.
