# Bounded speculative-decoding phase timing

## Status

Status: implemented with synthetic CPU tests. Importing the package does not
install hooks or access a GPU. Live measurement requires the explicit probe
bootstrap, matching source fingerprints, and CUDA event allocation.

The worker bootstrap in `spark_q2r_probe_bridge.py` calls `live_installer.py`
when the combined probe is enabled with `SPARK_Q2R_PROBE=1`.
Calling its module-level `install()` requires
`SPARK_Q2R_PHASE_TIMING=1`. It pins the exact deployed vLLM version and every
source seam before mutating one method, then preallocates all events before
arming.

Timing only `CudaGraphManager.run_fullgraph` leaves draft generation, other
graph modes, eager transitions, and collectives outside that call unmeasured.
The difference from whole-round wall time cannot be labelled CPU overhead
without measuring those regions.

The installed hooks measure the execution envelopes, graph calls, and draft
generation described below. Eager-transition and collective-boundary timing
remain unimplemented. The recorder adds no CUDA synchronization to the serving path.

## Hot-path contract

`PhaseTimingCollector`:

- allocates exactly `2 * capacity` CUDA events before `arm`;
- registers a fixed set of phase descriptors before `arm`
  (`register_descriptors` may finalize them after manager construction);
- records a start and end event on the caller's current stream;
- optionally emits a balanced NVTX range using the same fixed descriptor;
- never calls event/stream/device `synchronize`;
- never calls `query` or `elapsed_time` in `measure`;
- never allocates another CUDA event or grows a duration list while armed;
- stops accepting samples at capacity and increments `dropped.capacity`;
- executes operations with unregistered descriptors without recording timing
  and increments `dropped.unregistered_descriptor`;
- propagates operation exceptions and excludes aborted operations from timing
  aggregates; `errors.operation` counts failures after a start event is recorded.

There is a small Python mutex around slot reservation and counter updates.
That is a host bookkeeping lock, not a CUDA synchronization. The intended
integration runs on the single model-execution thread; the lock protects
low-rate snapshots from torn counters.

`snapshot()` copies counters and completed timing samples without polling CUDA.
The live-session snapshot also reports whether hooks are installed, cleanup
is pending, and manager binding is complete. An empty startup registry can be
reported before model initialization, but cannot be armed. Cleanup disarms the
recorder; if cleanup fails, retry it before arming again.
Module-level `live_installer.uninstall()` retries retained cleanup and clears
the installed session only after successful hook restoration.
Session lifecycle and reporting calls are serialized separately from model
callbacks. Install, arm, disarm, reset, and uninstall outside measured model
execution; this serialization does not wait for GPU work to finish.

`drain()` is a separate, nonblocking completion poll: an unready event remains
pending. It must run in the low-rate reporter or after timed execution, never
inside a model phase. `reset()` reuses the preallocated event pairs only after
the epoch is disarmed and all pending events have completed or errored.

The seven phase families are:

1. `step_envelope`
2. `target_full_graph`
3. `draft_multistep_graph`
4. `draft_generation`
5. `other_graph`
6. `eager_transition`
7. `collective`

Sublabels must be finite and registered at construction. Recommended initial
labels are target query lengths five and six (`Q5`/`Q6`), observed draft manager stage
(`prefill`/`decode`), graph mode plus token count, eager boundary reason, and
collective family plus eligibility reason. Do not synthesize per-step draft
counts from the configured speculative depth.

## Fail-closed adapter

`vllm_adapter.py` validates the complete hook set before mutation. Every
owner/method/source SHA-256 is validated before any monkeypatch occurs. A
missing method, changed source, uninspectable function, duplicate target, or
pre-existing wrapper aborts installation without mutating vLLM. Uninstall
also refuses to overwrite a method changed by somebody else.
An assignment failure triggers rollback. If restoration also fails, the
adapter reports the failure and retains pending cleanup for an explicit retry;
it does not guarantee atomic restoration under arbitrary setter failures.

The pinned runtime version and graph-replay seam are:

```text
vLLM:
  0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea.fi25dd814.cu132.20260626
CudaGraphManager.run_fullgraph source SHA-256:
  4d58b8ef1a5023af0c11eb7a659620faca15f8a0303b37774ed0d28f4a5919db
```

Do not infer target versus draft solely from `num_tokens`. Target and draft
managers may share a query length, and the first draft vocabulary operation
can also be captured. Set semantic context at the caller that owns target
verification or the draft iteration.

Live source inspection confirmed that this deployed `CudaGraphManager` has
both `run_fullgraph(desc)` and `run_pw_graph(model, model_inputs)`, and that
multiple manager instances serve verification and speculative work. The
descriptor does not contain the manager's semantic role.

The deployed launch uses speculative method `mtp`, not `dspark`. vLLM
therefore creates an `MTPSpeculator` through `AutoRegressiveSpeculator`.
That path owns two draft managers:

```text
prefill_cudagraph_manager -> draft position 0, Q = MTP tokens + 1
decode_cudagraph_manager  -> draft positions 1..K-1, Q = 1
```

The target uses a third `ModelCudaGraphManager`. Query length remains metadata,
not semantic ownership. Manager ownership also does not prove invocation:
an eager draft loop need not call the registered draft-decode graph manager.

The source identity associated with the pinned `AutoRegressiveSpeculator`
call boundaries is:

```text
/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/
  autoregressive/speculator.py
file SHA-256:
  dad44b274e2a2fd8ab2196bcaaec5989b24359cb149cc2dbf879a25f90894cad
```

The similarly named `/opt/vllm/vllm/.../speculator.py` has different source
and is not the imported module. The imported `_multi_step_decode` executes:

```text
for step in range(1, self.num_speculative_steps):
    FULL     -> decode_cudagraph_manager.run_fullgraph(...)
    non-FULL -> self._generate_draft(...)
```

For configured MTP5, draft position 0 is produced by `_prefill`; the loop has
four real positions, 1--4. There is no fifth decode-loop invocation. Fixed
MTP4 similarly has three continuation positions, 1--3. The non-FULL path
does not invoke the draft-decode graph manager.

`draft_step_timing.py` establishes an ordinal context around that exact loop.
It records the actual callable taken for each position: either the FULL graph
replay or `_generate_draft`. It uses host loop ordinals and preallocated CUDA
events; it never reads `current_draft_step`, queries an event, or synchronizes
in the hot path. A successful fixed-depth loop must expose exactly `K - 1`
real continuation calls: four for MTP5 or three for MTP4. With adaptive depths
disabled, `live_installer.py` accepts only the configured fixed depth in each round.

Adaptive timing is accepted only for the exact attested contract
`configured=4`, `depths={2,4}`, and `window=32`. The installer derives that
contract from `VLLM_SPARK_MTP_TOKENS=4`,
`VLLM_ADAPTIVE_SPEC_DEPTHS=2,4`, and
`VLLM_SPARK_MTP_ADAPTIVE_WINDOW=32`; any other nonzero window, maximum, or
depth set is rejected before installation. Descriptors for positions 1--3
are still preallocated, but each round records only the positions the real
loop executes:

```text
round depth 2 -> position 0 prefill + continuation position 1
round depth 4 -> position 0 prefill + continuation positions 1, 2, 3
```

At loop entry the adapter reads `speculator.num_speculative_steps`, the same
host-side Python value that bounds the source loop. This is not a device
scalar read and adds no GPU query or synchronization. A runtime value outside
the attested set fails before the original loop runs. Successful armed rounds
increment `completed_rounds_by_depth`; unarmed warmups and failed rounds do
not enter that histogram, and `reset()` clears it with the timing epoch.

This is **generation-only timing**, not the complete loop iteration.
Per-position attention preparation occurs before both callable branches and
has no enclosing callable in the deployed source. Complete-step timing
requires a tiny source patch that brackets the body of the existing `for`
loop (events recorded on the current stream and drained later), or an upstream
callback/context-manager seam at the same indentation. Do not add together a
guessed attention share or manufacture a position 5.

The complete-step patch should expose one prebound no-allocation context at
the loop body, not read the device scalar:

```python
for step in range(1, self.num_speculative_steps):
    with self.q2r_draft_step(step):  # host ordinal; CUDA events only
        # existing prepare_attn + fill_(step) + generation branch unchanged
        ...
```

`q2r_draft_step.__enter__` and `.__exit__` record a preallocated event on
`torch.cuda.current_stream(self.device)`. They must not query, synchronize,
allocate an event, or append to a growing container. The existing low-rate
`drain()` remains the only place that queries readiness and reads elapsed
time.

`manager_roles.py` assigns every manager a bounded, stable startup identity.
The source-pinned `GPUModelRunner.initialize_kv_cache` seam explicitly assigns
`TARGET_VERIFY`, `DRAFT_PREFILL`, and `DRAFT_DECODE`. Draft-prefill and
draft-decode both resolve to `draft_multistep_graph`, with distinct stage,
manager, and query-length labels.

Both `run_fullgraph` and `run_pw_graph` should use a dynamic
`MethodHook.descriptor` resolver backed by this registry.

The implemented live ownership seam is
`GPUModelRunner.initialize_kv_cache`. After the original returns, it requires
three distinct objects and explicitly binds:

```text
self.cudagraph_manager                         -> TARGET_VERIFY
self.speculator.prefill_cudagraph_manager      -> DRAFT_PREFILL
self.speculator.decode_cudagraph_manager       -> DRAFT_DECODE
```

The pinned `GPUModelRunner` splits a speculative round across two sequential calls.
`GPUModelRunner.execute_model` is the target-forward envelope, while
`GPUModelRunner.sample_tokens` contains sampling and the later draft proposal.
Both are independently source-pinned `step_envelope` descriptors. Their
durations are additive; nested target/draft graph events are attribution and
must not be added again.

`ModelCudaGraphManager` overrides `run_fullgraph`, but the deployed override
delegates the actual replay to `CudaGraphManager.run_fullgraph`. Wrapping both
methods therefore double-counts one target replay as two nested timings. The
live adapter source-pins the override but wraps only the base method. That one
wrapper observes the delegated target replay and the plain base-class MTP draft
replays exactly once each.

The deployed execution class is
`vllm.v1.worker.gpu.model_runner.GPUModelRunner`; the similarly named legacy
`vllm.v1.worker.gpu_model_runner` class is not on this execution path.

Pinned deployed source hashes in `live_installer.py`:

| Seam | SHA-256 |
|---|---|
| `CudaGraphManager.__init__` | `48af9e1aa167af1c0914dd96c2d686a73c2e95c29095b5ae2901356629f0d01d` |
| `CudaGraphManager.run_fullgraph` | `4d58b8ef1a5023af0c11eb7a659620faca15f8a0303b37774ed0d28f4a5919db` |
| `CudaGraphManager.run_pw_graph` | `d515cf6e2e5b9e9fd4516c93b9028b16788a213b5529d0304a3792a4d49833a8` |
| `ModelCudaGraphManager.run_fullgraph` | `9601425569906383209d353ff82e2fa49bafdd3778f9ee9cf094dec29f1a33ed` |
| `AutoRegressiveSpeculator.init_cudagraph_manager` | `f17ff547954777855bb33cd9796a1c82e6d412de859c5cd5dfd481af147304f6` |
| `GPUModelRunner.initialize_kv_cache` | `c606851a60fef594fb231c7c68e695d3a1d52396d2e12a0304819bef8c21e808` |
| `gpu.model_runner.GPUModelRunner.execute_model` | `f04573f6477367594a977c54dc5048d10ad7aa4364fe85642fa56e657ea2e081` |
| `gpu.model_runner.GPUModelRunner.sample_tokens` | `7125f499709171ea88e141c72ec98330cac1ab82fd64876ccf683a28d7daf951` |
| `AutoRegressiveSpeculator._multi_step_decode` | `0e7c45ef90ae463db0a3bd4a9142301c9a2261d11837fac1215eb538d4e1ac26` |
| `AutoRegressiveSpeculator._generate_draft` | `0f0df160847853d7e359d86f9b4de05863eff5e596f0557aa03c2279a53afdda` |

The live snapshot reports both observed envelope counts and their additive
total. Final campaign validation requires nonzero `execute_model` and
`sample_tokens` samples and rejects disagreement with either descriptor. Graph
coverage separately reports actual target, draft-prefill, and draft-decode
manager calls; zero draft-decode calls are valid evidence, not a missing-step
error. For draft-generation coverage, final validation accepts only fixed
MTP4, fixed MTP5, or the exact adaptive `{2,4}`/window-32 attestation. It
requires the completed-depth histogram to equal the `sample_tokens` envelope
count, derives the only valid per-position totals from that histogram, rejects
missing, extra, negative, unscoped, or fabricated samples, reconciles coverage
with the actual phase descriptors, and requires identical evidence on all
ranks. Ordered raw samples preserve the observed call sequence. The snapshot
also states that eager-transition and collective-boundary timing are
unimplemented with zero samples.
After all managers have registered, resolve the finite manager/method/step
descriptor set and pass it to `register_descriptors` before arming. Descriptor
registration is rejected during or after an epoch.

## Source verification before enabling a runtime bundle

Before enabling this component, inspect and hash these methods from the exact
running image:

1. `vllm.v1.worker.gpu.cudagraph_utils.CudaGraphManager.run_fullgraph`
2. `CudaGraphManager.run_pw_graph(model, model_inputs)`
3. `CudaGraphManager.__init__`, specifically the stored
   `decode_query_len`; this value is metadata, never semantic classification.
   The live installer binds explicit roles after `initialize_kv_cache`
4. `AutoRegressiveSpeculator.init_cudagraph_manager`, which owns the
   draft-prefill and draft-decode managers
5. the target manager's owning construction seam, which must explicitly
   assign `TARGET_VERIFY`
6. the speculative decoding driver method that enters target verification
7. the method containing the serial MTP draft loop and its per-step call
8. the eager fallback branch immediately around graph dispatch
9. the original/custom collective boundaries already exposed by
   `spark_collective_audit` and `spark_tp4_backend`

At each candidate seam, record:

```text
module and qualified method
source SHA-256
arguments carrying descriptor/token count
device/current-stream expression
whether call is capture, replay, or eager
semantic owner: target, draft step N, transition, or collective family
```

Then instantiate one `MethodHook` per exact seam. Validate every hook before
adding the opt-in `sitecustomize` call. The production adapter should also
assert the vLLM version and immutable source-bundle manifest before calling
`install`.

For graph dispatch the stream expression should be equivalent to the existing
timed graph call:

```python
torch.cuda.current_stream(instance.device)
```

For collectives, use the tensor/current communicator stream actually used by
that implementation; do not silently substitute the default stream.

## One-load evidence procedure

1. Build the complete descriptor registry and event pool during startup.
2. Take an unarmed baseline snapshot.
3. Arm one named epoch immediately before the controlled request window.
4. Run 100--500 target rounds across the fixed prompt strata.
5. Disarm immediately after the window.
6. Drain from the reporter until `pending == 0`; never synchronize to force
   completion.
7. Take the after snapshot and compute `snapshot_delta`.
8. Reject the evidence if any operation, record, or drain errors, capacity drops, unknown
   descriptor drops, or cross-rank count disagreement exists.
9. Reconcile whole-round wall time against the mutually exclusive semantic
   phases. Nested graph and collective timers are useful attribution but must
   not be summed as if they were disjoint.

This probe attributes elapsed time to measured call boundaries. It does not
establish a performance improvement.
