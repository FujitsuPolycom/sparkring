# sparkring — vLLM plugin for the SIRCL direct-cable transport

Status: implemented, offline-validated; not deployment-qualified. Version `0.1.0.dev0`. Packages the SparkRing vLLM adapter as a
standard `vllm.general_plugins` entry point, replacing the container-era
`sitecustomize` + per-file bind-mount deployment. No serving evidence is
attached to this packaging; the transport itself carries the evidence
labels documented in the repository.

## What installing does

Nothing, by itself. `register()` returns immediately unless a
`VLLM_SPARK_*` mode variable is set. When one is set and installation
fails, the process exits 78 before vLLM can serve traffic — the same
fail-closed contract the container deployment enforced via
`sitecustomize`.

## Enabling (shadow-first)

```bash
pip install sparkring
export VLLM_SPARK_TP4_MODE=shadow          # or custom, after qualification
export VLLM_SPARK_TP4_EAGER_WIDTHS=5120    # admitted eager hidden widths
export SPARK_TP4_PEER0=... SPARK_TP4_PEER1=...        # ring control peers
export SPARK_TP4_DEVICE0=... SPARK_TP4_DEVICE1=...    # per-rank HCAs
vllm serve <model> -tp 4 --enforce-eager --dtype bfloat16
```

The core family is the width-generic eager TP4 all-reduce. The
GLM-geometry families (generic all-gather, vocabulary all-gather, DCP)
remain individually opt-in via their own `VLLM_SPARK_TP4_*_MODE`
variables. Unsupported collectives fall back to the stock vLLM path and
are counted by the collective audit.

## Environment variables each installer acts on

Two installers exist for these adapters. The container overlay is
[`spark_transport/integrations/vllm/sitecustomize.py`](../spark_transport/integrations/vllm/sitecustomize.py),
which CPython imports at interpreter startup when that directory is on
`PYTHONPATH`. The plugin installer is `register()` in
[`src/sparkring/plugin.py`](src/sparkring/plugin.py), which vLLM calls
through the `vllm.general_plugins` entry point.

They install different sets of hooks. `register()` gates on four
variables — one per collective family — and on nothing else. The overlay
gates on those four and on ten further startup hooks. A variable that
only the overlay gates on is inert under the plugin: `register()` never
reads it, no hook is installed, startup succeeds, and neither the plugin
nor `sparkring-preflight` reports it. Exit 78 is the failure contract for
a family that was enabled and could not install; it is not a check that
every `VLLM_SPARK_*` or `SPARK_*` variable in the environment is one some
installer reads. An overlay environment block ported to the plugin
therefore loses the ten hooks below without an error, a warning, or a
log line.

### Acted on by both installers

| Variable | Family it installs |
|---|---|
| `VLLM_SPARK_TP4_MODE` | Width-generic eager TP4 all-reduce |
| `VLLM_SPARK_TP4_ALLGATHER_MODE` | Generic all-gather |
| `VLLM_SPARK_TP4_VOCAB_MODE` | Vocabulary all-gather |
| `VLLM_SPARK_TP4_DCP_MODE` | DCP query and attention combine |

`VLLM_SPARK_TP4_MODE=disabled` installs nothing under either installer.
Per-family tuning variables — control ports, shadow windows and
promotion, admitted widths, query-row policy, combine tolerances,
`SPARK_TP4_GRAPH_STATUS_PATH` — are read by the family modules
themselves, not by either installer, and `tests/test_vendor_parity.py`
keeps the vendored copies byte-identical to the overlay's sources, so
those variables behave the same on both paths. One of them,
`SPARK_TP4_GRAPH_STATUS_PATH`, reaches into overlay-only hooks; see
note 2.

### Acted on by the overlay only

| Overlay gate | Hook it installs | Under the plugin |
|---|---|---|
| `VLLM_SPARK_NF3_STARTUP_PROFILE_MAX_TOKENS` non-empty | NF3 startup-profile cap around `GPUModelRunner.profile_run`, extended to `Worker._compile_or_warm_up_model_impl` when `VLLM_SPARK_NF3_SINGLE_COMPILE_RANGE=1` | inert |
| `VLLM_SPARK_NF3_PROFILE` set to `reference-four-spark`, `reference-four-spark-adaptive-2-4`, or `reference-four-spark-adaptive-2-4-c8`, or `VLLM_SPARK_NF3_WORKSPACE_RESERVE_BYTES` non-empty | NF3 workspace reserve around `GPUModelRunner.capture_model` | inert |
| `SPARK_ADAPTIVE_MTP_CONTROL=1` | adaptive-MTP controller runtime hooks | inert; note 2 |
| `SPARK_GLM52_MTP_INDEX_REUSE=1` | GLM-5.2 MTP index reuse for the V2 speculator | inert |
| `VLLM_SPARK_TRUE_ADAPTIVE_DRAFT=1` | true selected-depth drafting, which attests `VLLM_SPARK_MTP_TOKENS`, `VLLM_SPARK_MTP_ADAPTIVE_WINDOW`, and `VLLM_ADAPTIVE_SPEC_DEPTHS` | inert; note 2 |
| `SPARK_Q2R_PROBE=1` | Q2R probe bridge on `Worker.initialize_from_config` | inert; note 2 |
| `SPARK_TP4_FLIGHT_RECORDER=1` | B12X fused-indexer import hook | partial; note 1 |
| `SPARK_CUDAGRAPH_REPLAY_TIMING=1` | CUDA-graph replay timing, armed through `SPARK_CUDAGRAPH_REPLAY_TIMING_ARM_PATH` | inert; note 2 |
| `SPARK_TP4_DCP_COLLECTIVE_AUDIT=1` | DCP combine and reduce-scatter audit on `vllm.v1.attention.ops.common` | inert |
| `VLLM_SPARK_TRACE_ALLREDUCE=1` | all-reduce shape tracer writing `VLLM_SPARK_TRACE_PATH` | inert |

**Note 1 — the flight recorder is partial, not inert.**
`spark_tp4_backend` and `spark_tp4_allgather_backend` read
`SPARK_TP4_FLIGHT_RECORDER` themselves. With the all-reduce family
installed, the recorder is activated when the first native backend is
constructed, so the `op=AR` and `op=AG:*` trace lines are emitted under
either installer. Only `install_b12x_import_hook()` is overlay-gated, so
the `op=B12X` indexer lines are the part that disappears.

**Note 2 — the graph-status snapshot reads four overlay gates.**
With `SPARK_TP4_GRAPH_STATUS_PATH` set, the collective families start the
low-rate status reporter on both paths. Its collector branches on
`SPARK_CUDAGRAPH_REPLAY_TIMING`, `SPARK_Q2R_PROBE`,
`VLLM_SPARK_TRUE_ADAPTIVE_DRAFT`, and `SPARK_ADAPTIVE_MTP_CONTROL`, so
under the plugin those sections describe hooks that were never installed.
`SPARK_ADAPTIVE_MTP_CONTROL=1` additionally makes the collector import
`adaptive_mtp_controller.runtime_installer`, which the wheel does not
vendor: where that package is not importable in the serving process, the
collector raises on every poll, the reporter thread swallows the
exception, and no status file is written at all.

Four of the overlay-only modules ship in the wheel because they are in
the import closure of the collective families —
`spark_q2r_probe_bridge`, `spark_cudagraph_replay_timing`,
`spark_tp4_flight_recorder`, and `spark_true_adaptive_draft` — and
`register()` never calls their `install()`. Present is not installed. The
remaining six hooks (both NF3 adapters, the adaptive-MTP controller, MTP
index reuse, the DCP collective audit, and the shape tracer) are not in
the wheel at all. The stock-path counting named above is
`spark_collective_audit`, which is vendored and active under the plugin;
it is a different module from the one `SPARK_TP4_DCP_COLLECTIVE_AUDIT`
gates.

Scope of this comparison: both installer files and the vendored modules
were read in this checkout, and every gate listed above is the expression
those files test. Whether an overlay-only hook would work if it were
installed from the plugin was not tested, and none of this was exercised
on a serving process. `VLLM_USE_V2_MODEL_RUNNER` is a vLLM variable that
the overlay's NF3 cap requires to be exactly `0` or `1`; the plugin adds
no check of its own, and what vLLM does with it is outside this
comparison.

## Native library

`spark_tp4_backend` loads `libspark_transport_capi.so` from
`SPARK_TP4_LIBRARY`. If that variable is unset and the wheel ships
`sparkring/_native/libspark_transport_capi.so` (linux-aarch64/sbsa,
CUDA 13), the plugin points the variable there. The binary is a build
artifact, not committed: a wheel built from a clean source checkout
contains no native library, and serving with such a wheel requires
`SPARK_TP4_LIBRARY` to name an externally built one. Produce it from
the repository source in any CUDA 13 aarch64 devel container:

```bash
cmake -S spark_transport -B build -DCMAKE_BUILD_TYPE=Release
make -C build -j8 spark_transport_capi
cp build/libspark_transport_capi.so \
   sparkring_plugin/src/sparkring/_native/
```

Distribution wheels containing the binary must be platform-tagged
(`linux_aarch64`); editable installs on the Sparks do not care.

## vLLM compatibility

The deployed runtime is a vLLM fork with a dev version string, so
compatibility is feature-detected rather than version-ranged:
`sparkring/_compat.py` resolves `CudaCommunicator.all_reduce` at its
known path and verifies the method signature, refusing with a clear
message when a future vLLM moves the integration point. Verified
targets at packaging time: the pinned fork build
(`0.1.dev1+ge2666d9a6.d20260810`) and upstream `v0.27.1` (same class,
same signature). Any newer vLLM that keeps the class works unchanged.

## Vendored runtime

`src/sparkring/_vendor/` holds byte-identical copies of the flat modules
from `spark_transport/integrations/vllm` (the adapter's proven bare-name
import layout, unchanged). `scripts/sync_vendor.py` refreshes them;
`tests/test_vendor_parity.py` fails on any drift, so a stale vendor
cannot ship silently.

## Not in scope for 0.1

CUDA-graph admission (open correctness gate; `--enforce-eager` only),
expert-parallel all-to-all, non-BF16 collective dtypes, topologies beyond
directly cabled two- and four-Spark rings, and any performance claim.
