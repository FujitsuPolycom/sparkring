# sparkring — vLLM plugin for the SIRCL direct-cable transport

Status: skeleton, `0.1.0.dev0`. Packages the SparkRing vLLM adapter as a
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

## Native library

`spark_tp4_backend` loads `libspark_transport_capi.so` from
`SPARK_TP4_LIBRARY`. If that variable is unset and the wheel ships
`sparkring/_native/libspark_transport_capi.so` (linux-aarch64/sbsa,
CUDA 13), the plugin points the variable there. The binary is a build
artifact, not committed; produce it from the repository source in any
CUDA 13 aarch64 devel container:

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
