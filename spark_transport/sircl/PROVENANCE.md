# Provenance of the SIRCL ring-session package

This page names the origin of every file of `spark_transport/sircl`, the
SIRCL ring-session package. Paths below are relative to that directory
unless they start with a repository directory. Status of each component:
[STATUS.md](STATUS.md#component-status).

## Origins

| Label | Source | Identity |
|---|---|---|
| **b12x** | `local-inference-lab/b12x`, Apache-2.0 (Local Inference Lab) | commit `236ddff04a5f5064084b46464800b70b34ec7d8c`. b12x, including its RoCE transport RoCEnante (`b12x/comm/roce`), is developed inside FlashInfer at `flashinfer/experimental/b12x` (`local-inference-lab/flashinfer`; added to `flashinfer-ai/flashinfer` as a submodule in `fc8fdc17f702371302a0ea780c604b099b927196`) |
| **SparkRing** | the SparkRing repository, Apache-2.0 | commit `6d27dd10a9f97da21278ae504ea6235f8634f303`, or written for SparkRing and first published in this package where a row says so |
| **new** | written for the SIRCL ring sessions | |

Blob ids below are `git rev-parse <commit>:<path>` of the origin file.

## Licenses and documents

| File | Origin |
|---|---|
| `LICENSE` | b12x `LICENSE` (blob `261eeb9e`), the Apache License 2.0 text, unchanged |
| `NOTICE` | new |
| `README.md`, `RUNBOOK.md`, `STATUS.md`, `PROVENANCE.md`, `pyproject.toml` | new |
| `sparkring_sircl/vllm/README.md`, `RUNBOOK.md`, `STATUS.md`, `SURVEY.md` | new |

## Package `sparkring_sircl`

| File | Origin |
|---|---|
| `__init__.py`, `env.py`, `protocol.py`, `routes.py`, `agreement.py`, `build.py`, `pieces.py`, `cpus.py`, `references.py`, `bounds.py`, `teardown.py` | new |
| `groups.py` | new; reproduces vLLM's rank layout (read from vLLM `distributed/parallel_state.py`) |
| `roce_gid.py` | new; imports SparkRing's resolver `integrations/vllm/spark_roce_gid.py` and re-exports it, so the package carries no copy of it |
| `oneshot/__init__.py`, `oneshot/_compile.py` | new; `make_pointer` prefers b12x's runtime pointer wrapper (`b12x/_lib/utils.py`) when b12x is installed |
| `oneshot/runtime.py` | new; the setup exchange, stream ordering, capture handling, alignment scratch and padded all-gather follow b12x `roce_oneshot.py` (blob `b3436b99`) |
| `oneshot/_proxy.py` | new; the build-and-bind pattern follows b12x `_proxy.py` (blob `4a9ce229`) |
| `oneshot/_roce_proxy.c` | new; the arena protocol, doorbell catch-up, queue-pair parameters and completion handling follow b12x `_roce_proxy.c` (blob `27b45a43`) and SparkRing's per-peer copy `third_party/b12x_roce/.../_roce_proxy.c` (blob `c8f9d44b`) |
| `oneshot/_cute_intrinsics.py` | SparkRing `third_party/b12x_roce/.../_cute_intrinsics.py` (blob `6517dc3f`), which is b12x `_cute_intrinsics.py` (blob `1e0ac9db`) with two clarified docstrings; line endings normalized |
| `oneshot/_oneshot_cute.py` | derived from b12x `_oneshot_cute.py` (blob `92fe911c`); the lane-count flag wait, the command-ring words and the compile helper are new |
| `oneshot/_allgather_cute.py` | derived from b12x `_allgather_cute.py` (blob `0c4de63d`); the lane-count flag wait, the command-ring words, the tiled launcher and the compile helper are new |
| `oneshot/_twoshot_cute.py` | new; the pack arithmetic and the stage, doorbell, wait and epoch steps follow `oneshot/_oneshot_cute.py` |
| `oneshot/_chain_cute.py` | new; the pack arithmetic and the fence, flag and counter patterns follow `oneshot/_oneshot_cute.py` |
| `oneshot/_links_cute.py` | new; the fence, flag, timed-wait and counter patterns follow `oneshot/_chain_cute.py` |
| `oneshot/_timed_wait.py` | new; the inline-assembly helper follows b12x `_cute_intrinsics.py` (blob `1e0ac9db`) |
| `oneshot/_fast_launch.py` | new |
| `oneshot/_scatter_cute.py` | SparkRing: the reduce-scatter and all-to-all kernel written for SparkRing's eight-Spark serving and first published here, with this package's compile and timed-wait calls; its dtype pack arithmetic and stage, doorbell, wait and epoch steps follow b12x `_oneshot_cute.py` (blob `92fe911c`) |
| `oneshot/_swing_cute.py` | SparkRing: the Swing all-reduce kernel written for SparkRing's eight-Spark serving and first published here, with the schedule of `swing_plan.py` and this package's compile and timed-wait calls; pack arithmetic after b12x `_oneshot_cute.py` (blob `92fe911c`) |
| `oneshot/_cute_batch.py` | SparkRing: the batched-load intrinsic written for SparkRing's eight-Spark serving and first published here; the inline-assembly call follows b12x `_cute_intrinsics.py` (blob `1e0ac9db`) |
| `oneshot/_scatter_ops.py` | SparkRing: the host code of the scatter collectives (geometry, launch) written for SparkRing's eight-Spark serving, restructured as functions of a session; the relay-safe op split is new |
| `oneshot/_swing_ops.py` | new; the launch arguments follow the same SparkRing serving code |
| `fused_norm/__init__.py`, `_geometry.py`, `_reference.py`, `_ptx.py`, `_kernel.py`, `runtime.py` | SparkRing: the fused all-reduce + residual add + RMSNorm package written for SparkRing's eight-Spark serving and first published here, with this package's session import and API check, timed flag wait, compile helper, `bind`, `allreduce_add_rms_norm` and vLLM's `try_fused_add_rms_norm` interface; `_kernel.py` follows the layout of b12x's PCIe fused all-reduce + RMSNorm (`b12x/comm/pcie`) |
| `scatter_plan.py`, `swing_plan.py`, `posting.py`, `latency_model.py`, `tuning.py`, `callprofile.py` | new |
| `p2p/__init__.py`, `protocol.py`, `settings.py`, `budget.py`, `build.py`, `_native.py`, `session.py` | new; the setup exchange and agreement follow `oneshot/runtime.py` |
| `p2p/_p2p_proxy.c` | new; queue-pair setup, connection records and completion handling follow `oneshot/_roce_proxy.c` |
| `p2p/_kernels.py` | new; the fence, flag, batched-load and timed-wait patterns follow `oneshot/_chain_cute.py` and `oneshot/_timed_wait.py` |
| `ring/*` | new |
| `fabric/mesh_marker.c` | SparkRing: derived from the RDMA-TX rewrite probe `spark_transport/fabric/cx7_hairpin_diagonal/native/mlx5_rdma_tx_rewrite_probe.c` (blob `49948e7a`); one process per RDMA device with one rule per destination address |
| the other files of `fabric/` | new |
| `testing/*` | new; `testing/fake_verbs/infiniband/verbs.h` declares the libibverbs subset the native layer uses with libibverbs' names and values |
| `vllm/*` | new; its shims replace the vLLM functions that `vllm/hooks.py` names, pinned to vLLM builds by file hash in `vllm/pins.py` |

## Tests

| File | Origin |
|---|---|
| `tests/*` | new |

## Used in place

The four-rank SIRCL sources (`spark_transport/include`, `spark_transport/src`,
`spark_transport/CMakeLists.txt`), the four-rank vLLM adapter modules in
`integrations/vllm` (among them the RoCE GID resolver `spark_roce_gid.py`)
and `third_party/b12x_roce` are used where they are and are not copied into
this package.
