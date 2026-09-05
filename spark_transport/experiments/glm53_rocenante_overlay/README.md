# Source-bound GLM-5.3 collective routing

Status: **research-only**. This package composes a RoCEnante all-reduce wrapper
with SIRCL and the existing NCCL fallback. It targets the exact GLM-5.3 vLLM
Python sources identified in [`overlay_contract.json`](overlay_contract.json).
It is not a generic vLLM command-line option or an unmodified installation of
an upstream PR.

The [native-MTP3 profile](../../../runtime/glm53-spark-mtp3-mesh/README.md)
selects the serving policy and creates the complete content-addressed bundle.
Only `b12x.comm.roce` is overlaid; attention, MoE, KDA, and linear kernels stay
in the pinned image's B12X package.

## Dispatch contract

Eligible RoCEnante operations are contiguous CUDA BF16 TP4 all-reduces shaped
`[Q,4096]`, with Q from 1 through 32. Other signatures call the already
installed SIRCL wrapper, which independently selects graph-native SIRCL,
fused eager SIRCL, or NCCL. RoCEnante all-gather is disabled.

For the MTP3 profile, captured Q16, Q20, Q24, Q28, and Q32 also delegate to
SIRCL. Captured Q4, Q8, and Q12 and eligible eager Q1 through Q32 use
RoCEnante. Larger configured target captures remain on SIRCL. These are
transport choices for tensor rows, not a mapping from prompt length.

Six persistent origin QPs per rank supply two paths to each of three peers.
The runtime uses two operation slots and CPU 13 for its proxy. Q24 is the
direct-then-diagonal threshold; smaller eligible operations use both path
sets. The four-rank HCA order and peer-path mapping are source-bound inputs,
not topology autodetection.

## Initialization and failure boundary

`sitecustomize.py` installs the SIRCL integration and the source-bound
RoCEnante wrapper. The wrapper verifies its contract and source preimages,
then uses the existing TP CPU/Gloo group for rank-wide capability agreement.
It does not create a second device process group or NCCL communicator.

Native health is checked at vLLM's synchronized output boundary. A failure
after operation publication is fatal to the worker; it is not silently
retried through a different collective. Mixing fallback decisions after
publication could diverge collective order across ranks. The implementation
exits with code 70 on the specified fatal boundary.

These code paths do not qualify marker-expiry or mesh-path failure handling.
The `sidecars` section of `overlay_contract.json` records requirements for an
external host orchestrator. It does not mean that bundle composition installs
host rules, renews markers, or performs automatic host cleanup.

## Bundle construction and inspection

Use `runtime/glm53-spark-mtp3-mesh/profile.py bundle` for the pinned MTP3
policy. `build_bundle.py` is the lower-level composer: it verifies the base
SIRCL manifest, copies the selected communication sources, records exact
file hashes, and refuses an existing output directory. It makes no host or
network changes.

```bash
python -m pytest spark_transport/experiments/glm53_rocenante_overlay -q
```

The tests cover source identity, signature routing, bundle manifests,
capability voting, and the Python health boundary. Live CUDA graphs, RDMA,
cache correctness, and fault containment require separate hardware tests.

## Attribution and design origins

RoCEnante is Local Inference Lab's communication implementation, not a
SparkRing invention. Credit includes Luke (`lukealonso`), the project's
contributors, and PR author `original-el8`. The included implementation derives from
[B12X PR 295](https://github.com/local-inference-lab/b12x/pull/295).
[`third_party/b12x_roce`](../../../third_party/b12x_roce/README.md) contains
the selected modified source, provenance, and retained Apache-2.0 license.
[vLLM PR 597](https://github.com/local-inference-lab/vllm/pull/597) supplied
integration ideas, including rank-wide agreement and post-step health checks;
the adapter here is bound to the separately pinned serving runtime. Neither
link implies that the complete PR is installed unchanged.

The two upstream PRs motivated SparkRing's hardware-forwarded diagonal
exploration on a physical ring. The topology adaptation, hybrid SIRCL
dispatch, and managed deployment are SparkRing's integration work.
