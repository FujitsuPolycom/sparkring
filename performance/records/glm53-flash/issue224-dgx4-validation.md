# Issue 224: DGX4 GPU validation

Status: qualified for the bounded publication-order regression and the tested
single-GPU indexer cases. End-to-end incident resolution remains unqualified.

## Conditions

Tests ran on 2026-09-06 on DGX4's four NVIDIA GB10 GPUs (SM121), using isolated
containers from the affected operator image:
`sha256:5e32aaa1bbe3559e81db7706ed4286248f18d27cfdb186f6b851bf786eb43075`.
The runtime reported PyTorch `2.13.0+cu130`; the installed indexer file matched
the affected B12X source byte-for-byte:
`d3ec6274e142a4e7d1062ea6d2d99b97db0a02e92bb976c6570ae990b836b18d`.

The four existing `sparkring-linked-test-20260905-r*` serving containers were
left running. Their image is newer (`2e41b1e934a8`); inspection of rank 0
confirmed that its barrier still lacks the entry synchronization. The patch was
not installed in those serving containers. All four remained healthy after the
tests, with zero GPU utilization in the final snapshot.

## Measurement

The bounded GPU probe compiles the installed `_fused_group_barrier()` through
CuTe. Three cooperatively launched blocks publish a small histogram; one
nonleader warp delays its contribution by a bounded GPU-clock loop. Each block
records the histogram sum it observes after the barrier. The expected sum is
513. There is only one group barrier, so the test exposes premature consumption
without intentionally executing the divergent second round that would hang.

Each GPU ran 10 eager and 10 graph-replay probes with the original helper, then
the same 20 probes with the proposed entry synchronization. The first graph
harness attempt incorrectly used the pre-capture stream and was rejected; the
recorded successful run uses the active capture stream and verifies nonzero
results. No result from the rejected attempt is included in the totals.

The full-indexer tests mounted the patched source read-only into a separate
container on rank 0. Its SHA-256 is
`c63ac2712dc19bc67cb6e892de24751d58a534348e29121623daee64776e2679`.
The patch adds the block synchronization and increases the fused-kernel compile
revision from 1 to 2. Compilation caches were private to the test directory.

The image-builder transform emits the same source with LF line endings:
`49f6fd916fd1ccf94311ee99427551edbd0dc3a5de23aeeb426418370f76f66d`.
That exact output was compiled again in an isolated container on rank 0; all 15
selected GPU tests passed again. The original 2,000-replay results below belong
to the mixed-line-ending file identified above.

Fifteen existing GPU tests covered reference top-k values and selected-index
sets, partial pages, short contexts, repeated cooperative merges, padding,
counter cleanup, and graph replay switching between serial/cooperative merge.

An additional stress harness ran 500 graph replays for each GLM-shaped case
below. Every replay checked sorted top-k values against the reference with
absolute tolerance 0.01, exact selected-index sets, and cleared merge state.
Lengths alternated between 4,097 and the case's maximum. Each replay also
submitted a 64 MiB device-to-device copy on a separate stream.

## Result

| Test | Result |
|---|---|
| Original helper, all four GPUs | 80/80 probes observed incomplete data |
| Entry synchronization, all four GPUs | 80/80 probes observed complete data |
| Existing patched full-indexer GPU tests, rank 0 | 15 passed; 79 unrelated cases deselected |
| 3 rows, 32 heads, top-k 512, maximum 65,536 tokens | 500/500 graph replays passed |
| 4 rows, 32 heads, top-k 512, maximum 200,000 tokens | 500/500 graph replays passed |
| 8 rows, 32 heads, top-k 512, maximum 65,536 tokens | 500/500 graph replays passed |
| 16 rows, 32 heads, top-k 512, maximum 65,536 tokens | 500/500 graph replays passed |

Original probes consistently returned `[512, 513, 512]`: two blocks passed the
barrier before the delayed contribution. Fixed probes consistently returned
`[513, 513, 513]`, in both eager execution and CUDA graph replay.

## Conclusion

The publication-order race is confirmed on all four DGX4 GPUs. The proposed
entry synchronization fixes that measured race. The patched full indexer
compiles and passes the tested numerical and graph-replay cases on GB10,
including 2,000 replays with concurrent device copies.

This supports testing a patched serving image. It does not establish that all
reported EngineCore stalls have the same cause or that the production incident
is resolved.

## Limitations and next serving test

The probe deliberately forces a delayed publisher; it does not measure the
natural failure rate. The stress test exercises one indexer on one GPU, uses
synthetic inputs and device copies, and does not run actual SparkCache restores,
TP4 collectives, the full model, or a multi-hour workload. No performance claim
is made from these tests.

The next gate is a coordinated four-rank canary using a child image that changes
only the barrier fix and its compile revision, with a separate JIT namespace.
Repeat the issue's concurrency-three-or-higher restore/decode workload, retain
full worker-thread stacks on stalls, and compare against the same unpatched
image/configuration. The running test stack has not been replaced.

## Reproduction artifacts

[Raw evidence](issue224-dgx4-evidence.json) contains probe and stress observations
and artifact hashes. The [GPU harnesses](../../harnesses/indexer_barrier/README.md)
and the image builder's
[checked transform](../../../runtime/glm53-flash-jj-r8-gb10/patch_indexer_barrier.py)
are checked in. The existing test file comes from the pinned B12X repository.

Run `gpu_barrier_probe.py` inside the affected image with GPU access. For the
full-indexer tests, apply the checked transform to that image's B12X source in
an isolated child image or use a read-only file mount. Install pytest
8.4.2 in the test environment, then run:

```bash
python3 -m pytest test_fused_indexer.py -q -x -k 'paged_matches_reference or partial_last_page or short_context_no_radix or preinitialized_state_graph_replay or cooperative_merge_repeated_launches or cooperative_pack_path'
python3 stress_indexer.py
```
