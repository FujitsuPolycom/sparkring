# Reconciled B12X tuning selections on multi-rank starts

Status: **implemented**. Installer image
`dev-20260930-spinwait-cuda1342-nccl2323-status033`, which every installer
profile runs, carries it from its ancestor `dev-20260927-b12xcache-cuda1342-nccl2323-status032`.

## Condition

B12X prepares kernels by measuring candidate configurations and storing each
winner in a per-rank selection cache. vLLM's B12X warmup
(`vllm/model_executor/warmup/b12x_prepare.py`, `get_b12x_session`) shards that
tuning across the tensor-parallel ranks. At the start of a sharded preparation
job, `PreparationJob._run` reconciles the ranks' caches
(`SelectionCache.reconcile`); afterwards every rank reads the same agreed
records and `PreparationSession._tuning_cache_synchronized` is true.

SparkRing's B12X source composition `c8e461281a38`, which the installer image
carries, also makes `PreparationJob._lookup` skip the selection cache whenever
tuning is sharded and autotuning is active, including after reconciliation.
Upstream B12X does not have that skip. Every two- and four-rank start therefore
measures every kernel family again although the cache already holds the
selections. The DeepSeek-V4.1-Flash four-Spark installation with a warm cache
spent 6 min 12 s in B12X preparation
([installation record](../../../performance/records/images/dev-20260927-h2dstaging-deepseek-v41-tp4-20260927.md#installation)).

## Behavior

[`sparkring_b12x_selection_cache.py`](sparkring_b12x_selection_cache.py)
replaces `PreparationJob._lookup` after `b12x.preparation.session` loads. The
replacement is the composition's method with one added condition: the
sharded-tuning skip applies only while `_tuning_cache_synchronized` is false.

- After reconciliation, a sharded lookup reads the agreed record, so a cached
  selection is used on every rank or on none.
- Before reconciliation, for example in a warmup-only job, a sharded lookup
  still skips the cache, so a rank-local hit cannot remove a rank from a shared
  race.
- Single-rank, cache-only and stopped-tuning sessions behave as before.

The correction applies only when the loaded `session.py` has SHA-256
`83bf552115fdb2bafa399ccfa8d7adbe4958121a3891243eff117a47ea20a31e`, the
composition's file. Any other B12X source is left unchanged and the process
writes `SparkRing B12X selection-cache correction not applied` to standard
error. A failure while correcting the module is reported the same way and never
fails the import; the unchanged method is correct, only slower. A corrected
process writes `SparkRing B12X selection-cache correction: PreparationJob._lookup
reads reconciled selections`.

The prepared RoCE transport ([rocenante_prepared](../../vllm/rocenante_prepared/README.md))
hashes six B12X sources, including `b12x/preparation/session.py`, at startup
and refuses to start if one differs from its manifest. The correction leaves
every B12X file byte-identical and changes only the loaded class.

## Installation in an image

Both files go into the serving interpreter's site-packages,
`/usr/local/lib/python3.12/dist-packages/` in the installer image:

- `sparkring_b12x_selection_cache.py`, the correction module;
- [`sparkring_b12x_selection_cache.pth`](sparkring_b12x_selection_cache.pth),
  which imports it at interpreter startup, including in spawned workers.

The module inserts its import finder immediately before Python's path finder,
so finders that SparkRing inserts at the front of `sys.meta_path`, such as the
transport selector, keep their position. The derived-layer
[descriptor](../../../runtime/images/compositions/installer-b12xcache-status032/descriptor.json)
pins both files for [`runtime/images/derived_layer.py`](../../../runtime/images/README.md#derived-installer-layers).

Remove the correction from derived layers once SparkRing's B12X composition
carries the condition in `session.py`; the correction is inert on that source
because its SHA-256 differs.

## Evidence

- **CPU tests.** [`test_sparkring_b12x_selection_cache.py`](test_sparkring_b12x_selection_cache.py)
  drives the replaced lookup with session, cache and obligation doubles: a hit
  after reconciliation, a skip before it, single-rank, cache-only,
  stopped-tuning and autotune-off hits, a miss, and a rejected record. It
  compares the replacement's syntax tree with the composition's method and
  exercises the import finder on a stand-in package. Set
  `SPARKRING_B12X_SOURCE_ROOT` to a checkout of the composition to compare
  against the complete `session.py`.
- **B12X tests.** With the correction imported, B12X's
  `tests/preparation/test_session.py` from the composition, with upstream's
  expectations for `test_two_ranks_agree_on_cached_choices_and_shard_remaining_races`
  (cached ranks select `cached` and race two of four candidates), passes on
  CPU with import stubs for Triton and CUTLASS; without it the four two-rank
  cases fail. One unrelated test needs `triton.language` and fails in that
  environment either way.
- **Two starts of an installer deployment.** `qwen38-flash-next-tp2` on image
  `dev-20260927-b12xcache-cuda1342-nccl2323-status032` on one pair. The first
  start of the deployment compiled 136, 200 and 38 kernels across the RoCE and
  two GEMM families and was ready 342.7 s after start. After `sparkring down`
  and `sparkring up`, the second start took every selection from the cache,
  compiled nothing and was ready in 181.1 s.
- **One four-Spark start.** Conditions: DeepSeek-V4.1-Flash on four GB10 Sparks
  with the installer profile's serving arguments, the installer image, a B12X
  tuning cache that already held the installer deployment's selections, and a
  bind-mounted module with the same replacement method but without the source
  SHA-256 check and with its finder at the front of `sys.meta_path`. Result:
  the server was healthy 200 s after start; B12X logged
  `gemm.block_fp8_linear: 857/857 ready, 0 measured, 281 cached` and
  `attention.compressed_sparse_mla: 857/857 ready, 20 measured, 273 cached`.
  Greedy decode of 512 tokens ran at 49.3, 97.3 and 112.0 tokens/s for prose,
  code and JSON, and 16K-token prefill at 4,028-4,461 tokens/s, against 49.8,
  100.0, 110.8 and 4,406 tokens/s for the installation without it. That
  installation reached rank-0 API readiness in 652.7 s, measured by the
  installer rather than the same harness. This is one start, not a serving
  qualification; an image that carries the correction needs its own hardware
  check.

```bash
python -m pytest integrations/b12x/selection_cache -q
```
