# Kraken distributed tuning-cache oracle, version 2

This independent CPU oracle covers the cache agreement protocol of the reconciled
Kraken candidate based on vLLM `af9e4dca109e0348323c0182e98a3aaf7282bfc3`
and B12X `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`. The adjacent JSON records
the test digest, observed source-file digests, and result. These are source-test
receipts, not evidence of the complete source trees' upstream provenance.
The consuming gate must verify the source manifest and test digest before
accepting results. Existing protected tests and suite manifests remain intact.

## Contract and source prerequisites

The source roots must explicitly be supplied through
`SPARKRING_B12X_SOURCE_ROOT` and `SPARKRING_VLLM_SOURCE_ROOT`; there is no
installed-package or baseline fallback.

B12X must expose the actual immutable `TuningCacheRequirement`,
`TuningRequirement`, and `CollectiveRequirement` types, schema-5
`SelectionCache`, and the `PreparationJob._advance`/`_run` handshake.
vLLM must expose `B12xPreparationCoordinator._decision` and its cache/tuning
authorization helpers. The source AST loader requires every selected method;
it never inserts missing production methods or rewrites their bodies.

Before lookup or racing, each TP participant supplies a snapshot. vLLM requires
the complete participant set and orders snapshots by rank. B12X validates
identity and exhaustive-race coverage, then uses the first completed record in
rank order for each key. Unequal local records are therefore permitted;
unequal installed selections are rejected by the oracle.

The reconciled candidate retains SparkRing's policy of reracing distributed
autotune even after cache agreement. Cold, warm, mixed, and conflicting local
caches must all reach the same tuning boundary. Cache-only preparation instead
consumes the agreed record, including a record that was absent locally.
This is a candidate contract, not an assertion that pristine upstream must
retain the SparkRing reracing policy.

## Coverage and seams

The 27 cases cover:

- TP2/TP4 cold, warm, mixed, and conflicting caches; cache exchange precedes
  races and installation; shared winner installation precedes collective warmup.
- TP2/TP4 cache-only mixed/conflicting agreement and cold-cache rejection.
- Cancellation from one control payload at either the cache or tuning boundary:
  no snapshots or partial winners are authorized; each job reaches the same
  default selection, and a cancelled cache handshake stays unsynchronized.
- Missing peers, wrong participant sets, identity mismatch, incomplete measured
  coverage, saved-config/lowering disagreement, and incompatible query identities.
- Rejection of an empty agreement without cancellation and reuse of a completed
  session agreement by a later job.
- A mutation that incorrectly retains rank-local choices: both cache agreement
  and final-selection equality assertions reject the divergence.

The test executes source-owned protocol types in full, real cache validation,
lookup, and reconciliation, selected complete `PreparationJob` methods, and
vLLM's complete decision/authorization functions. It initializes session state
as unsynchronized and changes it only by executing the production handshake.
The `_warmup_only` property is source-owned, including cancellation behavior.

Fixtures replace compile/configuration setup, candidate measurements, kernel
installation, progress packaging, and physical cache publication. They provide
valid records at the cache-storage seam and drive metadata exchange in-process.
Tests stop at the collective warmup boundary, before GPU execution and final
resource teardown. vLLM's stop decision is real; delivery of that decision to
each session's stop event is the explicit transport seam.

Not covered: TCPStore/process failure or deadlock behavior, actual GPU candidate
sharding and timing, kernel compilation or graph replay, executable artifact
verification, disk locking/atomic publication, full coordinator cancellation
delivery, installed-image admission, or serving quality/performance.
The upstream threaded coordinator tests and device preparation checks remain
necessary for those layers. CPU success is not GPU qualification.

## Run

From the integration repository in PowerShell:

```powershell
$env:SPARKRING_B12X_SOURCE_ROOT = 'C:/path/to/reconciled-b12x'
$env:SPARKRING_VLLM_SOURCE_ROOT = 'C:/path/to/reconciled-vllm'
python -m pytest runtime/images/upgrades/checks/kraken/distributed_cache_v2.py -q
```

Recorded result: **27 passed, 0 skipped**, Windows CPU.
