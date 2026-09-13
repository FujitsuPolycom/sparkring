# GLM-5.3-Flash TP4: LIL R37 source upgrade

Status: **Experimental**. These bounded checks cover one ARM64 candidate using
LIL R37 application sources and SparkRing integrations. They do not establish
long-duration stability or resolve the previously observed native TP4 stall.

The [machine-readable record](r37-tp4-source-upgrade.json) binds image
`sha256:beeb32253aa754cf8e22f7c054b21d659e041388a06c0427473bebe77d564f97`
to the integrated source trees, benchmark settings and worker-log hashes.
The candidate is local; there is no registry pull command for it.

## Configuration and build

Four GB10 Sparks run GLM-5.3-Flash NVFP4-Spark checkpoint revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`, TP4/DCP1 and MTP3, with 1M
configured context and 24 GiB KV per rank. SparkCache uses read-write storage;
mHC sharding, continuation-prefill coalescing, direct-doorbell SIRCL and the
patched dual-domain NCCL configuration are retained. OMP uses one thread, with
graph submission/progress assigned to CPUs 10/11.

Pinned LIL vLLM and B12X sources are applied over the verified R35 ARM64 image.
The fully patched native/build inputs match the integrated foundation references
listed in the record. Source packaging is deterministic. Installed-image checks
verified 57,226 files, and every rank passed host admission against the candidate
descriptor, lease contract and inherited native hashes. No CUDA/PyTorch rebuild
or published artifact replacement was performed.

## Bounded comparison

The same harness ran two cold 8K prefill samples and one short decode measurement
at each concurrency. Prefill requests contained 8,194 actual prompt tokens with
server-confirmed zero cache reuse. Decode used zero added context, 10 seconds
of warmup and a 20-second measurement window per concurrency.

| Measurement | R35 | R37 candidate |
| --- | ---: | ---: |
| Cold 8K prefill, tokens/s | 3,330 | 3,330 |
| TTFT, seconds | 2.469 | 2.468 |
| Decode C1, tokens/s | 57.69 | 55.59 |
| Decode C4, aggregate tokens/s | 123.39 | 137.11 |

Both decode cells passed validity checks with zero errors. This single short
comparison does not establish a statistically significant improvement or regression.

## Cache and integration checks

Three exact arithmetic requests passed. A cold 8K lookup request returned the
expected answer with 8,192 created cache tokens. An immediate warm request was
also correct, reporting 7,680 cached tokens and 512 created tokens. That resident
cache path is distinct from external restore after a process restart.

After stopping and restarting all four model processes, the identical lookup
request reported **8,192 cached tokens and zero created cache tokens**. Every
worker logged an external 8,192-token restore. Two changed-tail requests returned
the exact expected answers while reusing that prefix. Container IDs and image
identity remained unchanged across restart; process start times changed.

Worker logs confirm 8,192-row mHC prefill with 2,048 owner rows per rank,
90 reduce-scatter and 90 all-gather calls, and no auxiliary gathers. Scheduler
logs confirm coalesced prefill. Final API health passed and request queues were
empty. No long soak was run.

## Adoption boundary

The candidate remains separate from published R35 defaults. Its source archives,
build context, original requests, benchmark outputs and rollback inputs are
retained with the local evaluation workspace; this summary is not a standalone
release recipe. Promote only after reviewing the desired model/topology scope
and outstanding stability requirements. Preserve the R35 image, containers and
cache for rollback; stop every candidate rank before restarting that deployment.
