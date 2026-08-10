# ADR-0001: SparkRing ownership and extension extraction

- **Status:** Proposed
- **Date:** 2026-08-09
- **Decision owners:** SparkRing maintainers

## Context

SparkRing currently contains several kinds of work with different natural
owners: a recovered vLLM runtime delta, Spark-specific model and kernel work,
native transport, two cache integrations, launch and topology configuration,
attestation, qualification gates, and evidence. Treating all of this as one
independent inference runtime creates avoidable fork-maintenance pressure and
obscures which claims have actually been established.

The established product identity is:

> SparkRing is a qualification-driven four-Spark distribution and operational
> contract, with explicitly versioned configurations at accepted or
> live-validated maturity.

That sentence does not transfer maturity between configurations. The
public-functional NF3 configuration is the accepted deterministic alternative.
The default EXL3 3.25-bpw plus LMCache CS512 configuration is live-validated,
not accepted; its broader correctness, resource and rollback acceptance, and
release-promotion gates remain open. The generic runtime tooling is
offline-validated and is not a plugin implementation or a live-validated
generic runtime. These scopes remain governed by
[`STATUS.json`](../STATUS.json),
[`PUBLIC_FUNCTIONAL_TARGET.md`](../PUBLIC_FUNCTIONAL_TARGET.md), and the
applicable configuration evidence.

SIRCL is SparkRing's candidate differentiator, not yet its established core.
Its native transport and reference-lane validation do not yet prove
attributable value over the same serving binary's fallback transport. The
existing public probes also do not constitute the required same-stack A/B
campaign; the limits and evidence contract are recorded in
[`SIRCL_BASELINE.md`](../agents/SIRCL_BASELINE.md).

The runtime overlay must likewise be described by provenance rather than as
"SparkRing hacks." It consists of a recovered 71-file reference delta: 59
preimage-pinned modifications and 12 content-addressed, non-overwriting
additions. Two independently written, preimage-pinned SparkCache compatibility
modifications are then applied. The public builders therefore execute 73
ordered, fail-closed operations: 61 modifications and 12 additions. The
recovered files have best-effort attribution markers, not a conclusive
ownership assignment.
See the
[`reference overlay README`](../../runtime/patches/00-reference-vllm/README.md),
its [`provenance manifest`](../../runtime/patches/00-reference-vllm/provenance.json),
the [`SparkCache patch README`](../../runtime/patches/vllm/README.md), and the
canonical [`runtime lock`](../../runtime/runtime-lock.json).

## Decision

We propose a three-layer ownership model.

| Layer | Natural ownership | Responsibilities |
|---|---|---|
| Upstream inference engine | vLLM and SparkInfer | Generic GLM/DCP/MTP model and scheduler semantics belong in vLLM. GB10 kernels, quantization, sparse MLA, and Spark-specific kernel enablement belong in SparkInfer. Small missing extension interfaces should be proposed upstream rather than maintained as broad edits in place. Generic NCCL topology behavior belongs in NCCL; a pinned SparkRing patch may remain only as a temporary, qualified compatibility measure. |
| Independent extensions | SIRCL and, conditionally, SparkCache | SIRCL owns switchless RoCE/RDMA collectives, topology-aware schedules, CUDA-graph-safe command rings, and their native verification. A narrow vLLM-to-SIRCL adapter satisfies the upstream communicator interface at a plugin seam. SparkCache, if retained, is an independently versioned external KV-connector module behind vLLM's connector seam. |
| SparkRing distribution | SparkRing | Four-Spark cabling and topology contracts, immutable recipes and images, launchers, extension composition, rollback, attestation, qualification, acceptance gates, and sanitized evidence. SparkRing qualifies exact combinations; it does not acquire ownership of upstream engine semantics merely by distributing or patching them. |

This is a target architecture. Until extraction and live validation are
complete, the locked known-good distribution remains available and its
existing lane and maturity labels remain unchanged.

### Overlay inventory and retirement contract

Before an overlay operation is migrated or retired, all 73 operations must be
recorded in one reviewable inventory. Each row must contain at least:

1. **Provenance:** recovered reference modification/addition or independently
   written SparkCache patch, with the source evidence and attribution
   confidence.
2. **Active consumption:** whether the operation is consumed by each current
   locked configuration, with executable or source evidence; presence in the
   build is not proof of execution.
3. **Natural home:** vLLM, SparkInfer, SIRCL, the vLLM-to-SIRCL adapter,
   SparkCache, NCCL, SparkRing distribution, or historical/reference-only.
4. **Missing seam:** the specific upstream interface needed to remove the edit
   in place, or `none` with justification. A proposed seam is not treated as
   real until an adapter uses it; where behavior genuinely varies, at least two
   adapters should exercise the interface.
5. **Retirement proof:** the replacement commit and artifact identity,
   preimage/operation removal, focused tests, same-configuration behavioral
   evidence, rollback rehearsal, and updated lock/attestation result required
   before deletion.

Migration is incremental. Each replacement lands behind an explicit selector
or other reversible rollout mechanism, is tested in the locked distribution,
and has a documented rollback to the last attested artifact set. An operation
is not retired merely because equivalent-looking code exists upstream.

### SIRCL decision test

The decisive question is:

> Does native SIRCL provide repeatable, attributable value—performance,
> correctness, graph compatibility, topology determinism, or operational
> reliability—over the same binary using its fallback transport?

The comparison must use the same serving image, executable, libraries,
checkpoint, runtime lock, four-Spark topology, launch arguments, workload,
graph/cache state, and measurement procedure. The declared transport selector
is the only intended variable. Comparing different SIRCL and NCCL images or
library builds is not transport attribution.

Both arms must fail closed unless they prove which path executed. Synchronized
per-rank before/after counters must show native SIRCL work in the SIRCL arm and
account for every relevant stock/fallback path; a requested mode string,
process success, or positive custom count alone is insufficient. Required
gates are:

- numerical and request correctness, including the FP32 audit where
  applicable, non-finite rejection, and token-level evidence for serving;
- complete custom/stock/fallback accounting, with no unexplained or ambiguous
  fallback;
- eager and CUDA-graph capture/replay compatibility for every claimed
  collective family;
- deterministic behavior on the exact four-direct-cable topology, including
  rank/edge mapping and pre/post fabric counters;
- repeated startup, steady-state, timeout, teardown, and restart reliability,
  including failure disposition rather than discarded failed trials; and
- separately labelled host submission, GPU/transport completion, and
  end-to-end performance measurements with repeated, randomized A/B cells.

Correctness and execution attribution are mandatory acceptance conditions, not
tradeable benefits. If they pass, repeatable value on performance, graph
compatibility, topology determinism, or operational reliability can justify
SIRCL; speed alone is not required. If native SIRCL cannot demonstrate such
value, SparkRing retains its distribution and qualification identity but must
stop presenting SIRCL as an independently differentiated runtime capability.

### SparkCache disposition

SparkCache remains unpromoted while its public-runtime integration is not
accepted. If retained, it must be extracted behind vLLM's external KV-connector
seam and versioned independently from the SparkRing distribution. The locked
default may continue to use LMCache while this work proceeds.

Promotion requires evidence of a useful capability not adequately supplied by
LMCache—principally durable persistence across a full stack restart—plus
cold/store, engine-restart and full-restart restore, corruption withdrawal,
rollback, correctness/equivalence, and C8 interference gates on the same
versioned configuration. If it does not establish that benefit, SparkRing will
use LMCache and deemphasize or retire SparkCache rather than preserve a second
cache implementation solely because it exists in this repository.

## Consequences

- SparkRing has a coherent product interface even if most inference-engine
  changes migrate upstream: an exact, qualification-driven four-Spark
  distribution and operational contract.
- SIRCL and SparkCache can evolve and be tested through narrow extension seams
  without requiring a broad vLLM overlay, improving locality of transport and
  persistence changes.
- Overlay maintenance becomes measurable: every operation has an owner,
  active-use evidence, a migration dependency, and a retirement proof.
- Extraction takes multiple releases. During that period, both the locked
  overlay path and plugin/adapter candidates may exist, increasing temporary
  test and release work.
- Upstream acceptance is outside SparkRing's control. Hash-pinned temporary
  patches may remain where necessary, but their presence and maturity must stay
  explicit.
- No accepted or live-validated maturity transfers to a newly extracted
  module, adapter, runtime, checkpoint, or configuration. Each must pass its
  own named gates and publish sanitized evidence before promotion.

## Non-goals

- This ADR does not claim that SIRCL already wins the decision test or that a
  vLLM communicator plugin has been live-validated.
- It does not accept EXL3, promote SparkCache, or elevate the generic runtime
  beyond its current offline-validated tooling scope.
- It does not deprecate the accepted NF3 alternative or replace the default
  EXL3 configuration.
- It does not authorize a flag-day rewrite, deletion of the recovered overlay,
  or retirement of the known-good locked distribution.
- It does not transfer generic inference semantics to SparkRing or
  hardware-specific kernel ownership to vLLM by default.
- It does not treat performance evidence as functional acceptance, nor move
  measurements between lanes, binaries, hardware, or configurations.

## Implementation sequence

1. Review and accept this ownership decision.
2. Publish and review the 73-operation inventory using the required fields.
3. Add only the upstream extension interfaces required by the first vertical
   slice.
4. Build the smallest SIRCL communicator-adapter tracer bullet and execute the
   same-binary, transport-selector-only decision campaign.
5. Extract SparkCache through the external connector seam and either promote it
   through its persistence and interference gates or choose LMCache.
6. Retire overlay operations one at a time after their retirement proof passes,
   updating the runtime lock, attestation, rollback artifacts, status, and
   evidence labels at each step.
