# Promote a rebuilt EXL3 3.5-bpw operator profile

This checklist promotes a clean-checkout build of the operator-accepted EXL3
3.5-bpw fixed-MTP4 profile, whose durable recipe identifier is `R7`. The
existing acceptance applies to one four-DGX-Spark appliance and immutable image
ID. It does not transfer to another build.

## Publication and offline qualification

- [ ] Every runtime input is public, license-audited, and hash-pinned.
- [ ] `runtime/exl3-r7/prepare_context.py` creates and verifies its source
  receipt from a clean checkout.
- [ ] The ARM64 build receipt records the parent image ID, parent license
  expression, SparkRing revision, source receipt, wheel hashes, and resulting
  image ID.
- [ ] Offline tests, Ruff, the repository-relative Markdown link checker, and
  the release-safety/private-identifier scan pass.
- [ ] Numeric documentation claims have a retained evidence file and explicit
  hardware and measurement scope.

Offline checks qualify the builder only. They do not promote a deployment.

## Four-rank live qualification

- [ ] The identical rebuilt image ID is present on all four directly cabled
  DGX Sparks.
- [ ] Every check in the reviewed read-only preflight plan passes after the
  serving stack is stopped.
- [ ] Startup attests every baked runtime byte and every generated SIRCL and
  exact-Q40 bundle byte.
- [ ] Exact-Q40 pre-graph receipts pass on all ranks, including all 75 target
  layers and the unchanged uniform draft layer.
- [ ] Q1 through Q40 graph capture completes on every rank.
- [ ] `/health` returns HTTP 200; `/v1/models` reports the immutable served
  model identity, 262,144-token limit, and 1,156,864 reported KV tokens.
- [ ] Fixed-seed 16K and 32K response, token, and finite-logprob equality pass.
- [ ] The sealed C8 16K baseline-candidate-baseline bracket passes.
- [ ] Matched 8K, 16K, 32K, 64K, and 128K cold-prefill checks pass or retain a
  predeclared machine failure plus an explicit engineering waiver.
- [ ] Post-test API, queue, graph-sequence, fatal, overflow, transport, resource,
  and KV-idle gates pass.
- [ ] Starting the preserved rollback profile restores health and capacity.

## Promotion record

- [ ] Sanitized receipts and evidence identify the tested image ID and all four
  ranks without publishing site addresses, accounts, or local paths.
- [ ] Any machine failure remains labelled as a failure even when a documented
  engineering waiver accepts the configuration.
- [ ] `AGENTS.md`, `docs/STATUS.json`, `README.md`, the recipe, results, and the
  pull-request evidence table describe the same lane, maturity, hardware, and
  evidence scope.
- [ ] The promoted image is published by immutable digest or its local-only
  distribution procedure and limitation are stated explicitly.
