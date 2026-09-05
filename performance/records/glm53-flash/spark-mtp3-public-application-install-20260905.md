# Public four-Spark application-install rehearsal

Status: **research-only** functional evidence. Installation, initial serving,
native correctness, fresh cache publication, and a post-restart restoration
recall passed for the conditions below. This is not a throughput benchmark
or a factory-reset qualification.

## Conditions

Four prepared DGX Sparks used TP4/DCP4/PP1, the
`GLM-5.3-Flash-NVFP4-Spark` checkpoint at revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`, native MTP depth three,
16 sequences, batch 8,192, and the
[managed-mesh quickstart](../../../docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md).

| Artifact | Immutable identity |
|---|---|
| Public application source | `91c313d028877ada5fb1f04610f83c6465428657` |
| Public image manifest | `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:23f00af873ccc784cfb742b7be2a29c6d3c20ebec9741843c025320bb9c04685` |
| Image configuration ID | `sha256:26273b8e358df139ae913610a5d43084ff0fd08aafe282ef633a3bc74afefe47` |
| Extracted transport manifest SHA-256 | `4204fabc93303226b9a120b094ef3c82ed4aadd1d7f97cfbe291204c027ed45f` |
| Extracted forwarding helper SHA-256 | `2828c07e4255c4962c77425be2c88969e7eb7dd4b1bf9e36485bc705bb5d6d64` |

Every rank obtained a fresh public checkout and resolved the registry image
without authentication. Transport files were extracted from that image.
Rendered launch directories, cache roots, key/epoch, containers, and managed
installation targets were fresh. The exact planned mesh objects were removed
after the model-stop barrier and recreated by the installed service. The
prior containers and application state were retained for rollback.

OS, firmware, driver configuration, physical cables, IP addresses, MTUs,
GID assignments, and hardware hairpin prerequisites were reused. Docker
could reuse existing image layers. Model weights were **not redownloaded**:
all 44 indexed shards per rank were present, configuration/index hashes
matched, and safetensors headers were checked. This did not independently
hash every byte of the weight files.

## Measurement

The [sanitized JSON record](spark-mtp3-public-application-install-20260905.json)
preserves aggregate observations and SHA-256 fingerprints for 40 private raw
receipts. Full host inspection, resolved addresses, secrets, and private
filesystem paths are not included. The retained receipts cover inventory,
preparation, archive, installation, native checks, readiness, semantic checks,
per-rank publication state, coordinated restart, and restoration.

The audit compared each of 13 installed source files on each rank against
the public checkout and installation plan. It compared model commands,
environment, image, entrypoint, and mounts with the retained configuration
and rendered profile. Native Q4/Q20/Q28/Q64 checks counted five correctness
cases per rank and verified operation/path counters against their expected
values.

Initial readiness used controller `time.monotonic()` polling. The endpoint
was four running, Docker-health-healthy containers plus rank-zero API health
and scheduler liveness HTTP 200. Its elapsed time starts when that readiness
wait begins; it excludes preceding installation/download work and is not a
complete model-load timing.

Semantic and prefix-recall requests used temperature one. The prefix request
had 27,274 prompt tokens and zero external or RAM-prefix cache hits. Per-rank
logs and ownership journals were retained after publication and before the
coordinated restart. The same request was submitted after model restart,
with external-hit metric deltas and per-rank restore logs retained. There was one deployment and one observation per canary;
no variability estimate or performance comparison is claimed.

## Result

| Check | Result |
|---|---|
| Public checkout, anonymous image resolution, extracted artifacts | Passed on all four ranks |
| Empty cache roots and absent installation targets | Confirmed on all four ranks |
| Installed source identity | 52/52 file-hash comparisons passed |
| Create-only containers | All four stopped; installer did not start or enable services |
| Model command, environment, image, entrypoint | Equal to the retained configuration on every rank; matched rendered settings |
| Mount changes | Fresh transport/cache source paths only; model path and mount access modes unchanged |
| Retained containers | All four prior container IDs preserved |
| Mesh network ownership | Two routes, two qdiscs, and two rules created-and-owned per rank; none adopted |
| Native correctness | 80/80 cases passed across Q4/Q20/Q28/Q64 |
| Native counters | 3,168 payload writes, 3,168 flag writes, 3,168 send completions; zero errors or expectation mismatches |
| Initial combined readiness | Passed after 407.609 seconds of readiness polling, across 95 samples |
| Temperature-one semantic canary | Expected `SPARKCACHE_GLM53_OK`; finish reason `stop` |
| Temperature-one uncached recall | Expected `cobalt orchard lantern`; 27,274 prompt tokens, zero cache hits |
| Persistent publication | 26,624 tokens committed on each of four ranks |
| Restart combined readiness | Passed after 299.344 seconds of readiness polling, across 68 samples |
| Post-restart recall | Expected phrase at temperature one; 26,624 cached/external-hit tokens of 27,274 prompt tokens |
| Per-rank restoration | 26,624 tokens restored on every rank; 79.1 MiB per rank |
| Post-restart semantic canary | Passed at temperature one |
| Same-fabric restart identity | Container IDs, supervisor generations, shared readiness view, and helper processes unchanged |
| Final service state | All four containers healthy/running; supervisors armed and locally ready, peer health not degraded, two helpers each |

Each native shape exercised 24 origin paths and 33 operations per rank.
Container names and the two artifact/cache mount source paths intentionally
changed. Docker additionally represented `HostConfig.OomKillDisable` as
`false` rather than `null`; neither configuration disabled the OOM killer.
No model/scheduler/transport parameter change was found.

For the restored recall, logged rank 0–3 restore times were 153.7, 163.9,
184.2, and 178.0 milliseconds respectively. The full request elapsed time
was 2.718 seconds. These values describe different timing scopes and are
not a throughput comparison.

The raw request tool retains `persistent_restore_proven=false` by design:
it does not inspect remote rank logs. This record's restoration pass is based
on the combined evidence: identical request conditions, coordinated model
restart, unchanged container IDs, 26,624 external-hit tokens, zero RAM-prefix
hits, matching expected output, and explicit 26,624-token restoration logs
on all four ranks. No raw receipt was altered to assert the combined gate.

## Conclusion

The public source, image, extracted transport artifacts, and installer were
sufficient to create the application on four already prepared hosts with
fresh application and cache state. Selected native checks, initial serving
readiness, temperature-one output, fresh persistent publication, coordinated
model restart, and one persistent restoration recall passed.

This establishes an application-install result without relying on private
transport files or an existing application cache. It does not establish a
four-factory-reset setup result. The restoration conclusion is limited to
the recorded request and four-rank evidence, not a general cache-correctness
claim.

## Limitations

- Physical networking, drivers, firmware, GID/MTU setup, and hairpin
  configuration were not rebuilt from factory defaults.
- Existing Docker layers and model files were reused. Anonymous resolution
  proves public artifact accessibility, not a full cold-network transfer.
- Weight verification covered file presence, headers, and configuration/index
  hashes, not independent full-shard content authentication.
- Readiness polling elapsed is not total deployment time or a model-throughput
  measurement. Fresh caches and JIT compilation affect startup behavior.
- One stochastic semantic/recall observation does not establish comprehensive
  model correctness, cache correctness, or unattended availability.
- Publication alone does not prove restoration. The restore gate here uses
  the separate post-restart response, counters, and all-rank logs and covers
  only that bounded workload.
