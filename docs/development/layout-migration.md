# Repository layout migration inventory

Base: `c65a9981e2a69f821ac716f6f13484d99d23f4ea`, fetched from
FujitsuPolycom/sparkring main on 2026-09-11. Work is isolated on
`refactor/repository-layout`. No deployment or publication is part of this change.

| Responsibility / current source | Decision and owner | Callers and verification |
|---|---|---|
| `recipes/*.json`, `recipes/sparkcache/*.json` | Profile inputs move to `profiles/`; legacy recipe files remain generated compatibility exports | Recipe checks, docs, deployment scripts; require byte equality at migration and generator freshness |
| `runtime/profiles/*/profile.json` | Identity-bound legacy contracts stay frozen; catalog explicitly distinguishes them from shared-image selections | Source-image lock hashes and installed asset verification; never substitute defaults from README |
| `runtime/sparkring/jovian-r33/` | Published build inputs remain at original paths; release index records their hashes | Publication receipts, image artifact lock, relative scripts, archived source; no rebuild or rehash as a published image |
| `runtime/sparkring/source_image/` | Frozen source composition; maintained image entry point under `runtime/images/` dispatches to the selected builder | Offline prepare-context and asset tests; preserve installed `/opt` paths |
| `runtime/exl3-r7`, `deepseek0731-gb10`, `deepseek-v41-gb10`, `qwen38` | Distinct dependency compositions remain separate builders, indexed by responsibility | Different engine commits/loaders/model formats; cannot consolidate into one image by renaming |
| `spark_transport/integrations/vllm` | Maintained adapters belong to `integrations/vllm`; byte-identical generated compatibility exports preserve overlay inputs | Python imports, public overlay allowlist, runtime build scripts, standalone injection and tests |
| `spark_transport/experiments/cx7_hairpin_diagonal` | Active fabric dependency graduates to maintained transport | `scripts/deploy_network.py`, managed profile, LIL, installed-service allowlist and source-bound bundle hashes |
| `spark_transport/experiments/glm53_rocenante_overlay` | Active integration dependency graduates to `integrations/vllm` | Mesh bundle builder and qualification; retain original source-bound exports |
| Other `spark_transport/experiments` | Audit individually; native substrate and source snapshots are not disposable | Dynamic imports, CMake includes, content manifests and runtime receipts preclude blanket relocation |
| `performance/harnesses`, `methodology`, `records`, `receipts` | Keep established ownership; do not relocate public evidence without verified replacement | Receipt URLs, source hashes and benchmark records |
| Operator quickstarts in `docs/` | Primary profile entry points link shared operations; relocate shared guides with compatibility pointers | Markdown links including existing anchors; preserve warnings next to operations |
| `integrations/lil` | Keep companion lifecycle boundary | Pinned Go companion tests; no vendoring of its implementation |
| Retired GLM variants | Retain explicitly indexed historical configurations and licenses | Existing published evidence remains valid only for its named configuration |

Generated compatibility exports have one authored source and are checked in CI.
They may be removed only after all documented callers, package inputs and
supported release consumers have migrated, with an announced breaking release.
Frozen release inputs are not generated from changing implementation.

## Pending contributions

PR #262 is merged and already present in the base. PRs #258, #263 and #265 remain open. PR #259 landed in main as
`84f2a01160e9d7542c06d1f52e0163907c1bbe77`; the restructuring branch was
rebased onto that commit. PR #266 adds a separate proposed management-address
loss mechanism and remains open. Their changes are not merged by this branch. Preserve #258's model
variant and cache-namespace separation, #259's bounded peer-silence mitigation,
#263's README wording, and #265's TP4/DCP4 selection/evidence when adopted.
An increase in peer silence tolerance does not recover disappearance of the
local management address. Contributor credit stays in normal Git history.

## Implementation checklist

- [x] Fetch and isolate; inspect instructions, CI and PR state.
- [x] Baseline Markdown links and release-safety scan.
- [x] Authoritative catalog, deterministic resolution and compatibility exports.
- [x] Component ownership, maintained launch/configuration and build entry points.
- [x] Documentation, contributor policy, maintainer prompt and templates.
- [x] Structural checks, regression tests and migration comparisons.
- [x] Reconcile main; prepare the adoption plan and local PR description.

The [adoption report](repository-layout-adoption.md) records verification, pending
contribution handling, compatibility limits and rollback.
