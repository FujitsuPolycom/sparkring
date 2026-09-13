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

## Contribution reconciliation

Integration base: `506c8db0c09c95a75467e006110242cb5d0bcc7d` from main.
It includes PRs #259, #262, #265 and #269. The DCP4 evidence describes the
verified global-KV gather and arithmetic owner mapping; it does not claim a
separate top-k owner-exchange primitive.

The [adoption report](repository-layout-adoption.md#pending-contributions)
records pending PR heads and how their work fits the maintained owners.
Increasing peer-silence tolerance does not recover loss of the local
management address. Contributor credit remains in Git history.

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
