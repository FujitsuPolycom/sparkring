# Repository layout adoption plan

## Result and base

`refactor/repository-layout` is the public integration branch. Its local history
includes main through `0b7c08d3e4b3b88c568fb719c365ab523de6db3b`, the canonical
Qwen TP2 quickstart through `22a6dd1`, and shared Docker/Compose deployment.
Repository adoption and image deployment are separate decisions.

The [profile catalog](../../profiles/README.md) owns deployment discovery.
Profiles select model settings and image releases; private site files select
hosts, interfaces and paths. Shared adapters render effective settings for
Docker and supported [Compose deployments](../operations/compose.md).
[Repository ownership](layout.md) identifies implementation owners and retained
compatibility paths. [AGENTS.md](../../AGENTS.md) and the
[maintainer prompt](maintainer-prompt.md) reference one writing policy.

Contributions require neither Spark hardware nor an issue before a PR.
Incomplete reports are welcome; maintainers own hardware qualification.

## Compatibility and migration

| Surface | Contract |
|---|---|
| Recipes, serving ENV examples and legacy adapter imports | Generated from canonical owners; retain public paths and recorded defaults |
| Retained GLM TP2 launcher | Delegates to `runtime/common/tp2.py`; exposes only selected HCA functions and indexes peer maps within that list |
| Switched launcher | Frozen image source remains unchanged; the maintained host adapter owns development |
| Docker and Compose | Share container specifications; Compose coordinates separate per-host projects and rejects project collisions |
| Guides | Canonical instructions retain compatibility URLs and heading anchors |
| Published images and source inputs | Preserve image identities, installed paths, 430 inventoried inputs and 22 locked profile assets |
| Native substrate and incompatible builders | Retain identity-bound locations documented in the ownership guide |

Run `python scripts/generate_profiles.py` after changing compatibility sources
and `python scripts/generate_compose_examples.py` after changing Compose inputs.
Generated exports are not independent implementations. Retiring a public path
requires a separate compatibility decision after its consumers migrate.

GLM TP4/DCP1 remains the default; DCP4 is an alternative with its own evidence.
The DeepSeek vLLM profile uses 1M context; its SGLang profile retains the
contributor's 262K default. Measurements retain their actual configurations.

## Verification

Evidence applies to the named source, image, model and topology. Passing CPU
checks, publishing an image and qualifying model serving are distinct outcomes.

| Evidence | Proven scope |
|---|---|
| Integration at `4ac872b` | 354 shared/profile/Compose tests passed on Windows with real Compose resolution required; layout, 1,241 links and release-preservation checks passed |
| [R37 shared image build](../../runtime/images/compositions/lil-r37-shared/local-build.json) | Exact descriptor/image identity, full installed inventory and explicitly recorded feature checks; read its serving qualification field |
| [Qwen TP4 prefill evidence](../../integrations/vllm/qwen38_prefill/README.md) | Bounded compute-bundle correctness and performance on the recorded overlay deployment; not automatic qualification of a baked image |
| [Baked QAD TP4 serving](../../performance/records/qwen38-flash-next/r37-shared-tp4.json) | Complete image admission, four-rank Compose launch, matched repository-harness prefill/decode, text checks and coordinated stop/fresh-deployment restart |
| [Qwen TP2 guide](../../profiles/qwen38-flash-next-tp2/README.md) | Published aligned-cache image and separate private request-boundary results are identified explicitly |
| Historical Linux/native/serving checks | Revision-specific results summarized below; they do not establish current-head CI or serving qualification |

### Historical validation snapshots

- `ae3d60af4a9f1da55fb128b896930b3cf6644569`: maintained Linux CI selection
  passed 5,006 tests, with 29 skips, using Python 3.12.3 and CPU Torch 2.11.0.
- `3dbb25d702a134929f04064254e8f78d333bb537`: local Linux selection passed
  5,135 tests, with 29 hardware/optional-input skips and the CI-pinned tools.
- `06e4925046db3b8ae203f9f745d7dbe01d0dbe73`: ARM64 GB10/CUDA 13.0 native
  build passed 31 CTests. Four-rank tiled-prefill, dual-rail, graph-reuse and
  changing-input checks passed their recorded gates. Existing-image NCCL
  2.31.2 DCP2/DCP4 checks do not qualify a separate NCCL rebuild.
- `699e12bb82b863b211f85add10848ed65ccd495c`: GLM TP4/DCP1 restored an
  8,192-token prefix after model restart; a 30-minute C4 run passed 956 checks;
  a 1,047,552-token retrieval returned the requested value. These results do not
  establish crash recovery, concurrent full-context capacity or model quality.

The R35 native TP4 stall remains unresolved; its
[performance results](../../performance/records/glm53-flash/r35-tp4-direct-performance.md)
do not establish long-duration stability. SGLang contributor evidence is separate
from GLM tests. Hosted Linux CI, including the pinned LIL companion, must run on
the adoption revision. The workflow runs on pull requests and pushes to main;
a refactor-branch push alone does not run it.

## Pending contributions

At the 2026-09-14 source review:

| PR | Head | Integration requirement |
|---|---|---|
| [#258](https://github.com/FujitsuPolycom/sparkring/pull/258) | `5a00273a7ea1` | Preserve NVIDIA NVFP4 versus NVFP4-Spark identity, overrides and cache namespaces |
| [#266](https://github.com/FujitsuPolycom/sparkring/pull/266) | `544f498c6369` | Review local-address recovery separately from peer-silence mitigation |
| [#267](https://github.com/FujitsuPolycom/sparkring/pull/267) | `fa77ac48f821` | Compare any merged SGLang changes with the contributor-authored local integration; retain its context default and image identity |

Recheck main and open PR heads before adoption. Preserve contributor credit
through normal history; adapt fixes to maintained owners instead of replacing
compatibility shims with duplicate implementations.

## Review and adoption sequence

<a id="r35-image-integration"></a>

### Image integration

1. Retain the tested Qwen QAD TP4 shared-image admission and quickstart. Its
   bounded serving record does not qualify other models or feature combinations.
   Existing TP2 instructions keep their published pins.
2. Migrate GLM TP4 to the shared specification while preserving fabric,
   source-verification, readiness and recovery contracts.
3. Validate the exact baked image through the documented path, without private
   source mounts. Check effective features, inference, bounded performance and
   rollback. Arrange TP2 Compose/cache-restore tests with the serving owner.
4. Evaluate a pinned published ARM64 LIL base as a separate composition. Compare
   installed sources and native libraries; retain required SparkRing features.
   Promote a profile's image selection only after matching acceptance checks.

### Repository adoption

1. Review ownership, profiles, compatibility exports and configuration defaults.
2. Reconcile contributor changes and run current-head Linux/CI checks. Report
   skipped hardware checks separately; do not transfer historical results.
3. With explicit authorization, push the refactor branch and open a draft PR for
   hosted Linux/LIL CI and review. Keep image publication separate.
4. Adopt through a normal reviewed merge. Do not force-push or replace main.
5. Promote deployments separately with exact image/model/site evidence and an
   operational rollback procedure. A repository merge starts no services.

## Rollback

Before adoption, the isolated branch leaves main and deployments unchanged.
After adoption, revert through normal Git history while retaining compatibility
paths. Deployment rollback follows the selected profile's procedure and image
receipt. Retain known-working containers, model files and cache ownership until
that deployment's acceptance and rollback checks are complete.

## Profile-default verification

Compare effective plans and generated exports after integration. Defaults,
checkpoint identities, image selections and source hashes must agree with their
canonical profiles. Feature-dependent KV capacity remains a measured estimate,
not a per-request context limit. Software configuration checks do not promote a
profile from Development or Experimental to Validated.
