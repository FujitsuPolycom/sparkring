# Repository layout adoption plan

## Result and base

The restructuring branch is `refactor/repository-layout`. Its integration
base is main commit `f575d421d72c7fbdef3d6165eb6bbe241517fa87`, including
PRs #259, #265 and #269. The initial preservation inventory is based on
`c65a9981e2a69f821ac716f6f13484d99d23f4ea`. Adoption requires completing the
repository-wide review and the checks below; no deployment is included.

The branch provides 21 indexed deployment profiles, deterministic configuration
resolution, a shared guarded launch entry point, ENV rendering for the DeepSeek
and Qwen adapters, and a shared image-builder selector. Framework adapters and
active fabric planning have explicit maintained owners. Documentation separates
operations, architecture, development and retained historical configurations.
AGENTS.md and the reusable maintainer prompt reference one writing policy.

Contributions remain open to people without Spark hardware. There is no
mandatory issue before a PR, corporate paperwork, or requirement to supply a
complete report. Only the observed-problem field is required in the bug form.

## Compatibility and migration

| Public surface | Result |
|---|---|
| `recipes/*.json` and SparkCache composition recipes | Generated exports of the authoritative `profiles/` source; existing consumers keep their paths |
| Six serving ENV examples | Generated from profile templates; migration-baseline tests preserve their existing defaults |
| TP2 Python launcher | Existing path delegates to `runtime/common/tp2.py`; plan/receipt/guard behavior is retained |
| Switched Python launcher | Frozen image asset remains byte-identical; maintained host adapter lives in `runtime/common/switched.py` and produces an equivalent plan |
| vLLM adapter source paths | Canonical owner is `integrations/vllm`; generated legacy exports preserve overlay/package consumers |
| Fabric and RoCEnante integration | Maintained imports use their component owners; managed-service allowlist includes the relocated dependencies |
| Moved Markdown guides | Original URLs and heading anchors remain as pointers; the migration map names canonical destinations |
| Published source images, locks, receipts and evidence | 430 inputs protected by a preserved-content check; public image names/digests and installed paths are unchanged |
| Native tiled-prefill substrate and incompatible image builders | Retained at identity-bound paths; their native/build dependencies justify the exception |

No documented command was intentionally removed. Some internal test filenames
and source locations changed; the entire original CI suite remains selected,
with tests for relocated owners added to discovery. Compatibility exports are
not separate authored copies. Run `python scripts/generate_profiles.py` after
editing their sources; CI rejects drift.

The layout does not promise a lower tracked-file count. Published source-bound
paths require compatibility exports. Removing those paths requires a separate
announced compatibility change after package and release consumers migrate.
No useful evidence was deleted or uploaded to another location.

## Verification

Local verification ran on Windows with Python 3.12:

| Check | Result |
|---|---|
| Full maintained pytest selection from CI | 3,709 passed; 118 skipped; no failures |
| Shared profile/configuration tests | 56 passed |
| Focused image-documentation and shared-runtime checks after navigation refinement | 76 passed |
| Ruff over maintained Python trees | Passed |
| Repository structural check | 21 profiles, 52 generated outputs, 427 preserved inputs, 439 Python sources, 7 builders |
| Repository Markdown links | Passed; 933 local links at report preparation |
| Release-safety scan | Zero findings |
| Managed-service source closure | Imported from the installed allowlist in an isolated Python process, without the checkout on its import path |
| Launch compatibility | Existing TP2 tests pass; maintained switched renderer matches the frozen renderer's plan |
| Configuration equivalence | Omitted defaults, explicit defaults and equivalent legacy recipes resolve consistently; preferred DCP selections and composition base references are validated; ENV examples match migration-baseline hashes |

The initial baseline subset passed 1,072 tests with 77 skips. The migration's
full-suite documentation regressions were corrected by following canonical
paths while retaining the original content assertions. Test discovery was not
reduced to hide failures.

The standalone, unchanged DeepSeek Engram probe could not import vLLM in this
environment. It requires the serving image and model/packed-row inputs, so it is
not part of hosted CPU CI. No model files were accessed or changed for this work.

Windows skips include POSIX mode/symlink behavior, Bash/native compilers, optional
LIL integration and serving-runtime checks. Hosted Linux CI has not run because
the reviewed changes have not been published. CUDA, RDMA, native ARM64 image assembly, live failure
recovery, full-context serving and performance were not validated. Local CPU
results do not qualify a reorganized build or authorize a deployment promotion.

## Pending contributions

| PR | State at reconciliation | Adoption handling |
|---|---|---|
| #258 | Open, head `5a00273a7ea12d41db965c50a3aedb02e23add83` | Preserve NVIDIA NVFP4 versus NVFP4-Spark identity, launcher overrides and cache namespaces; extend the profile catalog when the implementation lands |
| #259 | Merged in the integration base | Preserve the 300-second peer-silence mitigation and bounded claim; no local-address recovery is implied |
| #263 | Closed, head `2ae60f01774131d07e7370104599bacafb39473d` | Retain contributor wording improvements around the generated README region; edit the generator for table changes |
| #265 | Merged in the integration base | TP4/DCP1 remains the default; TP4/DCP4 is a validated alternative; its contract/entrypoint overlay and activation record are pinned separately from the published image |
| #266 | Open, head `544f498c6369a72c1afba0a856dfb870cfc0d302` | Review local-address recovery separately from peer-silence handling; relocation leaves its managed-service paths intact |
| #267 | Open, head `5a74a07a0b8e11d660036600ce64ea7eedc6903b` | Keep the proposed SGLang runtime distinct from vLLM; add its profile only with its own builder and evidence |
| #269 | Merged in the integration base | Preserve the corrected DCP4 arithmetic-owner mapping and no-overlap evidence; published image identity is unchanged |
| #270 | Open, head `27d6b0dbd2b5a2113fb8c14923d70fb354d5a0ee` | Apply selected-HCA rendering to the maintained TP2 owner and regenerate its compatibility launcher if adopted; retain transport-change validation scope |

PR states above were checked against GitHub with this integration base.

PR #262 is already part of the initial base. No pending PR was merged, closed,
rejected or rewritten by this branch. Maintainers should recheck main and open
PR heads immediately before adoption; preserve contributor credit in normal
Git history.

## Review and adoption sequence

1. Review the profile catalog, configuration/launch contracts, migration map and
   retained-source exceptions. Compare default plans and generated legacy inputs.
2. With explicit authorization, push the branch and open the prepared PR. Run
   hosted Linux CI, including the pinned LIL companion job. Resolve any platform
   failures before adopting the layout.
3. Reconcile main changes that landed after this report. Add new release/profile
   selections rather than rewriting preserved inputs or transferring evidence.
4. Adopt through a normal reviewed merge. Do not replace or force-push main.
5. Keep deployment promotion separate. Any rebuilt image or operational default
   needs its exact source/image/model/topology checks and relevant qualification
   on separately authorized hosts. This layout change starts no services.

## Rollback

Before adoption, abandoning this isolated branch leaves main and deployments
unchanged. After adoption, revert the adoption commit or merge through normal
Git history. Keep the generated compatibility paths while the revert is reviewed.
No cluster rollback is needed for the repository-only change: this work did not
alter images, site configuration, model files or running services. A later
operational promotion must carry its own image/site rollback procedure.

## Profile-default verification

After incorporating main's DCP4 change, the relevant runtime, image, mesh and
shared-profile suite passed 1,082 tests with 27 platform/optional skips. The
published DCP4 activation receipt passed the verifier for all four ranks.
The original entrypoint/contract/publication bytes are retained as archival
inputs; the added release selection pins the merged overlay and its evidence.

DeepSeek-V4.1-Flash defaults to 1,048,576 context tokens in the recipe and generated
ENV example. Its earlier measurements retain their actual settings. The ENV
migration baseline records this intentional default change instead of rewriting
its historical hash. KV displays carry a feature-dependent capacity footnote.
