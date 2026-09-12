# Repository layout adoption plan

## Result and base

The restructuring branch is `refactor/repository-layout`. Its integration
base is main commit `506c8db0c09c95a75467e006110242cb5d0bcc7d`, including
PRs #259, #262, #265, #269 and #271. The initial preservation inventory is
based on `c65a9981e2a69f821ac716f6f13484d99d23f4ea`. Local validation results
and remaining review requirements are listed below; adoption includes no deployment.

The branch provides 22 indexed deployment profiles, deterministic configuration
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
| Six serving ENV examples | Generated from profile templates; ordered-assignment checks preserve baseline defaults except recorded changes |
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
The preserved-content check protects the 430 inventoried inputs against deletion
or byte changes.

## Verification

The file inventory at `41491c7` contains 1,566 tracked files. Exhaustive semantic review is
not established: several completion records assigned full-review status without
recovering the underlying review or inspecting all intervening changes. Those
records require reconciliation before an exhaustive-review claim is warranted.
At that revision, 210 files still require review or evidence reconciliation.
Coverage records include provenance checks and do not all establish semantic
correctness. The checks below ran on Windows with Python 3.12 at commit `ef8de04`;
their inventory counts describe that snapshot.

| Check | Result |
|---|---|
| Full maintained pytest selection from CI | 4,573 passed; 123 skipped; no failures |
| Shared profile/configuration tests | 119 passed |
| Ruff over maintained Python trees | Passed |
| Repository structural check | 21 profiles, 52 generated outputs, 430 preserved inputs, 448 Python sources, 7 builders |
| Repository Markdown links | 999 local links checked |
| Release-safety scan | Zero findings |
| Managed-service source closure | Imported from an extracted deployment archive without the checkout on its import path |
| Launch compatibility | Existing TP2 entry point and maintained switched-renderer contracts pass |
| Configuration equivalence | Omitted and explicit defaults agree; ENV assignments match the initial baseline except declared default changes; comment edits do not change compatibility |

Source snapshot `06e4925046db3b8ae203f9f745d7dbe01d0dbe73` passed the full
maintained Linux selection in Ubuntu WSL with Python 3.12.3 and CPU Torch
2.11.0: 4,591 passed and 105 skipped. Linux lint, links, release safety and
layout checks passed. Skips include PowerShell and optional serving dependencies.

The same snapshot built on ARM64 GB10 with CUDA 13.0, architecture `sm_121`,
Release mode and `BUILD_TESTING=ON`, including the optional fused probe and GPU
smoke targets. All 31 CTests passed. Isolated four-rank tests established:

- All 18 tiled-prefill cases passed their per-rank receipt gates, including
  boundary sizes, backpressure and expected poison outcomes.
- The fused dual-rail probe passed exact and noninteger input checks on all ranks.
- Tiered mixed-query graph replay and two-slot reuse passed without mismatches
  or command overflows. The ctypes/PyTorch boundary passed 100 alternating-input
  all-reduces per rank using the rebuilt native library.
- NCCL DCP4 passed 19 eager/graph result rows per rank, and DCP2 passed nine,
  using the existing image's NCCL 2.31.2. This does not qualify a rebuild of the
  separate NCCL 2.30.7 patch.
- All four ring cables passed bidirectional 12,288-byte and 16,384-byte integrity
  checks, 1,000 measured iterations per direction and size, with the latency
  target met. The source-route checker fix at `1f439d0` passed 14 regression tests.

Managed-serving source snapshot `699e12bb82b863b211f85add10848ed65ccd495c`
was tested separately on four Sparks with TP4/DCP1, SparkCache, a 1M context
limit, the existing published image, and read-only checkpoint mounts:

- A coordinated model-process restart restored an 8,192-token prefix from an
  identical 8,220-token request. Transfer counters and all four worker logs
  confirmed the restore.
- A 1,801.875-second mixed-prompt soak at concurrency four passed 956 exact-answer
  and normal-finish checks with thinking enabled.
- One retrieval request with 1,047,552 prompt tokens and a 1,024-token output
  budget returned the exact requested value in 434.203 seconds.

These are bounded restart, soak and capacity results, not crash-recovery,
general long-context quality or throughput qualification. A separate
thinking-disabled batch failed because responses exposed reasoning text;
the profile guide documents the installed template limitation.

The installed managed mesh services were in a failed state when inspected and remained
untouched. Separately authorized temporary controllers and test containers were
removed after testing. Cleanup receipts show no owned network objects remaining
and six pre-existing adopted objects retained per host. Model files were not
modified. Test evidence is retained locally; these results do not qualify
arbitrary later source revisions or rebuilt images.

The SGLang integration snapshot `4b6d9c311d61644fb1d738b7a31c8308dc864baf`
passed 4,621 tests with 105 skips in the maintained Linux selection. Subsequent
focused Windows and Linux checks passed 26 SGLang launcher tests, 95 telemetry
tests and 18 public-overlay tests. The contributor's SGLang serving records
remain separate from the GLM hardware results.

Local WSL validation with Python 3.12.3 also passed 97 shell-guard,
cleanup and deployment-documentation tests, plus 66 staging, trust and
existing-asset tests, with no skips in either selection. These checks used
fixtures and temporary directories; they did not deploy to hosts. The pinned
[LIL deployment companion](../../integrations/lil/README.md) was built at revision
`329cde801b847294005cb16765692032a6cdf206` with Go 1.26.0: its tests and
vet passed, and all 87 SparkRing LIL integration tests passed in WSL without
skips. These local checks do not constitute a hosted GitHub Actions run.

The standalone, unchanged DeepSeek Engram probe could not import vLLM in this
environment. It requires the serving image and model/packed-row inputs, so it is
not part of hosted CPU CI. The GLM serving tests used read-only model mounts.

Windows skips include POSIX mode/symlink behavior, Bash/native compilers, optional
LIL integration and serving-runtime checks. Hosted Linux CI has not run because
the reviewed changes have not been published. Native ARM64 image assembly,
crash recovery and representative performance remain unverified. The near-limit
retrieval check above does not establish concurrent full-context serving.
Standalone hardware and CPU results do not authorize a deployment promotion.

## Pending contributions

| PR | State at reconciliation | Adoption handling |
|---|---|---|
| #258 | Open, head `5a00273a7ea12d41db965c50a3aedb02e23add83` | Preserve NVIDIA NVFP4 versus NVFP4-Spark identity, launcher overrides and cache namespaces; extend the profile catalog when the implementation lands |
| #259 | Merged in the integration base | Preserve the 300-second peer-silence mitigation and bounded claim; no local-address recovery is implied |
| #263 | Closed, head `2ae60f01774131d07e7370104599bacafb39473d` | Retain contributor wording improvements around the generated README region; edit the generator for table changes |
| #265 | Merged in the integration base | TP4/DCP1 remains the default; TP4/DCP4 is a validated alternative; its contract/entrypoint overlay and activation record are pinned separately from the published image |
| #266 | Open, head `544f498c6369a72c1afba0a856dfb870cfc0d302` | Review local-address recovery separately from peer-silence handling; relocation leaves its managed-service paths intact |
| #267 | Open, head `fa77ac48f821a8c760129914f58158bb4700c3c4` | Integrated locally with contributor authorship, a separate SGLang catalog entry, builder, launcher and qualification records; retain the contributor's 262144-context default |
| #269 | Merged in the integration base | Preserve the corrected DCP4 arithmetic-owner mapping; PR #271's overlap correction and attribution withdrawal are integrated locally |
| #270 | Open, head `27d6b0dbd2b5a2113fb8c14923d70fb354d5a0ee` | Apply selected-HCA rendering to the maintained TP2 owner and regenerate its compatibility launcher if adopted; retain transport-change validation scope |

PRs #258, #266, #267 and #270 remain open at the upstream recheck.
Main commit `506c8db0c09c95a75467e006110242cb5d0bcc7d` adds merged
[PR #271](https://github.com/FujitsuPolycom/sparkring/pull/271), which corrects
the DCP4 measurement overlap and withdraws the C4 gather attribution. Its
record and publication-reference changes are integrated locally, with the
DCP4 release selection updated to bind the corrected evidence. The frozen
published-input archive remains unchanged. The conflicting RouteFinal times
in that historical correction do not establish an independently verified
non-overlap result.

PR #262 is already part of the initial base. No pending PR was merged, closed,
rejected or rewritten by this branch. Maintainers should recheck main and open
PR heads immediately before adoption; preserve contributor credit in normal
Git history.

## Review and adoption sequence

1. Review the profile catalog, configuration/launch contracts, migration map and
   retained-source exceptions. Compare default plans and generated legacy inputs.
2. Reconcile main changes that landed after this report. Preserve upstream
   contributor changes when integrating them into maintained owners. If PR #267
   merges, compare its merged content with the locally integrated SGLang changes
   and resolve overlaps without losing either contributor fixes or layout adapters.
   Add release/profile selections rather than rewriting preserved inputs or
   transferring evidence.
3. With explicit authorization, push the branch and open a pull request against
   main. Run hosted Linux CI, including the pinned LIL companion job, on the
   resulting revision. Resolve failures and rerun affected checks after any
   further source reconciliation before adopting the layout.
4. Adopt through a normal reviewed merge. Do not replace or force-push main.
5. Keep deployment promotion separate. Any rebuilt image or operational default
   needs its exact source/image/model/topology checks and relevant qualification
   on separately authorized hosts. This layout change starts no services.

## Rollback

Before adoption, abandoning this isolated branch leaves main and deployments
unchanged. After adoption, revert the adoption commit or merge through normal
Git history. Keep the generated compatibility paths while the revert is reviewed.
No cluster rollback is needed for adopting the repository layout. The isolated
hardware tests have their own completed cleanup described above. A later
operational promotion must carry its own image/site rollback procedure.

## Profile-default verification

After incorporating main's DCP4 change, the relevant runtime, image, mesh and
shared-profile suite passed 1,082 tests with 27 platform/optional skips. The
published DCP4 activation receipt passed the verifier for all four ranks.
The original entrypoint/contract/publication bytes are retained as archival
inputs; the added release selection pins the merged overlay and its evidence.

The vLLM DeepSeek-V4.1-Flash profile defaults to 1M context tokens in the recipe
and generated ENV example. The SGLang profile retains its contributor's 262,144-token
default. Measurements retain their actual settings. The ENV
migration baseline records this intentional default change instead of rewriting
its historical hash. KV displays carry a feature-dependent capacity footnote.
