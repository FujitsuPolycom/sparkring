# Release 2026.10.3: installer and profile changes on release 2026.10.2's image

GitHub release `2026.10.3` carries SparkRing package changes that need no new
container image. It installs installer image release
`dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`, the image
that release 2026.10.2 published; that image's
[release record](../dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/README.md)
describes its layers, evidence and publication. This page is the record of
release 2026.10.3. [RELEASE_NOTES.md](RELEASE_NOTES.md) is the summary for
users, and [status.json](status.json) states the same status for programs.

Status: **implemented**; release maturity **candidate**. No tag, package or
GitHub release exists for it yet ([open gates](#qualification)); its image is
already published.

## Identity

| Item | Value |
|---|---|
| GitHub release tag | `2026.10.3`, not yet created; `2026.10.2` is the release before it |
| Image | Release 2026.10.2's, unchanged: image configuration ID `sha256:d52737a109e083d3eef05c0fc0db4d09fc1bf34485a13467382130963ecfe66b`, registry reference `ghcr.io/fujitsupolycom/sparkring@sha256:4fffc4dc3074d5539f4e9d3a013ff9ef4e0be570a95b74d4646ee341da1f6911`. A Spark that holds it downloads nothing |
| Lock | [installer-image.json](../dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/installer-image.json), SHA-256 `3458beca108dccfe9df301adf3f62c347bac10114a835ded06d5e633c064c541`, which that directory's `release.json` pins. It is release 2026.10.2's lock with `glm53-nvfp4-csf-tp8` added to its `profiles`, 14 in all; an admission changes only that list ([release procedure](../../../docs/development/releases.md)) |
| Image sources | The sources the image carries are byte-identical to release 2026.10.2's: `spark_transport/libsircl` is git tree `030419b8a2b61acc1a74010308a1c3e758863e2f`, the lock's `libsircl.source_tree`; `spark_transport/sircl` is tree `8a54c1abf591`, `integrations/vllm/runtime_status` `de6358a86f8c`, and the GLM-5.3 plugin directories are unchanged |
| Relay marker | Source `c64c74535c06` is unchanged, so package builds take the binary that [relay-marker-artifact.json](../../../spark_transport/fabric/relay-marker-artifact.json) records |
| Release-tag index | [installer-releases.json](../installer-releases.json) names no `2026.10.3`: this release publishes no image. Without `--image`, or with `--image 2026.10.2`, the package installs this image |
| Rollback | Release 2026.10.2's package on the same image; release 2026.10.1's image on the prepared transport with `--image 2026.10.1` |

Every installer deployment's identity includes its lock, so the first
`sparkring install` of any profile with this package plans a separate
deployment from the one 2026.10.2's package installed.

## Changes

| Change | Status | Where |
|---|---|---|
| `sparkring dashboard`, a read-only ring status page on Node A (port 8090, its LAN address and 127.0.0.1) that sends no inference request and opens no SSH session; `sparkring dashboard tui`; the dashboard's quiet mode during benchmark load; placement and pacing settings of the host agent, the recovery timer and the tunnel check | implemented | [dashboard](../../../docs/operations/dashboard.md), `runtime/host/dashboard.py` |
| An install with `--on` places the serving image only on the arc's Sparks; a Spark outside the arc writes nothing and forwards only when no way to an arc Spark avoids it | implemented | [image distribution](../../../docs/operations/install-reference.md#spreading-along-the-cables), `runtime/host/spread.py`, `runtime/host/install_assets.py` |
| Automatic recovery judges a four-rank group on a fabric other than a four-Spark ring by its containers and API, without the mesh reports that only a four-Spark ring's Sparks give | implemented | [automatic recovery](../../../docs/operations/install-reference.md#automatic-recovery), `runtime/host/recovery.py` |
| `qwen38-flash-next-tp4` and `qwen38-flash-next-tp8` select `qwen38-flash-next-qad-tp4` and `qwen38-flash-next-qad-tp8` | implemented | `runtime/host/models.py` |
| `qwen38-flash-next-tp2 --checkpoint csf` installs local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD at `dc5b8ab2fa14`, on an image whose vLLM reads the nvfp4_csf format (this one) | research-only | [install reference](../../../docs/operations/install-reference.md), `runtime/common/image_lock.py` |
| `glm53-nvfp4-csf-tp8`: GLM-5.3-NVFP4-CSF at `a4d406932f2d` on eight Sparks with every setting of `glm53-nvfp4-tp8`, served from a serving directory derived without a donor; the lock admits it | research-only | [profile guide](../../../profiles/glm53-nvfp4-csf-tp8/README.md) |
| The column-split start check reads only `glm53full_speedups`' own log lines | implemented | `runtime/host/start_checks.py` |
| SparkCache variants `glm53-flash-nvfp4-sparkcache-tp2` and `-tp4` of the GLM-5.3-Flash profiles, with 64 MiB CUDA placement arenas inside a 256 MiB restore budget, and the recipe of their derived image layer, `installer-sparkcache-gab86b70`. They are listed in the research catalog. Only the lock that layer's record step writes admits them, and no Spark has built it | research-only | [layer recipe](../../images/compositions/installer-sparkcache-gab86b70/README.md), [recipe record](../dev-20261010-sparkcache-gab86b70-plugins-status036/README.md) |
| The README's attribution notice names Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD beside the GLM-5.3 CSF checkpoints; the libsircl architecture page states the eight routing variables the installer transport carries | implemented | [README](../../../README.md), [libsircl](../../../docs/architecture/libsircl.md) |

The changes came from these commits, merged or applied on release
2026.10.2's branch head `dc0300ac`: `claude/qwen-csf-tp2` at `8e398fd4`,
`claude/dashboard-feature` at `86baf73b`, `glm/sparkcache-flash` at
`2f12d483` with `a6d1e309` from `claude/ring-test-20261010`,
`claude/glm53-csf-tp8` at `809fc4b1`, and `3e991b93` from
`claude/csf-start-check-fix`; the rest are commits of `release/2026.10.3`.

## Evidence

The ring test of 2026-10-10 on the eight-Spark ring, with this image,
reported the following; no record of it is under `performance/records/`:

- `qwen38-flash-next-tp2 --checkpoint csf` loaded on two Sparks and answered
  correctly.
- The dashboard's page, terminal view, loopback listener and reload worked.
- Every worker of `glm53-flash-nvfp4-sparkcache-tp4` refused to start on
  Sparks 0-3 with 256 MiB placement arenas (SparkCache needs the restore
  budget to hold one lane's two arenas); the arenas are 64 MiB in this
  release. No start with them is recorded.
- A start of `glm53-nvfp4-csf-tp8` logged both `glm53full_speedups` patches
  and both split layers (rank 0's excerpt is the test fixture
  `runtime/host/fixtures/start_checks/glm53-nvfp4-csf-tp8-rank0.log`). The
  start check without commit `3e991b93` judged that column-split loading did
  not take effect; this release's check judges that it did. Its serving is not
  recorded.
- Automatic recovery reported a serving TP4 group on Sparks 0-3 as `unknown`
  ("its mesh units are not reported"), so it would not have restarted it.
- An install on Sparks 0-3 wrote the image, 14.2 GiB, to each of Sparks 4-7
  while they served.

The last two conditions are corrected in this release and tested offline
only.

## Qualification

| Gate | Result | Record |
|---|---|---|
| `image-lock`: the lock validates for its 14 profiles | passed (CPU) | `runtime/common/test_installer_image.py`, `runtime/common/test_image_lock.py` |
| `image-sources`: the image's sources are byte-identical to release 2026.10.2's | passed | [Identity](#identity) |
| `cpu-checks`: the CPU jobs of `.github/workflows/ci.yml` that run on Windows 11 with Python 3.12.10: ruff 0.15.17, the generators' `--check`, the layout, link and release-safety checks, the image composition check, the Compose and Install Builder jobs, and pytest over the tests job's paths, `runtime/images` and `spark_transport/sircl` | passed apart from tests that also fail on release 2026.10.2's head `dc0300ac` on the same host. Pytest: 11,161 passed, 77 failed, 37 errors, 682 skipped; 111 of the 114 failing tests also fail on `dc0300ac`, and the other 3 (a timing-bound spread test, a container-cleanup test and a sweep-harness timeout) pass and fail from run to run on both | [Validation](#validation) |
| `qwen-csf-tp2`: an installation of `qwen38-flash-next-tp2 --checkpoint csf` that passes the installer's checks | reported, not recorded | [Evidence](#evidence) |
| `dashboard-hardware`: the dashboard on the ring | reported, not recorded | [Evidence](#evidence) |
| `arc-image`: an install with `--on` on a ring of eight leaves the Sparks outside the arc without image writes | pending | |
| `arc-recovery`: automatic recovery restarts a stopped four-rank group on part of a ring of eight | pending | |
| `glm53-nvfp4-csf-tp8-install`: an installation that passes the installer's checks | pending | |
| `sparkcache-variants`: the derived layer built and recorded, and an installation of a variant | pending | |
| `publication`: tag, package built with the relay marker, GitHub pre-release | pending | |

## Validation

The CPU checks ran on Windows 11 with Python 3.12.10. The libsircl job
(`make check`, which needs `gcc` and `nvcc`) and the pinned LIL bridge's Go
build did not run, and the SIRCL tests ran without that job's private mount
namespace. Most pytest failures on this
host come from Windows paths, line endings and process timing, and they fail
on release 2026.10.2's head too. CPU checks do not validate CUDA, RDMA,
serving or performance.

## Known limitations

- Release 2026.10.2's limitations hold
  ([known limitations](../dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/README.md#known-limitations)).
- No image this package carries runs the SparkCache variants: only the lock
  their derived layer records admits them, and no Spark has built that layer.
  They are listed in the research catalog.
- libsircl's README and NOTICE do not state the attribution rule for a file
  that adapts NCCL source. That change (`claude/libsircl-provenance-policy`,
  commit `014dd1f6`) alters `spark_transport/libsircl`'s git tree, which this
  image's lock records, so it belongs to a release that rebuilds libsircl.
