# Release selections and preserved inputs

Each `release.json` selects an immutable published contract or a separately
pinned builder. Recipe selections reference their authoritative runtime section
rather than copying model-specific image settings into the catalog.

[preserved-inputs.json](preserved-inputs.json) records SHA-256 values of release
inputs and evidence retained for reproduction. It includes the base commit
for provenance. These hashes are checked locally and in CI; they are not a
receipt for a rebuilt image. The original paths remain usable by published
instructions and source snapshots.

Do not update a preserved hash to make a changed build look like the same
release. Add a distinct release selection and evidence instead. Historical
records retain their original scope even when navigation retires a profile.

## Shared serving versions

[Shared runtime 2026.09.3](shared-2026.09.3/README.md) is selected by Qwen TP2
and TP4, with and without SparkCache, and GLM TP2/TP4 DCP1 with SparkCache.
GLM supports separate Spark and QAD checkpoint selections. Other profiles
retain independent image pins and qualification records.

GitHub Releases provides the version landing page; GHCR Packages stores the
container layers. A serving Release links an exact image digest, build/source
receipts and profile-specific evidence. Auxiliary host tools and build inputs
retain their separate identities.

The [shared ARM64 candidate](shared-2026.09.0-rc.1/README.md) records published
image version `shared-2026.09.0-rc.1`, its verified registry digest, source inputs
and bounded qualification. The matching GitHub prerelease links its assets. Its release
selection is opt-in; profile defaults remain on their pinned releases until
each configuration is qualified and explicitly promoted.

## Published image and DCP4 overlay

`sparkring-r33/published-inputs/` retains the original contract, publication
metadata and entrypoint bytes superseded by main's DCP4 overlay support. These
are archival snapshots: internal relative references retain their original
repository context, recorded by the preservation manifest's relocation map.
They are not a standalone build directory.

The `sparkring-r33-dcp4` selection pins the published image and the merged
contract/entrypoint overlay separately. It does not change the published image
identity. DCP4 activation requires the overlay and managed fabric; DCP1 remains
available without that additional deployment choice.

Its evidence pins include the correction in upstream commit
`f575d421d72c7fbdef3d6165eb6bbe241517fa87`: gathered global KV entries are
mapped to their owners arithmetically, without a separate owner-exchange step.
The evidence also includes upstream commit
`506c8db0c09c95a75467e006110242cb5d0bcc7d`, which marks overlapping measurement
windows and withdraws the attribution of the C4 throughput gap to a gather.
These corrections change the report and its publication hash, not image or
overlay code. Preserved release inputs retain their existing hashes.
