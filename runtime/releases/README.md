# Release selections and preserved inputs

Each `release.json` selects an immutable published contract or a separately
pinned builder. Recipe selections reference their authoritative runtime section
rather than copying model-specific image settings into the catalog.

[preserved-inputs.json](preserved-inputs.json) records SHA-256 values of release
inputs and evidence retained during this migration. It includes the base commit
for provenance. These hashes are checked locally and in CI; they are not a
receipt for a rebuilt image. The original paths remain usable by published
instructions and source snapshots.

Do not update a preserved hash to make a changed build look like the same
release. Add a distinct release selection and evidence instead. Historical
records retain their original scope even when navigation retires a profile.
