# Reproducible vLLM and B12X sources

[manifest.json](manifest.json) binds each upstream commit, complete patch,
accepted source-tree digest and source-archive SHA256. Each patch was applied to
an independent clean snapshot of its pinned upstream commit and reproduced the
admitted tree. Repository-local agent instruction directories excluded by the
build are also excluded by reconstruction.

The source archives accompany the GitHub prerelease rather than being committed
as binary files. Other inherited components keep their separate source and
license records in the [component index](../components.md). These archives prove
source identity, not full-model qualification.
