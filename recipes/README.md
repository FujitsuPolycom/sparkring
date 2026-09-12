# Serving recipes

These recipe paths are generated compatibility exports. Edit the matching
source under [profiles/](../profiles/README.md), then run
`python scripts/generate_profiles.py`. Do not maintain a second copy here.

The machine-readable [profile catalog](../profiles/catalog.json), with its
[human index](../profiles/README.md), owns deployment discovery,
recommendations, settings and primary quickstarts. Some deployments select
release-profile contracts rather than a standalone recipe JSON, so this
directory is not a complete inventory of supported deployments.

[compatibility.json](../profiles/compatibility.json) maps each exported recipe
to its authoritative source. The JSON records artifact and model identities,
topology, serving values, evidence and limitations. Use it for inspection and
reproduction; follow the selected profile's guide for host operations.

```bash
python scripts/profiles.py list
python scripts/profiles.py resolve deepseek-v41-flash-cycle
```

[`sparkcache/`](sparkcache/README.md) contains compatibility exports for
compositions that add persistent rank-local prefix storage. Their evidence can
be narrower than the corresponding base serving profile.

## MTP3 mesh profile

The retained [compute-image mesh recipe](glm53-spark-mtp3-managed-mesh-tp4.json)
records a TP4/DCP4 native-MTP3 composition. TP and DCP use the same four
processes: TP4 shards model tensors across the ranks, while DCP4 shards decode
context work across those ranks; the shape does not require 16 processes. Its
dedicated site contract is
consumed by the [mesh renderer](../runtime/glm53-spark-mtp3-mesh/README.md),
not by treating recipe JSON as an executable installer. Its settings and
receipts do not define the defaults of the published-image profiles in the
catalog.

## Status definitions

A status applies to the exact configuration and evidence named by its record.
See the [writing and evidence policy](../docs/development/writing.md) for
`implemented`, `qualified`, `research-only`, and `unsupported` scope.

[Profile validation](../docs/operations/profile-validation.md) describes
performance, accuracy, and restart checks.
