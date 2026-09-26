# Qwen QAD TP4 Compose files

The four rank files are generated from the [canonical configuration](../config.json)
and [example site](site.example.yaml). The [quickstart](../README.md) explains
published image preparation and required fabric checks. Replace example site inputs
and regenerate; do not edit the generated YAML.

Status: **implemented**. The rank files run the installer image
`dev-20260925-qwendecode-cuda1342-nccl2323-status031` with the containers
[`sparkring install`](../../../docs/operations/install.md) deploys, without
its per-container runtime-binding file. `sparkring install` is the supported
way to run this profile: it also creates the four-rank native mesh the ranks
need. Compose deployments of these files have no hardware evidence; the
installer deployments are measured in the
[installer tuning record](../../../performance/records/qwen38-flash-next/installer-tuning-20260925.md).
