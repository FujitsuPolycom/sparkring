# Generated Qwen TP2 Compose with SparkCache

Status: **qualified for bounded correctness and restart checks**. These examples select QAD and the shared serving image.
The [qualification record](../../../runtime/releases/shared-2026.09.2/qualification.json)
states exact-image deployment and persistent-cache coverage.

[Rank 0](compose.rank0.yaml) and [rank 1](compose.rank1.yaml) are generated from
the [SparkCache configuration](../../qwen38-flash-next-tp2/sparkcache.json) and
the [public site example](../../qwen38-flash-next-tp2/compose/site.example.yaml).
They show the effective container settings; their documentation addresses and
paths are not a deployment for your hosts.

Use the [Compose deployment guide](../../../docs/operations/compose.md) to render
a private deployment with profile `qwen38-flash-next-tp2-sparkcache` and coordinate
both hosts. Complete the [SparkCache quickstart](../../qwen38-flash-next-tp2/SPARKCACHE.md)
prerequisites first. Edit profile/site inputs and regenerate; do not maintain a
second set of serving defaults in these YAML files.
