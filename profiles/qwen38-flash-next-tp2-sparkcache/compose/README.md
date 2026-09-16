# Generated Qwen TP2 Compose with SparkCache

Status: **Development**. [Bounded TP2 checks](../../../performance/records/qwen38-flash-next/compose-tp2.json)
cover coordinated startup/shutdown, text responses and SparkCache restore across
a fresh deployment with the original non-QAD checkpoint. Generated examples
select QAD; those historical checks do not qualify QAD serving or cache restore.

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
