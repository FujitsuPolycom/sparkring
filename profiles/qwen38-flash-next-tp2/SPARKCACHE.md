# Qwen TP2 with persistent prefix caching

The [Qwen TP2 quickstart](README.md) owns checkpoint verification, image selection,
planning, startup, restart and safety instructions for both cache choices.
Use its default SparkCache configuration, [sparkcache.json](sparkcache.json).

## Plan, create and start

Follow [Plan and create](README.md#plan-and-create) and
[Controlled startup](README.md#controlled-startup). Existing containers use
[the restart procedure](README.md#restart-existing-containers), not another `create`.

## Verification scope

See [Evidence and remaining checks](README.md#evidence-and-remaining-checks).
The published cache64 image uses aligned checkpoints; it does not contain the
complete request-boundary connector. Its
[build recipe](../../runtime/images/compositions/lil-r37-cache64/README.md)
and immutable publication remain available.
