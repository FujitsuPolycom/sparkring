# Qwen TP2 with persistent prefix caching

The [Qwen TP2 quickstart](README.md) owns checkpoint verification, image selection,
planning, startup, restart and safety instructions for both cache choices.
Select `PROFILE_ID=qwen38-flash-next-tp2-sparkcache` before its image/checkpoint
block to choose [sparkcache.json](sparkcache.json).

## Plan, create and start

Follow [Plan and create](README.md#plan-and-create) and
[Controlled startup](README.md#controlled-startup). Existing containers use
[the restart procedure](README.md#restart-existing-containers), not another `create`.

## Verification scope

See [SparkCache limits](README.md#sparkcache-limits).
[SparkRing 2026.09.3](../../runtime/releases/shared-2026.09.3/README.md) uses
aligned checkpoints and a release-specific persistent-cache namespace. Its
[qualification record](../../runtime/releases/shared-2026.09.3/qualification.json)
owns exact-image correctness and restart/restore evidence. Request-boundary
caching and request-salt isolation are not enabled by this profile.
