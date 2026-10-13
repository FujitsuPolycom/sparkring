# GLM-5.3-Flash on two Sparks (CSF checkpoint, or NVFP4-Spark on an image that cannot read CSF)

Use the [installer](../../docs/operations/install.md) to deploy this profile.

## Request defaults

When a request chooses no effort, this profile always thinks at `high`.
An install option such as `--reasoning-effort max` replaces that deployment
default; a request can still choose `low`, `high` or `max`. GLM has no
thinking-off switch.

`--generation-config auto` loads the pinned checkpoint's sampling defaults:
`temperature: 1.0` and `top_p: 0.95`. The generation-config hashes and values
are recorded in the [thinking records](../thinking.json) and
[GLM-5.3 record](../research-thinking.json); no sampling override is needed.

Recorded measurements that omitted an effort choice used the chat template's
`max` default. They do not measure this profile's `high` deployment default.
Measurements with explicit request settings retain those stated conditions.
