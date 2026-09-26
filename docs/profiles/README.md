# Deployment profile index

The machine-readable [profile catalog](../../profiles/catalog.json) and its
generated human [profile index](../../profiles/README.md) form the maintained
deployment index. The [README table](../../README.md#profiles) lists the
profiles that `sparkring install` sets up.
Run `python3 scripts/profiles.py list` to discover stable IDs and
`python3 scripts/profiles.py resolve PROFILE` to inspect selected defaults.

Profile definitions keep evidence status separate from recommendation.
Older profile overview URLs remain available for compatibility; the selected
profile's primary guide owns current deployment instructions.
