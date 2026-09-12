# Deployment profile index

The machine-readable [profile catalog](../../profiles/catalog.json), its human
[profile index](../../profiles/README.md), and the generated
[README table](../../README.md#profiles) form the maintained deployment index.
Run `python scripts/profiles.py list` to discover stable IDs and
`python scripts/profiles.py resolve PROFILE` to inspect selected defaults.

Profile definitions keep evidence status separate from recommendation.
Older profile overview URLs remain available for compatibility; the selected
profile's primary guide owns current deployment instructions.
