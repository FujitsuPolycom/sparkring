# Repository ownership

| Path | Authored responsibility |
|---|---|
| `profiles/` | Deployment discovery, configuration, evidence scope and primary quickstarts |
| `runtime/common/` | Shared resolution, site parsing, command planning and guarded rank launching |
| `runtime/images/` | Image-builder selection and shared build entry point |
| `runtime/releases/` | Release selection and protection of retained immutable inputs |
| `spark_transport/` | Communication kernels, native transport and maintained fabric planning |
| `integrations/vllm/` | Framework adapters, including RoCEnante bundle composition |
| `integrations/sparkcache/` | Selection of external SparkCache compositions; implementation stays upstream |
| `integrations/lil/` | Pinned companion lifecycle/export bridge |
| `docs/architecture/` | Component design and communication architecture |
| `docs/operations/` | Shared host prerequisites, deployment and validation procedures |
| `docs/development/` | Contribution, configuration, testing and maintenance contracts |
| `docs/history/` | Explicitly retired deployment guides |
| `performance/` | Harnesses, methodology, immutable receipts and measured findings |
| `experiments/` | Prototype index and admission boundary |
| `third_party/` | Vendored source, licenses and provenance |
| `scripts/` | User/developer entry points and existing deployment adapters |

## Compatibility and frozen inputs

[compatibility.json](../../profiles/compatibility.json) names generated legacy
recipe and source exports. Edit their canonical source, then run
`python scripts/generate_profiles.py`. CI checks freshness. Retained source
paths serve published overlays and imported scripts; they are not additional
implementation owners. Tests live with the canonical implementation.

Source-image assembly and version-specific builders remain at their original
paths when relative inputs, source receipts or installed paths bind them there.
The [builder catalog](../../runtime/images/builders.json) explains these
exceptions. The switched launcher included in the frozen source image remains
byte-identical; maintained host launching uses `runtime/common/switched.py`.
The TP2 compatibility launcher delegates to `runtime/common/tp2.py`.

The native tiled-prefill substrate remains under its recorded path in
`spark_transport/experiments/tiled_prefill`. It is consumed by native libraries
and snapshot manifests, so its directory name does not make it disposable.
Its [classification](../../experiments/README.md) explains the maintained boundary.
Maintained fabric code belongs to `spark_transport/`; RoCEnante runtime
composition belongs to `integrations/vllm/`.

Existing Markdown URLs retain heading anchors and point to their authoritative
guides. [documentation-paths.json](documentation-paths.json) records the moves.
Preserve evidence and traces according to their release and measurement contracts.

See the [migration inventory](layout-migration.md) for callers, dispositions and
conditions for retiring compatibility paths. New work belongs with its owner;
adding another version-named launcher is not the default extension mechanism.
