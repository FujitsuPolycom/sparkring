# Model-neutral SparkRing runtime packaging

Status: **research-only — testing in progress**. The candidate packages the
working ARM64/SM121 runtime and its three externally mounted code/library
files into a model-neutral image. Models and site-specific serving settings
belong in profiles; model weights are not included.

Repository: `ghcr.io/fujitsupolycom/sparkring`. The verified release reference
is in [publication.json](publication.json); the builder itself never pushes.
Existing model-specific images remain
unchanged. A neutral name does not qualify all profiles against this runtime.

| Profile | Evidence scope |
|---|---|
| GLM-5.3 Flash NVFP4-Spark, TP2/DCP1, native MTP3 | Bounded source-deployment checks; child image still requires serving qualification |
| Other SparkRing profiles | Not qualified against this candidate; use their existing pinned images |

## Build locally

The Docker host must already contain the exact parent image ID in
`manifest.json`. Place the three listed assets in a directory; the builder
rejects missing or changed files before generating a build context.

```bash
python3 package_image.py --assets /path/to/assets --context /path/to/new-context
# Inspect the printed command and generated Dockerfile, then use a new path:
python3 package_image.py --assets /path/to/assets --context /path/to/build-context --build
```

No GPU access, model load, push, stop, or deployment change is performed.
The image preserves vLLM serve as its entrypoint; its default argument is
--help. The inherited model-specific healthcheck is disabled: probe the
profile's actual API endpoint and then test inference.

The build receipt identifies the child and manifest. The publication receipt
records distribution status separately from the embedded pre-publication
manifest. The manifest, not
inherited historical parent labels, defines the candidate support scope.
This packager is not yet a source-complete replacement for existing builders.
The matching source pin is public; see [distribution notes](DISTRIBUTION.md).
Before image publication: audit inherited layer provenance
and redistribution licenses, produce the SBOM, and qualify the child image.
Do not replace missing registry pins with local image IDs in public pull commands.

See the [draft TP2 quickstart](../../docs/GLM53_FLASH_SPARK_TP2_EXPERIMENTAL_QUICKSTART.md).
