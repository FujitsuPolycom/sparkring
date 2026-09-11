# Model-neutral SparkRing runtime packaging

For the source-pinned GLM TP2/TP4 image with mesh, dual-domain NCCL, coalescing,
mHC, and optional SparkCache, use the [shared GLM source recipe](source_image/README.md).
It downloads public source bases and applies checked-in patches. The asset
packager below describes the separately published three-asset image.

Status: **research-only — testing in progress**. The published ARM64/SM121
runtime image includes a model attention module and two native cache libraries,
identified by hashes in manifest.json. Models and site-specific serving settings
belong in profiles; model weights are not included.

Repository: `ghcr.io/fujitsupolycom/sparkring`. The verified release reference
is in [publication.json](publication.json); the builder itself never pushes.
Existing model-specific images remain
unchanged. A neutral name does not qualify all profiles against this runtime.

| Profile | Evidence scope |
|---|---|
| GLM-5.3 Flash NVFP4-Spark, TP2/DCP1, MTP3 | [Public-image installation, chat, still-image and restore checks](../../performance/records/glm53-flash/tp2-public-image-install-20260906.md) passed; blue-video recognition failed |
| Other SparkRing profiles | Not qualified against this image; use their profile-specific pinned images |

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
inherited historical parent labels, defines the package support scope.
This packager is not yet a source-complete replacement for existing builders.
The source pin, image digest and filesystem SBOM are public; see
[distribution notes](DISTRIBUTION.md) for the audit scope. File/contract
verification is distinct from GPU serving qualification. Qualification of
each model/topology/workload remains required before relying on that profile.
Use the registry digest, not a local Docker image ID, in pull commands.

See the [GLM-5.3 Flash TP2 quickstart](../../docs/GLM53_FLASH_SPARK_TP2_EXPERIMENTAL_QUICKSTART.md).
