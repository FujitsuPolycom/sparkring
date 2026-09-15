# GLM-5.3-Flash with NVIDIA NVFP4 weights

Status: **Development** for the R37 Docker/Compose adaptation. Contributor
serving results apply to a separate, identified image; see the
[NVIDIA checkpoint evidence](../performance/records/glm53-flash/nvidia-nvfp4.md).
NVFP4-Spark remains the default target for every existing GLM profile.

The optional target is
[`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/423acf37583782c51c142d145aef733d72943d93).
Its [target record](glm53-target-variants.json) pins the revision, configuration,
weight index, file manifest and cache fingerprint. It uses `modelopt`
quantization and the `safetensors` loader. The launcher preserves the verified
checkpoint's quantization settings and excludes its BF16 MTP predictor layers.

## R37 deployment

Follow the [GLM TP4 quickstart](glm53-flash-spark-tp4-dcp1-sparkcache/README.md)
through image verification to set `SPARKRING_RECEIPT`. Replace the example
addresses and SSH aliases below with your controller and four hosts. Discover
and plan into a separate deployment directory:

```bash
sr() { python3 scripts/sparkring.py deploy "$@"; }
STATE="$PWD/.sparkring/glm-nvidia-tp4"
sr discover --controller-address 192.0.2.10 \
  --node spark0=192.0.2.20 --node spark1=192.0.2.21 \
  --node spark2=192.0.2.22 --node spark3=192.0.2.23 \
  --output "$STATE/inventory.json"
sr plan --inventory "$STATE/inventory.json" --name glm-nvidia-tp4 \
  --workspace /srv/sparkring/glm-nvidia-tp4 --preserve-existing-network \
  --image-receipt "$SPARKRING_RECEIPT" --runtime-profile tp4-dcp1-sparkcache \
  --target-model-variant nvidia-nvfp4 --output "$STATE/preparation.json"
```

Select `tp4-dcp1` to disable SparkCache. Continue with network planning, staging
and managed Docker or Compose creation from the same quickstart, using this
`STATE` throughout. Staging downloads or verifies NVIDIA weights at the pinned
revision. Model and cache directories are distinct from the Spark default;
the cache namespace also receives the `-nvidia-nvfp4` suffix. Both target and
MTP cache identities use the NVIDIA fingerprint.

Container planning requires the pinned `config.json` and weight index on the
rank host. Changed or missing metadata blocks an executable plan. Creation
rechecks that metadata and the plan's source identities. The normal readiness
budget is 900 seconds; this target uses 1500 seconds to accommodate its loader.
Fabric-failure timers and readiness requirements are unchanged.

The [DCP4 procedure](glm53-flash-spark-tp4-dcp1-sparkcache/README.md#dcp4-alternative)
also accepts `"target_model_variant": "nvidia-nvfp4"` in the private site file.
Use the pinned NVIDIA model revision and distinct model/cache roots on all
four ranks. The deployment-suite CLI remains DCP1-only.

## Image and validation limits

The registered `lil-r37-glm-spark` composition contains the ModelOpt loader,
wildcard exclusion matching and B12X dense NVFP4 kernel. The host adaptation
does not modify that image. CPU checks cover model selection, pinned metadata,
MTP exclusions, cache identity, Docker/Compose settings and readiness budgets.
NVIDIA loading, responses and cache restoration on this exact R37 image remain
unqualified; use a bounded maintainer test before promoting this target.

R33, R35 and the frozen source-image contracts reject this target. Their baked
loader contracts cannot be changed by a host setting. The retained compute
launcher supports the contributor's separately documented image and settings;
its results do not transfer to R37 or to other images.
