# Compose files

These Docker Compose files run a profile's model without the installer: one
file per Spark (rank), and for three two-Spark profiles one standalone file
that both Sparks share. [`sparkring install`](install.md) is the recommended
path. It also prepares the network, image and checkpoint that the files
expect ([images and checkpoints](#images-and-checkpoints)).

[Compose deployments](compose.md) · [`compose` and `validate-compose` commands](commands.md#compose-and-validate-compose) ·
[Install Builder](compose-builder.md): these files for your own Sparks, checkpoint and settings, as a zip

## Published files

| Model | Sparks | Profile | Profile status | Files | Instructions |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | 2 | `qwen38-flash-next-tp2` | implemented | Ranks [0](../../profiles/qwen38-flash-next-tp2/compose/compose.rank0.yaml), [1](../../profiles/qwen38-flash-next-tp2/compose/compose.rank1.yaml) · [standalone](../../profiles/qwen38-flash-next-tp2/compose/standalone.yaml) · [site example](../../profiles/qwen38-flash-next-tp2/compose/site.example.yaml) | **Step-by-step:** [Qwen on two Sparks with Compose](../../profiles/qwen38-flash-next-tp2/compose/README.md) |
| Qwen3.8-Flash-Next | 4 | `qwen38-flash-next-qad-tp4` | implemented | Ranks [0](../../profiles/qwen38-flash-next-qad-tp4/compose/compose.rank0.yaml), [1](../../profiles/qwen38-flash-next-qad-tp4/compose/compose.rank1.yaml), [2](../../profiles/qwen38-flash-next-qad-tp4/compose/compose.rank2.yaml), [3](../../profiles/qwen38-flash-next-qad-tp4/compose/compose.rank3.yaml) · [site example](../../profiles/qwen38-flash-next-qad-tp4/compose/site.example.yaml) | [Render and run](../../profiles/qwen38-flash-next-qad-tp4/compose/README.md) |
| GLM-5.3-Flash | 2 | `glm53-flash-nvfp4-spark-tp2` | implemented | Ranks [0](../../profiles/glm53-flash-nvfp4-spark-tp2/compose/compose.rank0.yaml), [1](../../profiles/glm53-flash-nvfp4-spark-tp2/compose/compose.rank1.yaml) · [site example](../../profiles/glm53-flash-nvfp4-spark-tp2/compose/site.example.yaml) | [Per-rank files](#per-rank-files) |
| GLM-5.3-Flash | 4 | `glm53-flash-nvfp4-spark-tp4` | implemented | Ranks [0](../../profiles/glm53-flash-nvfp4-spark-tp4/compose/compose.rank0.yaml), [1](../../profiles/glm53-flash-nvfp4-spark-tp4/compose/compose.rank1.yaml), [2](../../profiles/glm53-flash-nvfp4-spark-tp4/compose/compose.rank2.yaml), [3](../../profiles/glm53-flash-nvfp4-spark-tp4/compose/compose.rank3.yaml) · [site example](../../profiles/glm53-flash-nvfp4-spark-tp4/compose/site.example.yaml) | [Per-rank files](#per-rank-files) |
| MiMo-V2.6-Flash-MOPD | 2 | `mimo-v26-flash-mopd-tp2` | implemented | Ranks [0](../../profiles/mimo-v26-flash-mopd-tp2/compose/compose.rank0.yaml), [1](../../profiles/mimo-v26-flash-mopd-tp2/compose/compose.rank1.yaml) · [site example](../../profiles/mimo-v26-flash-mopd-tp2/compose/site.example.yaml) | [Per-rank files](#per-rank-files) |
| MiMo-V2.6-Flash-MOPD | 4 | `mimo-v26-flash-mopd-tp4` | implemented | Ranks [0](../../profiles/mimo-v26-flash-mopd-tp4/compose/compose.rank0.yaml), [1](../../profiles/mimo-v26-flash-mopd-tp4/compose/compose.rank1.yaml), [2](../../profiles/mimo-v26-flash-mopd-tp4/compose/compose.rank2.yaml), [3](../../profiles/mimo-v26-flash-mopd-tp4/compose/compose.rank3.yaml) · [site example](../../profiles/mimo-v26-flash-mopd-tp4/compose/site.example.yaml) | [Per-rank files](#per-rank-files) |
| DeepSeek-V4.1-Flash | 4 | `deepseek-v41-flash-tp4` | implemented | Ranks [0](../../profiles/deepseek-v41-flash-tp4/compose/compose.rank0.yaml), [1](../../profiles/deepseek-v41-flash-tp4/compose/compose.rank1.yaml), [2](../../profiles/deepseek-v41-flash-tp4/compose/compose.rank2.yaml), [3](../../profiles/deepseek-v41-flash-tp4/compose/compose.rank3.yaml) · [site example](../../profiles/deepseek-v41-flash-tp4/compose/site.example.yaml) | [Per-rank files](#per-rank-files) |
| Swift-1.5-Qwen3.8-Flash-Next | 2 | `swift15-qwen38-flash-next-tp2` | implemented | Ranks [0](../../profiles/swift15-qwen38-flash-next-tp2/compose/compose.rank0.yaml), [1](../../profiles/swift15-qwen38-flash-next-tp2/compose/compose.rank1.yaml) · [site example](../../profiles/swift15-qwen38-flash-next-tp2/compose/site.example.yaml) | [Per-rank files](#per-rank-files) |
| Swift-1.5-Qwen3.8-Flash-Next | 4 | `swift15-qwen38-flash-next-tp4` | research-only | Ranks [0](../../profiles/swift15-qwen38-flash-next-tp4/compose/compose.rank0.yaml), [1](../../profiles/swift15-qwen38-flash-next-tp4/compose/compose.rank1.yaml), [2](../../profiles/swift15-qwen38-flash-next-tp4/compose/compose.rank2.yaml), [3](../../profiles/swift15-qwen38-flash-next-tp4/compose/compose.rank3.yaml) · [site example](../../profiles/swift15-qwen38-flash-next-tp4/compose/site.example.yaml) | [Per-rank files](#per-rank-files) |
| Qwen3.8-Flash-Next with SparkCache | 2 | `qwen38-flash-next-tp2-sparkcache` | qualified | Ranks [0](../../profiles/qwen38-flash-next-tp2-sparkcache/compose/compose.rank0.yaml), [1](../../profiles/qwen38-flash-next-tp2-sparkcache/compose/compose.rank1.yaml) · [standalone](../../profiles/qwen38-flash-next-tp2-sparkcache/compose/standalone.yaml) · [site example](../../profiles/qwen38-flash-next-tp2/compose/site.example.yaml) | [Notes](../../profiles/qwen38-flash-next-tp2-sparkcache/compose/README.md) |
| Qwen3.8-Flash-Next with SparkCache | 4 | `qwen38-flash-next-qad-tp4-sparkcache` | qualified | Ranks [0](../../profiles/qwen38-flash-next-qad-tp4-sparkcache/compose/compose.rank0.yaml), [1](../../profiles/qwen38-flash-next-qad-tp4-sparkcache/compose/compose.rank1.yaml), [2](../../profiles/qwen38-flash-next-qad-tp4-sparkcache/compose/compose.rank2.yaml), [3](../../profiles/qwen38-flash-next-qad-tp4-sparkcache/compose/compose.rank3.yaml) · [site example](../../profiles/qwen38-flash-next-qad-tp4/compose/site.example.yaml) | [Notes](../../profiles/qwen38-flash-next-qad-tp4-sparkcache/compose/README.md) |
| GLM-5.3-Flash with SparkCache | 2 | `glm53-flash-spark-tp2-dcp1-sparkcache` | qualified | [Standalone](../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/compose/standalone.yaml) | [Standalone file](#standalone-file-two-sparks); hosts: [profile guide](../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) |

- Profile status is the profile's own; the files add no serving
  qualification.
- `sparkring compose start` coordinates only the two Qwen SparkCache variants.
  For the nine installer profiles, `render` and `check` work and `start` stops
  at image admission ([supported profiles](compose.md#supported-profiles)).
- Qwen SparkCache variants use their base profile's site example.
- Every file sets `SPARKRING_TRANSPORT_PROFILE: tp2-rocenante-adaptive-prepared`,
  the image's prepared B12X RoCE transport bundle, on two and four Sparks
  alike; the `tp2-` name is kept for compatibility
  ([transport bundle](compose.md#what-each-input-controls)).
- Each installer-profile rank is the container `sparkring install` runs, with
  the same image and serving settings but no per-rank runtime-binding file or
  managed telemetry mount.
  The status plugin then reports worker identity as `binding_not_configured`;
  serving is unaffected.
- GLM-5.3-Flash with SparkCache on four Sparks publishes no YAML: its
  [Compose creation guide](../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/compose/README.md)
  (`implemented`) generates private files through `sparkring deploy`.

## Images and checkpoints

The nine installer profiles default to release `2026.10.4`'s image and the
same checkpoints as `sparkring install`. This applies to generated per-rank
files, `sparkring compose render`, `sparkring init` and its ZIP export, and
standalone installer-profile exports. GLM-5.3-Flash and two-Spark Qwen use CSF
by default on that image. The manifest records the selected image and checkpoint.

Use `sparkring compose render PROFILE --image 2026.10.1` or
`sparkring init --profile PROFILE --image 2026.10.1` with the other required
arguments to select release `2026.10.1`. Its image uses the profile's own
checkpoint because it cannot read CSF.

The Qwen and GLM SparkCache exports listed above use their frozen
`shared-2026.09.3` image and checkpoints; they do not run on `2026.10.4`.
Installer SparkCache variants (`glm53-flash-nvfp4-sparkcache-tp2` and
`glm53-flash-nvfp4-sparkcache-tp4`) have no public Compose exports and require an explicit development image
lock that lists them. Relayed six- and eight-Spark profiles use the
installer's SIRCL transport and have no public Compose exports. The eight-Spark
profiles `qwen38-flash-next-qad-tp8` and `glm53-nvfp4-csf-tp8` also require an
explicit compatible development lock: no packaged image lists them. Use
`sparkring images --profile PROFILE` to see which packaged images list a profile.

## How to use them

### Standalone file (two Sparks)

Each Spark needs:

- Docker with the NVIDIA Container Toolkit and `docker compose`;
- the image named in the file;
- the complete checkpoint at the revision in the file's header;
- the p0-to-p0 cable with both functions addressed (MTU 9000).

On each Spark, in an empty directory:

1. Save `standalone.yaml` as `compose.yaml`, or write the same file with
   `sparkring export --format compose --profile PROFILE --output compose.yaml`.
   For `qwen38-flash-next-tp2`, also copy
   [`runtime/common/loader-seccomp.json`](../../runtime/common/loader-seccomp.json)
   to that path relative to `compose.yaml`.
2. Create `.env` next to `compose.yaml`:
   ```bash
   SPARKRING_MODEL_DIR=/path/to/checkpoint       # complete, verified checkpoint
   SPARKRING_CACHE_DIR=/var/tmp/sparkring-cache  # existing writable directory
   SPARKRING_MASTER_ADDR=198.18.0.1              # Spark 0's fabric address, on both Sparks
   SPARKRING_HOST_IP=198.18.0.1                  # this Spark's fabric address
   SPARKRING_INTERFACE=enp1s0f0np0               # this Spark's fabric interface
   ```
3. Run `docker compose --profile rank1 up -d` on Spark 1, then
   `docker compose --profile rank0 up -d` on Spark 0, which serves the API.

The [Qwen two-Spark quickstart](../../profiles/qwen38-flash-next-tp2/compose/README.md)
covers fabric addresses, checkpoint download, checks and failures.

### Per-rank files

The published rank files use the site example's placeholder addresses and
paths. Render your own from a SparkRing checkout; take `PROFILE` and
`SITE_EXAMPLE` from the table:

```bash
mkdir -p .sparkring
cp SITE_EXAMPLE .sparkring/site.yaml
# Set name, master and every rank's host, host_ip, interface, hcas, gid,
# model, cache, repository and deployment_root; on four Sparks also fabric.
python3 scripts/sparkring.py compose render PROFILE \
  --site .sparkring/site.yaml --output .sparkring/deployments/PROFILE
python3 scripts/sparkring.py compose check --deployment .sparkring/deployments/PROFILE
```

Before you start, on every Spark:

- `model` holds the complete checkpoint;
- the `cache` and `deployment_root` directories exist;
- the image is pulled by the digest on the file's `image:` line (the files use
  `pull_policy: never`).

Four-Spark profiles need the prepared mesh fabric, which
`sparkring install` creates and these files do not; each rank's `fabric` entry
in the site file references it.

To start:

- Installer profiles: copy each `rankN/compose.yaml` to its Spark and run
  `docker compose -f compose.yaml up -d`, workers first, then rank 0. Compose
  reads `runtime/common/loader-seccomp.json` under the site's `repository`
  directory on each Spark.
- Qwen SparkCache variants: `compose start` stages and starts every rank after
  you approve its plan ([Check and coordinate hosts](compose.md#check-and-coordinate-hosts)).

[`scripts/generate_compose_examples.py`](../../scripts/generate_compose_examples.py)
generates the YAML from each profile's configuration and site example; change
those inputs and regenerate instead of editing the YAML.
