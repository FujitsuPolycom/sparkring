# Compose files

SparkRing publishes Docker Compose files that run a profile's model without the
installer: one file per Spark (rank), and for three two-Spark profiles one
standalone file that both Sparks share. [`sparkring install`](install.md) is
the recommended path; it also prepares the network, image and checkpoint these
files expect. For the six installer profiles each rank is the container
`sparkring install` runs, with the same image and serving settings but without
the per-rank runtime-binding file, so the status plugin reports worker identity
as `binding_not_configured`; serving is unaffected.

[Compose deployments](compose.md) · [`compose` and `validate-compose` commands](commands.md#compose-and-validate-compose)

## Published files

| Model | Sparks | Profile | Files | Instructions |
|---|---|---|---|---|
| Qwen3.8-Flash-Next | 2 | `qwen38-flash-next-tp2` | Ranks [0](../../profiles/qwen38-flash-next-tp2/compose/compose.rank0.yaml), [1](../../profiles/qwen38-flash-next-tp2/compose/compose.rank1.yaml) · [standalone](../../profiles/qwen38-flash-next-tp2/compose/standalone.yaml) · [site example](../../profiles/qwen38-flash-next-tp2/compose/site.example.yaml) | **Step-by-step:** [Qwen on two Sparks with Compose](../../profiles/qwen38-flash-next-tp2/compose/README.md) |
| Qwen3.8-Flash-Next | 4 | `qwen38-flash-next-qad-tp4` | Ranks [0](../../profiles/qwen38-flash-next-qad-tp4/compose/compose.rank0.yaml), [1](../../profiles/qwen38-flash-next-qad-tp4/compose/compose.rank1.yaml), [2](../../profiles/qwen38-flash-next-qad-tp4/compose/compose.rank2.yaml), [3](../../profiles/qwen38-flash-next-qad-tp4/compose/compose.rank3.yaml) · [site example](../../profiles/qwen38-flash-next-qad-tp4/compose/site.example.yaml) | [Render and run](../../profiles/qwen38-flash-next-qad-tp4/compose/README.md) |
| GLM-5.3-Flash | 2 | `glm53-flash-nvfp4-spark-tp2` | Ranks [0](../../profiles/glm53-flash-nvfp4-spark-tp2/compose/compose.rank0.yaml), [1](../../profiles/glm53-flash-nvfp4-spark-tp2/compose/compose.rank1.yaml) · [site example](../../profiles/glm53-flash-nvfp4-spark-tp2/compose/site.example.yaml) | [Per-rank files](#per-rank-files) |
| GLM-5.3-Flash | 4 | `glm53-flash-nvfp4-spark-tp4` | Ranks [0](../../profiles/glm53-flash-nvfp4-spark-tp4/compose/compose.rank0.yaml), [1](../../profiles/glm53-flash-nvfp4-spark-tp4/compose/compose.rank1.yaml), [2](../../profiles/glm53-flash-nvfp4-spark-tp4/compose/compose.rank2.yaml), [3](../../profiles/glm53-flash-nvfp4-spark-tp4/compose/compose.rank3.yaml) · [site example](../../profiles/glm53-flash-nvfp4-spark-tp4/compose/site.example.yaml) | [Per-rank files](#per-rank-files) |
| MiMo-V2.6-Flash-RL | 2 | `mimo-v26-flash-rl-tp2` | Ranks [0](../../profiles/mimo-v26-flash-rl-tp2/compose/compose.rank0.yaml), [1](../../profiles/mimo-v26-flash-rl-tp2/compose/compose.rank1.yaml) · [site example](../../profiles/mimo-v26-flash-rl-tp2/compose/site.example.yaml) | [Per-rank files](#per-rank-files) |
| MiMo-V2.6-Flash-RL | 4 | `mimo-v26-flash-rl-tp4` | Ranks [0](../../profiles/mimo-v26-flash-rl-tp4/compose/compose.rank0.yaml), [1](../../profiles/mimo-v26-flash-rl-tp4/compose/compose.rank1.yaml), [2](../../profiles/mimo-v26-flash-rl-tp4/compose/compose.rank2.yaml), [3](../../profiles/mimo-v26-flash-rl-tp4/compose/compose.rank3.yaml) · [site example](../../profiles/mimo-v26-flash-rl-tp4/compose/site.example.yaml) | [Per-rank files](#per-rank-files) |
| DeepSeek-V4.1-Flash | 4 | `deepseek-v41-flash-tp4` | Ranks [0](../../profiles/deepseek-v41-flash-tp4/compose/compose.rank0.yaml), [1](../../profiles/deepseek-v41-flash-tp4/compose/compose.rank1.yaml), [2](../../profiles/deepseek-v41-flash-tp4/compose/compose.rank2.yaml), [3](../../profiles/deepseek-v41-flash-tp4/compose/compose.rank3.yaml) · [site example](../../profiles/deepseek-v41-flash-tp4/compose/site.example.yaml) | [Per-rank files](#per-rank-files) |
| Qwen3.8-Flash-Next with SparkCache | 2 | `qwen38-flash-next-tp2-sparkcache` | Ranks [0](../../profiles/qwen38-flash-next-tp2-sparkcache/compose/compose.rank0.yaml), [1](../../profiles/qwen38-flash-next-tp2-sparkcache/compose/compose.rank1.yaml) · [standalone](../../profiles/qwen38-flash-next-tp2-sparkcache/compose/standalone.yaml) · [site example](../../profiles/qwen38-flash-next-tp2/compose/site.example.yaml) | [Notes](../../profiles/qwen38-flash-next-tp2-sparkcache/compose/README.md) |
| Qwen3.8-Flash-Next with SparkCache | 4 | `qwen38-flash-next-qad-tp4-sparkcache` | Ranks [0](../../profiles/qwen38-flash-next-qad-tp4-sparkcache/compose/compose.rank0.yaml), [1](../../profiles/qwen38-flash-next-qad-tp4-sparkcache/compose/compose.rank1.yaml), [2](../../profiles/qwen38-flash-next-qad-tp4-sparkcache/compose/compose.rank2.yaml), [3](../../profiles/qwen38-flash-next-qad-tp4-sparkcache/compose/compose.rank3.yaml) · [site example](../../profiles/qwen38-flash-next-qad-tp4/compose/site.example.yaml) | [Notes](../../profiles/qwen38-flash-next-qad-tp4-sparkcache/compose/README.md) |
| GLM-5.3-Flash with SparkCache | 2 | `glm53-flash-spark-tp2-dcp1-sparkcache` | [Standalone](../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/compose/standalone.yaml) | [Standalone file](#standalone-file-two-sparks); hosts: [profile guide](../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) |

- Only `qwen38-flash-next-tp2` has a step-by-step quickstart.
- Installer profiles run the installer image lock's image; SparkCache variants
  run release `shared-2026.09.3`'s image. Files name images by registry digest.
  Qwen SparkCache variants use their base profile's site example.
- `sparkring compose start` coordinates only the two Qwen SparkCache variants;
  for the six installer profiles it supports `render` and `check` only.
- GLM-5.3-Flash with SparkCache on four Sparks publishes no YAML: its
  [Compose creation guide](../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/compose/README.md)
  (status Development) generates private files through `sparkring deploy`.

## How to use them

### Standalone file (two Sparks)

Each Spark needs Docker with the NVIDIA Container Toolkit and `docker compose`,
the image named in the file, the complete checkpoint at the revision in the
file's header, and the p0-to-p0 cable with both functions addressed (MTU 9000).
On each Spark, in an empty directory:

1. Save `standalone.yaml` as `compose.yaml`; `sparkring export --format compose
   --profile PROFILE --output compose.yaml` writes the same file. For
   `qwen38-flash-next-tp2`, also copy [`runtime/common/loader-seccomp.json`](../../runtime/common/loader-seccomp.json)
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

The [TP2 quickstart](../../profiles/qwen38-flash-next-tp2/compose/README.md)
covers fabric addresses, checkpoint download, checks and failures in full.

### Per-rank files

The published rank files show each rank's effective settings for the site
example's placeholder addresses and paths. Render files for your own Sparks
from a SparkRing checkout, substituting your profile and its site example:

```bash
mkdir -p .sparkring
cp profiles/qwen38-flash-next-tp2/compose/site.example.yaml .sparkring/qwen.site.yaml
# Set name, master and every rank's host, host_ip, interface, hcas, gid,
# model, cache, repository and deployment_root.
python3 scripts/sparkring.py compose render qwen38-flash-next-tp2 \
  --site .sparkring/qwen.site.yaml --output .sparkring/deployments/qwen
python3 scripts/sparkring.py compose check --deployment .sparkring/deployments/qwen
```

- On every Spark: `model` holds the complete checkpoint, the `cache` and
  `deployment_root` directories exist, and the image is pulled by the digest on
  the file's `image:` line (the files use `pull_policy: never`).
- Four-Spark profiles need the prepared mesh fabric, which `sparkring install`
  creates and these files do not; each rank's `fabric` entry references it.
- Installer profiles: copy each `rankN/compose.yaml` to its Spark and run
  `docker compose -f compose.yaml up -d`, workers first, then rank 0. Compose
  reads `runtime/common/loader-seccomp.json` under the site's `repository`
  directory on each Spark.
- Qwen SparkCache variants: `compose start` stages and starts every rank after
  you approve its plan ([Check and coordinate hosts](compose.md#check-and-coordinate-hosts)).

The YAML files are generated by [`scripts/generate_compose_examples.py`](../../scripts/generate_compose_examples.py)
from each profile's configuration and site example; change those inputs and
regenerate instead of editing the YAML.
