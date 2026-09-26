# Set up SparkRing manually

This page is the manual setup for the four profiles below: you prepare the
hosts and data network, then follow the profile guide to pull the image, fetch
the model and start it. The normal path is `sudo sparkring install`, which does
all of this in one command; see [Install SparkRing](install.md).

The [SparkRing image](images.md) holds the inference software, not model
weights or host configuration. Every rank needs the complete checkpoint. Rank 0
serves the API and is the controller in these instructions.

## 1. Choose your deployment

| Sparks | Model | Profile ID | Guide |
|---|---|---|---|
| 2 | GLM-5.3-Flash | `glm53-flash-spark-tp2-dcp1-sparkcache` | [GLM pair](../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) |
| 2 | Qwen3.8-Flash-Next | `qwen38-flash-next-tp2-sparkcache` | [Qwen pair](../../profiles/qwen38-flash-next-tp2/README.md) |
| 4 | GLM-5.3-Flash | `glm53-flash-spark-tp4-dcp1-sparkcache` | [GLM ring](../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) |
| 4 | Qwen3.8-Flash-Next | `qwen38-flash-next-qad-tp4-sparkcache` | [Qwen ring](../../profiles/qwen38-flash-next-qad-tp4/README.md) |

All four use the shared 2026.09.3 image with SparkCache on. The Qwen profiles
without SparkCache, `qwen38-flash-next-tp2` and `qwen38-flash-next-qad-tp4`,
install with `sudo sparkring install`. The [profile catalog](../../profiles/README.md)
lists every other deployment; six-node and switched deployments have their own guides.

Record privately: the profile, each rank's label, management IP and username,
and the storage paths. Management addresses carry SSH and API traffic; fabric
addresses belong to the directly cabled ConnectX ports. Do not swap them.

## 2. Prepare hosts and install one checkout

On every Spark, in Bash, follow [host preparation](host-preparation.md).

**Done when:** every rank has the same checkout revision, the login user can
run Docker, `host check` passes and `setup storage` passes.

## 3. Configure and verify the data network

| Sparks | Procedure | Done when |
|---|---|---|
| 2 | [Prepare a pair](pair-network.md) | Both Socket Direct functions have the expected IP/GID mapping; neighbor and MTU checks pass on both ranks |
| 4 | [Prepare the ring hosts](../GLM53_SPARK_MESH_HOST_SETUP.md#2-cable-the-four-node-data-ring), sections 2–7 | Primary and secondary fabric checks pass; mesh driver prerequisites are met |

On hosts that already serve a model, check and reuse the existing network
settings instead of configuring them again. Keep the model stopped while running
standalone GPU or RDMA probes.

## 4. Obtain the image and checkpoint

Follow the guide's image and checkpoint section. Its `setup show` command reads
the image and checkpoint from the profile; the commands that follow pull and
verify the image and download or verify the checkpoint on every rank.

**Done when:** every rank has the same image ID and the complete pinned
checkpoint, and each rank has its own writable cache directory.

## 5. Review and start the deployment

Use the guide's plan and start commands. Check ranks, model and cache paths,
image and checkpoint before creating containers. If creation refuses an
existing container name, do not delete the existing deployment to make room.

**Done when:** all ranks finish startup and the API is ready. The first start
prepares kernels and can take many minutes; the GLM ring allows 30. Watch the
guide's logs and readiness checks instead of restarting.

## 6. Test a response and save restart instructions

Run the guide's health, model-list and short generation requests. Require the
documented model name and a successful response, and check every rank's log for
startup or transport errors.

Save privately: the checkout revision, `.sparkring/selection.env`, the image
ID, the site or deployment path, container names and test output. Save the
guide's stop and restart commands; `create` is not a restart command.

To measure speed, accuracy and cache restore, use
[Validate a serving profile](profile-validation.md). To rehearse these
instructions on prepared or blank hosts, use the [rehearsal checklist](setup-rehearsal.md).
