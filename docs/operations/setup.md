# Set up SparkRing manually

Manual setup for the four profiles below: prepare the hosts and data network,
then follow the profile guide to pull the image, fetch the model and start it.
The normal path is `sudo sparkring install`, which does all of this in one
command; see [Install SparkRing](install.md).

- Check the [prerequisites](prerequisites.md) first.
- The [SparkRing image](images.md) holds only the inference software: no model
  weights or host configuration. Every rank needs the complete checkpoint.
- Rank 0 serves the API and is the controller in these instructions.

## 1. Choose your deployment

| Sparks | Model | Profile ID | Guide |
|---|---|---|---|
| 2 | GLM-5.3-Flash | `glm53-flash-spark-tp2-dcp1-sparkcache` | [GLM pair](../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) |
| 2 | Qwen3.8-Flash-Next | `qwen38-flash-next-tp2-sparkcache` | [Qwen pair](../../profiles/qwen38-flash-next-tp2/README.md) |
| 4 | GLM-5.3-Flash | `glm53-flash-spark-tp4-dcp1-sparkcache` | [GLM ring](../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) |
| 4 | Qwen3.8-Flash-Next | `qwen38-flash-next-qad-tp4-sparkcache` | [Qwen ring](../../profiles/qwen38-flash-next-qad-tp4/README.md) |

All four use the shared 2026.09.3 image with SparkCache on. Without SparkCache,
the Qwen profiles `qwen38-flash-next-tp2` and `qwen38-flash-next-qad-tp4`
install with `sudo sparkring install`. The [profile catalog](../../profiles/README.md)
lists the other profiles; six-node and switched deployments are outside it and
have their own guides.

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

- On hosts that already serve a model, check and reuse the existing network
  settings; do not configure them again.
- Keep the model stopped while running standalone GPU or RDMA probes.

## 4. Obtain the image and checkpoint

Follow the guide's image and checkpoint section: `setup show` reads both from
the profile, then the guide pulls and verifies the image and downloads or
verifies the checkpoint on every rank.

**Done when:** every rank has the same image ID and the complete pinned
checkpoint, and each rank has its own writable cache directory.

## 5. Review and start the deployment

Use the guide's plan and start commands.

- Before creating containers, check the ranks, model and cache paths, image and
  checkpoint.
- If creation refuses an existing container name, do not delete the existing
  deployment to make room.
- The first start prepares kernels and can take many minutes; the GLM ring
  allows 30 minutes. Watch the guide's logs and readiness checks instead of
  restarting.

**Done when:** all ranks finish startup and the API is ready.

## 6. Test a response and save restart instructions

Run the guide's health, model-list and short generation requests.

**Done when:** the response succeeds with the documented model name and no
rank's log shows startup or transport errors.

Save privately: the checkout revision, `.sparkring/selection.env`, the image ID,
the site or deployment path, container names, test output, and the guide's stop
and restart commands. `create` is not a restart command.

To measure speed, accuracy and cache restore, use
[Validate a serving profile](profile-validation.md). To rehearse these
instructions on prepared or blank hosts, use the [rehearsal checklist](setup-rehearsal.md).
