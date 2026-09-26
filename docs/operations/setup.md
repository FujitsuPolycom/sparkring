# Set up SparkRing

To install a model with one command, use the Debian package and
`sudo sparkring install` as described in [Install SparkRing](install.md); that
guide covers first boot requirements, networking and model assets for the
installer profiles. This page is the manual procedure for the profile guides
listed below.

SparkRing runs a language model across NVIDIA GB10 machines. Manual
installation has three parts: prepare the hosts and network, obtain the
container and model weights, then start and test a selected deployment.

The [SparkRing image](images.md) contains the inference software. It does not
contain model weights or configure your hosts. Each rank needs the complete
checkpoint. Rank means a machine's fixed position; rank 0 serves the API and is
the default controller for these instructions.

Status: **implemented**. Existing profile records cover prepared systems and
bounded serving checks. This manual sequence has not been run from
factory-reset hosts. Installation checks, network checks and model
qualification are separate results.

## 1. Choose your deployment

Pick one row and keep it for the entire installation. Profile guides own their
image, checkpoint and settings. Start with the row's default configuration;
advanced variants have their own instructions and evidence.

| Machines | Model | Profile ID | Serving guide |
|---|---|---|---|
| 2 | GLM-5.3-Flash | `glm53-flash-spark-tp2-dcp1-sparkcache` | [GLM pair](../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) |
| 2 | Qwen3.8-Flash-Next | `qwen38-flash-next-tp2` | [Qwen pair](../../profiles/qwen38-flash-next-tp2/README.md) |
| 4 | GLM-5.3-Flash | `glm53-flash-spark-tp4-dcp1-sparkcache` | [GLM ring](../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) |
| 4 | Qwen3.8-Flash-Next | `qwen38-flash-next-qad-tp4` | [Qwen ring](../../profiles/qwen38-flash-next-qad-tp4/README.md) |

The GLM defaults enable SparkCache. The Qwen defaults disable it; the Qwen
guides name the cache-enabled profile as an explicit alternative. The
[full catalog](../../profiles/README.md) includes other deployments. Six-node
and switched deployments use their own experimental instructions.

Record the chosen profile, physical rank labels, management IPs/usernames and
storage paths privately. Management addresses carry SSH and client traffic;
fabric addresses belong to the directly cabled ConnectX interfaces. Do not
substitute one for the other.

## 2. Prepare hosts and install one checkout

**Run on:** every Spark, with rank 0 as controller. **Shell:** Bash on Linux.
Follow [host preparation](host-preparation.md) once. It covers first boot,
permissions, software, the exact checkout and storage planning.

**Finished when:** the same checkout revision is present on every rank, the
login user can use Docker, host checks pass and destination storage is adequate.
A successful host check alone does not verify the network or model assets.

## 3. Configure and verify the data network

| Machines | Procedure | Finished when |
|---|---|---|
| 2 | [Prepare a pair](pair-network.md) | Both Socket Direct functions have the expected IP/GID mapping and neighbor/MTU checks pass on both ranks |
| 4 | [Prepare the ring hosts](../GLM53_SPARK_MESH_HOST_SETUP.md#2-cable-the-four-node-data-ring), sections 2–7 | Primary and secondary fabric checks pass and mesh driver prerequisites are satisfied |

For prepared hosts, inspect and reuse verified settings. Do not run fresh-network
configuration over an existing deployment. IP ping success is not RDMA collective
qualification. The serving guide retains its native transport checks and startup
verification. Leave the complete model stopped during standalone GPU/RDMA probes.

## 4. Obtain the image and checkpoint

Continue to the selected serving guide's image/checkpoint section. Run its
`setup show` command to obtain the release values from the profile. The commands
then pull and verify the image and download or verify the checkpoint on every rank.

**Finished when:** every rank has the same image identity and the complete pinned
checkpoint. Use separate writable cache directories. A directory name or image
tag alone is not verification. No image build is required for these published
selections.

## 5. Review and start the deployment

Use the selected guide's planner and lifecycle. Inspect ranks, model/cache paths,
image and checkpoint before creating containers. Creation may refuse an existing
name; do not delete the existing deployment merely to make the example work.

**Finished when:** all ranks complete startup and the model's API is ready. Cold
kernel preparation can take many minutes; GLM permits 30 minutes. Use the guide's
logs and readiness checks instead of restarting during preparation.

## 6. Test a response and save restart instructions

Run the guide's health, model-list and short generation requests. Require the
documented model name and a successful response. Inspect every rank for startup
or transport errors. This establishes a basic installation smoke test, not
maximum-context, media-quality or sustained-load qualification.

Record the checkout revision, setup selection, image receipt where applicable,
private site/deployment path, container names and test output. Keep them outside
Git. Save the guide's stop and restart sequence; `create` is not a restart command.

When persistent caching matters, run the guide's coordinated restart/restore
test. The [full validation runbook](profile-validation.md) is for subsequent
workload qualification and benchmarks. Use the [rehearsal checklist](setup-rehearsal.md)
to evaluate these instructions on prepared or blank hosts.
