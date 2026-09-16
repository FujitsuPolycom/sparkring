# Qwen3.8-Flash-Next QAD on four Sparks

Status: **Development**. This profile selects the published R37 shared image
with Qwen collective selection, HC fusion and MTP prefill GEMMs. The baked image
passed inventory, four-rank Compose startup, bounded text checks and matched
prefill/decode measurements. The [serving record](../../performance/records/qwen38-flash-next/r37-shared-tp4.json)
identifies the exact image, conditions and remaining limits. It is not a general
media, full-context or long-duration qualification.

| Setting | Selection |
|---|---|
| Model quant | [NVFP4 QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/629bc3218833a38b475b719f34aa571666f4a03e) |
| Parallelism | TP4/DCP1 on a four-node ring with hardware-forwarded mesh paths |
| Context / sequences / batch | 262K / 16 / 8192 |
| KV allocation | 24 GiB FP8 per rank |
| Speculation | MTP3 |
| Collectives | RoCEnante all-reduce up to 20 KiB; larger reductions use dual-domain NCCL |
| Prefill | Qwen HC up/gate fusion and large-row MTP GEMMs |
| SparkCache | [Optional](../qwen38-flash-next-qad-tp4-sparkcache/README.md); disabled by default. Native prefix caching remains enabled |
| Media | Three images / one video, 16 configured frames; QAD TP4 media acceptance is pending |

The [configuration](config.json) owns these settings. The original PTQ checkpoint
and published SparkCache image use the [two-Spark quickstart](../qwen38-flash-next-tp2/README.md).
GLM-specific mHC/KDA and SIRCL serving switches are disabled in this Qwen profile.

## Prepare image, model and fabric

Use four Linux ARM64 Sparks with Docker Compose and the
[host prerequisites](../../docs/operations/prerequisites.md). Pull the exact
[shared feature image](../../runtime/images/compositions/lil-r37-shared/README.md)
on every host:

```bash
BASE_IMAGE='ghcr.io/fujitsupolycom/sparkring@sha256:f5a7e01c6112c8ef85a51b24bfacfd3934ee9cfff06b7e8c72abcf5d90b50270'
IMAGE_REF='ghcr.io/fujitsupolycom/sparkring@sha256:aef597a5ee70f7b4e0807901e43456b6ac8d2234247ab4df6a6cfe031e5169c6'
# The base supplies the pinned parent receipt; Docker reuses shared layers.
docker pull --platform linux/arm64 "$BASE_IMAGE"
docker pull --platform linux/arm64 "$IMAGE_REF"
docker image inspect --format '{{.Id}}' "$IMAGE_REF"
```

The image ID must be
`sha256:2540686d726a28eb07784f9d2db5dc1f795404c7874fc1d6c11f018cd789adc2`.
The readable tag is `ghcr.io/fujitsupolycom/sparkring:r37-shared-arm64-2540686d726a`;
the [publication receipt](../../runtime/images/compositions/lil-r37-shared/publication.json)
binds it to the immutable digest. The coordinator verifies the complete image
contents before serving. Local image builds remain a separate developer workflow.

Use an existing verified checkpoint or download the pinned
[`qad-step-4000` revision](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/629bc3218833a38b475b719f34aa571666f4a03e)
into a dedicated directory on each host. The repository's `main` branch contains
the separate PTQ checkpoint; it does not substitute for this QAD revision.

```bash
hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
  --revision 629bc3218833a38b475b719f34aa571666f4a03e \
  --local-dir /srv/models/Qwen3.8-Flash-Next-NVFP4-QAD/629bc3218833
REPO=$PWD
(cd /srv/models/Qwen3.8-Flash-Next-NVFP4-QAD/629bc3218833 && \
  sha256sum --check "$REPO/profiles/qwen38-flash-next-qad-tp4/SHA256SUMS")
```

Verify the complete shard checksums before launch. The coordinator checks model
metadata identity; it does not substitute that for full weight verification.
Model mounts are read-only, and caches must be outside the model directory.

Prepare the [managed ring fabric](../../runtime/glm53-spark-mtp3-mesh/MANAGED_MESH.md)
and its private mesh site before starting a model. The
[fabric owner](../../spark_transport/fabric/cx7_hairpin_diagonal/README.md)
defines the network contract independently of Qwen. The coordinator does not
install routes, hardware rules or source markers. Do not restart a fabric
controller while dependent model ranks are running.

The private Compose site binds the mesh-site file hash and canonical plan hash.
Its HCA order is primary clockwise/counterclockwise followed by secondary
clockwise/counterclockwise. Host checks verify the requested devices, GIDs,
routes, hardware rules and live marker attachment. This is a readiness snapshot,
not an end-to-end RDMA test or continuing fabric supervision.

## Render, check and start

Follow the [shared Compose guide](../../docs/operations/compose.md) for controller
and host dependencies. Use the same source checkout on every host. Prepare
dedicated cache and deployment directories, then copy and edit the site example:

```bash
mkdir -p .sparkring
cp profiles/qwen38-flash-next-qad-tp4/compose/site.example.yaml .sparkring/qwen-qad.site.yaml
# Fill all four hosts, directories and the prepared fabric's actual identities.
python3 scripts/sparkring.py compose render qwen38-flash-next-qad-tp4 \
  --site .sparkring/qwen-qad.site.yaml --output .sparkring/deployments/qwen-qad
python3 scripts/sparkring.py compose check --deployment .sparkring/deployments/qwen-qad
python3 scripts/sparkring.py compose check --deployment .sparkring/deployments/qwen-qad --hosts
python3 scripts/sparkring.py compose start --deployment .sparkring/deployments/qwen-qad
```

Review the printed plan and repeat `start` with `--approve` and its exact hash.
The coordinator creates all four stopped containers, starts workers before the
API rank, and checks readiness. TP4 preflight uses `sudo -n` to inspect root-owned
fabric processes. Existing workloads must be stopped explicitly during the
agreed test window; the coordinator never replaces them.

On rank 0, use the container name from `rank0/container.json`:

```bash
docker logs --follow --tail 100 sr-qwen-qad-example-r0
curl --fail http://127.0.0.1:8015/health
curl --fail http://127.0.0.1:8015/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next-NVFP4-QAD","messages":[{"role":"user","content":"What is 17 + 25?"}],"temperature":0,"max_tokens":128}'
```

Confirm a correct response and feature-activation evidence on every rank before
benchmarking. API readiness alone does not establish correctness or throughput.
Use the [bounded image comparison](../../performance/qwen-image-comparison.md)
for matched prefill/decode measurements and their interpretation.

## Stop and rollback

```bash
python3 scripts/sparkring.py compose stop --deployment .sparkring/deployments/qwen-qad
# Review and repeat with the printed --approve hash.
```

Stop retains the exact containers and cache directories. Restore the previous
deployment only after all four test ranks have stopped. Keep its image and
fabric configuration intact; do not change marker ownership underneath live
collectives. See the shared guide's [recovery behavior](../../docs/operations/compose.md#stop-and-recover).
