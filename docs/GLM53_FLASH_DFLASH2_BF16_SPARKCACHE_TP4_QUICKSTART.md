# Serve GLM-5.3 Flash with BF16 DFlash2 and SparkCache on four DGX Sparks

> Historical exact-artifact procedure. Use the
> [published JJ r7-compatible quickstart](GLM53_JJ_R7_GB10_TP4_QUICKSTART.md) for a public GB10
> deployment. The identities below remain valid only for their recorded
> evidence.

Use the [GLM-5.3 routing guide](GLM53_FLASH_QUICKSTARTS.md) to compare this
published immutable image with the source-built DFlash7, adaptive-MTP, and
`e10536a` paths.

Runtime performance and correctness come primarily from Local Inference Lab's
[Jovian Judgement vLLM branch](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement),
with Blackwell kernels from
[B12X](https://github.com/local-inference-lab/b12x). This recipe uses
[`GLM-5.3-Flash-NVFP4@520de24e`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/520de24eabf507659eaef7c70f14fd584527facc)
and the BF16
[`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410).
The draft is not the separate Local Inference Lab
[MXFP8 DFlash2 checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8).

Status: **qualified** for startup, semantic generation, runtime health, and one
8,192-token persistent restore using the immutable image, model revisions, and
TP4/DCP1 settings in this guide. The configured 524,288-token request limit and
32-sequence limit were not exercised at their limits. The image is a
FujitsuPolycom community derivative,
not an official NVIDIA, vLLM, local-inference-lab, B12X, Inco AI, Z.AI, or
SparkCache release.

This guide starts one vLLM service across four directly cabled NVIDIA DGX
Spark systems. Tensor parallelism spans all four GPUs. SparkCache persists an
aligned 8,192-token target-model context on each rank's NVMe and can restore
it after all four containers are replaced.

## Qualified configuration

| Setting | Value |
|---|---|
| Target | `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc` |
| Draft | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, BF16 |
| Parallelism | TP4, DCP1, PP1; four hosts |
| Model and scheduler limits | 524,288 tokens; 8,192 batched tokens; 32 sequences |
| GPU KV memory | 12 GiB FP8 per rank; measured capacity 549,950 tokens |
| Speculation | DFlash2; seven tokens; draft TP4 |
| Graphs | target `FULL_AND_PIECEWISE`; DFlash FULL; capture sizes 8–256 |
| Scheduler | asynchronous scheduling; chunked prefill; native prefix caching |
| KDA prefill | Triton |
| SparkCache | 48 GiB maximum and 40 GiB low watermark per rank |

The target repository publishes ModelOpt mixed precision: NVFP4 routed
experts in target layers 3–44 and MXFP8 in the embedded MTP expert layer. The
repository does not identify the unquantized base-checkpoint revision. The
external DFlash checkpoint is not quantized and is licensed CC BY-NC-ND 4.0;
obtain it under Inco AI's published terms.

## Prerequisites

- Four Linux/ARM64 DGX Spark systems with Docker, NVIDIA Container Toolkit,
  passwordless SSH from the operator host, and the direct-cycle RoCE network
  described by `scripts/config/glm53-flash-tp4-site.example.yaml`.
- At least 13 GiB of free GPU memory per rank for the configured KV slab and
  runtime overhead.
- At least 48 GiB of dedicated free local storage per rank for SparkCache,
  plus storage for model files and compilation caches.
- The target and draft revisions downloaded to the same host paths on every
  rank. Model weights are not included in either container image.
- Python 3.11 or newer with PyYAML on the operator host.

## Obtain the images and qualification client

The qualified image can be pulled directly:

```bash
runtime_image='ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd'
sparkcache_image='ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943'
docker pull "${sparkcache_image}"
```

The runtime image is the source-built parent. The SparkCache image adds the
connector and vLLM compatibility patch; it contains the complete runtime, so
the service profile uses only `sparkcache_image`.

The deterministic qualification client is versioned with the SparkCache image
source. Obtain that source at the revision recorded by
`runtime/glm53-flash/pins.json`, even when using the prebuilt image:

```bash
git clone https://github.com/FujitsuPolycom/sparkcache.git
git -C sparkcache checkout --detach 3860a2250193a6679ac6bac857af53e0757841f8
sparkcache_root="$PWD/sparkcache"
```

To build both images instead, follow
[`runtime/glm53-flash/BUILD.md`](../runtime/glm53-flash/BUILD.md) for the
runtime and then run:

```bash
python "${sparkcache_root}/deploy/glm53_flash/build_public_image.py" \
  --repository "${sparkcache_root}" \
  --base-image "${runtime_image}" \
  --output-image sparkring-glm53-sparkcache:local \
  --output glm53-sparkcache-build-receipt.json
```

A rebuilt image has **implemented** status. It does not acquire the
qualification of the published digest until its own image ID passes the live
checks below. Publication procedures, SBOM generation, licenses, and required
OCI labels are in
[`runtime/glm53-flash/PUBLISHING.md`](../runtime/glm53-flash/PUBLISHING.md)
and the SparkCache repository's `deploy/glm53_flash/PUBLISHING.md`.

## Configure the cluster

Clone SparkRing and detach at the reviewed 40-character commit that contains
this profile. A branch name is not an immutable deployment identity:

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git
sparkring_revision=REPLACE_WITH_REVIEWED_40_CHARACTER_COMMIT
git -C sparkring checkout --detach "${sparkring_revision}"
sparkring_root="$PWD/sparkring"
cp "${sparkring_root}/scripts/config/glm53-flash-tp4-site.example.yaml" site.yaml
cp "${sparkring_root}/scripts/config/glm53-flash-dflash2-bf16-tp4-dcp1-sparkcache.example.json" profile.json
```

Edit `site.yaml` with each rank's SSH target, management address, two RoCE
addresses, interfaces, devices, GID indices, and direct peers. Edit
`profile.json` and replace only these host paths unless changing the qualified
contract deliberately:

- `/REPLACE/TARGET_MODEL_HOST_PATH`
- `/REPLACE/DFLASH_MODEL_HOST_PATH`
- `/REPLACE/GLM53_CACHE_HOST_ROOT`

The cache root must be a dedicated rank-local directory. Do not place it on a
shared filesystem. The checked-in image digest and image ID must remain exact
for reproduction of the qualification record.

Validate placeholders and the physical fabric:

```bash
python "${sparkring_root}/scripts/preflight.py" \
  --site site.yaml --strict-placeholders --json preflight.json
```

Distribute the same immutable image to every rank and record parity:

```bash
python "${sparkring_root}/scripts/pull_glm53_image_cluster.py" \
  --site site.yaml --image "${sparkcache_image}" \
  --execute --confirmation PULL_GLM53_IMAGE \
  --output cluster-image.json
```

## Start and observe

Review the dry-run plan, then start all ranks:

```bash
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site site.yaml --profile profile.json start > start-plan.json
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site site.yaml --profile profile.json \
  --execute --confirmation START_GLM53_FLASH_DFLASH2_TP4 start
```

Tail rank 0:

```bash
ssh operator@rank0.example.net \
  'docker logs --follow --tail 120 glm53-flash-dflash2-bf16-sparkcache-tp4-r0 2>&1'
```

Replace the SSH target with rank 0 from `site.yaml`. The container name comes
from `profile.json`; use that literal name if it was changed. Wait for API
readiness, then verify the served model:

```bash
api_endpoint='http://rank0.example.net:8015'
until curl --fail --silent "${api_endpoint}/health" >/dev/null; do sleep 5; done
curl --fail --silent "${api_endpoint}/v1/models"
```

Cold startup with stock safetensors took about 8 minutes for model loading and
about 45–103 seconds for graph capture in the qualification runs. API health,
not a particular progress line, defines readiness.

## Verify persistent restore

Use SparkCache's deterministic request program:

```bash
qualification_script="${sparkcache_root}/deploy/glm53_flash/qualification_request.py"
served_model='glm-5.3-flash-nvfp4-dflash7-bf16-tp4'
python "${qualification_script}" --endpoint "${api_endpoint}" \
  --model "${served_model}" --kind persistent --output cold.json
```

Every worker must log `committed 8192 tokens` with one common digest. Replace
the containers without deleting the cache roots:

```bash
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site site.yaml --profile profile.json \
  --execute --confirmation START_GLM53_FLASH_DFLASH2_TP4 stop
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site site.yaml --profile profile.json \
  --execute --confirmation START_GLM53_FLASH_DFLASH2_TP4 start
```

After readiness, the first request may safely recompute while all worker
inventories reach the scheduler. Run it once as a prime, save metrics, then
repeat it:

```bash
python "${qualification_script}" --endpoint "${api_endpoint}" \
  --model "${served_model}" --kind persistent --output post-restart-prime.json
curl --fail --silent "${api_endpoint}/metrics" > metrics-before-restore.prom
python "${qualification_script}" --endpoint "${api_endpoint}" \
  --model "${served_model}" --kind persistent --output post-restart-restore.json
python "${qualification_script}" --endpoint "${api_endpoint}" \
  --model "${served_model}" --kind semantic --output post-restore-semantic.json
curl --fail --silent "${api_endpoint}/metrics" > metrics-after-restore.prom
```

Require every worker to log `restored 8192 tokens async`. Across the metrics
snapshots, require an 8,192-token increase in both
`external_prefix_cache_hits_total` and the `external_kv_transfer` source.
Require `draft_tokens = 7 × drafts`, zero preemptions, a passing semantic
receipt, the same image ID on every rank, no restarts or OOMs, and 24 RTS
`VLLM::Worker` queue pairs per rank.

The qualification result is recorded in
[`performance/records/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828.md`](../performance/records/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828.md).

## Provenance and support

[`runtime/glm53-flash/pins.json`](../runtime/glm53-flash/pins.json) records the
model hashes, quantization attribution, vLLM fork commits and merged pull
requests, B12X commit, original SparkRing NCCL patch, Git trees, image digests,
SBOM hashes, and license scope. SparkCache problems belong at
<https://github.com/FujitsuPolycom/sparkcache/issues>; runtime, transport, and
profile problems belong at <https://github.com/FujitsuPolycom/sparkring/issues>.

The runtime pins `local-inference-lab/vllm` branch
`dev/jovian-judgement@da4d7be6c97434f6942292ed8abbf4b32dc44355` and
`local-inference-lab/b12x@2fcf23a0ce269be27b2e03fece73d46e90e6aeea`.
The target repository does not record its base-checkpoint revision.

The optional `deep_ep` import can emit a duplicate-NCCL warning. The qualified
run proceeded after vLLM selected `/opt/sparkring/nccl/libnccl.so.2` through
`VLLM_NCCL_SO_PATH`. Treat an engine initialization failure, NCCL runtime
error, OOM, container restart, or semantic failure as fatal; do not classify
the documented import warning alone as a service failure.
