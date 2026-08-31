# Serve GLM-5.3 Flash with BF16 DFlash2 without an external KV cache

> Historical exact-artifact procedure. Use the
> [published JJ r7-compatible quickstart](GLM53_JJ_R7_GB10_TP4_QUICKSTART.md) for a public GB10
> deployment. The identities below remain valid only for their recorded
> evidence.

Use the [GLM-5.3 routing guide](GLM53_FLASH_QUICKSTARTS.md) to compare this
cache-disabled profile with the SparkCache and source-built GLM-5.3 paths.

Runtime performance and correctness come primarily from Local Inference Lab's
[Jovian Judgement vLLM branch](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement),
with Blackwell kernels from
[B12X](https://github.com/local-inference-lab/b12x). This recipe uses
[`GLM-5.3-Flash-NVFP4@520de24e`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/520de24eabf507659eaef7c70f14fd584527facc)
and the BF16
[`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410).
The draft is not the separate Local Inference Lab
[MXFP8 DFlash2 checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8).

Status: **qualified** for startup, semantic generation, and runtime health
using the immutable image, model revisions, and TP4/DCP1 settings in this
guide. The configured 524,288-token request limit and 32-sequence limit were
not exercised at their limits. The image is a FujitsuPolycom community
derivative.

This profile is the controlled comparison for the SparkCache-enabled service.
It uses the same image, target checkpoint, BF16 DFlash2 checkpoint, 12 GiB FP8
GPU KV memory per rank, scheduler, CUDA graphs, asynchronous scheduling,
chunked prefill, native prefix caching, Triton KDA prefill, and source-built
NCCL. It omits only `--kv-transfer-config`.

## Prepare the four-rank service

Complete the prerequisites, model acquisition, image pull or source-build,
qualification-client checkout, cluster inventory, and provenance review in
[`GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md`](GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md).
The qualified service image is:

```bash
image='ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943'
docker pull "${image}"
```

The source-built parent runtime is available separately at
`ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd`.
The service uses the SparkCache image because it contains the complete parent
runtime and lets both profiles share one byte-identical artifact.

Copy and edit the sanitized inputs:

```bash
sparkring_root="$PWD/sparkring"
sparkcache_root="$PWD/sparkcache"
cp "${sparkring_root}/scripts/config/glm53-flash-tp4-site.example.yaml" site.yaml
cp "${sparkring_root}/scripts/config/glm53-flash-dflash2-bf16-tp4-dcp1.example.json" profile.json
```

Set every SSH, management, RoCE, interface, device, GID, peer, and host-path
placeholder. The required host paths are the target checkpoint, BF16 DFlash2
checkpoint, and a rank-local compilation-cache root. No SparkCache storage is
read or written by this profile.

Validate the fabric and distribute one digest:

```bash
python "${sparkring_root}/scripts/preflight.py" \
  --site site.yaml --strict-placeholders --json preflight.json
python "${sparkring_root}/scripts/pull_glm53_image_cluster.py" \
  --site site.yaml --image "${image}" \
  --execute --confirmation PULL_GLM53_IMAGE \
  --output cluster-image.json
```

## Start and verify

```bash
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site site.yaml --profile profile.json start > start-plan.json
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site site.yaml --profile profile.json \
  --execute --confirmation START_GLM53_FLASH_DFLASH2_TP4 start
```

Tail the API rank using the container name in `profile.json`:

```bash
ssh operator@rank0.example.net \
  'docker logs --follow --tail 120 glm53-flash-dflash2-bf16-tp4-r0 2>&1'
```

Wait for health and run the deterministic semantic canary:

```bash
api_endpoint='http://rank0.example.net:8015'
served_model='glm-5.3-flash-nvfp4-dflash7-bf16-tp4'
until curl --fail --silent "${api_endpoint}/health" >/dev/null; do sleep 5; done
python "${sparkcache_root}/deploy/glm53_flash/qualification_request.py" \
  --endpoint "${api_endpoint}" --model "${served_model}" \
  --kind semantic --output no-external-cache-semantic.json
curl --fail --silent "${api_endpoint}/metrics" > no-external-cache-metrics.prom
```

Require `semantic_match: true`, `finish_reason: stop`,
`draft_tokens = 7 × drafts`, zero external-cache queries, zero preemptions,
the same image ID on all ranks, no SparkCache connector log lines, no restarts
or OOMs, and 24 RTS `VLLM::Worker` queue pairs per rank.

The qualified observation completed the semantic request in 2.703 seconds,
produced 231 draft tokens from 33 drafts, and satisfied all health conditions.
It is a functional startup result, not a throughput or soak result.

## Provenance and limitations

The target is
`local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`.
Its published ModelOpt configuration uses NVFP4 target experts and an MXFP8
embedded MTP expert; the unquantized base-checkpoint revision is not recorded.
The draft is
`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`,
BF16, under CC BY-NC-ND 4.0. vLLM is
`local-inference-lab/vllm` at
`dev/jovian-judgement@da4d7be6c97434f6942292ed8abbf4b32dc44355`;
B12X is `2fcf23a0ce269be27b2e03fece73d46e90e6aeea`.

Complete source, pull-request, patch, quantization, image, SBOM, and license
attribution is in
[`runtime/glm53-flash/pins.json`](../runtime/glm53-flash/pins.json). The
functional record is
[`performance/records/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828.md`](../performance/records/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828.md).

FlashKDA prefill, InstantTensor checkpoint loading, MTP drafting, other
topologies, and other model revisions are unsupported by this image record.
Report runtime or transport problems at
<https://github.com/FujitsuPolycom/sparkring/issues> and SparkCache-specific
problems at <https://github.com/FujitsuPolycom/sparkcache/issues>.
