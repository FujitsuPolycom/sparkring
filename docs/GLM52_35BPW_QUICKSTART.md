# GLM-5.2 EXL3 3.5-bpw four-Spark quickstart

This quickstart deploys the 16-sequence GLM-5.2 EXL3 **candidate** on four
directly cabled NVIDIA DGX Sparks. The machine-readable settings are in
[`recipes/glm52-exl3-r7-3.5bpw.json`](../recipes/glm52-exl3-r7-3.5bpw.json).

The older 262,144-token, eight-sequence setup is **qualified** only on the one
appliance that was tested. The settings below are still being tested.

A clean-checkout rebuild has **implemented** status. One image built with these
steps started on four ranks and matched the pinned checkpoint and runtime, as
recorded in
[the rebuilt-image bring-up record](../performance/records/glm-3.5bpw/rebuilt-image-20260821.md).

Each rebuilt image must pass the
[full acceptance checklist](GLM52_35BPW_PROMOTION_CHECKLIST.md) before it can be
called qualified.

## Serving contract

| Setting | Value |
|---|---|
| Checkpoint | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f` |
| Config SHA-256 | `fabb73eb513ec64f3a365da396b38de8d55b3930edfb11baeecbf34ecafa6126` |
| Index SHA-256 | `9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd` |
| Parallelism | TP4, DCP4 `ag_rs` |
| Speculation | fixed MTP4 |
| Request limit | 1,048,576 tokens |
| Key-value cache | `nvfp4_ds_mla`, 9,250,000,000 bytes per rank |
| Maximum sequences | 16 |
| Key-value block size | 64 tokens |
| Tensor-parallel transport | SIRCL with patched NCCL fallback |

## 1. Prepare the four ranks

Complete [the prerequisites](PREREQUISITES.md). Copy the site template to the
ignored local configuration, replace every placeholder, and use one identical
model path on every rank.

```bash
cp scripts/config/exl3-r7-site.example.yaml scripts/config/site.yaml
$EDITOR scripts/config/site.yaml
python scripts/sparkring_site.py scripts/config/site.yaml
python scripts/preflight.py --site scripts/config/site.yaml --print-plan
```

The printed plan is offline. Run the command without `--print-plan` only after
reviewing it; that contacts the configured hosts but does not mutate them.

## 2. Download and verify the checkpoint

Download the pinned checkpoint to the model path configured in the site file.

```bash
python scripts/download_exl3_r7.py download \
  --model-path /var/tmp/sparkring-r7-model
python scripts/download_exl3_r7.py verify \
  --model-path /var/tmp/sparkring-r7-model
```

The pinned checkpoint contains 157 weight shards. The index total is
346,218,639,128 bytes.

## 3. Build and identify the runtime image

Pull the published ARM64 serving image as the builder's pinned parent:

```bash
docker pull ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028
docker image inspect \
  ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028 \
  --format '{{.Id}}'
```

Build on an ARM64 host from the tracked runtime inputs. Supply that immutable
parent reference, the local image ID reported by Docker, and the audited SPDX
license expression for the exact parent:

```bash
BASE_IMAGE=<parent-image-tag> \
BASE_IMAGE_ID=<parent-image-sha256-id> \
BASE_IMAGE_LICENSES=<parent-image-spdx-expression> \
  ./runtime/exl3-r7/build-image.sh
```

The exact-Q40 attestation generator requires `--image-id` and binds its output
to the derived image. Record the immutable Docker image ID and set both runtime image fields in
`scripts/config/site.yaml` before generating the launch profile.

```bash
docker image inspect <your-image-ref> --format '{{.Id}}'
```

```yaml
runtime:
  container_image: <your-image-ref>
  container_image_digest: <your-image-id>
```

## 4. Generate the complete pre-exact-Q40 profile

Build and test the native SIRCL library, then create a local candidate template
whose image and host paths match `scripts/config/site.yaml`.

```bash
cmake -S spark_transport -B build/sircl-tiered \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build/sircl-tiered \
  --target spark_transport_capi \
  --parallel
ctest --test-dir build/sircl-tiered --output-on-failure

mkdir -p .sparkring/exl3-r7
cp scripts/config/exl3-r7-candidate.example.json \
  .sparkring/exl3-r7/candidate.json
$EDITOR .sparkring/exl3-r7/candidate.json

python scripts/glm35_profile.py plan --execute \
  --site scripts/config/site.yaml \
  --template .sparkring/exl3-r7/candidate.json \
  --transport-library build/sircl-tiered/libspark_transport_capi.so \
  --backend spark_transport/integrations/vllm/spark_tp4_backend.py \
  --port-namespace spark_transport/integrations/vllm/spark_tp4_port_namespace.py
```

The command writes only `.sparkring/exl3-r7/`. The complete dynamic-NVFP4,
bounded full-CKV-gather, tiered-SIRCL serving inputs are:

```text
.sparkring/exl3-r7/pre-q40-profile.json
.sparkring/exl3-r7/pre-q40-site.yaml
.sparkring/exl3-r7/pre-q40-receipt.json
.sparkring/exl3-r7/pre-q40-bundle/
```

The output directory also contains the fixed-MTP4 foundation, each
intermediate profile and site, and byte-identical rollback inputs. The receipt
binds the complete profile, site, and five SIRCL artifacts by SHA-256.

## 5. Bind the exact-Q40 overlays

The exact-Q40 tools remain separate because they accept only pinned vLLM source
bytes and bind the serving profile to the built image ID. Prepare the pinned
vLLM tree, generate `exl3.py` with
`q40_exact_state_overlay.py`, and generate `model_runner.py` with
`q40_exact_state_attestation_overlay.py`. Then bind those outputs to the
complete pre-Q40 profile:

```bash
python scripts/glm35_q40/prepare_q40_exact_state_serving.py \
  --base-profile .sparkring/exl3-r7/pre-q40-profile.json \
  --expected-base-profile-sha256 <pre-q40-profile-sha256> \
  --exl3 .sparkring/exl3-r7/q40-overlay/exl3.py \
  --model-runner .sparkring/exl3-r7/q40-overlay/model_runner.py \
  --expected-model-runner-sha256 <model-runner-sha256> \
  --bundle .sparkring/exl3-r7/q40-bundle \
  --output-profile .sparkring/exl3-r7/exact-q40-profile.json \
  --output-manifest .sparkring/exl3-r7/exact-q40-receipt.json
```

Use `pre_q40_profile_sha256` from the compiler's printed receipt. The exact-Q40
generators reject source-byte or image-identity drift rather than rewriting an
unrecognized runtime.

Inspect the final launch plan before starting any remote process.

```bash
python scripts/preflight.py \
  --site .sparkring/exl3-r7/pre-q40-site.yaml \
  --print-plan
python scripts/sparkring_generic_launcher.py \
  --site .sparkring/exl3-r7/pre-q40-site.yaml \
  --profile .sparkring/exl3-r7/exact-q40-profile.json \
  plan
```

## 6. Start and verify

The complete profile has status **implemented**. Copy the two generated
bundles to their declared remote roots on all four ranks before launch.
Staging bundles and starting the service mutate the named hosts; starting can
replace a running serving stack and requires explicit authorization.

```bash
python scripts/sparkring_generic_launcher.py \
  --site .sparkring/exl3-r7/pre-q40-site.yaml \
  --profile .sparkring/exl3-r7/exact-q40-profile.json \
  --execute \
  --confirmation START-SIRCL-Q40-EXACT-STATE-CANARY-ALL-FOUR \
  start

python scripts/sparkring_generic_launcher.py \
  --site .sparkring/exl3-r7/pre-q40-site.yaml \
  --profile .sparkring/exl3-r7/exact-q40-profile.json \
  --execute status
```

Require HTTP 200 from `/health`, model name `glm-5.2-exl3-r7-3.5bpw`, and a
1,048,576-token maximum length in `/v1/models`.

## Benchmark snapshot for these settings

The 1,048,576-token, 16-sequence setup started successfully and was benchmarked
on four directly cabled DGX Sparks with SparkCache disabled.

| Context | Prefill tok/s | T=0 C1 | C2 | C4 | C8 | T=1 C1 | C2 | C4 | C8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 694 | 23.13 | 35.34 | 50.95 | 71.82 | 22.001 | 28.283 | 46.981 | 67.622 |
| 8K | 675 | 22.87 | 34.17 | 51.26 | 74.50 | 19.153 | 30.207 | 47.697 | 65.534 |
| 16K | 671 | 21.60 | 33.45 | 49.14 | pending | 20.151 | 32.384 | pending | pending |
| 32K | 661 | 22.40 | 34.62 | 48.98 | pending | 21.611 | 30.524 | pending | pending |
| 64K | 649 | — | — | — | — | pending | pending | pending | pending |
| 128K | 635 | — | — | — | — | pending | pending | pending | pending |

Decode values are aggregate generated tok/s. Temperature 0 and 1.0 are
separate datasets with `top_p` unset. The
[full GLM benchmark record](../performance/records/glm-3.5bpw/normalized-base-20260822.md)
contains the temperature-0.9 probe, temperature-1 Coding Peak N=5 summary,
machine-readable data, accounting gates, exclusions, and pending coordinates.

This launch used fresh rank-specific JIT and create-once receipt
namespaces. A same-namespace restart currently fails before model startup when
the exact-Q40 producer finds its existing receipt. Keep the receipt and use a
fresh namespace until the launcher can safely reuse a matching receipt. This is
a restart problem, not a model-performance failure.

## Older qualified result

The qualified appliance measured the profile with fixed MTP4, dynamic NVFP4 MLA
key-value cache, bounded full-CKV gather, and SIRCL TP collectives. Results and
limits are in [the serving record](GLM52_35BPW_FIXED_MTP4_PROFILE.md) and
[results](RESULTS.md). DCP and indexer collectives use the stock path.
