# Reproduce the GLM-5.2 EXL3 3.5-bpw deployment profile

This procedure derives the qualified GLM serving configuration from tracked
inputs. It is for four directly cabled DGX Sparks. The resulting image has
status **implemented** until it completes
[the promotion checklist](GLM52_35BPW_PROMOTION_CHECKLIST.md).

## Inputs

- Recipe: `recipes/glm52-exl3-r7-3.5bpw.json`
- Checkpoint: `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f`
- Site template: `scripts/config/exl3-r7-site.example.yaml`
- Candidate template: `scripts/config/exl3-r7-candidate.example.json`
- Runtime builder: `runtime/exl3-r7/build-image.sh`

Complete [the GLM quickstart](GLM52_35BPW_QUICKSTART.md) through image build
before deriving the deployment foundation.

## Derive the complete pre-exact-Q40 profile

The consolidated compiler performs the fixed-MTP4, dynamic-NVFP4,
bounded full-CKV-gather, and tiered-SIRCL transformations in one hash-bound
local operation. The candidate template and site must bind the same image
identity, model path, and JIT-cache path. The SIRCL artifact paths must name
the exact locally built library and tracked adapter modules.

```bash
python scripts/glm35_profile.py plan --execute \
  --site scripts/config/site.yaml \
  --template .sparkring/exl3-r7/candidate.json \
  --output-dir .sparkring/exl3-r7 \
  --transport-library build/sircl-tiered/libspark_transport_capi.so \
  --backend spark_transport/integrations/vllm/spark_tp4_backend.py \
  --port-namespace spark_transport/integrations/vllm/spark_tp4_port_namespace.py
```

The module validates all inputs before creating the final bundle. The
deployment input immediately before exact-Q40 binding is:

```text
.sparkring/exl3-r7/pre-q40-profile.json
.sparkring/exl3-r7/pre-q40-site.yaml
.sparkring/exl3-r7/pre-q40-receipt.json
.sparkring/exl3-r7/pre-q40-bundle/
```

The same output directory contains the fixed-MTP4 foundation, dynamic-NVFP4
and CKV-gather intermediates, and byte-identical rollback files. The printed
receipt records SHA-256 values for the complete profile, site, receipt, and
each mounted SIRCL artifact.

## Bind the exact-Q40 runtime bytes and image identity

Generate the exact-Q40 `exl3.py` and runtime-attestation `model_runner.py` from
the pinned vLLM source tree. These generators remain separate because they
reject unexpected source bytes and bind the attestation to the image ID.
Use the compiler receipt's `pre_q40_profile_sha256` as the baseline digest:

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

## Plan and launch

Inspect the complete site and exact-Q40 profile before authorizing a start.

```bash
python scripts/preflight.py \
  --site .sparkring/exl3-r7/pre-q40-site.yaml \
  --print-plan

python scripts/sparkring_generic_launcher.py \
  --site .sparkring/exl3-r7/pre-q40-site.yaml \
  --profile .sparkring/exl3-r7/exact-q40-profile.json \
  plan
```

Copy both generated bundles to their declared remote roots before launch.
Staging files and starting are host-mutating, and starting can stop a running
service. Perform those actions only with authorization for the four named
hosts. Verify `/health`, the served model name, and the 1,048,576-token maximum.
The rebuilt profile has status **implemented** until it completes the promotion
checklist.
