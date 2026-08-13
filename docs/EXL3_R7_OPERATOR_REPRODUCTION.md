# Reproduce the operator-accepted EXL3 R7 profile composition

This document composes the public source layers used by the operator-accepted
EXL3 R7 3.5-bpw profile on four directly cabled NVIDIA DGX Sparks. The profile
uses fixed MTP4, DCP4, dynamic per-token NVFP4 MLA KV with FP8 RoPE, bounded
full-CKV gather, tiered/deferred SIRCL tensor-parallel transport, and a
target-only exact-40-row routed-MoE state.

## Status and scope

| Attribute | Value |
|---|---|
| Lane | public-functional |
| Operator profile | accepted |
| Public composition source | published and offline-validated |
| Clean-checkout image built from these instructions | requires live validation |
| Public-functional default | no |
| Hardware scope | four DGX Sparks / GB10, direct 200-Gb/s cycle |

The public generator chain is fail-closed and preserves rollback inputs. It
does not make a locally rebuilt image equivalent to the operator's historical
image ID. The local image, generated model-runner overlay, SIRCL library, and
launch profile receive new hashes and must pass the complete live gate.

Three prerequisites still lack complete zero-context public build chains: the
weight-utils local-I/O overlay, two quack annotation overlays, and the ARM64
`tvm-ffi` bundle. They are listed in
[`EXL3_R7_QUICKSTART.md`](EXL3_R7_QUICKSTART.md#3-build-or-obtain-the-arm64-image).
Do not remove their profile mounts or bypass their startup hashes. A builder
must supply independently audited copies until those source/build chains are
published.

## Resulting serving contract

| Setting | Value |
|---|---|
| Checkpoint | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f` |
| TP / DCP | TP4 / DCP4 |
| Speculation | fixed MTP4; Q1 through Q40 graphs |
| Model limit | 262,144 tokens |
| Batched-token and EXL3 prefill capacity | 4,096 |
| KV | `nvfp4_ds_mla`, dynamic per-token scale, FP8 RoPE, 368-byte record |
| KV allocation | 9,250,000,000 bytes/rank |
| Reported KV capacity | 1,156,864 tokens |
| CKV gather | transient full-CKV gather, maximum 262,144 logical tokens |
| SIRCL graph protocol | `two_slot_deferred_ack` |
| SIRCL kernel selector | `tiered_64k` |
| Exact-Q40 policy | target mixed-EXL3 only; capacity 40, route block 8 |

Dual-port graph transport and the prefill capacity pool remain disabled. They
are research-only options, not part of the accepted profile.

## 1. Prepare immutable source trees and build the image

Run this on an ARM64 build host. The prepared tree is retained because the
exact-Q40 generators verify the installed vLLM source preimages.

```bash
python runtime/exl3-r7/prepare_context.py .sparkring/r7-prepared-sources
python runtime/exl3-r7/prepare_context.py \
  --verify .sparkring/r7-prepared-sources

PREPARED_SOURCES="$PWD/.sparkring/r7-prepared-sources" \
BASE_IMAGE='<audited-parent-reference>' \
BASE_IMAGE_ID='sha256:<audited-parent-image-id>' \
BASE_IMAGE_LICENSES='<audited-SPDX-expression>' \
IMAGE='sparkring-r7:arm64-sm121' \
  runtime/exl3-r7/build-image.sh

image_id="$(docker image inspect sparkring-r7:arm64-sm121 \
  --format '{{.Id}}')"
printf '%s\n' "$image_id"
```

The builder compiles `libspark_transport_capi.so` for SM121 and installs the
manifest-bounded vLLM adapter from the same SparkRing revision. See
[`runtime/exl3-r7/README.md`](../runtime/exl3-r7/README.md) for parent-image,
source-receipt, and license requirements.

## 2. Generate the fixed-MTP4 KV9.25 foundation

Copy and fill the ignored site file first:

```bash
cp scripts/config/exl3-r7-site.example.yaml scripts/config/site.yaml
${EDITOR:-vi} scripts/config/site.yaml
python scripts/sparkring_site.py scripts/config/site.yaml
python scripts/preflight.py --site scripts/config/site.yaml --print-plan
```

Generate the conservative profile chain without contacting a Spark:

```bash
python scripts/exl3_r7_standup.py plan
python scripts/exl3_r7_standup.py plan --execute
```

The source for the following stages is:

```text
.sparkring/exl3-r7/mtp4-kv925-profile.json
.sparkring/exl3-r7/mtp4-kv925-site.yaml
```

## 3. Derive 262K dynamic-NVFP4

```bash
out=.sparkring/exl3-r7
mtp4_profile_sha="$(sha256sum "$out/mtp4-kv925-profile.json" | awk '{print $1}')"
mtp4_site_sha="$(sha256sum "$out/mtp4-kv925-site.yaml" | awk '{print $1}')"

python scripts/prepare_exl3_r7_mtp4_nvfp4.py \
  --source-profile "$out/mtp4-kv925-profile.json" \
  --source-site "$out/mtp4-kv925-site.yaml" \
  --expected-profile-sha256 "$mtp4_profile_sha" \
  --expected-site-sha256 "$mtp4_site_sha" \
  --candidate-profile "$out/mtp4-nvfp4-profile.json" \
  --candidate-site "$out/mtp4-nvfp4-site.yaml" \
  --rollback-profile "$out/mtp4-nvfp4-rollback.json" \
  --rollback-site "$out/mtp4-nvfp4-rollback-site.yaml"
```

This stage changes only the maximum model length, KV dtype, dynamic scale,
FP8-RoPE flag, batched-token ceiling, EXL3 prefill capacity, profile identity,
and KV-contract label.

## 4. Enable bounded full-CKV gather

```bash
nvfp4_profile_sha="$(sha256sum "$out/mtp4-nvfp4-profile.json" | awk '{print $1}')"
nvfp4_site_sha="$(sha256sum "$out/mtp4-nvfp4-site.yaml" | awk '{print $1}')"

python scripts/prepare_exl3_r7_mtp4_ckv_gather.py \
  --source-profile "$out/mtp4-nvfp4-profile.json" \
  --source-site "$out/mtp4-nvfp4-site.yaml" \
  --expected-profile-sha256 "$nvfp4_profile_sha" \
  --expected-site-sha256 "$nvfp4_site_sha" \
  --candidate-profile "$out/mtp4-nvfp4-ckv-profile.json" \
  --candidate-site "$out/mtp4-nvfp4-ckv-site.yaml" \
  --rollback-profile "$out/mtp4-nvfp4-ckv-rollback.json" \
  --rollback-site "$out/mtp4-nvfp4-ckv-rollback-site.yaml"
```

The receipt reports the two-lane transient workspace prediction of 434,534,400
bytes/rank. SparkRing's custom CKV all-gather remains disabled; DCP and the
sparse indexer remain on their stock paths.

## 5. Compile and bind tiered/deferred SIRCL

```bash
cmake -S spark_transport -B .sparkring/sircl-sm121-build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DBUILD_TESTING=OFF
cmake --build .sparkring/sircl-sm121-build \
  --target spark_transport_capi --parallel 8

ckv_profile_sha="$(sha256sum "$out/mtp4-nvfp4-ckv-profile.json" | awk '{print $1}')"
python scripts/prepare_exl3_r7_sircl_tiered.py \
  --base-profile "$out/mtp4-nvfp4-ckv-profile.json" \
  --expected-base-profile-sha256 "$ckv_profile_sha" \
  --transport-library .sparkring/sircl-sm121-build/libspark_transport_capi.so \
  --backend spark_transport/integrations/vllm/spark_tp4_backend.py \
  --port-namespace spark_transport/integrations/vllm/spark_tp4_port_namespace.py \
  --capacity-pool spark_transport/integrations/vllm/spark_tp4_prefill_capacity_pool.py \
  --bundle "$out/sircl-tiered-bundle" \
  --output-profile "$out/mtp4-nvfp4-ckv-sircl-profile.json" \
  --output-manifest "$out/mtp4-nvfp4-ckv-sircl-manifest.json"
```

The generated profile mounts and startup-attests all four SIRCL artifacts. It
enables only the qualified deferred-ack protocol and tiered 64-KiB selector.
The native ABI, graph invariants, and standalone qualification are documented
in [`TIERED_DEFERRED_GRAPH.md`](../spark_transport/TIERED_DEFERRED_GRAPH.md).

## 6. Generate the target-only exact-Q40 overlays

Generate the EXL3 insertion-only state from the prepared vLLM result tree:

```bash
q40=.sparkring/exl3-r7/q40-exact
mkdir -p "$q40"

python spark_transport/experiments/moe_round_floor/q40_exact_state_overlay.py \
  --source .sparkring/r7-prepared-sources/vllm/vllm/model_executor/layers/quantization/exl3.py \
  --output "$q40/exl3.py"

python spark_transport/experiments/moe_round_floor/q40_exact_state_attestation_overlay.py \
  --source .sparkring/r7-prepared-sources/vllm/vllm/v1/worker/gpu/model_runner.py \
  --output "$q40/model_runner.py" \
  --image-id "$image_id" \
  --checkpoint-revision 9ab9579774cc432df91567a36f6e9e863e0d4c9f
```

The EXL3 input/output hashes are fixed at `8e0051fa...` and `8fad5330...`.
The model-runner output hash intentionally depends on the local immutable image
ID. The operator image reproduces `0e2e0150...`; another image must produce and
attest its own hash.

Compose the final profile:

```bash
sircl_profile="$out/mtp4-nvfp4-ckv-sircl-profile.json"
sircl_profile_sha="$(sha256sum "$sircl_profile" | awk '{print $1}')"
runner_sha="$(sha256sum "$q40/model_runner.py" | awk '{print $1}')"

python spark_transport/experiments/moe_round_floor/prepare_q40_exact_state_serving.py \
  --base-profile "$sircl_profile" \
  --expected-base-profile-sha256 "$sircl_profile_sha" \
  --exl3 "$q40/exl3.py" \
  --model-runner "$q40/model_runner.py" \
  --expected-model-runner-sha256 "$runner_sha" \
  --bundle "$out/q40-exact-bundle" \
  --output-profile "$out/operator-profile.json" \
  --output-manifest "$out/operator-profile-manifest.json"
```

The model-runner hook executes after the eager 4,096-row profile and before
CUDA graph capture. Every DCP rank must prove:

- exactly 75 mixed target layers and one uniform draft layer;
- Q1 through Q32 retain the decode state and route block 8;
- exact Q40 uses capacity 40, route block 8, and prefill tiers/tile;
- all other target prefill shapes retain capacity 4,096 and block 32 or 64;
- draft geometry is unchanged;
- exact BF16 Q40 output equals the general-prefill comparator on every target
  layer and is finite and nonzero;
- runtime caches and storage identities remain stable.

## 7. Stage the two generated bundles

This step **MUTATES HOST**. Replace the four placeholders with the SSH targets
from the reviewed ignored site file. Do not run it without authorization for
those hosts.

```bash
for host in <rank0> <rank1> <rank2> <rank3>; do
  ssh "$host" 'install -d -m 0755 \
    /var/tmp/sparkring-sircl-tiered-v1 \
    /var/tmp/sparkring-q40-exact-state-v1'
  scp "$out"/sircl-tiered-bundle/* \
    "$host":/var/tmp/sparkring-sircl-tiered-v1/
  scp "$out"/q40-exact-bundle/* \
    "$host":/var/tmp/sparkring-q40-exact-state-v1/
done
```

Compare every remote SHA-256 with the two local manifests before launch.

## 8. Plan, preflight, and launch

Use the derived 262K site, not the 65K foundation site:

```bash
site="$out/mtp4-nvfp4-ckv-site.yaml"
profile="$out/operator-profile.json"

python scripts/sparkring_generic_launcher.py \
  --site "$site" --profile "$profile" validate
python scripts/sparkring_generic_launcher.py \
  --site "$site" --profile "$profile" plan
```

Review every path, hash, rank, interface, port, image ID, and command. Starting
the profile **STOPS SERVING** and requires explicit authorization:

```bash
python scripts/sparkring_generic_launcher.py \
  --site "$site" --profile "$profile" \
  --execute \
  --confirmation START-SIRCL-Q40-EXACT-STATE-CANARY-ALL-FOUR \
  start
```

## 9. Required live acceptance gates

A locally rebuilt profile remains a candidate until all gates pass:

1. all four startup artifact hashes and image IDs match;
2. all four exact-Q40 pre-graph receipts pass and are archived;
3. Q1-through-Q40 graph capture completes on every rank;
4. `/health` returns HTTP 200 and KV capacity reports 1,156,864 tokens;
5. fixed-seed 16K and 32K response, token, and finite-logprob equality passes;
6. identical sealed C8 16K payloads pass a baseline-candidate-baseline bracket;
7. matched 8K, 16K, 32K, 64K, and 128K cold-prefill regression checks pass;
8. post-test API, queue, graph sequence, fatal, overflow, and KV-idle checks pass.

The operator evidence measured a 19.34% warm C8 aggregate decode gain. It is
evidence for the exact operator profile and four-Spark appliance, not a
prediction for a clean rebuild. See
[`EXL3_R7_FIXED_MTP4_PROFILE.md`](EXL3_R7_FIXED_MTP4_PROFILE.md).

## Rollback

- Remove only the exact-Q40 layer by starting
  `mtp4-nvfp4-ckv-sircl-profile.json`.
- Remove the SIRCL selector by starting `mtp4-nvfp4-ckv-profile.json`.
- The NVFP4 and CKV generators each preserve byte-identical rollback inputs.
- The fixed-MTP4 foundation preserves the fixed-MTP3 KV9.25 rollback.

Never edit a candidate profile to approximate rollback. Start the hash-bound
preserved input profile and repeat health/capacity checks.
