# EXL3 3.5-bpw operator-profile quickstart

This is the stand-up path for the **public-functional-lane, operator-accepted**
EXL3 3.5-bpw fixed-MTP4, DCP4, 9.25 GB KV/rank profile. Its durable recipe
identifier is `R7`. Acceptance applies to one four-Spark appliance and does
not transfer to a rebuilt image. It is not the repository default or a
reference-lane result. The advertised public default is
[EXL3 3.25-bpw plus LMCache CS512](EXL3_QUICKSTART.md). NF3 is an
[accepted deterministic alternative](NF3_QUICKSTART.md).

This quickstart builds the source-complete ARM64 runtime and links to the exact
commands that derive the accepted 262K dynamic-NVFP4, full-CKV-gather,
tiered-SIRCL, target-only exact-Q40 composition. A clean-checkout rebuild is an
offline-validated candidate until it passes the
[promotion checklist](EXL3_R7_PROMOTION_CHECKLIST.md) against its image ID.

SparkRing native transport and the exact-Q40 optimization are separate
layers. SparkRing supplies the native TP all-reduce and vocabulary paths with
NCCL fallback. The measured exact-Q40 decode gain comes from a target-only
EXL3 routed-MoE state that selects capacity 40 and route block 8 only for
exactly 40 rows; compiling SparkRing alone does not install that policy.

## Maturity

| Attribute | Value |
|---|---|
| Lane | public-functional |
| Operator deployment maturity | accepted |
| Clean-checkout builder maturity | offline-validated |
| Clean-checkout rebuilt image maturity | candidate until the live promotion gate passes |
| Default | no |
| Hardware | four directly cabled DGX Sparks / GB10 GPUs |
| Evidence | [EXL3_R7_FIXED_MTP4_PROFILE.md](EXL3_R7_FIXED_MTP4_PROFILE.md) |

## Accepted operator profile contract

| Setting | Candidate value |
|---|---|
| Model | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f` |
| Config SHA-256 | `fabb73eb513ec64f3a365da396b38de8d55b3930edfb11baeecbf34ecafa6126` |
| Index SHA-256 | `9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd` |
| Parallelism | TP4 plus DCP4 `ag_rs`, interleave size one |
| Speculation | fixed MTP4, greedy draft sampling, adaptive depth disabled |
| Maximum sequences | 8 |
| Query-row contract | Q1 through Q40; `8 * (4 + 1) = 40` verification rows |
| KV representation | `nvfp4_ds_mla`, dynamic per-token scale, FP8 RoPE, B12X block size 64 |
| KV allocation | 9,250,000,000 bytes/rank; 37,000,000,000 bytes aggregate |
| Reported KV capacity | 1,156,864 tokens |
| Model limit | 262,144 tokens |
| Graphs | `FULL_AND_PIECEWISE`, Q1 through Q40 |
| TP transport | hybrid SparkRing native plus patched NCCL 2.30.7 NET/IB fallback |
| DCP and indexer transport | stock `ag_rs` DCP and stock indexer collectives |
| Online quantization | EXL3 K6, target-only scope |
| Cache | native prefix caching enabled; LMCache and SparkCache disabled |

## Rollback profile

The exact rollback is the fixed-MTP3, 9.25 GB KV profile. It differs from
the candidate only in:

```text
profile and mode:              fixed-mtp4 -> fixed-mtp3
VLLM_SPARK_MTP_TOKENS:         4          -> 3
num_speculative_tokens:        4          -> 3
VLLM_SPARK_MAX_QUERY_ROWS:     40         -> 32
CUDA graph capture sizes:      Q1-Q40     -> Q1-Q32
maximum graph capture size:    40         -> 32
site serving.mtp_tokens:       4          -> 3
```

The rollback profile and site are byte-identical to the MTP3 KV9.25 inputs.
The MTP3 rollback is documented in
[EXL3_R7_FIXED_MTP3_PROFILE.md](EXL3_R7_FIXED_MTP3_PROFILE.md).

## 1. Complete the prerequisites

Read [PREREQUISITES.md](PREREQUISITES.md) completely. You need four ARM64
DGX Sparks with the direct 200-Gb/s cycle cabled and qualified, management
SSH from rank 0 to every rank, Docker with GPU access, enough storage for
the 346-GB model plus build/image headroom, and a filled ignored site
configuration.

```bash
cp scripts/config/exl3-r7-site.example.yaml scripts/config/site.yaml
$EDITOR scripts/config/site.yaml
```

Do not commit `scripts/config/site.yaml`. It contains local identities
and paths. Review every resolved rank, NIC, GID, direct-ring peer, model
path, and storage path before proceeding. The image fields remain unresolved
until step 3, so validate the complete file after updating them there.

## 2. Download and verify the immutable checkpoint

The model is `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` at revision
`9ab9579774cc432df91567a36f6e9e863e0d4c9f`. The downloader verifies
every runtime file against metadata at that pinned revision.

```bash
python scripts/download_exl3_r7.py download \
  --model-path /var/tmp/sparkring-r7-model
```

This downloads 157 weight shards plus 10 pinned metadata files and
verifies every SHA-256. The index total size is 346,218,639,128 bytes.
The downloader rejects a stale payload total, missing LFS metadata, hash
mismatch, or unmanifested files. It quarantines corrupted local files
instead of silently overwriting them.

Verify an existing download without re-downloading:

```bash
python scripts/download_exl3_r7.py verify \
  --model-path /var/tmp/sparkring-r7-model
```

## 3. Obtain the ARM64 image

The runtime filesystem is published. `ghcr.io/fujitsupolycom/gb10-vllm-serving`
holds one layer, `sha256:233970de794aec61170d16ee266015e0760e674974f4843294bc6e24d6b03c98`,
which is the same layer the image built from `runtime/exl3-r7/` carries. Its
entrypoint is `/usr/local/bin/sparkring-r7-entrypoint` and it is labelled
`org.sparkring.runtime_id=glm52-gb10-faststart-19523482c298`. Pulling it is
therefore an alternative to building, and it needs no registry credentials.

### Option A: Pull the published image

```bash
docker pull ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028
```

`runtime/faststart-lock.json` owns that pin; the command repeats it rather than
establishing a second one. The digest it names carries a `quack-kernels`
compatibility correction over its parent
`sha256:35b29616dc05677b98f647282e81a99fbca1969791ccbfca711c11a44285385e`, which
a checkpoint other than GLM requires and which is inert for this profile.
`docs/DEEPSEEK_V4_FLASH_QUICKSTART.md` states what the correction is.

Container labels differ between a pulled image and one built locally, so the
two carry the same filesystem under different configuration digests, and a
configuration digest is what Docker reports as an image ID. Record whichever
identity the image you run reports, by the command under Option B, and use it
everywhere the stand-up asks for one. Every section of this page holds for a
pulled image.

The exact-Q40 layer binds to that identity rather than to a fixed one.
`spark_transport/experiments/moe_round_floor/q40_exact_state_attestation_overlay.py`
takes a required `--image-id` and embeds it in the model-runner source it
emits, so its output hash depends on the image it was generated for, and
`prepare_q40_exact_state_serving.py` sets `SPARK_Q40_EXACT_STATE_IMAGE_ID` from
the same site identity. Generating the overlay against the image you run is
what makes the two agree; section 7 covers that step.

### Option B: Use an existing local image by immutable ID

If you have built the image or received it from a trusted builder, replace the
placeholder in the site template:

```bash
# Derive your image ID:
docker image inspect <your-image-ref> --format '{{.Id}}'
```

Update `scripts/config/site.yaml`:
```yaml
runtime:
  container_image: <your-image-ref>
  container_image_digest: <your-image-id>
```

### Option C: Local ARM64 build

Use the receipt-gated builder in [`runtime/exl3-r7/`](../runtime/exl3-r7/README.md),
then reference the resulting image ID in your site configuration. The builder
is offline-validated source; a clean-checkout image built from it still needs
the runtime overlay and live acceptance gates described below before it can be
called live-validated.

After either option, create an ignored candidate template with the same image
identity and local paths. Do not edit the tracked example in place:

```bash
mkdir -p .sparkring/exl3-r7
cp scripts/config/exl3-r7-candidate.example.json \
  .sparkring/exl3-r7/candidate.json
$EDITOR .sparkring/exl3-r7/candidate.json
```

Replace the image tag and image ID, then make `model_host_path` and
`jit_cache_host_path` match `runtime.model_path` and `paths.jit_cache_dir` in
the complete site file. The stand-up command rejects placeholders and
cross-file drift. Validate the resolved site and inspect the preflight plan:

```bash
python scripts/sparkring_site.py scripts/config/site.yaml
python scripts/preflight.py --site scripts/config/site.yaml --print-plan
```

The built image contains and startup-attests these runtime files:

| File | SHA-256 | Public source status |
|---|---|---|
| entrypoint | `bbc72446e9a7d811c903e76e37e7d9dfce3d21108b2ea7c3db278bb71e84f95e` | `runtime/exl3-r7/entrypoint.sh` |
| weight_utils local-I/O correction | `da5e6c3429293870d0de611183818fa57c0e9e0ad896784bc739c8a812343102` | hash-bound edit over the pinned vLLM result tree |
| EXL3 SM121 scratch overlay | `8e0051faf9b8bac9eefd6f38a5f0133a30bca4c0b5ab41962537e2f13cf968f4` | pinned vLLM result tree |
| cudagraph shared-stream overlay | `ef03d64297ed2d1a5161847b48a435bf8ae5feda7a5b81b668d00ae9a1d65a2a` | pinned vLLM result tree |
| QuACK layout correction | `3199dc3f55f346183e3d284f6da98f4394eaf14f28b7616d147e6e49ec896194` | public QuACK 0.5.0 wheel plus hash-bound compatibility edit |
| QuACK copy correction | `2ce88b0d7ee9afe025e52c02fcb32e772a429f1ee626b59546ab8b61d7a37929` | public QuACK 0.5.0 wheel plus hash-bound compatibility edit |
| stock-DCP audit overlay | `077a234e4edff8b8dd44784953aef713884b4dd7a3f7c46589b14c6bb8b40745` | `spark_transport/integrations/vllm/spark_dcp_collective_audit.py` |
| shared target/draft capture implementation | `b087e93463e9a2d9bede71d3a6e4d696c8f2657449e8dc1119b38613d5750e4e` | generated and baked by `runtime/exl3-r7/build_parallel_state_shared_capture_overlay.py` |
| ARM64 tvm-ffi 0.1.10 wheel | `3829216a8500c2f61062e48c627f6db6c3fa49416b3ffa85bc04243ae5d759f7` | pinned public wheel installed into the image |

The complete operator-profile generators, SIRCL source, exact-Q40 overlays,
wheel inputs, and compatibility edits are public. Do not bypass their hashes.

## 4. Generate the offline profile chain

The stand-up entrypoint derives the complete profile chain from tracked
inputs. It is **dry-run by default** — no files are written and no
hosts are contacted.

```bash
python scripts/exl3_r7_standup.py plan \
  --site scripts/config/site.yaml \
  --template .sparkring/exl3-r7/candidate.json
```

Review the planned steps. All steps in this mode are OFFLINE. To execute
the offline chain (writes files under `.sparkring/exl3-r7/`):

```bash
python scripts/exl3_r7_standup.py plan --execute \
  --site scripts/config/site.yaml \
  --template .sparkring/exl3-r7/candidate.json
```

This produces:

```text
.sparkring/exl3-r7/
  stock-dcp4-profile.json      # stock-DCP4 baseline (MTP-off, Q24, 9 GB)
  mtp2-profile.json            # fixed-MTP2 derivative
  mtp3-profile.json            # fixed-MTP3 derivative
  mtp3-kv925-profile.json     # KV9.25 profile (byte-identical to MTP3)
  mtp3-kv925-site.yaml         # KV9.25 site (9.25 GB KV/rank)
  mtp4-kv925-profile.json     # fixed-MTP4 candidate
  mtp4-kv925-site.yaml         # candidate site (mtp_tokens: 4)
  mtp4-kv925-rollback.json     # byte-identical to MTP3 KV9.25
  mtp4-kv925-rollback-site.yaml # byte-identical to MTP3 KV9.25 site
```

The receipt JSON includes SHA-256 hashes for every profile and site, plus
the rollback identity assertion.

## 5. Validate the generated profiles

```bash
python -m pytest scripts/test_generate_exl3_r7_candidate.py \
  scripts/test_generate_exl3_r7_stock_dcp4.py \
  scripts/test_prepare_exl3_r7_mtp2.py \
  scripts/test_prepare_exl3_r7_mtp3.py \
  scripts/test_prepare_exl3_r7_mtp3_kv925.py \
  scripts/test_prepare_exl3_r7_mtp4.py \
  scripts/test_download_exl3_r7.py \
  scripts/test_exl3_r7_standup.py -q
```

## 6. Inspect the dry-run launch plan (OFFLINE)

After filling `scripts/config/site.yaml`:

```bash
python scripts/preflight.py --site scripts/config/site.yaml --print-plan
```

This plan-only command is **OFFLINE** and prints the remote checks without
contacting the configured hosts. Review it before running the full
**READ-ONLY REMOTE** preflight:

```bash
python scripts/preflight.py --site scripts/config/site.yaml
```

## 7. Complete the operator-profile derivation

Run sections 3 through 6 of the
[operator reproduction guide](EXL3_R7_OPERATOR_REPRODUCTION.md) to create:

```text
.sparkring/exl3-r7/mtp4-nvfp4-ckv-site.yaml
.sparkring/exl3-r7/operator-profile.json
```

These files contain the dynamic-NVFP4, CKV-gather, tiered-SIRCL, and exact-Q40
layers. The 65K `mtp4-kv925-profile.json` is only their reproducible foundation.
Stage the two generated bundles using section 7 of the reproduction guide;
that step mutates the four named hosts and requires explicit authorization.
Then inspect and run the final site's preflight:

```bash
python scripts/preflight.py \
  --site .sparkring/exl3-r7/mtp4-nvfp4-ckv-site.yaml \
  --print-plan
python scripts/preflight.py \
  --site .sparkring/exl3-r7/mtp4-nvfp4-ckv-site.yaml
```

## 8. Start the reviewed profile (MUTATES HOST + STOPS SERVING)

Starting the candidate requires explicit authorization for the four named
hosts, every baked artifact at its exact container path, both generated bundles
at their attested remote paths, and a preserved rollback path. Use the generic
launcher:

```bash
python scripts/sparkring_generic_launcher.py \
  --site .sparkring/exl3-r7/mtp4-nvfp4-ckv-site.yaml \
  --profile .sparkring/exl3-r7/operator-profile.json \
  plan
```

A rank that has run this profile before holds adaptive-MTP receipts in its
JIT cache directory, and two guards refuse to start against them: the
exact-state attestation requires a receipt whose attested flag is unset to be
archived first, and the adaptive controller refuses to overwrite its own. Both
name the file they found. Archive every `adaptive-mtp-*.json` and
`adaptive-mtp-*.jsonl` under `paths.jit_cache_dir` on each rank before
starting, keeping the contents under a timestamped name rather than deleting
them:

```bash
for receipt in "$JIT_CACHE_DIR"/adaptive-mtp-*.json "$JIT_CACHE_DIR"/adaptive-mtp-*.jsonl; do
  [ -e "$receipt" ] || continue
  case "$receipt" in *.archived-*) continue ;; esac
  mv "$receipt" "$receipt.archived-$(date -u +%Y%m%dT%H%M%SZ)"
done
```

A first start on a rank that has never run the profile finds no receipt and
needs nothing archived.

Review the plan. Then start (this **STOPS SERVING** if a stack is running):

```bash
python scripts/sparkring_generic_launcher.py \
  --site .sparkring/exl3-r7/mtp4-nvfp4-ckv-site.yaml \
  --profile .sparkring/exl3-r7/operator-profile.json \
  --execute \
  --confirmation START-SIRCL-Q40-EXACT-STATE-CANARY-ALL-FOUR \
  start
```

## 9. Health and model checks

```bash
python scripts/sparkring_generic_launcher.py \
  --site .sparkring/exl3-r7/mtp4-nvfp4-ckv-site.yaml \
  --profile .sparkring/exl3-r7/operator-profile.json \
  --execute status
```

Require `/health` HTTP 200, the exact served model name
`glm-5.2-exl3-r7-3.5bpw`, and 262,144 maximum model length from
`/v1/models`.

## 10. MTP3 rollback

If the MTP4 candidate must be rolled back, the exact MTP3 KV9.25 profile
and site are in the rollback artifacts:

```bash
python scripts/sparkring_generic_launcher.py \
  --site scripts/config/site.yaml \
  --profile .sparkring/exl3-r7/mtp4-kv925-rollback.json \
  --execute \
  --confirmation START-EXL3-R7-CANDIDATE-ALL-FOUR \
  start
```

The rollback profile is byte-identical to the MTP3 KV9.25 input. Verify:

```bash
sha256sum .sparkring/exl3-r7/mtp4-kv925-rollback.json
sha256sum .sparkring/exl3-r7/mtp3-kv925-profile.json
```

These must match.

## Limitations

- The operator profile is accepted on one four-Spark appliance. A rebuilt
  image is not accepted until the promotion checklist passes against its ID.
- MTP4 improves the measured C1-C4 cells but regresses the matched
  C8 cell by 11.63%. See
  [EXL3_R7_FIXED_MTP4_PROFILE.md](EXL3_R7_FIXED_MTP4_PROFILE.md).
- The published image carries the runtime filesystem, and the exact-Q40
  attestation names the identity of an image built locally, so a pulled image
  serves the profile without that layer. The source builder is
  offline-validated; its clean-checkout image has not completed the published
  four-rank live gate.
- Dynamic-NVFP4, CKV-gather, tiered-SIRCL, and exact-Q40 composition commands
  are published in
  [`EXL3_R7_OPERATOR_REPRODUCTION.md`](EXL3_R7_OPERATOR_REPRODUCTION.md).
- DCP and indexer collectives use the stock path. Only the
  qualified TP all-reduce and vocabulary families use the SparkRing native
  transport.
- Fixed MTP5 requires a Q48 Python contract extension and the matching
  native library, which the deployed overlay set supplies rather than the
  bare image. A four-rank launch at `num_speculative_tokens: 5` serves and
  reports mean acceptance length between 4.38 and 5.52. Whether it decodes
  faster than the fixed-MTP4 contract above is unmeasured: no matched
  comparison between the two has been run, so this page continues to
  specify MTP4.

## Input chain

The public input chain is:

```text
recipes/glm52-exl3-r7-3.5bpw.json          # tracked recipe (model + serving contract)
scripts/config/exl3-r7-pins.json             # public pins (derived from recipe)
scripts/config/exl3-r7-candidate.example.json # candidate template (placeholders)
scripts/config/exl3-r7-site.example.yaml       # site template (placeholder addresses)
scripts/generate_exl3_r7_candidate.py         # baseline profile generator
scripts/generate_exl3_r7_stock_dcp4.py        # stock-DCP4 baseline generator
scripts/prepare_exl3_r7_mtp2.py               # MTP2 derivative
scripts/prepare_exl3_r7_mtp3.py                # MTP3 derivative
scripts/prepare_exl3_r7_mtp3_kv925.py          # KV9.25 derivative (site-only)
scripts/prepare_exl3_r7_mtp4.py                # MTP4 derivative (the candidate)
scripts/prepare_exl3_r7_mtp4_nvfp4.py          # 262K dynamic-NVFP4 derivative
scripts/prepare_exl3_r7_mtp4_ckv_gather.py     # bounded full-CKV derivative
scripts/prepare_exl3_r7_sircl_tiered.py         # tiered/deferred SIRCL bundle/profile
spark_transport/experiments/moe_round_floor/q40_exact_state_overlay.py
spark_transport/experiments/moe_round_floor/q40_exact_state_attestation_overlay.py
spark_transport/experiments/moe_round_floor/prepare_q40_exact_state_serving.py
scripts/download_exl3_r7.py                     # checkpoint downloader/verifier
scripts/exl3_r7_standup.py                      # stand-up entrypoint (dry-run default)
```

A user does not need any maintainer-held stock-DCP4 profile. The
stock-DCP4 baseline is derived from the tracked candidate generator plus
the recipe's serving contract.
