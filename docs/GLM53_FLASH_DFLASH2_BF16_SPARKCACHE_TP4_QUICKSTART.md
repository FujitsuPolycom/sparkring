# GLM-5.3 Flash TP4 with BF16 DFlash2 and SparkCache

Status: **qualified** for the exact artifacts and four parent/derived image pairs recorded in
[`recipes/sparkcache/glm53-flash-nvfp4-dflash2-bf16-tp4-dcp1.json`](../recipes/sparkcache/glm53-flash-nvfp4-dflash2-bf16-tp4-dcp1.json).
The operator template is **implemented** for a rebuilt immutable image; that
image requires its own live qualification before it inherits qualified status.

This procedure starts GLM-5.3 Flash on four directly cabled NVIDIA DGX Spark
systems at TP4/DCP1. It uses the public BF16 DFlash2 model for seven-token
speculation and SparkCache for persistent target-context store and restore.
Asynchronous scheduling, native vLLM prefix caching, and chunked prefill remain
enabled.

## Public reproducibility requirement

This procedure is not a standalone public build. The qualified GLM-5.3 ARM64
parent images are not published, and the loaded NCCL binary is checksum-bound
but lacks a complete public source/build receipt. A reader must already have an
equivalent GLM-5.3 parent image and the exact patched NCCL binary, or construct
and validate replacements independently. Stop here if those artifacts are not
available; the remaining steps cannot produce the qualified runtime from the
model downloads and these repositories alone.

## Required artifacts

Download each model by immutable revision on every rank:

```bash
hf download local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --revision 520de24eabf507659eaef7c70f14fd584527facc \
  --local-dir <target-model-directory>

hf download incoai/GLM-5.3-Flash-DFlash2 \
  --revision dc77ff1c99eeb2df044ee3d4f0094eb033fee410 \
  --local-dir <draft-model-directory>

printf '%s  %s\n' \
  676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996 \
  <target-model-directory>/config.json | sha256sum --check --strict
printf '%s  %s\n' \
  0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb \
  <target-model-directory>/model.safetensors.index.json | sha256sum --check --strict
printf '%s  %s\n' \
  c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573 \
  <draft-model-directory>/config.json | sha256sum --check --strict
printf '%s  %s\n' \
  b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b \
  <draft-model-directory>/model.safetensors | sha256sum --check --strict
```

The Inco DFlash2 checkpoint is licensed CC BY-NC-ND 4.0 for research and
evaluation. Review that license before downloading or using the artifact.

Use SparkCache commit `2d6a222f04fcb7b903cb899aba3ed3fdc75edc11`, whose normalized source SHA-256 is
`6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`.
From that checkout, verify the digest and build the derived image from an
immutable GLM-5.3 runtime image:

```bash
git clone https://github.com/FujitsuPolycom/sparkcache.git <sparkcache-source>
git -C <sparkcache-source> checkout --detach \
  2d6a222f04fcb7b903cb899aba3ed3fdc75edc11
cd <sparkcache-source>
python -c 'from pathlib import Path; from deploy.deployment_contract.source import source_tree_sha256; print(source_tree_sha256(Path("sparkcache")))'

base_image_ref='<immutable-glm53-runtime-image@sha256:manifest-digest>'
base_image_id='sha256:<resolved-parent-image-id>'
test "$(docker image inspect --format '{{.Id}}' "${base_image_ref}")" = \
  "${base_image_id}"

docker build \
  --file deploy/glm53_flash/Containerfile \
  --build-arg BASE_IMAGE="${base_image_ref}" \
  --build-arg BASE_IMAGE_ID="${base_image_id}" \
  --build-arg SPARKCACHE_SOURCE_SHA256=6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2 \
  --iidfile <image-id-file> \
  .
```

Build once, then distribute that exact image to all four ranks with a registry
manifest digest or `docker save`/`docker load`. Do not rebuild independently
on each rank. Record the common value printed to `<image-id-file>`.

## Prepare sanitized deployment inputs

Copy the site and runtime templates to files that remain outside version
control:

```bash
cp scripts/config/glm53-flash-tp4-site.example.yaml <site-yaml>
cp scripts/config/glm53-flash-dflash2-bf16-tp4-dcp1-sparkcache.example.json <profile-json>
```

In `<site-yaml>`, replace all documentation-only addresses, SSH targets,
interfaces, RDMA devices, GID indices, and image values. In `<profile-json>`:

- set `image` to an immutable registry reference or the exact local image ID;
- set `image_id` to the common `sha256:` image ID on all ranks;
- set required image label `org.sparkcache.parent-image-id` to the exact
  `base_image_id` used by the build;
- set `model_host_path` to the target revision directory present on every rank;
- set the DFlash volume host path to its pinned revision directory; and
- set the writable cache volume host path. The same path string is safe across
  hosts because each physical rank owns a different local filesystem.

Do not put site-resolved files into the repository. Validate the complete
plan offline:

```bash
python scripts/sparkring_site.py --strict-placeholders <site-yaml>
python scripts/sparkring_generic_launcher.py \
  --site <site-yaml> --profile <profile-json> validate
python scripts/sparkring_generic_launcher.py \
  --site <site-yaml> --profile <profile-json> explain
python scripts/sparkring_generic_launcher.py \
  --site <site-yaml> --profile <profile-json> plan > <reviewed-plan-json>
```

`validate` must fail while a zero image ID or another unresolved template value
remains. The resolved plan must show TP4/DCP1, 524,288 maximum model length,
12,884,901,888 key-value bytes per rank, 32 sequences, and the SparkCache
connector configuration. It must retain `--async-scheduling`,
`--enable-prefix-caching`, and `--enable-chunked-prefill`.

## Start and observe

Starting the profile changes all four hosts and can replace a serving stack.
Review `<reviewed-plan-json>`, then run:

```bash
python scripts/sparkring_generic_launcher.py \
  --site <site-yaml> --profile <profile-json> \
  --execute --confirmation START_GLM53_FLASH_DFLASH2_TP4 start
```

The launcher stops before container creation if the image ID, required image
labels, DFlash hashes, NCCL binary hash, vLLM configuration postimage, or vLLM
lease contract differs. It also rolls back containers that started if another
rank fails.

Tail the API rank's vLLM log:

```bash
ssh <rank-0-ssh-target> \
  'docker logs --follow --tail 120 glm53-flash-dflash2-bf16-sparkcache-tp4-r0 2>&1'
```

Check the API after all ranks finish graph capture:

```bash
curl --fail http://<rank-0-management-address>:8015/health
curl --fail http://<rank-0-management-address>:8015/v1/models
```

## Verify persistent restore

Use one deterministic prompt with at least 8,192 reusable aligned tokens.
Require every rank to log the same committed digest. Stop all four containers
without removing the rank-local cache roots, start the same resolved profile,
and wait for all ranks to report manifest discovery. Repeat the exact prompt
until the scheduler reports all-rank inventory quorum.

Qualification requires all of the following:

- the repeated request reports 8,192 external-prefix hit tokens;
- every rank logs a restore rather than unverified-byte consumption;
- DFlash draft tokens equal seven times its draft count;
- a separate uncached semantic canary ends with `stop` and its expected final
  answer;
- no rank records a preemption, restart, OOM, or fatal error; and
- every rank retains its qualified `VLLM::Worker` RTS queue-pair count.

The connector's `recompute` policy makes an incomplete inventory or rejected
manifest a cache miss. A recompute is safe but is not a restore result.

Stop the stack with the same explicit confirmation:

```bash
python scripts/sparkring_generic_launcher.py \
  --site <site-yaml> --profile <profile-json> \
  --execute --confirmation START_GLM53_FLASH_DFLASH2_TP4 stop
```

## Evidence and limitations

Conditions, measurement, result, and conclusion for the qualified 8,192-token
restore are in
[`performance/records/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828.md`](../performance/records/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828.md).
The qualified source digest is `6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`
with seven-file lease-contract SHA-256
`2e3b17fd6a34f2dbb8e91a99b83dbf18629cf0e718f9f814236da4bbfc9ae3f1`.
The restored request served 8,192 external hits in 1.509 seconds; per-rank
restore times were 155.6, 147.2, 194.0, and 151.8 milliseconds. DFlash
produced 301 draft tokens from 43 drafts and accepted 112. The 1.176-second
semantic canary passed. All ranks remained healthy with 24 RTS worker QPs.
Ranks 0, 2, and 3 passed strict verification of all 59 target files. Rank 1
matched those files and contained additional `.cache/huggingface` metadata.
The record does not establish throughput neutrality, larger-span restore
performance, streaming snapshots, native direct restore, MTP compatibility,
or compatibility with another image, source tree, checkpoint, topology, or
cache geometry.

## Docker image publication checklist

Any published image is a **FujitsuPolycom community derivative**, not an
official vLLM, Z.AI, local-inference-lab, NVIDIA, or Inco AI image. The
qualified receipt records these parent/derived pairs:

| Rank | Parent image ID | Derived image ID |
|---:|---|---|
| 0 | `sha256:ddd13fb1ea8ca61aaf771715dc8c5a52dfe6860f0cc62c145d155916bf381fc9` | `sha256:56f051b1b1b6f9f858ea5d21b7933b64af81c22bee2c417a3f8b4466220e37e6` |
| 1 | `sha256:7fb81337ba088a6bf0bbce71b22a5881f812a21af9ac1d6deea9533a8e9eed92` | `sha256:8506935b369bd4f0d5d73495ded9a2fcb52bbe2f310ea093818e5d3d5366ae38` |
| 2 | `sha256:9bd97e3d77de969ee0788aaac31b2888fd4c6a3d893ac5fc544ca85363927935` | `sha256:b969a49ec091157c686a3bc3f52816b6aa910e495af0c92780a321ea5fbd5324` |
| 3 | `sha256:d592c83cc04106532adf7d8d410347062ac1b80fc1b6981deca414b5335efff4` | `sha256:c9f0be4dccfd8fdcec80a3edce1ad217604fa09afee0f14d13a2839fb97eed9f` |

The four rank-local builds are not one distributable image. A publication
must build once from one recorded parent, distribute the resulting image ID,
and repeat the live qualification on all ranks.

Publish only one build distributed unchanged to all ranks. Its receipt must
include:

- the derived registry manifest digest and local image ID, and the parent
  registry manifest digest and resolved local parent image ID;
- the exact `deploy/glm53_flash/Containerfile`, build arguments, build command,
  Docker/BuildKit versions, platform, size, and creation timestamp;
- all model, vLLM, B12X, NCCL, SparkCache, patch, preimage, lease-contract, and
  source identities from the provenance section;
- inherited parent content: vLLM, B12X, CUDA/toolchain, GLM/DFlash runtime
  support, and patched NCCL;
- SparkCache-owned changes: connector source, `glm53-flash-hybrid`, the narrow
  VMM exemption, lease verifier, and OCI source/profile labels. SparkCache
  does not modify or include model weights and does not replace NCCL;
- the tested TP4/DCP1 configuration, all build and validation commands,
  CPU pass/skip counts, image-label and in-container attestation output,
  all-rank launch command, and sanitized store/restart/restore/canary results;
- every unsupported configuration listed above, plus commercial DFlash use
  without an applicable license; and
- FujitsuPolycom support links:
  [SparkRing issues](https://github.com/FujitsuPolycom/sparkring/issues) for
  deployment/transport and
  [SparkCache issues](https://github.com/FujitsuPolycom/sparkcache/issues) for
  connector/image behavior.

The SparkRing repository validation for this profile change is Ruff passed and
1,877 CPU tests passed with nine skips. A publication receipt must rerun and
record those commands plus the SparkCache repository's CPU suites. CPU results
do not replace the live GPU/RDMA record.

Minimal announcement template:

> **FujitsuPolycom community image — GLM-5.3 Flash TP4/DCP1, BF16 DFlash2,
> SparkCache**
> Image `<repository>@sha256:<manifest-digest>` / local ID
> `sha256:<derived-image-id>`; parent
> `<parent-repository>@sha256:<parent-manifest-digest>` / local ID
> `sha256:<parent-image-id>`. Built and validated under
> `runtime/glm53-flash/pins.json`. Qualified scope and unsupported settings:
> `docs/profiles/GLM53_FLASH_DFLASH2_BF16_TP4.md`. Community derivative, not
> an official upstream image. Support:
> `https://github.com/FujitsuPolycom/sparkring/issues` and
> `https://github.com/FujitsuPolycom/sparkcache/issues`.

## Provenance

The following facts are verified. No base-checkpoint, pull-request, or
binary-build lineage beyond the listed records is inferred.

| Component | Verified provenance | Limitation |
|---|---|---|
| Target quantization | `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`; repository owner `local-inference-lab`; uploaded by `lukealonso`; MIT; ModelOpt `0.39.0.dev290+gf9d9a71de.d20260407` `MIXED_PRECISION`; NVFP4 target expert layers 3-44; MXFP8 MTP expert layer 45. | The repository does not record a base-checkpoint revision. |
| Target artifact verification | `config.json` SHA-256 `676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996`; weight-index SHA-256 `0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb`; 59 expected files matched on all ranks. | Rank 1 also contained `.cache/huggingface` metadata. |
| Public BF16 drafter | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`; produced by Inco AI; uploaded by `zhijianliu`; config SHA-256 `c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573`; weights SHA-256 `b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`; CC BY-NC-ND 4.0. | The model card limits use to research and evaluation and directs commercial licensing inquiries to Inco AI. |
| vLLM | `local-inference-lab/vllm`, `dev/jovian-judgement@da4d7be6c97434f6942292ed8abbf4b32dc44355`; direct commits `e0db84abedb4a85f93d130252e54b73c0f3ed695`, `0c878821cf46c99729c7936bcbd4d868ad40e44e`, `4dbd82b9ced13114f90e93b8b6fae0966c942a3b`, `1036123e935177900122c14d3cf02ad67b5422aa`, and `e7097feb6fcdf57911cd68884420af2d80600dd7`; merged PR/commit pairs `#486@15d3f79439eadc396a57e253c955aa149def94ea`, `#489@015dcd423d6aabf843c8ad69074ff67d35c2a395`, `#493@067c37d974ca2b775d95e51e8fec234929f4e2c4`, `#494@e91c7e68f5863a27c79d2773205678be7d8ff132`, `#497@05d85f603097fe7678d7dda2d522613d9dc61f46`, and `#499@da4d7be6c97434f6942292ed8abbf4b32dc44355`; their roles are recorded in `runtime/glm53-flash/pins.json`; `#499` depends on `#493` and `#494`. | No other upstream pull-request lineage is claimed. |
| B12X | `local-inference-lab/b12x`, `master@2fcf23a0ce269be27b2e03fece73d46e90e6aeea`, Apache-2.0, commit title `Accept runtime QSA cache page sizes`. | No associated pull request was found. |
| SparkRing NCCL | NVIDIA NCCL 2.30.7; skip-Tree/PAT patch SHA-256 `097656d07a5774919f0d51558b51ec05de8168c0097ed6cb7764c33230ba6eb2`; listener-GID patch SHA-256 `dccfce86d14c15c39f0e0a742863960205a3d9823c464b31a7f7389354844178`; qualified loaded binary SHA-256 `ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f`. | The binary is not bound to an NVIDIA NCCL source commit and complete patch-build receipt. |
| SparkCache | `FujitsuPolycom/sparkcache@2d6a222f04fcb7b903cb899aba3ed3fdc75edc11` on branch `codex/glm53-flash`, normalized source-tree SHA-256 `6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`, profile `glm53-flash-hybrid`, vLLM lease-contract SHA-256 `2e3b17fd6a34f2dbb8e91a99b83dbf18629cf0e718f9f814236da4bbfc9ae3f1`, VMM exemption patch SHA-256 `370b498eebf44b4e52a2d2751fa249ad4bd3d0b6fd951b063a161fb06febbe99`, patch-preimage manifest SHA-256 `e0eb1b64d15812f122450f2e32323f0c907c640b8f8ccc270c77037bb9909b85`, Containerfile SHA-256 `ccc6b39173df80f604820959c3f19f8bc363f79d11f7d4f2d913054a4161b3f5`, and builder SHA-256 `c130e5c2fdd5f33e73f90f04ef85fa1247d93bfe6db409cd99508841f8d84547`. | The immutable commit and the source, contract, and build-recipe digests are authoritative. |
| SparkRing profile | `FujitsuPolycom/sparkring`, branch `codex/glm53-flash-sparkcache-tp4`, based on `510556275ed3b77fc56a14367d319417072eeb8c`. | A PR or image receipt must record the immutable commit containing this uncommitted profile branch. |
| Adapted launch inputs | Four-rank launcher snapshot SHA-256 `fef84dda87bab36f36f993f21a3e582438f3b0d1e3239b292ef0ef39e8c44b23`; service-settings snapshot SHA-256 `2c4d81d04060d92f4419d3f17d3c51b2f195d66376c9271617a167c18de14df1`; source-lock snapshot SHA-256 `913d54bd68fdea1280a8dd2baf15cf3461e04645f50be5bda9eafc027d03e4a8`. SparkRing expresses their settings through validated site and runtime schemas; no implementation source was copied. | The snapshots were uncommitted operator artifacts. Base Git revision `f3ba67fa476fd28109868811d6edbb4085c8f0a0` does not reproduce them without the recorded snapshots. |

The machine-readable provenance manifest is
[`runtime/glm53-flash/pins.json`](../runtime/glm53-flash/pins.json).
