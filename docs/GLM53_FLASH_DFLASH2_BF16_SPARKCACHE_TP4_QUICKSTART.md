# GLM-5.3 Flash TP4 with BF16 DFlash2 and SparkCache

**Qualified:** only the exact artifacts and four parent/derived image pairs in
[`recipes/sparkcache/glm53-flash-nvfp4-dflash2-bf16-tp4-dcp1.json`](../recipes/sparkcache/glm53-flash-nvfp4-dflash2-bf16-tp4-dcp1.json).

**Implemented, unqualified:** the deployment template with any other image
identity. Mark a rebuilt image **qualified** only after it produces the store,
coordinated restart, restore, and semantic-canary evidence required by
[Verify persistent restore](#verify-persistent-restore).

**Public reproduction: unsupported.** The required parent image and exact NCCL
build are not published.

This procedure starts GLM-5.3 Flash on four directly cabled NVIDIA DGX Spark
systems at TP4/DCP1. It uses the public BF16 DFlash2 model for seven-token
speculation and SparkCache for persistent target-context store and restore.
Asynchronous scheduling, native vLLM prefix caching, and chunked prefill remain
enabled.

## Public artifact boundary

The four qualified GLM-5.3 ARM64 parent images are local Docker objects rather
than pullable registry artifacts. Their loaded NCCL binary is checksum-bound
but lacks a complete public source/build receipt. The public repositories and
model downloads therefore cannot construct the qualified runtime by
themselves.

An operator-supplied parent must provide Linux ARM64 with CUDA SM121 support,
Python 3.12, vLLM commit
`da4d7be6c97434f6942292ed8abbf4b32dc44355`, B12X commit
`2fcf23a0ce269be27b2e03fece73d46e90e6aeea`, and NCCL 2.30.7 binary SHA-256
`ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f` at
`/opt/sparkring/nccl/libnccl.so.2`. The parent must also satisfy every vLLM
postimage check and provide the inherited `org.jovian.*` labels required by
[`scripts/config/glm53-flash-dflash2-bf16-tp4-dcp1-sparkcache.example.json`](../scripts/config/glm53-flash-dflash2-bf16-tp4-dcp1-sparkcache.example.json).
The completed derived image must satisfy every required OCI-label check in that
profile. No public recipe is claimed to produce the parent. Stop before the
build if an attested parent is unavailable.

## Operator prerequisites

Complete [SparkRing prerequisites](PREREQUISITES.md) and, for an unconfigured
four-Spark cycle, [cluster bootstrap](BOOTSTRAP.md). Run the commands below in a
Bash shell on the controller Spark with Git, Python 3 and PyYAML, Docker with
NVIDIA GPU access, and Hugging Face CLI `hf`. Every rank requires passwordless
SSH from the controller, two active 200 Gb/s RoCE links, the same absolute
model paths, at least 346,218,639,128 bytes for the target checkpoint, and
writable local JIT/context-cache storage. Port 8015 on rank 0 and port 29755 on
every rank must be free before startup.

Create immutable source checkouts on the controller:

```bash
operator_root="${PWD}/glm53-flash-operator"
sparkring_root="${operator_root}/sparkring"
sparkcache_root="${operator_root}/sparkcache"
site_yaml="${operator_root}/glm53-flash-tp4.site.yaml"
profile_json="${operator_root}/glm53-flash-sparkcache-tp4.profile.json"
plan_json="${operator_root}/glm53-flash-sparkcache-tp4.plan.json"
receipt_dir="${operator_root}/receipts"

mkdir -p "${operator_root}" "${receipt_dir}"
git clone https://github.com/FujitsuPolycom/sparkring.git "${sparkring_root}"
git -C "${sparkring_root}" checkout --detach \
  d45572dbd2adc7afa1d3208fb801c8ad9eac7864
git clone https://github.com/FujitsuPolycom/sparkcache.git "${sparkcache_root}"
git -C "${sparkcache_root}" checkout --detach \
  2d6a222f04fcb7b903cb899aba3ed3fdc75edc11
```

## Required artifacts

On every rank, choose identical absolute model paths, download both immutable
revisions, and verify every file published by those revisions. Edit the first
two assignments before running this block:

```bash
TARGET_MODEL_DIR=''
DFLASH_MODEL_DIR=''
: "${TARGET_MODEL_DIR:?Set TARGET_MODEL_DIR to an absolute model directory}"
: "${DFLASH_MODEL_DIR:?Set DFLASH_MODEL_DIR to an absolute model directory}"
target_model_dir="${TARGET_MODEL_DIR}"
draft_model_dir="${DFLASH_MODEL_DIR}"

mkdir -p "${target_model_dir}" "${draft_model_dir}"
hf download local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --revision 520de24eabf507659eaef7c70f14fd584527facc \
  --local-dir "${target_model_dir}"

hf download incoai/GLM-5.3-Flash-DFlash2 \
  --revision dc77ff1c99eeb2df044ee3d4f0094eb033fee410 \
  --local-dir "${draft_model_dir}"

hf cache verify local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --revision 520de24eabf507659eaef7c70f14fd584527facc \
  --local-dir "${target_model_dir}" --fail-on-missing-files
hf cache verify incoai/GLM-5.3-Flash-DFlash2 \
  --revision dc77ff1c99eeb2df044ee3d4f0094eb033fee410 \
  --local-dir "${draft_model_dir}" --fail-on-missing-files

test "$(sha256sum "${target_model_dir}/config.json" | awk '{print $1}')" = \
  676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996
test "$(sha256sum "${target_model_dir}/model.safetensors.index.json" | awk '{print $1}')" = \
  0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb
test "$(sha256sum "${draft_model_dir}/config.json" | awk '{print $1}')" = \
  c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573
test "$(sha256sum "${draft_model_dir}/model.safetensors" | awk '{print $1}')" = \
  b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b
```

The Inco DFlash2 checkpoint is licensed CC BY-NC-ND 4.0 for research and
evaluation. Review that license before downloading or using the artifact.

The SparkCache checkout above has normalized source SHA-256
`6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`.
Verify it and build once on one Spark. Replace the immutable parent reference
and expected local parent ID with values from the operator's attested parent:

```bash
sparkcache_source_sha256="$(
  cd "${sparkcache_root}" &&
  python -c 'from pathlib import Path; from deploy.deployment_contract.source import source_tree_sha256; print(source_tree_sha256(Path("sparkcache")))'
)"
test "${sparkcache_source_sha256}" = \
  6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2

base_image_ref='registry.example.invalid/REPLACE/glm53-runtime@sha256:0000000000000000000000000000000000000000000000000000000000000000'
base_image_id='sha256:0000000000000000000000000000000000000000000000000000000000000000'
temporary_build_tag='sparkring/glm53-flash-sparkcache:da4d7be-glm53-hybrid'
test "$(docker image inspect --format '{{.Id}}' "${base_image_ref}")" = \
  "${base_image_id}"

python "${sparkcache_root}/deploy/glm53_flash/build_image.py" \
  --repository "${sparkcache_root}" \
  --base-image "${base_image_ref}" \
  --base-image-id "${base_image_id}" \
  --source-sha256 "${sparkcache_source_sha256}" \
  --output-image "${temporary_build_tag}"
derived_image_id="$(docker image inspect --format '{{.Id}}' "${temporary_build_tag}")"
printf 'derived image ID: %s\n' "${derived_image_id}"
```

`temporary_build_tag` identifies only the local build command; it is not a
deployment identity. Push and pull one registry manifest, or transfer one
`docker save` archive and load it on every rank. Do not rebuild independently
per rank. Configure `image` with the immutable registry manifest reference or
`derived_image_id`, never the temporary tag. On every rank,
`docker image inspect --format '{{.Id}}' "${derived_image_id}"` must print the
same `derived_image_id`. Record the registry manifest digest, local image ID,
parent reference, and parent local image ID.

## Prepare sanitized deployment inputs

Copy the site and runtime templates to files that remain outside version
control:

```bash
cp "${sparkring_root}/scripts/config/glm53-flash-tp4-site.example.yaml" \
  "${site_yaml}"
cp "${sparkring_root}/scripts/config/glm53-flash-dflash2-bf16-tp4-dcp1-sparkcache.example.json" \
  "${profile_json}"
```

In `site_yaml`, replace all documentation-only addresses, SSH targets,
interfaces, RDMA devices, GID indices, and image values. In `profile_json`:

- set `image` to an immutable registry reference or the exact local image ID;
- set `image_id` to the common `sha256:` image ID on all ranks;
- set required image label `org.sparkcache.parent-image-id` to the exact
  `base_image_id` used by the build;
- set `model_host_path` to the target revision directory present on every rank;
- set the DFlash volume host path to its pinned revision directory; and
- set the writable `/cache/jit` volume host path to the site file's
  `paths.jit_cache_dir`. Set `paths.context_cache_dir` to the
  `sparkcache-context` child of that host path. Create both writable
  directories on every rank before preflight. The same path string is safe
  across hosts because each physical rank owns a different local filesystem.

The site and profile must contain the same image reference, image ID, and
target container path. All four ranks must have the target model, DFlash model,
and cache directories at the configured host paths.

Do not put site-resolved files into the repository. Validate the complete
plan offline:

```bash
python "${sparkring_root}/scripts/sparkring_site.py" \
  --strict-placeholders "${site_yaml}"
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site "${site_yaml}" --profile "${profile_json}" validate
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site "${site_yaml}" --profile "${profile_json}" explain
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site "${site_yaml}" --profile "${profile_json}" plan > "${plan_json}"
python "${sparkring_root}/scripts/preflight.py" \
  --site "${site_yaml}" --strict-placeholders --print-plan
```

`validate` must fail while a zero image ID or another unresolved template value
remains. The resolved plan must show TP4/DCP1, 524,288 maximum model length,
12,884,901,888 key-value bytes per rank, 32 sequences, and the SparkCache
connector configuration. It must retain `--async-scheduling`,
`--enable-prefix-caching`, and `--enable-chunked-prefill`.

Review the printed remote probes, then run the read-only preflight and retain
its JSON receipt:

```bash
python "${sparkring_root}/scripts/preflight.py" \
  --site "${site_yaml}" --strict-placeholders \
  --json "${receipt_dir}/preflight.json"
```

Preflight must pass every rank. If it reports port 8015 or 29755 in use, stop
the owning service with that service's documented command and rerun preflight.
The launcher does not remove a foreign container or process.

## Start and observe

Starting the profile changes all four hosts. Review `plan_json`, then run:

```bash
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site "${site_yaml}" --profile "${profile_json}" \
  --execute --confirmation START_GLM53_FLASH_DFLASH2_TP4 start
```

The launcher stops before container creation if the image ID, required image
labels, DFlash hashes, NCCL binary hash, vLLM configuration postimage, or vLLM
lease contract differs. It also rolls back containers that started if another
rank fails.

Tail the API rank's vLLM log:

```bash
rank0_ssh='operator@rank0.example.net'
api_endpoint='http://rank0.example.net:8015'
served_model='glm-5.3-flash-nvfp4-dflash7-bf16-tp4'

ssh "${rank0_ssh}" \
  'docker logs --follow --tail 120 glm53-flash-dflash2-bf16-sparkcache-tp4-r0 2>&1'
```

Replace `rank0_ssh` and `api_endpoint` with the values in `site_yaml`. API
readiness, rather than an informal log phrase, defines completion of graph
capture and startup. In another shell, wait for readiness and inspect the
served-model identity:

```bash
timeout 7200 bash -c \
  'until curl --fail --silent --show-error "$1/health" >/dev/null; do sleep 5; done' \
  _ "${api_endpoint}"
curl --fail --silent --show-error "${api_endpoint}/v1/models"
```

## Verify persistent restore

The pinned SparkCache checkout contains the deterministic 8,192-token request
and semantic canary. Store the context and retain the request receipt:

```bash
qualification_script="${sparkcache_root}/deploy/glm53_flash/qualification_request.py"
python "${qualification_script}" \
  --endpoint "${api_endpoint}" --model "${served_model}" \
  --kind persistent --output "${receipt_dir}/cold.json"
```

Require every rank log to contain `committed 8192 tokens` for one common
digest. Stop all four containers without deleting the rank-local cache roots,
then start the exact same resolved site and profile:

```bash
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site "${site_yaml}" --profile "${profile_json}" \
  --execute --confirmation START_GLM53_FLASH_DFLASH2_TP4 stop
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site "${site_yaml}" --profile "${profile_json}" \
  --execute --confirmation START_GLM53_FLASH_DFLASH2_TP4 start
```

Wait for API readiness again. Every rank must log
`manifest discovery checked=3 offered=3 rejected=0`. The first request can
cleanly recompute if it reaches the scheduler before the complete inventory;
record that request as a prime, then repeat the identical request for the
restore result:

```bash
python "${qualification_script}" \
  --endpoint "${api_endpoint}" --model "${served_model}" \
  --kind persistent --output "${receipt_dir}/post-restart-prime.json"
curl --fail --silent --show-error "${api_endpoint}/metrics" \
  > "${receipt_dir}/metrics-before-restore.prom"
python "${qualification_script}" \
  --endpoint "${api_endpoint}" --model "${served_model}" \
  --kind persistent --output "${receipt_dir}/post-restart-restore.json"
python "${qualification_script}" \
  --endpoint "${api_endpoint}" --model "${served_model}" \
  --kind semantic --output "${receipt_dir}/post-restore-semantic.json"
curl --fail --silent --show-error "${api_endpoint}/metrics" \
  > "${receipt_dir}/metrics-after-restore.prom"

grep -E '^vllm:(external_prefix_cache_(queries|hits)_total|prompt_tokens_by_source_total|spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total|num_preemptions_total)' \
  "${receipt_dir}/metrics-before-restore.prom" \
  "${receipt_dir}/metrics-after-restore.prom"
```

### Inspect all ranks

Inspect every rank with the SSH targets and rank-specific container names from
`site_yaml`. Replace the four example SSH targets before running the block:

```bash
set -euo pipefail
rank_ssh=(
  'operator@rank0.example.net'
  'operator@rank1.example.net'
  'operator@rank2.example.net'
  'operator@rank3.example.net'
)
container_prefix='glm53-flash-dflash2-bf16-sparkcache-tp4'
fatal_pattern='Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|NCCL.*(unhandled|error|failed)|Fatal Python error|Segmentation fault|ProcessGroupNCCL.*exception'

for rank in 0 1 2 3; do
  host="${rank_ssh[$rank]}"
  container="${container_prefix}-r${rank}"
  if ! log_text="$(ssh "${host}" "docker logs '${container}' 2>&1")"; then
    printf 'log retrieval failed on rank %s\n' "${rank}" >&2
    exit 1
  fi
  if ! grep -Fq 'spark-context-cache: restored 8192 tokens async' <<<"${log_text}"; then
    printf '8,192-token restore log is absent on rank %s\n' "${rank}" >&2
    exit 1
  fi
  if ! container_state="$(ssh "${host}" "docker inspect --format '{{.RestartCount}} {{.State.OOMKilled}} {{.State.Status}}' '${container}'")"; then
    printf 'container inspection failed on rank %s\n' "${rank}" >&2
    exit 1
  fi
  if [[ "${container_state}" != '0 false running' ]]; then
    printf 'unhealthy container state on rank %s: %s\n' "${rank}" "${container_state}" >&2
    exit 1
  fi
  if ! qp_count="$(ssh "${host}" "rdma resource show qp 2>/dev/null | awk '/state RTS/ && /comm VLLM::Worker/ {n++} END {print n+0}'")"; then
    printf 'RDMA inspection failed on rank %s\n' "${rank}" >&2
    exit 1
  fi
  if [[ "${qp_count}" != '24' ]]; then
    printf 'rank %s has %s worker QPs in RTS; expected 24\n' "${rank}" "${qp_count}" >&2
    exit 1
  fi
  if grep -Eiq "${fatal_pattern}" <<<"${log_text}"; then
    printf 'fatal log match on rank %s\n' "${rank}" >&2
    exit 1
  fi
done
```

Qualification requires all of the following:

- the repeated request reports 8,192 external-prefix hit tokens;
- every rank logs a restore rather than unverified-byte consumption;
- DFlash draft tokens equal seven times its draft count;
- a separate uncached semantic canary ends with `stop` and its expected final
  answer;
- no rank records a preemption, restart, OOM, or fatal error; and
- every rank retains exactly 24 `VLLM::Worker` queue pairs in RTS.

The external-hit and `external_kv_transfer` prompt-token counters must each
increase by 8,192 across the restore request. DFlash draft tokens must increase
by seven times the draft-count increase. The semantic receipt must report
`semantic_match: true`; the all-rank inspection must exit successfully.

The connector's `recompute` policy makes an incomplete inventory or rejected
manifest a cache miss. A recompute is safe but is not a restore result.

Stop the stack with the same explicit confirmation:

```bash
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site "${site_yaml}" --profile "${profile_json}" \
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
| SparkRing deployment profile | `FujitsuPolycom/sparkring@d45572dbd2adc7afa1d3208fb801c8ad9eac7864` on branch `codex/glm53-flash-sparkcache-tp4`, based on `510556275ed3b77fc56a14367d319417072eeb8c`. | The branch has no pull request, release tag, or distributed image artifact. |
| Adapted launch inputs | Four-rank launcher snapshot SHA-256 `fef84dda87bab36f36f993f21a3e582438f3b0d1e3239b292ef0ef39e8c44b23`; service-settings snapshot SHA-256 `2c4d81d04060d92f4419d3f17d3c51b2f195d66376c9271617a167c18de14df1`; source-lock snapshot SHA-256 `913d54bd68fdea1280a8dd2baf15cf3461e04645f50be5bda9eafc027d03e4a8`. SparkRing expresses their settings through validated site and runtime schemas; no implementation source was copied. | The snapshots were uncommitted operator artifacts. Base Git revision `f3ba67fa476fd28109868811d6edbb4085c8f0a0` does not reproduce them without the recorded snapshots. |

The machine-readable provenance manifest is
[`runtime/glm53-flash/pins.json`](../runtime/glm53-flash/pins.json).
