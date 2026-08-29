# GLM-5.3 Flash TP4 with BF16 DFlash2 and external caching disabled

**Implemented, unqualified.** The runtime profile produces a fail-closed
TP4/DCP1 deployment plan and preserves the model, scheduler, graph, and
transport settings used by the qualified SparkCache composition. No live
receipt qualifies external-cache-disabled behavior or performance.

**Public reproduction: unsupported.** The required parent image and exact NCCL
build are not published.

This procedure uses the same SparkCache-capable image, target checkpoint,
public BF16 DFlash2 checkpoint, rank topology, CUDA-graph settings, key-value
geometry, asynchronous scheduling, native vLLM prefix caching, and chunked
prefill as the SparkCache-enabled profile. It omits only vLLM's external
key-value connector. Using the same image makes the cache-disabled service a
controlled comparison rather than a different runtime build.

## Public artifact boundary and prerequisites

This procedure uses the same image as the
[SparkCache deployment](GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md)
so an A/B comparison changes only the external connector. Complete that
guide's [public artifact boundary](GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md#public-artifact-boundary),
[operator prerequisites](GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md#operator-prerequisites),
and [required artifacts](GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md#required-artifacts)
before continuing. Those sections define `operator_root`, `sparkring_root`,
`sparkcache_root`, `site_yaml`, `receipt_dir`, and `derived_image_id`.

An operator without the unpublished GLM-5.3 ARM64 parent and checksum-pinned
NCCL binary can inspect the profile but cannot construct its runtime from the
public model downloads and repositories alone.

## Required artifacts

The target and draft directories on every rank must pass the complete
Hugging Face revision checks in the SparkCache guide. The two identity hashes
used by the launch-time check can also be verified directly:

```bash
test "$(sha256sum "${target_model_dir}/config.json" | awk '{print $1}')" = \
  676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996
test "$(sha256sum "${target_model_dir}/model.safetensors.index.json" | awk '{print $1}')" = \
  0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb
test "$(sha256sum "${draft_model_dir}/config.json" | awk '{print $1}')" = \
  c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573
test "$(sha256sum "${draft_model_dir}/model.safetensors" | awk '{print $1}')" = \
  b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b
```

The DFlash2 checkpoint is licensed CC BY-NC-ND 4.0 for research and
evaluation. Review that license before downloading or using it.

The cache-disabled profile still verifies the SparkCache source label and vLLM
lease contract so both modes run identical image content. The same
`derived_image_id` must be present on all four ranks. This profile does not
load or call the external connector.

## Prepare and validate inputs

Copy the shared site template and the cache-disabled runtime template to files
outside version control:

```bash
profile_json="${operator_root}/glm53-flash-cache-disabled-tp4.profile.json"
sparkcache_profile_json="${operator_root}/glm53-flash-sparkcache-tp4.profile.json"
plan_json="${operator_root}/glm53-flash-cache-disabled-tp4.plan.json"

cp "${sparkring_root}/scripts/config/glm53-flash-tp4-site.example.yaml" \
  "${site_yaml}"
cp "${sparkring_root}/scripts/config/glm53-flash-dflash2-bf16-tp4-dcp1.example.json" \
  "${profile_json}"
cp "${sparkring_root}/scripts/config/glm53-flash-dflash2-bf16-tp4-dcp1-sparkcache.example.json" \
  "${sparkcache_profile_json}"
```

Replace the documentation-only site values. In `profile_json`, set the exact
image reference, image ID, required `org.sparkcache.parent-image-id` label,
target model host path, DFlash model host path, and writable JIT/cache host
path. The external-cache mount remains present so
the container filesystem and compile-cache locations match the enabled mode;
no `--kv-transfer-config` argument consumes persistent context data.

Run all offline checks:

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

`validate` must fail until every zero image identity and unresolved path is
replaced. The plan must contain `--async-scheduling`,
`--enable-prefix-caching`, and `--enable-chunked-prefill`; it must not contain
`--kv-transfer-config`.

The site file's `paths.jit_cache_dir` must equal the host path mounted at
`/cache/jit`; its `paths.context_cache_dir` must be the
`sparkcache-context` child of that host path. Create those directories on every
rank. Review the printed probes, then run the read-only preflight. It must pass
all ranks, including the free-port, directory, storage, image, RoCE, and SSH
checks:

```bash
python "${sparkring_root}/scripts/preflight.py" \
  --site "${site_yaml}" --strict-placeholders \
  --json "${receipt_dir}/cache-disabled-preflight.json"
```

For an A/B review, apply the same resolved image IDs and host paths to
`sparkcache_profile_json`, then compare the profiles. Exit status 1 is expected
because the external connector is intentionally different:

```bash
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site "${site_yaml}" \
  --profile-a "${profile_json}" \
  --profile-b "${sparkcache_profile_json}" diff
```

The serving-argument difference must be the external connector and its
profile/container identity. Model, draft, scheduler, graph, prefill, native
prefix-cache, memory, and transport settings must match.

## Start and observe

Starting changes all four hosts. Review `plan_json`, ensure ports 8015 and
29755 are free, then run:

```bash
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site "${site_yaml}" --profile "${profile_json}" \
  --execute --confirmation START_GLM53_FLASH_DFLASH2_TP4 start
```

The launch fails before container creation on any image ID, required-label,
DFlash, NCCL, vLLM configuration, or lease-contract mismatch.

Tail the API rank's vLLM log:

```bash
rank0_ssh='operator@rank0.example.net'
api_endpoint='http://rank0.example.net:8015'
served_model='glm-5.3-flash-nvfp4-dflash7-bf16-tp4'

ssh "${rank0_ssh}" \
  'docker logs --follow --tail 120 glm53-flash-dflash2-bf16-tp4-r0 2>&1'
```

Replace `rank0_ssh` and `api_endpoint` with the values in `site_yaml`. In
another shell, use API readiness as the startup-completion condition:

```bash
timeout 7200 bash -c \
  'until curl --fail --silent --show-error "$1/health" >/dev/null; do sleep 5; done' \
  _ "${api_endpoint}"
curl --fail --silent --show-error "${api_endpoint}/v1/models"
```

Capture metrics around the exact semantic canary from the pinned SparkCache
checkout:

```bash
qualification_script="${sparkcache_root}/deploy/glm53_flash/qualification_request.py"
curl --fail --silent --show-error "${api_endpoint}/metrics" \
  > "${receipt_dir}/cache-disabled-metrics-before.prom"
python "${qualification_script}" \
  --endpoint "${api_endpoint}" --model "${served_model}" \
  --kind semantic --output "${receipt_dir}/cache-disabled-semantic.json"
curl --fail --silent --show-error "${api_endpoint}/metrics" \
  > "${receipt_dir}/cache-disabled-metrics-after.prom"
grep -E '^vllm:(external_prefix_cache_(queries|hits)_total|spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total|num_preemptions_total)' \
  "${receipt_dir}/cache-disabled-metrics-before.prom" \
  "${receipt_dir}/cache-disabled-metrics-after.prom"
```

The receipt must report `semantic_match: true`. Draft tokens must increase by
seven times the draft-count increase, preemptions must not increase, and
external-prefix counters must not increase.

Inspect all four ranks after replacing the example SSH targets:

```bash
set -euo pipefail
rank_ssh=(
  'operator@rank0.example.net'
  'operator@rank1.example.net'
  'operator@rank2.example.net'
  'operator@rank3.example.net'
)
container_prefix='glm53-flash-dflash2-bf16-tp4'
fatal_pattern='Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|NCCL.*(unhandled|error|failed)|Fatal Python error|Segmentation fault|ProcessGroupNCCL.*exception'

for rank in 0 1 2 3; do
  host="${rank_ssh[$rank]}"
  container="${container_prefix}-r${rank}"
  if ! log_text="$(ssh "${host}" "docker logs '${container}' 2>&1")"; then
    printf 'log retrieval failed on rank %s\n' "${rank}" >&2
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

The block must exit successfully. These checks establish basic operation; they
do not create a qualified performance or soak result.

Stop the stack with:

```bash
python "${sparkring_root}/scripts/sparkring_generic_launcher.py" \
  --site "${site_yaml}" --profile "${profile_json}" \
  --execute --confirmation START_GLM53_FLASH_DFLASH2_TP4 stop
```

## Evidence and limitations

The cache-disabled configuration is implemented from the same validated
launch contract as the cache-enabled composition. The
[SparkCache restart-and-restore record](../performance/records/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828.md)
does not qualify cache-disabled throughput, capacity, restart behavior, or
long-duration reliability. Publish a separate conditions/measurement/result/
conclusion record before assigning qualified status to those claims.
That record qualifies SparkCache source
`6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`
and seven-file lease contract
`2e3b17fd6a34f2dbb8e91a99b83dbf18629cf0e718f9f814236da4bbfc9ae3f1`.
Its 8,192-token restore completed in 1.509 seconds and its semantic canary in
1.176 seconds. All 59 expected target files matched on every rank; rank 1 also
contained Hugging Face cache metadata. These cache-enabled measurements are
evidence for artifact compatibility, not cache-disabled performance.

## Docker image publication checklist

Any published image is a **FujitsuPolycom community derivative**, not an
official vLLM, Z.AI, local-inference-lab, NVIDIA, or Inco AI image. The
SparkCache qualification records these parent/derived pairs:

| Rank | Parent image ID | Derived image ID |
|---:|---|---|
| 0 | `sha256:ddd13fb1ea8ca61aaf771715dc8c5a52dfe6860f0cc62c145d155916bf381fc9` | `sha256:56f051b1b1b6f9f858ea5d21b7933b64af81c22bee2c417a3f8b4466220e37e6` |
| 1 | `sha256:7fb81337ba088a6bf0bbce71b22a5881f812a21af9ac1d6deea9533a8e9eed92` | `sha256:8506935b369bd4f0d5d73495ded9a2fcb52bbe2f310ea093818e5d3d5366ae38` |
| 2 | `sha256:9bd97e3d77de969ee0788aaac31b2888fd4c6a3d893ac5fc544ca85363927935` | `sha256:b969a49ec091157c686a3bc3f52816b6aa910e495af0c92780a321ea5fbd5324` |
| 3 | `sha256:d592c83cc04106532adf7d8d410347062ac1b80fc1b6981deca414b5335efff4` | `sha256:c9f0be4dccfd8fdcec80a3edce1ad217604fa09afee0f14d13a2839fb97eed9f` |

The rank-local builds are not one distributable image, and their cache-enabled
result does not qualify the cache-disabled profile. A publication must build
once, distribute one derived image ID, and qualify the selected mode.

Publish only one build distributed unchanged to all ranks. Its receipt must
include:

- the derived registry manifest digest and local image ID, and the parent
  registry manifest digest and resolved local parent image ID;
- the SparkCache `deploy/glm53_flash/Containerfile`, build arguments, exact
  build command, Docker/BuildKit versions, platform, size, and timestamp;
- every model, vLLM, B12X, NCCL, SparkCache, patch, preimage, lease-contract,
  and source identity in the provenance section;
- inherited parent content: vLLM, B12X, CUDA/toolchain, GLM/DFlash runtime
  support, and patched NCCL;
- SparkCache-owned changes: connector source, `glm53-flash-hybrid`, narrow VMM
  exemption, lease verifier, and OCI source/profile labels. The derivative
  does not alter or include model weights and does not replace NCCL;
- the tested TP4/DCP1 cache-disabled configuration, all build/validation
  commands, CPU pass/skip counts, attestation output, all-rank launch command,
  semantic and health results, and an explicit `implemented` status;
- unsupported configurations: any untested image/checkpoint/topology/geometry,
  MTP, DFlash depth other than seven, streaming snapshots, native direct
  restore, throughput or soak claims, and commercial DFlash use without an
  applicable license; and
- FujitsuPolycom support links:
  [SparkRing issues](https://github.com/FujitsuPolycom/sparkring/issues) and
  [SparkCache issues](https://github.com/FujitsuPolycom/sparkcache/issues).

The SparkRing repository validation for this profile change is Ruff passed and
1,877 CPU tests passed with nine skips. Rerun and record those commands plus
the SparkCache repository's CPU suites for a published image. CPU results do
not create a cache-disabled GPU/RDMA qualification.

Minimal announcement template:

> **FujitsuPolycom community image — GLM-5.3 Flash TP4/DCP1 with BF16
> DFlash2; external caching disabled**
> Image `<repository>@sha256:<manifest-digest>` / local ID
> `sha256:<derived-image-id>`; parent
> `<parent-repository>@sha256:<parent-manifest-digest>` / local ID
> `sha256:<parent-image-id>`. Build/source contract:
> `runtime/glm53-flash/pins.json`. Status: implemented; no standalone live
> qualification. Community derivative, not an official upstream image.
> Support: `https://github.com/FujitsuPolycom/sparkring/issues` and
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
