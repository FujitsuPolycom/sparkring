# Run the public GLM-5.3 Jovian Judgement r7-compatible GB10 images

Status: **implemented and TP4 smoke-verified**, not generally qualified. This
guide selects either a cache-disabled base image or its SparkCache composition.
Both are immutable public Linux ARM64 artifacts.

The launcher defaults to `MAX_MODEL_LEN=524288` and
`MAX_NUM_BATCHED_TOKENS=8192`. The SparkCache variant also defaults to
`SPARKCACHE_MAX_SPAN_TOKENS=524288`. These operator limits are implemented but
unqualified. The bounded C4 smoke used 262,144, 4,096, and a 262,144-token
SparkCache span, so the launcher labels the 512K/8K configuration accordingly.

| Variant | Immutable image | Local image ID after pull |
|---|---|---|
| `base` | `ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:11922064b342de1fc98f0ef85e6648843c8fa7eb3e4f4353c6ad82d6e457dde0` | `sha256:8cff7a250f16bfb89df23d29f9233dbb1c700a780dcec86a64c535a71aee88be` |
| `sparkcache` | `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5` | `sha256:6af83baabb239db6b05e379401daf93c8f51694f81483c2781f6014c30e31db4` |

Local Inference Lab's
[Jovian Judgement vLLM branch](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement)
is the primary source of GLM runtime performance and correctness. These images
use community r7 plus the connector seams bound by
[`FujitsuPolycom/vLLM@331573d`](https://github.com/FujitsuPolycom/vllm/commit/331573d20bd47e78327ed8d8b4d2e6d350bbb1ab).
[B12X at `6255090a`](https://github.com/local-inference-lab/b12x/commit/6255090a03b12c3f7d552102a02fac0b542fb8c9)
supplies the Blackwell kernels and backend integration.

The exact target is
[`GLM-5.3-Flash-NVFP4@520de24e`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/520de24eabf507659eaef7c70f14fd584527facc).
The exact external draft is
[`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410),
whose weights are BF16. It is not Local Inference Lab's separate
[MXFP8 DFlash2 checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8).

`331573d` is the active Python composition, not a claim that every native
extension was rebuilt from that commit. The retained compiled extensions come
from lower image layers reporting build commit
`3633d61c3c7b04bb4d598cadbdc342f3be40482d` and intermediate source label
`da4d7be6c97434f6942292ed8abbf4b32dc44355`; their complete shared-object SHA-256 set was verified
byte-identical across parent and child. Exact full identities and the immutable
manifest/config digests are in
[`artifacts.json`](../runtime/glm53-flash-jj-r7-gb10/artifacts.json).

The images inherit
`org.glm53.dflash2.checkpoint-revision=b6d33aa93fc1ac5b23a88251a1c0ce0bfe2ad17c`
and
`org.glm53.dflash2.mxfp8-quant-plumbing=v2` from a lower image layer. Those
labels record lineage and available plumbing, not the active mounted draft.
The launcher verifies the complete `dc77ff1c` BF16 draft config and weight
hashes before starting a container.

## Requirements

- Four directly connected NVIDIA GB10 systems with Docker and NVIDIA Container
  Toolkit.
- `git`, Python 3, `hf`, `ssh`, `rsync`, `sha256sum`, `zstd`, and `curl` on
  rank 0; `ssh`, `sha256sum`, `zstd`, and Docker on each follower. Install the
  Hugging Face CLI with `python3 -m pip install --user --upgrade huggingface_hub`
  if `hf` is absent.
- One GPU per rank, passwordless operator access, and the rank addresses,
  RoCE HCAs, GID, and routable interface for the four-rank service.
- Enough space on rank 0 to download the target and draft once, plus identical
  destination paths on the other ranks.
- A writable rank-local cache directory on every host. The base variant uses
  it for compilation data; the SparkCache variant also stores persistent roots.
- Python 3 on each host. The launcher uses it to encode JSON arguments.

Obtain the launcher and machine contracts on rank 0, then use the same checkout
revision on every rank:

```bash
git clone --branch codex/glm53-readme-quickstart-consolidation --single-branch \
  https://github.com/FujitsuPolycom/sparkring.git
sparkring_revision="$(git -C sparkring rev-parse HEAD)"
git -C sparkring checkout --detach "${sparkring_revision}"
cd sparkring
```

The branch is a review branch and may advance. Recording the resolved commit
before detaching prevents an unnoticed branch change during one deployment.

## Download once and fan out over the local fabric

Run the Hugging Face downloads once on rank 0. The example uses the same
absolute paths on every rank:

```bash
target_model='/srv/models/glm53-target'
draft_model='/srv/models/glm53-dflash2-bf16'
hf download local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --revision 520de24eabf507659eaef7c70f14fd584527facc \
  --local-dir "${target_model}"
hf download incoai/GLM-5.3-Flash-DFlash2 \
  --revision dc77ff1c99eeb2df044ee3d4f0094eb033fee410 \
  --local-dir "${draft_model}"
```

Fan those two immutable directories from rank 0 to the three followers. Set
the SSH targets to the direct or management addresses appropriate for the
site:

```bash
followers=(
  operator@rank1.example.net
  operator@rank2.example.net
  operator@rank3.example.net
)
for host in "${followers[@]}"; do
  ssh "${host}" mkdir -p "${target_model}" "${draft_model}"
  rsync -aH --partial --info=progress2 "${target_model}/" "${host}:${target_model}/"
  rsync -aH --partial --info=progress2 "${draft_model}/" "${host}:${draft_model}/"
done
```

Check the identity-bearing files on rank 0 and each follower. Every host must
print the same four hashes:

```bash
check_models() {
  sha256sum \
    "${target_model}/config.json" \
    "${target_model}/model.safetensors.index.json" \
    "${draft_model}/config.json" \
    "${draft_model}/model.safetensors"
}
check_models
for host in "${followers[@]}"; do
  ssh "${host}" \
    sha256sum \
    "${target_model}/config.json" \
    "${target_model}/model.safetensors.index.json" \
    "${draft_model}/config.json" \
    "${draft_model}/model.safetensors"
done
```

Expected hashes, in command order:

```text
676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996
0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb
c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573
b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b
```

## Pull once and fan out one immutable variant

Select one variant on rank 0, then pull its immutable digest once:

```bash
IMAGE_VARIANT='sparkcache' # or base
case "${IMAGE_VARIANT}" in
  base)
    image='ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:11922064b342de1fc98f0ef85e6648843c8fa7eb3e4f4353c6ad82d6e457dde0'
    expected_image_id='sha256:8cff7a250f16bfb89df23d29f9233dbb1c700a780dcec86a64c535a71aee88be'
    ;;
  sparkcache)
    image='ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5'
    expected_image_id='sha256:6af83baabb239db6b05e379401daf93c8f51694f81483c2781f6014c30e31db4'
    ;;
esac
docker pull "${image}"
test "$(docker image inspect --format '{{.Id}}' "${image}")" = "${expected_image_id}"
```

Create one compressed OCI archive on rank 0, then stream that same archive to
each follower over SSH. This requires enough temporary disk on rank 0 for the
compressed image and enough free Docker storage on every rank. It does not
create a second archive on the followers:

```bash
archive="${HOME}/${IMAGE_VARIANT}-glm53-jj-r7-gb10.oci.tar.zst"
docker image save "${image}" | zstd -T0 -3 -o "${archive}"
sha256sum "${archive}"

followers=(
  operator@rank1.example.net
  operator@rank2.example.net
  operator@rank3.example.net
)
for host in "${followers[@]}"; do
  ssh "${host}" 'zstd -d | docker image load' < "${archive}"
  actual="$(ssh "${host}" docker image inspect --format '{{.Id}}' "${image}")"
  test "${actual}" = "${expected_image_id}"
  printf '%s %s\n' "${host}" "${actual}"
done
```

If rank 0 cannot hold the compressed archive or direct SSH streaming is not
available, `docker pull "${image}"` on each rank is the registry-based
alternative. The launcher rejects a local image ID that differs from the
expected identity.

## Configure common operator settings

Copy the complete environment template once per rank:

```bash
cp runtime/glm53-flash-jj-r7-gb10/runtime.env.example \
  "$HOME/glm53-jj-r7-gb10.env"
${EDITOR:-vi} "$HOME/glm53-jj-r7-gb10.env"
```

Set `IMAGE_VARIANT=base` or `IMAGE_VARIANT=sparkcache`, then replace these five
site values:

- `HOST_IP` — this rank's routable address;
- `MASTER_ADDR` — rank 0's address, identical on all ranks;
- `TARGET_MODEL_HOST_PATH` — target checkpoint directory;
- `DFLASH_MODEL_HOST_PATH` — BF16 draft checkpoint directory;
- `CACHE_HOST_ROOT` — writable rank-local cache and compilation directory.

The same file exposes image/container names, API and rendezvous ports,
TP/DCP/PP topology, maximum model length, sequence and batched-token limits,
KV bytes/utilization/dtype, speculative depth, kernel backends, load format,
CUDA graphs, per-variant namespaces, SparkCache capacity/TTL/restore workers,
network interfaces, NCCL channels, and CPU/loader threads.

The default namespaces are distinct:

- base: `jj-r7-gb10-base-v1`;
- SparkCache: `jj-r7-gb10-page-tail-cow-v1`.

A serving configuration that differs from the bounded smoke receives
`org.sparkring.launch.status=implemented-unqualified-configuration` and a
comma-separated `org.sparkring.launch.modified-settings` label. Site addresses,
bind-mount paths, and container names do not alter status. A modified launch
may work, but it does not inherit the C4 smoke evidence.

## Start one rank

Rank 0:

```bash
bash runtime/glm53-flash-jj-r7-gb10/launch-rank.sh \
  0 "$HOME/glm53-jj-r7-gb10.env"
```

Run the same command with rank arguments `1`, `2`, and `3` on the other
systems. Start all ranks within the distributed rendezvous window.

The image entrypoint is `vllm serve`. The launcher passes `/models/target`
directly after the image reference; it does not append a second `serve` token.

Tail rank 0, replacing the variant if necessary:

```bash
docker logs --follow --tail 120 glm53-jj-r7-gb10-sparkcache-r0
```

Check API health and model identity:

```bash
api_endpoint='http://rank0.example.net:8015'
until curl --fail --silent "${api_endpoint}/health" >/dev/null; do sleep 5; done
curl --fail --silent "${api_endpoint}/v1/models"
```

For the SparkCache variant, complete one small inference after every full
process restart before asserting persistent hits. HTTP health alone does not
prove that worker manifest inventory has reached the scheduler:

```bash
curl --fail --silent "${api_endpoint}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash-nvfp4-dflash2-bf16-t7-jj-r7-gb10","prompt":"Reply OK.","max_tokens":1,"temperature":0}' \
  > scheduler-inventory-readiness.json
```

## Evidence boundary

The base image returned exact `red`, `blue`, `green`, and `black` outputs for
one C4 cohort. The SparkCache image returned the same exact outputs during
fresh publication and, after full process replacement plus one readiness
inference, externally restored all four stored roots. Client restore time was
0.561595–1.582937 seconds; worker cache-service time was 277–394 ms.

The physical-page delta and shared-base mechanisms are implemented and have
bounded exact TP4 evidence in the
[split-page record](../performance/records/glm53-flash/split-page-shared-base-c8-20260830.md).
The public-image C4 smoke used complete roots, so it makes no shared-base claim
and did not exercise a page delta.

Other models and topologies, embedded MTP with SparkCache, C16 serving, soak,
and fault injection are unqualified by this procedure. The exact source and
measurement identities are in
[`artifacts.json`](../runtime/glm53-flash-jj-r7-gb10/artifacts.json) and the
[`validation receipt`](../performance/receipts/glm53-flash/jj-r7-gb10-tp4-smoke-20260830/validation.json).
