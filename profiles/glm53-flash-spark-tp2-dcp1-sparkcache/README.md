# GLM-5.3-Flash on two Sparks

Run the [NVFP4-Spark checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark)
on two GB10 Sparks with MTP3 and DCP1. Context defaults to 1M tokens.
SparkCache is optional. These commands select the published **R37** image with
SparkCache enabled and 7.5 GiB KV per rank. Status: **Experimental**. R37 TP2
launch plans are software-checked; the R37 hardware record covers TP4, not TP2.
Use the [R35 fallback](#r35-fallback) for the composition with bounded TP2 tests.

Run commands in Bash on each Spark from the repository root. Use the same
checkout revision on both hosts; record it with `git rev-parse HEAD`.
The catalog's recorded defaults remain R33; these commands select R37
explicitly through an image receipt. An R33 fallback is provided below.

## 1. Prepare the pair

Both hosts need Docker with NVIDIA Container Toolkit, Python 3, the Hugging
Face CLI, matching model files, and working direct-link RoCE. Check the
[host prerequisites](../../docs/operations/prerequisites.md).
Keep management access independent of the data cable.

**Check the selected transport after startup.** The launcher currently lists
all four RDMA functions. With only one physical DAC connected, an inactive
function can prevent RoCEnante initialization and leave the engine serving
through NCCL instead. [Issue #268](https://github.com/FujitsuPolycom/sparkring/issues/268)
tracks this problem; [PR #270](https://github.com/FujitsuPolycom/sparkring/pull/270)
proposes restricting the opened devices to the selected cable. That change is
not included here. A healthy API alone does not prove RoCEnante is active.

## 2. Download the image and model

On both Sparks:

```bash
IMAGE_REF='ghcr.io/fujitsupolycom/sparkring@sha256:f5a7e01c6112c8ef85a51b24bfacfd3934ee9cfff06b7e8c72abcf5d90b50270'
RELEASE=lil-r37-glm-spark
MODEL_DIR=/srv/models/GLM-5.3-Flash-NVFP4-Spark/df116c4
CACHE_DIR=/srv/cache/glm53-r37-tp2

docker pull --platform linux/arm64 "$IMAGE_REF"
SPARKRING_IMAGE=$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")
RECORD=$(mktemp -d "$HOME/sparkring-r37-receipt.XXXXXX")
docker run --rm --network none --pull never --entrypoint cat "$SPARKRING_IMAGE" \
  /opt/sparkring/receipts/candidate-installed.json > "$RECORD/installed.json"
docker run --rm --network none --pull never "$SPARKRING_IMAGE" verify \
  > "$RECORD/verification.json"
python3 runtime/common/candidate.py --composition "$RELEASE" \
  --image-id "$SPARKRING_IMAGE" --image-reference "$IMAGE_REF" \
  --installed-receipt "$RECORD/installed.json" \
  --verification "$RECORD/verification.json" --output "$RECORD/image.json"
SPARKRING_RECEIPT="$RECORD/image.json"

hf download local-inference-lab/GLM-5.3-Flash-NVFP4-Spark \
  --revision df116c4fb16b1d37ae43d2cfd624de26ffbc832e \
  --local-dir "$MODEL_DIR"
mkdir -p "$CACHE_DIR"
```

Use directories your account can write, or create them with appropriate
ownership first. A verified existing model directory can be reused. The
launcher checks for `config.json`; it does not authenticate model shards.
Keep the image receipt above distinct from `publication.json`, which is not
accepted as a launch receipt.

## 3. Install the memory guard

During a stopped-serving maintenance window, install the guard on both hosts:

```bash
sudo install -D -m 0755 runtime/sparkring/memory_guard.py /usr/local/libexec/sparkring-memory-guard
sudo install -m 0644 runtime/sparkring/sparkring-memory-guard.service /etc/systemd/system/
sudo install -D -m 0644 runtime/profiles/glm53-flash-spark-tp2/memory-guard.conf \
  /etc/systemd/system/sparkring-memory-guard.service.d/tp2.conf
sudo systemctl daemon-reload
sudo systemctl enable --now sparkring-memory-guard.service
systemctl is-active sparkring-memory-guard.service
systemctl show sparkring-memory-guard.service --property=ExecStart --value
```

The effective command must contain `--available-floor-bytes 2147483648`.
If a guard is already installed, inspect its unit and drop-ins before changing
it; `enable --now` does not reload a process that is already running.
The guard can stop opted-in model containers when host memory is exhausted.

## 4. Set private rank inputs

On each host, set `RANK` to 0 or 1 and `MASTER` to rank 0's reachable address.
Create a rank-local environment file using the template:

```bash
RANK=0
MASTER=rank0.example
ENV_FILE="$PWD/.sparkring/glm53-rank${RANK}.env"
mkdir -p "$PWD/.sparkring"
cp -n runtime/profiles/glm53-flash-spark-tp2/runtime.env.example "$ENV_FILE"
```

Edit the file and replace all three placeholders: `VLLM_HOST_IP`,
`NCCL_SOCKET_IFNAME`, and `GLOO_SOCKET_IFNAME`. Use this rank's address and the
interfaces for the configured communication path. Replace `rank0.example`
above with the real rendezvous address. The two ranks must use different
rank numbers and their own site files.

## 5. Plan, create and start

Choose the same cache mode on both hosts and define the command once:

```bash
CACHE_ARGS=(--sparkcache)
launch_rank() {
  python3 runtime/common/tp2.py "$1" \
    --rank "$RANK" --master "$MASTER" \
    --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" \
    --env-file "$ENV_FILE" --image "$SPARKRING_IMAGE" \
    --runtime-receipt "$SPARKRING_RECEIPT" "${CACHE_ARGS[@]}"
}
launch_rank plan
```

Inspect both plans for the resolved R37 image ID, 1,048,576 context tokens and
no activation blockers. Cache on selects `tp2-dcp1-sparkcache` and 8,053,063,680
KV bytes per rank; cache off selects `tp2-dcp1` and 9,395,240,960 KV bytes.
Stop any existing GPU workload explicitly before proceeding.

Run `launch_rank create` on each host to create stopped containers. Then run
`launch_rank start` on rank 1, followed by rank 0. These actions validate the
receipt, memory guard and stopped-container settings; they do not replace
existing containers automatically.

## 6. Check serving

On each host, inspect the selected backend and engine logs:

```bash
PROFILE=tp2-dcp1
if ((${#CACHE_ARGS[@]})); then PROFILE=tp2-dcp1-sparkcache; fi
docker logs --tail 150 "sparkring-${RELEASE}-${PROFILE}-r${RANK}"
curl --fail "http://${MASTER}:8000/health"
curl --fail "http://${MASTER}:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-NVFP4-Spark","messages":[{"role":"user","content":"What is 17 + 25? End with FINAL=42."}],"reasoning_effort":"low","max_tokens":1024}'
```

Require the expected answer and healthy logs on both ranks. Check for a
RoCEnante initialization failure or NCCL fallback, as described in issue #268.
This launcher binds the API without authentication; keep it on a trusted
network or behind an authenticated gateway.

## SparkCache off

Set `CACHE_ARGS=()` before planning and creating containers; retain the selected
image receipt. This selects `tp2-dcp1`, with InstantTensor loading, coalescing
disabled and 8.75 GiB KV per rank. Both modes use MTP3, mHC, DCP1 and 1M
configured context. Changing the array does not reconfigure a running container:
stop the previous workload before creating the other mode.

## R35 fallback

Follow [R35 image recording](../../docs/operations/r35-local-launch.md#record-the-image),
then set these variables on each host and reuse steps 3–6 above:

```bash
SPARKRING_IMAGE=$IMAGE
SPARKRING_RECEIPT="$RECORD/image.json"
RELEASE=r35
CACHE_DIR=/srv/cache/glm53-r35-tp2
mkdir -p "$CACHE_DIR"
CACHE_ARGS=(--sparkcache)
```

For cache off, use `CACHE_ARGS=()`. The [R35 TP2 record](../../performance/records/glm53-flash/r35-tp2-sparkcache.md)
contains the bounded cache-on hardware evidence; it does not establish long-duration
stability or a completed 1M request.

## R33 fallback

During a stopped-serving maintenance window, use these inputs instead of the
image-recording commands above. Use a separate cache directory and retain the
same model, memory guard and private rank inputs:

```bash
SPARKRING_IMAGE='ghcr.io/fujitsupolycom/sparkring@sha256:1328a4f6f483014021a66a757012793629bd054d28d0fe4d5e581fa4aed776ef'
SPARKRING_RECEIPT="$PWD/runtime/sparkring/jovian-r33/public-image-receipt.json"
RELEASE=r33
CACHE_DIR=/srv/cache/glm53-r33-tp2
docker pull "$SPARKRING_IMAGE"
python3 runtime/sparkring/jovian-r33/profiles/verify_profile.py image \
  --receipt "$SPARKRING_RECEIPT"
mkdir -p "$CACHE_DIR"
CACHE_ARGS=(--r33-sparkcache)
```

Reuse `launch_rank` from step 5. For cache off, set `CACHE_ARGS=()`.
The receipt selects R33; log names are `sparkring-r33-${PROFILE}-r${RANK}`,
with `PROFILE` selected as in step 6. API port and model alias are unchanged.
The [R33 evidence](../../performance/records/glm53-flash/r33-image020-tp2-sparkcache-20260911.md)
applies to that composition.

## Evidence and other builds

The [R37 record](../../performance/records/glm53-flash/r37-tp4-source-upgrade.md)
covers TP4 and does not qualify this TP2 selection. The R35 TP2 cache-on record
reported a 1,081,922-token pool; do not treat it as an R37 measurement or transfer
its cache-on evidence to cache off.
The [retained source-image guide](../../runtime/profiles/glm53-flash-spark-tp2/README.md)
uses different settings and is only for reproducing that separate composition.
