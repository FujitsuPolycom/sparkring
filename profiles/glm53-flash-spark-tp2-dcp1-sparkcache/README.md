# GLM-5.3-Flash on two Sparks

Run the [NVFP4-Spark checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark)
on two GB10 Sparks with MTP3 and DCP1. Context defaults to 1M tokens.
SparkCache is optional; the commands below select the validated cache-enabled
configuration, with 7.5 GiB KV per rank and a recorded 1.1M-token pool.

Run commands in Bash on each Spark from the repository root. Use the same
checkout revision on both hosts; record it with `git rev-parse HEAD`.
Do not switch to an integration branch or rebuild the published image for this setup.

## 1. Prepare the pair

Both hosts need Docker with NVIDIA Container Toolkit, Python 3, the Hugging
Face CLI, matching model files, and working direct-link RoCE. Check the
[host prerequisites](../../docs/operations/prerequisites.md).
Keep management access independent of the data cable.

**Check the selected transport after startup.** The launcher currently lists
all four RDMA functions. With only one physical DAC connected, an inactive
function can prevent RoCEnante initialization and leave the engine serving
through NCCL instead. [Issue #268](https://github.com/FujitsuPolycom/sparkring/issues/268)
tracks this problem. A healthy API alone does not prove RoCEnante is active.

## 2. Download the image and model

On both Sparks:

```bash
SPARKRING_IMAGE='ghcr.io/fujitsupolycom/sparkring@sha256:1328a4f6f483014021a66a757012793629bd054d28d0fe4d5e581fa4aed776ef'
SPARKRING_RECEIPT="$PWD/runtime/sparkring/jovian-r33/public-image-receipt.json"
MODEL_DIR=/srv/models/GLM-5.3-Flash-NVFP4-Spark/df116c4
CACHE_DIR=/srv/cache/glm53-tp2

docker pull "$SPARKRING_IMAGE"
python3 runtime/sparkring/jovian-r33/profiles/verify_profile.py image \
  --receipt "$SPARKRING_RECEIPT"
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

Define the command once on each host:

```bash
launch_rank() {
  python3 runtime/profiles/glm53-flash-spark-tp2/launch.py "$1" \
    --rank "$RANK" --master "$MASTER" \
    --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" \
    --env-file "$ENV_FILE" --image "$SPARKRING_IMAGE" \
    --runtime-receipt "$SPARKRING_RECEIPT" --r33-sparkcache
}
launch_rank plan
```

Inspect both plans. They must select `tp2-dcp1-sparkcache`, the published image,
1,048,576 context tokens and 8,053,063,680 KV bytes per rank, without activation
blockers. Stop any existing GPU workload explicitly before proceeding.

Run `launch_rank create` on each host to create stopped containers. Then run
`launch_rank start` on rank 1, followed by rank 0. These actions validate the
receipt, memory guard and stopped-container settings; they do not replace
existing containers automatically.

## 6. Check serving

On each host, inspect the selected backend and engine logs:

```bash
docker logs --tail 150 "sparkring-r33-tp2-dcp1-sparkcache-r${RANK}"
curl --fail "http://${MASTER}:8000/health"
curl --fail "http://${MASTER}:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-NVFP4-Spark","messages":[{"role":"user","content":"What is 17 + 25? End with FINAL=42."}],"temperature":1,"max_tokens":256}'
```

Require the expected answer and healthy logs on both ranks. Check for a
RoCEnante initialization failure or NCCL fallback, as described in issue #268.
This launcher binds the API without authentication; keep it on a trusted
network or behind an authenticated gateway.

## SparkCache off

Remove `--r33-sparkcache` from `launch_rank` and repeat the plan/create/start
procedure during a stopped-serving maintenance window. Keep the R33 image
receipt. This selects `tp2-dcp1`, with InstantTensor loading, coalescing disabled
and 8.75 GiB KV per rank. It is not the older 256K source-image profile.

## Evidence and other builds

The [TP2 record](../../performance/records/glm53-flash/r33-image020-tp2-sparkcache-20260911.md)
documents text checks, cold starts and cache restore. Startup reported
1,081,922 total KV tokens; this is pool capacity, not a completed 1M request.
The [retained source-image guide](../../runtime/profiles/glm53-flash-spark-tp2/README.md)
uses different settings and is only for reproducing that separate composition.
