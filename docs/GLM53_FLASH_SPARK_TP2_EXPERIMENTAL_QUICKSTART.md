# GLM-5.3 Flash NVFP4-Spark on two DGX Sparks

**Status: research-only.** This guide runs GLM-5.3 Flash NVFP4-Spark across
two DGX Spark computers, with native multi-token prediction and persistent
key/value cache storage provided by SparkCache. Each profile has its own
qualification boundary; this guide does not qualify the four-node profiles.

## Requirements and pinned inputs

Two ARM64 DGX Spark/GB10 nodes with a verified direct RoCE data path, Docker
GPU support, sufficient disk for checkpoint and runtime, and independently
reachable management access. Inspect per-node HCA names and GID indices;
do not assume interface names or GIDs survive hardware/OS changes unchanged.

- [Runtime package manifest](../runtime/sparkring/manifest.json).
- [Model identity and exact vLLM arguments](../runtime/profiles/glm53-flash-spark-tp2/profile.json).
- [Site-adjustable environment template](../runtime/profiles/glm53-flash-spark-tp2/runtime.env.example).
- Checkpoint: local-inference-lab/GLM-5.3-Flash-NVFP4-Spark,
  revision df116c4fb16b1d37ae43d2cfd624de26ffbc832e.

The matching SparkCache source is [public](https://github.com/FujitsuPolycom/sparkcache/commit/360f97dbd00b62b06fc5e7839b87514c12f2908f).
The immutable pull reference is recorded in
[publication.json](../runtime/sparkring/publication.json). Read the
[distribution terms and audit scope](../runtime/sparkring/DISTRIBUTION.md).
The container includes NVIDIA-licensed components; Apache terms on the
SparkRing repository do not relicense the whole image.

## Profile settings

The [reference runtime](../performance/records/glm53-flash/tp2-reference-runtime-20260906.md)
is a separately identified Docker image whose measured settings inform this
profile. Its measurements do not transfer automatically to the published image.

| Setting | Value |
|---|---|
| Parallel layout | TP2, DCP1, two nodes, one GPU/rank |
| Speculation | Native adaptive MTP3, Humming draft MoE |
| KV pin | 5 GiB per rank; reference-runtime capacity: 698,452 logical tokens |
| Per-request context limit | 524,288 tokens; this limit is not a completed-request qualification |
| Concurrent sequences / batch | 8 / 8192 |
| Prefill schedule interval | 2 |
| Target / recurrent / prefix-match blocks | 2048 / 256 / 256 |
| Graph mode | FULL_AND_PIECEWISE |
| Graph token sizes | 1,2,3,4,5,6,7,8,9,10,12,14,15,16,18,20,21,24,28,32 |
| Image / video prompt limits | 1 / 1 |
| KV dtype / weights | FP8 / modelopt_mixed |
| Loader | safetensors |
| SparkCache | Native capture/restore, one pending restore, one 3.5 GiB managed capture slot |
| Cache disk limit | 12 GiB/rank, low watermark 8 GiB/rank |

## Launch contract

The installation scripts and profile are distributed through
[pull request 228](https://github.com/FujitsuPolycom/sparkring/pull/228).
Obtain its head revision on each node and record the resolved commit:

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git
cd sparkring
git fetch origin pull/228/head
git checkout --detach FETCH_HEAD
git rev-parse HEAD
```

Run from the repository root on each node. Set site-specific paths and
addresses locally; do not commit the resolved runtime.env file.

```bash
profile=runtime/profiles/glm53-flash-spark-tp2
image=$(python3 -c 'import json; print(json.load(open("runtime/sparkring/publication.json"))["registry_digest"])')
docker pull "$image"
docker run --rm --network none --entrypoint python3 "$image" \
  /opt/sparkring/bin/verify-runtime-package.py

# Choose paths on a volume with sufficient free space.
model_dir=/absolute/path/to/glm53-flash-spark
cache_dir=/absolute/path/to/per-node-cache
mkdir -p "$model_dir" "$cache_dir"
hf download local-inference-lab/GLM-5.3-Flash-NVFP4-Spark \
  --revision df116c4fb16b1d37ae43d2cfd624de26ffbc832e --local-dir "$model_dir"
private_dir="$HOME/.config/sparkring/glm53-tp2"
mkdir -p "$private_dir"
cp "$profile/runtime.env.example" "$private_dir/runtime.env"
# Edit all angle-bracket placeholders and verify NCCL_IB_GID_INDEX.
```

If the exact checkpoint is already present, point model_dir to its existing
directory and skip hf download; do not copy or download the weights again.
For a clean-install test, use a new writable cache_dir rather than an old JIT
or SparkCache directory. Docker reuses existing layers when pulling the digest.
Install the Hugging Face CLI first if hf is not available. Site discovery can
start with ip -br addr, ibv_devinfo -l, and show_gids where installed. A verified
RoCE fabric is a prerequisite; this guide does not provision the physical links.

Install the guard on both nodes **only if one is not already configured**;
inspect any existing service instead of overwriting it blindly:

```bash
sudo install -d /usr/local/libexec
sudo install -m 0755 runtime/sparkring/memory_guard.py /usr/local/libexec/sparkring-memory-guard
sudo install -m 0644 runtime/sparkring/sparkring-memory-guard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sparkring-memory-guard.service
systemctl status sparkring-memory-guard.service --no-pager
```

Render and inspect each plan before executing. On rank1 first, then rank0:

```bash
rank=1 # use 0 on the API node
master=REPLACE_WITH_RANK0_ADDRESS
python3 "$profile/launch.py" --rank "$rank" --master "$master" \
  --model-dir "$model_dir" --cache-dir "$cache_dir" --env-file "$private_dir/runtime.env"
# Repeat the same command with --execute after reviewing its output.
```

The launcher creates a new named container and starts it. It refuses to
overwrite an existing container or launch alongside another GPU container.
It never stops another model deployment automatically. Loading can take
15 minutes or more, followed by kernel compilation for an empty cache.

Mount the checkpoint read-only at /models/target and a per-node writable cache
directory at /cache/jit. The image includes attention.py and native cache
libraries, so no separate host-code or native-library mounts are required.
Do not mount an older host library over the packaged versions. The launcher
substitutes MASTER_ADDR and NODE_RANK from its arguments and adds --headless
on rank1; do not edit profile.json manually. The JSON contains exact vLLM argv
entries, not a shell-expanded command.

Use host networking and IPC, GPU access, /dev/infiniband access, unlimited
memlock and IPC_LOCK capability as specified by the launcher. Keep
the two ranks on the exact same image digest. Start rank1 before rank0.
Configure a host-memory guard before loading: the profile uses a 4 GiB floor,
polled each second with two consecutive low samples. Label only model
containers for that guard. Do not enable automatic container restarts that
would defeat a guard stop. A guard is not proof against every host OOM.

## Verify

```bash
curl --fail http://RANK0_ADDRESS:8016/health
curl --fail http://RANK0_ADDRESS:8016/v1/models
curl --fail http://RANK0_ADDRESS:8016/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash-spark-pr646","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
docker logs --tail 60 -f sparkring-glm53-tp2-r0
```

After startup, query /health and /v1/models on API port 8016, then send a
short completion to model glm-5.3-flash-spark-pr646. Verify actual engine
occupancy for concurrency tests. Test cold prefill, cache capture and restore
separately. A healthy API does not prove multimodal or cache correctness.

The included functional smoke client uses Python's standard library:

```bash
python3 "$profile/smoke.py" --phase text
python3 "$profile/smoke.py" --phase image
python3 "$profile/smoke.py" --phase long-prompt
```

The text check submits eight distinct codeword requests. The image check
generates a solid-blue PNG. For the video check, create a solid-blue MP4 with
FFmpeg, then submit that file:

```bash
ffmpeg -f lavfi -i 'color=c=blue:s=224x224:r=8:d=1' \
  -c:v libx264 -pix_fmt yuv420p blue-smoke.mp4
python3 "$profile/smoke.py" --phase video --video-file blue-smoke.mp4
```

A correct long-prompt response proves only answer correctness. To verify
persistence, observe successful capture and commit logs on both ranks, stop
both model containers, and start rank1 then rank0 with the same cache mounts.
Repeat the identical long-prompt request after engine startup. Confirm native
restore completion on both ranks and nonzero cached prompt tokens; do not
infer an external-cache hit from answer correctness or response speed alone.

## Evidence and limits

The [reference-runtime evidence record](../performance/records/glm53-flash/tp2-reference-runtime-20260906.md)
identifies the tested image, workload, raw observations and limits. It covers
three single-request decode checks, eight overlapping requests with 2,048
input tokens each, and one request with 131,072 input tokens and persistent
cache capture. These measurements are not qualification of arbitrary
concurrency/context combinations or other runtime images.

Graph buffers, prefill workspaces and persistent-cache capture require memory
outside the KV pin. Keep the guard active and verify headroom under the intended
workload. A configured context limit does not guarantee enough working memory
for every request admitted at that length.

Synthetic repeated filler can produce repeated follow-on prose. Chat output
has also included reasoning in the content field despite a no-thinking
request; strict JSON-only formatting and reasoning separation are unqualified.

Only DCP1 is specified here. A DCP change requires a distinct cache identity
and separate correctness qualification. Do not force DCP1 cache identities
onto a different shard layout.

Preserve the previous image and launch contract for rollback. Stop both ranks
before replacement; never run competing full model stacks on these nodes.
