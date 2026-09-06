# GLM-5.3 Flash NVFP4-Spark on two DGX Sparks

**Research-only — testing in progress.** The packaged image is experimental;
the source deployment's tests do not qualify this child on every profile.
This profile records a two-node deployment without replacing the existing
four-node GLM recommendation. The model is GLM-5.3 Flash, not GLM-5.2.

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

| Setting | Value |
|---|---|
| Parallel layout | TP2, DCP1, two nodes, one GPU/rank |
| Speculation | Native adaptive MTP3, Humming draft MoE |
| KV pin | 5 GiB/rank; source deployment reported 698,452 logical tokens |
| Per-request context limit | 524,288, not requalified at that length after final changes |
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
cp "$profile/runtime.env.example" /path/to/private/runtime.env
# Edit all angle-bracket placeholders and verify NCCL_IB_GID_INDEX.
```

Install the guard on both nodes **only if one is not already configured**;
inspect any existing service instead of overwriting it blindly:

```bash
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
  --model-dir "$model_dir" --cache-dir "$cache_dir" --env-file /path/to/private/runtime.env
# Repeat the same command with --execute after reviewing its output.
```

The launcher creates a new named container and starts it. It refuses to
overwrite an existing container or launch alongside another GPU container.
It never stops production automatically. Loading can take15 minutes or more.

Mount the checkpoint read-only at /models/target and a per-node writable cache
directory at /cache/jit. The child packages attention.py and native libraries,
so it should not need the source deployment's separate code/library mounts.
Do not mount an older host library over the packaged versions. Substitute
MASTER_ADDR and NODE_RANK in profile.json; rank1 additionally uses --headless.
The JSON contains exact vLLM argv entries, not a shell-expanded command.

Use host networking and IPC, GPU access, /dev/infiniband access, unlimited
memlock and IPC_LOCK capability as required by the source deployment. Keep
the two ranks on the exact same image digest. Start rank1 before rank0.
Configure a host-memory guard before loading: the tested floor was4GiB,
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

After startup, query /health and /v1/models on the API port8016, then send a
short completion to model glm-5.3-flash-spark-pr646. Verify actual engine
occupancy for concurrency tests. Test cold prefill, cache capture and restore
separately. A healthy API does not prove multimodal or cache correctness.

## Evidence and limits

Conditions: source deployment with this profile, not the newly packaged child.
Measurement: three short C1 requests, one C8/2K cohort, and a cold C1/128K
codeword/capture request. Result: correct codewords, observed C8 overlap,
successful approximately914MiB/rank capture commits, no guard stop in these
bounded final checks. Conclusion: a usable experimental baseline, not broad
production qualification. Larger graphs/batches cost transient memory.

Earlier C8 attempts at9GiB and7GiB KV tripped memory protection during video
warmup and a128K capture respectively. Five GiB was selected for headroom.
Do not lower the guard to make an unqualified allocation appear to fit.
Some synthetic filler requests repeated follow-on prose. Chat responses
also exposed reasoning in content despite a no-thinking request; strict
JSON-only formatting and reasoning separation are not qualified.

DCP2 and the newer merged PR669 runtime are separate investigations. Changing
DCP requires different cache identity and correctness qualification. Do not
reuse the DCP1 cache namespace by manually forcing incompatible identities.

Preserve the previous image and launch contract for rollback. Stop both ranks
before replacement; never run competing full model stacks on these nodes.
