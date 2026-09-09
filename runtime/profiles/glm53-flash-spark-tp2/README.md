# GLM-5.3 Flash NVFP4-Spark on two Sparks

Status: **implemented**. This profile selects NVFP4-Spark with 8.75 GiB FP8
KV per node, static MTP3, and both PCI functions of one physical DAC. A
separate source deployment completed startup and one arithmetic smoke check.
The guarded shared-image configuration still requires serving qualification.

Select `glm53-flash-spark-tp2-mtp3` and the checkpoint
`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` at full revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`. The [profile](profile.json)
owns its arguments and environment; the [dependency matrix](dependencies.json)
records the common-source requirements and reference-trial limits.

| Setting | Value |
|---|---|
| Nodes / parallelism | Two GB10 nodes, one GPU each; TP2/DCP1 |
| Model / speculation | NVFP4-Spark; static MTP3, Humming draft MoE, B12X draft attention |
| Loader | B12X; `--model-loader-extra-config '{"allocation":"managed"}'` |
| KV allocation | FP8; 8.75 GiB (9,395,240,960 bytes) per node |
| Reference KV pool estimate | 1,050,118 tokens; allocator estimate, not demonstrated workload capacity |
| Maximum request context | 262,144 tokens |
| Scheduler | Eight sequences, 8,192 batched tokens, prefill interval 8 |
| Prefill | Token-sharded mHC and recurrent-checkpoint coalescing; sequential KDA projection |
| Graphs | `FULL_AND_PIECEWISE`, mode 0; `[1,2,4,8,12,16,20,24,28,32]` |
| Transport | HCA indices 0/2; two RoCEnante paths; 16 MiB all-reduce and all-gather input-shard limits |
| NCCL | Verified 2.30.7; eight channels; `=rocep1s0f0,roceP2p1s0f0` |
| Multimodal / SparkCache | Four images, zero videos; SparkCache disabled |
| Lifecycle | Active 2 GiB host-memory guard; manual create/start; Docker restart `no` |

The approximately one-million-token number describes the reference allocator's
KV pool, shared among requests. It does not raise the per-request context
limit or establish that eight long requests fit. The shared image's different
sources and runtime allocations may change that estimate.

## Prepare the image and checkpoint

Use the same source revision for this checkout, its prepared image context,
and its receipt. The [shared-image recipe](../../sparkring/source_image/README.md)
uses public source bases and the repository's locked native archive. With
the declared parent image present on an ARM64 Docker build host, download the
native archive as documented there and run:

```bash
python3 runtime/sparkring/source_image/prepare_image.py \
  --output /tmp/sparkring-glm-image-context \
  --source-cache /tmp/sparkring-glm-image-sources \
  --native-files /tmp/native-runtime-files-20260908.tar
docker build --platform linux/arm64 --network none \
  -t sparkring-glm53-source /tmp/sparkring-glm-image-context
SPARKRING_LOCAL_IMAGE_ID=$(docker image inspect --format '{{.Id}}' sparkring-glm53-source)
python3 runtime/sparkring/source_image/verify_image.py \
  --image "$SPARKRING_LOCAL_IMAGE_ID" \
  --context /tmp/sparkring-glm-image-context \
  --profile glm53-flash-spark-tp2-mtp3 \
  --output /srv/config/glm53-tp2-source-receipt.json
```

The context and source-cache directories must initially be absent. For a
verified existing source cache, add `--reuse-source-cache` and select a fresh
context directory. Create `/srv/config` before writing the receipt. The CPU
verifier checks installed packages, native files, this profile's hash, and
its transport bundle before producing the receipt; it does not load a model.

A published shared image is usable when its source lock and TP2 profile hash
match this checkout. Pull its immutable digest, obtain its local config ID
with `docker image inspect`, and run the same verifier against the matching
prepared context. An image with an older TP2 profile cannot pass this source
lock. Do not relabel an old receipt or infer compatibility from a tag name.

Download the exact checkpoint on both serving nodes, or reuse a verified copy:

```bash
hf download local-inference-lab/GLM-5.3-Flash-NVFP4-Spark \
  --revision df116c4fb16b1d37ae43d2cfd624de26ffbc832e \
  --local-dir /srv/models/GLM-5.3-Flash-NVFP4-Spark/df116c4
mkdir -p /srv/cache/glm53-spark-tp2
```

The launcher checks for `config.json` but does not authenticate shard
contents. Make the same image and receipt available on both nodes. The
writable directory stores compilation artifacts separated by profile hash
and rank; it is not an external prompt/KV cache.

## Inspect, create, and start

Copy [runtime.env.example](runtime.env.example) to a private file for each
rank. Replace its three placeholders with that node's fabric address and
socket interfaces. It accepts only those assignments; model, graph,
transport, memory-guard, and cache settings come from the profile.

Confirm this hardware inventory maps to the intended physical cage:
`rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1`. The reciprocal peer maps
are rank 0 `1=0/2` and rank 1 `0=0/2`; both selected functions belong to p0.
Install the host memory-guard service and apply
[memory-guard.conf](memory-guard.conf) as its systemd drop-in. Both `create`
and `start` require the active guard's effective 2 GiB floor. The launcher
does not install services or configure the network.

Run from the same checkout on each node, selecting its rank and replacing
`rank0.example` with rank 0's resolved fabric address:

```bash
RANK=1
python3 runtime/profiles/glm53-flash-spark-tp2/launch.py plan \
  --rank "$RANK" --master rank0.example \
  --model-dir /srv/models/GLM-5.3-Flash-NVFP4-Spark/df116c4 \
  --cache-dir /srv/cache/glm53-spark-tp2 \
  --env-file "/srv/config/glm53-rank${RANK}.env" \
  --image "$SPARKRING_LOCAL_IMAGE_ID"
```

Inspect the printed image/model paths, TP2/DCP1 settings, KV bytes, graph list,
and HCA map. Use `create` instead of `plan`, with the same arguments plus
`--runtime-receipt /srv/config/glm53-tp2-source-receipt.json`, to create a
stopped container. It refuses an existing name or another running GPU
container. Use `start` with the same arguments and receipt, starting rank 1
before rank 0. The launcher verifies the stopped container's complete plan.
Stop an existing serving pair explicitly before switching settings;
containers are not replaced automatically.

The common source verifier dispatches the profile to its verified TP2
transport entrypoint. `PYTHONPATH` is empty; the installed selector verifies
the TP2 bundle before vLLM imports, including spawned workers. The launcher
disables the inherited TP4 Docker healthcheck because this entrypoint does
not create its readiness marker. It provides no TP4 startup-admission or
sampling-warmup wrapper. Check API health and run a semantic smoke request
manually before admitting traffic:

```bash
curl --fail http://rank0.example:8000/health
curl --fail http://rank0.example:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-NVFP4-Spark","messages":[{"role":"user","content":"What is 17 + 25? End with FINAL=42."}],"temperature":1,"max_tokens":256}'
```

Rank 0 serves `GLM-5.3-Flash-NVFP4-Spark` on port 8000. It binds all interfaces
without API authentication; use a trusted network or authenticated gateway.
There is no automatic boot start or restart after a guard stop.

## Reference evidence and limits

**Conditions.** The separate source trial used this checkpoint and request
configuration on two GB10 nodes, adaptive TP2 transport, a loader with
hardcoded managed allocation, and NCCL 2.30.4. Both memory guards were stopped.
The shared image uses its locked common vLLM/B12X sources, explicit managed
allocation, NCCL 2.30.7, and a required active 2 GiB guard.

**Measurement and result.** Both ranks started, and one arithmetic response
returned `17 + 25 = 42` followed by `FINAL=42`. Rank 0 reported 35.61/2.33
seconds for target/draft loading. Graph capture reported 1.68/1.65 GiB across
ranks; the allocator estimated 1,050,118 KV tokens. Available host memory
after the response was 8,184/9,025 MiB. These are single observations.

**Conclusion and limits.** The reference establishes startup and one arithmetic
answer. It does not qualify the guarded shared image, throughput, sustained
memory stability, full-context/concurrent capacity, or multimodal accuracy.
All indexed model shards existed and config/index hashes matched; full shard
hashes were not recomputed during the checkpoint swap. Video remains disabled;
the separate [video-color issue](https://github.com/FujitsuPolycom/sparkring/issues/229)
is unresolved. Historical image measurements retain their own
[record](../../../performance/records/glm53-flash/tp2-public-image-install-20260906.md).
