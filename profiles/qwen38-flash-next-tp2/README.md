# Qwen3.8-Flash-Next NVFP4 on two Sparks

Status: **Experimental**. The profile fixes **262K context, 16 sequences,
8,192 batched tokens and 24 GiB KV per rank**. Both ranks completed startup;
inference, C16 and full-context tests at these settings are deferred. The
separate 64K/C1 baseline does not qualify these defaults. Resolving the profile
starts no model or test.

Use [LIL Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)
revision `ada4da32a583a78aa47299f45a70603c950490b8` with the published R37
ARM64 image. This is distinct from Qwen3.8-27B EXL3.

| Setting | Default |
| --- | --- |
| Parallelism | TP2/DCP1; one direct cable, two PCI domains |
| Context / sequences / batch | 262,144 / 16 / 8,192 |
| KV | 25,769,803,776 bytes per rank (24 GiB), FP8 |
| Loader / speculation | Managed B12X target and MTP3 draft |
| PLE / vocabulary head | Device placement; BF16 target head |
| Cache | Native vLLM prefix caching; SparkCache disabled |
| Runner / convolution layout | V2 / DS |
| Media | One image, zero videos; media correctness untested |
| Context extension | None; no YaRN or HF overrides |

[config.json](config.json) owns the arguments and environment. Inspect catalog
defaults with `python3 scripts/profiles.py resolve qwen38-flash-next-tp2`.

## Image and checkpoint

Run Bash from the same repository revision on both Sparks. Complete the
[host prerequisites](../../docs/operations/prerequisites.md) first.

```bash
REPO=$PWD
IMAGE_REF=ghcr.io/fujitsupolycom/sparkring@sha256:f5a7e01c6112c8ef85a51b24bfacfd3934ee9cfff06b7e8c72abcf5d90b50270
docker pull --platform linux/arm64 "$IMAGE_REF"
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")
MODEL_DIR=/srv/models/Qwen3.8-Flash-Next-NVFP4/ada4da32
CACHE_DIR=/srv/cache/qwen38-flash-next-r37
```

Use writable parent directories and allow approximately 106 GB for the
checkpoint on each rank. Download only if a verified copy is not available:

```bash
hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
  --revision ada4da32a583a78aa47299f45a70603c950490b8 --local-dir "$MODEL_DIR"
mkdir -p "$CACHE_DIR"
(
  cd "$MODEL_DIR"
  sha256sum --check "$REPO/profiles/qwen38-flash-next-tp2/SHA256SUMS"
)
```

The checksum list covers all 34 weight shards, large runtime metadata and
`config.json`. Directory names are not identity proof. The launcher rechecks
config/index hashes but does not repeat the full shard scan. Model mounts are
read-only; the writable cache must be outside the model tree.

## Plan and create

Replace these examples with rank zero's reachable bootstrap address, this
rank's address and its corresponding Linux interface. Rank 1 needs its own
rank number, address and local paths.

```bash
RANK=0
MASTER_ADDR=192.0.2.10
HOST_IP=192.0.2.10
INTERFACE=eth0
launch_rank() {
  python3 runtime/common/qwen_flash_next.py "$1" \
    --profile profiles/qwen38-flash-next-tp2/config.json \
    --rank "$RANK" --master "$MASTER_ADDR" --host-ip "$HOST_IP" \
    --interface "$INTERFACE" --image "$IMAGE_ID" \
    --model "$MODEL_DIR" --cache "$CACHE_DIR"
}
launch_rank plan
```

The RDMA defaults select `rocep1s0f0` and `roceP2p1s0f0`, the two PCI-domain
views of cage p0, on both nodes. Confirm the cable uses that cage. This adapter
does not autodetect or configure networking. Its RoCEnante peer map uses selected
indices 0/1; the bootstrap interface is separate from that device list.

Inspect both plans. `plan` is offline; `check` verifies the image and exercises
CLI help only. `create` verifies the exact R37 image/payload and creates stopped
containers. It refuses an existing name and never stops another workload:

```bash
launch_rank create
```

Capacity overrides are not accepted. Keep KV pinned at 24 GiB until further
testing supports changing the profile.

## Controlled startup

During a test window, stop other GPU workloads explicitly and preserve their
containers/configuration for rollback. Start rank 1, then rank 0 on their hosts:

```bash
docker start qwen-flash-next-tp2-r1 # rank 1 host
docker start qwen-flash-next-tp2-r0 # rank 0 host
docker logs --follow --tail 100 "qwen-flash-next-tp2-r${RANK}"
```

Monitor host `MemAvailable`: the 108 GiB container limit and 112 GiB combined
memory/swap limit do not bound all GB10 GPU allocations. Managed B12X loading
avoids the staging path that exhausted memory in the recorded fastsafetensors
attempt. No persistent memory guard, boot service or networking change is
installed by this profile.

```bash
curl --fail "http://${MASTER_ADDR}:8000/health"
curl --fail "http://${MASTER_ADDR}:8000/v1/models"
```

The API has no configured authentication; use a trusted network or authenticated
gateway. Health does not prove inference or capacity. Tests at the primary
settings remain deferred; this preparation does not authorize running them.

## Evidence and remaining checks

The [evidence record](../../performance/records/qwen38-flash-next/r37-tp2.json)
separates startup-only defaults from the [64K/C1 baseline](config-64k-c1.json).
Startup reported an estimated 2,954,103-token pool, not proof that sixteen 262K
requests fit simultaneously.

The baseline used 64K context, one sequence, 2,048 batched tokens and 1.5 GiB KV.
It passed three exact JSON requests and native prefix-cache reuse. Short C1
decode measured 46.84 tokens/s without added context and 47.14 at 8K; one
integrated 8K prefill scout measured 2,825 tokens/s. These are bounded observations,
not throughput claims for the primary profile. SparkCache and GLM-specific mHC
and coalescing remain disabled.

Remaining primary-profile checks: exact answers, native prefix reuse, C16
completion, near-limit correctness, multimodal behavior and sustained memory
use. The rejected 384K YaRN trial exceeded QSA's internal native limit; this
profile preserves checkpoint positional-encoding defaults.

Upstream: [LIL TP2/RDMA launcher](https://github.com/local-inference-lab/vllm/blob/c687594b8a8082e18af9fe2f64eb5f9ee442e251/scripts/serve-qwen38-flash-next-nvfp4-tp2-rdma.sh).
SparkRing uses the generic verified R37 entrypoint, not GLM serving admission.
