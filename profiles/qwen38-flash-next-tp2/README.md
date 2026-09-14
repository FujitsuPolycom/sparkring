# Qwen3.8-Flash-Next NVFP4 on two Sparks

[Experimental per-host Compose examples](compose/README.md) are available for the
published R37 image on an already prepared fabric. They preserve the profile’s
serving settings but have only been checked offline, not launched through Compose.

Status: **Experimental**. The profile fixes **262K context, 16 sequences,
8,192 batched tokens and 24 GiB KV per rank**. Bounded exact-answer, C16,
near-limit retrieval and native prefix-cache checks passed at these settings.
Synthetic C1 image/video checks also passed. General media behavior and
long-duration stability remain unqualified.
Resolving the profile starts no model or test.

Use [LIL Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)
revision `ada4da32a583a78aa47299f45a70603c950490b8` with the published R37
ARM64 image. 

| Setting | Default |
| --- | --- |
| Parallelism | TP2/DCP1; one direct cable, two PCI domains |
| Context / sequences / batch | 262,144 / 16 / 8,192 |
| KV | 25,769,803,776 bytes per rank (24 GiB), FP8 |
| Loader / speculation | Managed B12X target and MTP3 draft |
| PLE / vocabulary head | Device placement; BF16 target head |
| Cache | Native vLLM prefix caching; SparkCache disabled |
| Runner / convolution layout | V2 / DS |
| Media | Up to three images and one video per request |
| Video sampling | 16 configured frames; not a visual-token or memory bound |
| Context extension | None; no YaRN or HF overrides |

[config.json](config.json) owns the arguments and environment. Inspect catalog
defaults with `python3 scripts/profiles.py resolve qwen38-flash-next-tp2`.

For opt-in persistent prefix caching, see the
[SparkCache extension profile](SPARKCACHE.md). It requires a locally built,
verified R37 extension; the published base-image default remains cache-disabled.

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
gateway. Health alone does not prove inference or capacity; retain the validation
results for the actual image, profile and workload.

## Evidence and remaining checks

The [evidence record](../../performance/records/qwen38-flash-next/r37-tp2.json)
covers this single launch profile. Sixteen concurrent exact-JSON requests passed;
a 257,504-token prompt returned all three requested keys correctly. Three short
exact-JSON checks and native prefix reuse also passed, with 7,871 cached tokens
observed on repeat.

The short performance matrix contains C1/C2/C4/C8 measurements at 8K, 16K, 32K
and 64K input lengths, all under the same 262K launch configuration. These input
lengths are benchmark rows, not alternative presets. The 17-second windows and
single-sample prefill measurements do not establish statistical performance or
long-duration stability.

The media defaults are `--limit-mm-per-prompt '{"image":3,"video":1}'` and
`--media-io-kwargs '{"video":{"num_frames":16}}'`. OpenAI-style chat content
uses `image_url` and `video_url` items; the recorded tests supplied data URLs.
The following synthetic checks ran at C1 with all capacity settings unchanged:

| Request | Result | Elapsed |
| --- | --- | ---: |
| Three 256×256 red/green/blue PNGs | Correct color order | 8.45 s |
| Six-second 256×256 MP4, two seconds per color | Correct temporal order | 2.61 s |
| Three images plus that video | Both orders correct; fenced JSON | 3.40 s |

All returned HTTP 200. The combined answer was semantically correct but was
not strict bare JSON. Sampled minimum host available memory was 18.28/21.93 GiB
across the two ranks. These results do not qualify long or high-resolution
videos, arbitrary media, or C16 multimodal requests. Sixteen sampled frames do
not mean sixteen visual tokens or guarantee bounded memory for every video.
The earlier text measurements used a one-image/zero-video limit.

Startup reported an estimated 2,954,103-token pool, not proof that sixteen 262K
requests fit simultaneously. SparkCache and GLM-specific mHC/coalescing remain
disabled. Sustained memory use and longer stability checks remain open. The
profile preserves native positional encoding without YaRN.

Upstream: [LIL TP2/RDMA launcher](https://github.com/local-inference-lab/vllm/blob/c687594b8a8082e18af9fe2f64eb5f9ee442e251/scripts/serve-qwen38-flash-next-nvfp4-tp2-rdma.sh).
SparkRing uses the generic verified R37 entrypoint, not GLM serving admission.
