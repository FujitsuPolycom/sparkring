# Qwen3.8-Flash-Next on two DGX Sparks with Docker Compose

[`standalone.yaml`](standalone.yaml) runs the [`qwen38-flash-next-tp2`](../README.md)
profile on two DGX Sparks connected by one direct cable, with the same file on
each Spark. Spark 0 serves `Qwen3.8-Flash-Next-NVFP4-QAD-TP2` on port 8000.
The containers are the ones [`sparkring install`](../../../docs/operations/install.md)
runs; unlike the installer, this recipe assumes the network, image and
checkpoint are already set up.

## Files

On each Spark, in an empty directory:

```bash
BASE=https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer
mkdir -p runtime/common
curl -fsSLo compose.yaml "$BASE/profiles/qwen38-flash-next-tp2/compose/standalone.yaml"
curl -fsSLo runtime/common/loader-seccomp.json "$BASE/runtime/common/loader-seccomp.json"
curl -fsSLo SHA256SUMS "$BASE/profiles/qwen38-flash-next-tp2/SHA256SUMS"
```

Keep `loader-seccomp.json` at that path: the model loader needs `io_uring`,
which Docker's default seccomp policy blocks.

## Requirements on each Spark

1. **Docker** with the NVIDIA Container Toolkit and `docker compose`.
2. **The image** (31.6 GB):
   ```bash
   docker pull ghcr.io/fujitsupolycom/sparkring@sha256:451c5e23a90e0df2fc904e8851aab12c3ec9ffdcd1258b6f14cf502222e46b5f
   ```
3. **The fabric**: port p0 cabled to port p0, and both of its functions
   addressed, one subnet each, MTU 9000:

   | Interface | Spark 0 | Spark 1 |
   |---|---|---|
   | `enp1s0f0np0` | `198.18.0.1/24` | `198.18.0.2/24` |
   | `enP2p1s0f0np0` | `198.18.1.1/24` | `198.18.1.2/24` |

   `ping -c 3 -M do -s 8972 198.18.0.2` and `… 198.18.1.2` from Spark 0 must
   succeed.
4. **The checkpoint** (about 110 GB), as real files, verified:
   ```bash
   hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --revision 60215d26cf5e42c2db6128774032d57fc62678da --local-dir /path/to/Qwen3.8-Flash-Next-NVFP4
   (cd /path/to/Qwen3.8-Flash-Next-NVFP4 && sha256sum -c /path/to/SHA256SUMS)   # every line OK
   ```
   Already in your Hugging Face cache? Hard-link it instead (same filesystem,
   no extra space, the cache is not modified):
   ```bash
   SNAP=~/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/60215d26cf5e42c2db6128774032d57fc62678da
   DEST=/path/to/Qwen3.8-Flash-Next-NVFP4
   mkdir -p "$DEST" && for f in "$SNAP"/*; do ln "$(readlink -f "$f")" "$DEST/$(basename "$f")"; done
   ```
   Downloads of `main`, `qad-step-4000` and other quantizations are different
   weights; `sha256sum -c` catches them.
5. **A writable cache directory**, for example `/var/tmp/sparkring-cache`.

## Configure and start

Next to `compose.yaml`, create a `.env` file. On Spark 0:

```bash
SPARKRING_MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4
SPARKRING_CACHE_DIR=/var/tmp/sparkring-cache
SPARKRING_MASTER_ADDR=198.18.0.1
SPARKRING_HOST_IP=198.18.0.1
SPARKRING_INTERFACE=enp1s0f0np0
```

On Spark 1, the same except `SPARKRING_HOST_IP=198.18.0.2`.
`SPARKRING_MASTER_ADDR` is Spark 0's `enp1s0f0np0` address on both Sparks.

Start Spark 1 first, then Spark 0, which serves the API:

```bash
# On Spark 1
docker compose --profile rank1 up -d
# On Spark 0
docker compose --profile rank0 up -d
docker compose logs -f rank0
```

Never start both ranks on one Spark. The first start compiles and tunes
kernels and can take up to 15 minutes; later starts take about 4.

## Check

```bash
curl http://SPARK0_ADDRESS:8000/v1/models
curl http://SPARK0_ADDRESS:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3.8-Flash-Next-NVFP4-QAD-TP2", "messages": [{"role": "user", "content": "What is 17 * 23?"}]}'
```

The API listens on port 8000 on every interface of Spark 0, with no API key.
Keep Spark 0 on a trusted network or firewall the port. Stop with
`docker compose --profile rank0 stop` on Spark 0 and
`docker compose --profile rank1 stop` on Spark 1.

## Performance

One pair serving checkpoint step 4000 (revision `629bc3218833`), 512-token
single-stream requests at temperature 0; prefill is one cold prompt. The
Compose recipe matched the installer deployment. The recipe on this page pins
step 5500; the [profile page](../README.md#performance) gives its rates.

| Deployment | Decode prose / code / JSON (tokens/s) | Prefill 16K / 64K (tokens/s) |
|---|---|---|
| This recipe | 62.3 / 89.7 / 101.5 | 4,236 / 3,932 |
| `install.sh` | 61.0 / 89.2 / 100.8 | 4,258 / 3,939 |

With the installer's kernel cache mounted, rank 0 was healthy 240 seconds
after `docker compose up`. Measurements:
[installer tuning record](../../../performance/records/qwen38-flash-next/installer-tuning-20260925.md#installation-from-the-published-branch-and-the-standalone-compose-recipe).

## If it fails

- **`docker compose` reports a missing variable**: the `.env` file is not next
  to `compose.yaml`, or one of its five values is missing.
- **The container exits while loading with an `io_uring` or "Operation not
  permitted" error**: `runtime/common/loader-seccomp.json` is not at that path
  relative to `compose.yaml`.
- **The ranks hang at startup, or NCCL reports no usable device**: check the
  fabric (both functions addressed, MTU 9000, GID index 3 is RoCE v2 IPv4,
  jumbo ping works) and the `SPARKRING_*` values. `NCCL_IB_HCA` and
  `B12X_ROCE_HCA` name `rocep1s0f0` and `roceP2p1s0f0`, the DGX Spark device
  names for port p0.
- **Loading fails with a missing file or a shape mismatch**: the checkpoint
  folder is incomplete, holds unresolved symlinks, or is another revision.

## Per-rank files

[`compose.rank0.yaml`](compose.rank0.yaml) and [`compose.rank1.yaml`](compose.rank1.yaml)
are the same containers rendered for an [example site](site.example.yaml); see
[Compose deployments](../../../docs/operations/compose.md).
