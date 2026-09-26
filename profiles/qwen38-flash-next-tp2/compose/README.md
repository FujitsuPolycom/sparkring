# Qwen3.8-Flash-Next on two DGX Sparks with Docker Compose

[`standalone.yaml`](standalone.yaml) runs the `qwen38-flash-next-tp2` profile
on two DGX Sparks connected by one direct cable, from one Compose file on each
Spark. It is the container that [`sparkring install`](../../../docs/operations/install.md)
deploys, on the same image, without the installer's per-container
runtime-binding file; the image's status dashboard then reports worker
identities as `binding_not_configured`, and serving is unaffected.

Status: **implemented**. On one pair, this recipe served with `docker compose`
at 62.3 / 89.7 / 101.5 tokens per second single-stream decode (prose / code /
JSON) and 4,236 tokens per second prefill at 16K tokens, and passed counting,
arithmetic and code checks
([record](../../../performance/records/qwen38-flash-next/installer-tuning-20260925.md#installation-from-the-published-branch-and-the-standalone-compose-recipe)).
Serving is not qualified.

`sparkring install` sets up the network, image and checkpoint for you. This
recipe assumes you prepare them yourself.

## Files

On each Spark, in an empty directory:

```bash
BASE=https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer
mkdir -p runtime/common
curl -fsSLo compose.yaml "$BASE/profiles/qwen38-flash-next-tp2/compose/standalone.yaml"
curl -fsSLo runtime/common/loader-seccomp.json "$BASE/runtime/common/loader-seccomp.json"
curl -fsSLo SHA256SUMS "$BASE/profiles/qwen38-flash-next-tp2/SHA256SUMS"
```

- `compose.yaml`: both ranks; the same file goes on both Sparks.
- `runtime/common/loader-seccomp.json`: the seccomp policy the container runs
  under. The checkpoint loader uses `io_uring`, which Docker's default policy
  blocks. Compose reads it at this path relative to `compose.yaml`.
- `SHA256SUMS`: the pinned checksums of the checkpoint's `config.json`, index
  and 36 weight shards.

## Requirements on each Spark

1. **Docker** with the NVIDIA Container Toolkit and the Compose v2 plugin
   (`docker compose version` works).
2. **The image**, 31.6 GB, public:
   ```bash
   docker pull ghcr.io/fujitsupolycom/sparkring@sha256:451c5e23a90e0df2fc904e8851aab12c3ec9ffdcd1258b6f14cf502222e46b5f
   ```
3. **The fabric.** Cable port p0 of one Spark to port p0 of the other. Each p0
   cable appears as two network functions: `enp1s0f0np0` (RDMA device
   `rocep1s0f0`) and `enP2p1s0f0np0` (RDMA device `roceP2p1s0f0`). The
   container uses both. The measured pair used this layout:

   | Interface | Spark 0 | Spark 1 | MTU |
   |---|---|---|---|
   | `enp1s0f0np0` | `198.18.0.1/24` | `198.18.0.2/24` | 9000 |
   | `enP2p1s0f0np0` | `198.18.1.1/24` | `198.18.1.2/24` | 9000 |

   Any private subnets work, one subnet per function. GID index 3 of each
   function must be its RoCE v2 IPv4 GID (`NCCL_IB_GID_INDEX=3`):
   ```bash
   cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/3     # RoCE v2
   cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/ndevs/3     # enp1s0f0np0
   cat /sys/class/infiniband/roceP2p1s0f0/ports/1/gid_attrs/types/3   # RoCE v2
   ```
   From Spark 0, `ping -c 3 -M do -s 8972 198.18.0.2` and the same for
   `198.18.1.2` must succeed.
4. **The checkpoint, complete, as real files on both Sparks**: revision
   `629bc3218833a38b475b719f34aa571666f4a03e` (branch `qad-step-4000`) of
   [`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/629bc3218833a38b475b719f34aa571666f4a03e),
   about 99 GB. The container runs with `HF_HUB_OFFLINE=1` and never
   downloads.
   - To download it into a folder:
     ```bash
     hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --revision 629bc3218833a38b475b719f34aa571666f4a03e --local-dir /path/to/Qwen3.8-Flash-Next-NVFP4
     ```
   - A copy in the Hugging Face cache
     (`~/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/<commit>/`)
     is a directory of symlinks into `../../blobs`, which do not resolve
     inside the container. Give the container a folder of hard links instead;
     this uses no extra space and never modifies the cache, but the cache and
     the folder must be on the same filesystem:
     ```bash
     SNAP=~/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/629bc3218833a38b475b719f34aa571666f4a03e
     DEST=/path/to/Qwen3.8-Flash-Next-NVFP4
     mkdir -p "$DEST"
     for f in "$SNAP"/*; do ln "$(readlink -f "$f")" "$DEST/$(basename "$f")"; done
     ```
   - A copy of the `main` branch downloaded after 2026-09-16 20:03 UTC has the
     same 36 weight shards and index; only `config.json` and `README.md`
     differ. In a hard-linked folder, fetch the pinned `config.json` after
     removing its link, so the download does not write into the other copy:
     ```bash
     rm "$DEST/config.json"
     hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4 config.json --revision 629bc3218833a38b475b719f34aa571666f4a03e --local-dir "$DEST"
     ```
   - A `main` download from before 2026-09-16 20:03 UTC is an older checkpoint
     with different weights, although its `config.json` matches. The
     `qad-step5500-ple1000` branch and third-party quantizations are different
     weights too, even where file names and sizes match.
   - Verify the folder before the first start; this reads all 99 GB. Every
     line must print `OK`:
     ```bash
     (cd /path/to/Qwen3.8-Flash-Next-NVFP4 && sha256sum -c /path/to/SHA256SUMS)
     ```
5. **A writable cache directory**, for example `/var/tmp/sparkring-cache`.
   Kernel tuning results are kept there, so later starts are faster than the
   first.

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
kernels; the health check allows 900 seconds. With a warm kernel cache, rank 0
reported healthy after 240 seconds.

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
are the same containers rendered for the [example site](site.example.yaml)
with documentation addresses; the
[Compose deployment guide](../../../docs/operations/compose.md) describes
rendering them for a private site. They are generated by
`scripts/generate_compose_examples.py`; edit the profile or site input and
regenerate instead of editing them. Per-rank files on this image have no
hardware evidence.
