# MiMo-V2.6-Flash-RL with DFlash on two Sparks

Profile: `mimo-v26-flash-rl-tp2`. Status: **Experimental (research-only)**.
This profile uses pinned Karmic Kraken and B12X sources for native global,
sliding-window and DFlash draft attention, with BF16 target and draft KV.
The TP2 adaptation has not been hardware-tested; memory use, media and
performance remain unqualified.

Inspect the selected defaults with
`python scripts/profiles.py resolve mimo-v26-flash-rl-tp2`.

The [runtime contract](../../runtime/mimo-v26-flash/b12x-image.json) pins the
base image and both source revisions. The
[runtime guide](../../runtime/mimo-v26-flash/README.md) describes the image
build. The [serving recipe](recipe.json) records the serving settings.

| Setting | Value |
|---|---|
| Parallelism | TP2/DCP1; one direct cable, both Socket Direct functions |
| Context / sequences / batch | 262,144 / 16 / 8,192 |
| KV | 12 GiB per rank shared by BF16 target and draft caches; TP2 capacity unmeasured |
| Attention | Native B12X target global/SWA attention and B12X noncausal draft attention |
| Loading / speculation | Safetensors / DFlash, 5 drafted tokens |
| Runner | V2; `VLLM_USE_V2_MODEL_RUNNER=1` |
| Media | 3 images, 1 video (16 frames), 1 audio per prompt |
| Collectives | RoCEnante one-shot all-reduce up to 2 MiB; NCCL above |
| Capture | Full and piecewise cudagraphs up to 64 tokens |

## Prepare the two hosts

Complete the two-Spark steps in
[PREREQUISITES.md](../../docs/operations/prerequisites.md): one direct cable,
persistent addresses on both RoCE device functions, a verified RoCEv2/IPv4 GID
on each, `/dev/infiniband` available to containers and the NVIDIA persistence
daemon running. The management network carries torch rendezvous and NCCL/Gloo
bootstrap between the two management addresses. Scope the rank-zero API port
to its intended clients. Stop other model workloads first: the weights alone
take 81 GiB of each rank's unified memory.

The 12 GiB reservation is a starting point, not a measured memory margin
for this build. BF16 draft KV and the 64-row graph ceiling require TP2
startup and memory validation before unattended use.

## Image and checkpoint

Build the derived image on an ARM64 Spark from the repository root:

```bash
bash runtime/mimo-v26-flash/build-image.sh
IMAGE=sparkring:mimo-b12x-6afb999-4f3028
```

This image is not published to GHCR. Follow the
[build and distribution steps](../../runtime/mimo-v26-flash/README.md#build-and-distribute)
to load the same image on both ranks, then set `IMAGE` in each private
rank environment to its local image ID. The launcher's source-label check
rejects the unmodified base image.

Download the checkpoint at revision
`5711b268169967567844e1e560e8a3966da959b1` into the same absolute directory
on both ranks (about 166 GB, including the vision and audio encoders and the
`dflash/` draft):

```bash
hf download XiaomiMiMo/MiMo-V2.6-Flash-RL \
  --revision 5711b268169967567844e1e560e8a3966da959b1 \
  --local-dir /srv/models/MiMo-V2.6-Flash-RL
```

Verify `config.json` (SHA-256
`61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621`) and
`model.safetensors.index.json` (SHA-256
`09d9b96a77ed9765fa4e02a6a45f92797702eef432da426efc6b45c1131b1812`) and that
no `*.incomplete` files remain. Revisions before `b2674c72` ship
`dflash/config.json` with a trailing comma; the launcher detects the invalid
file and mounts a corrected copy.

## Configure each rank

Copy the environment template to a private file on each rank and replace
every `REPLACE_` value:

```bash
cp runtime/mimo-v26-flash/pair.env.example /srv/private/mimo-tp2-rank.env
chmod 0600 /srv/private/mimo-tp2-rank.env
```

`RANK` is 0 on the serving rank and 1 on the follower. `HOST_IP` and
`MGMT_IFNAME` are the rank's own management address and interface;
`MASTER_ADDR` is rank 0's management address on both ranks. `ROCE_HCA_PAIR`
lists one RoCE device per PCIe domain. `MODEL_DIR` is the verified checkpoint
directory and `CACHE_DIR` a separate writable compile-cache directory.
`IMAGE` is the derived image's local image ID.

Validate the inputs without starting anything:

```bash
python scripts/launch.py mimo-v26-flash-rl-tp2 --check /srv/private/mimo-tp2-rank.env
```

The check confirms the environment file, checkpoint files and derived-image source labels. It does not test the fabric.

## Start, check and stop

Start rank 1, then rank 0:

```bash
python scripts/launch.py --execute mimo-v26-flash-rl-tp2 --run /srv/private/mimo-tp2-rank.env
docker logs --follow mimo-v26-flash-rl-tp2-r0
```

Cold startup includes safetensors loading, B12X selection, kernel preparation
and graph capture. Retain `CACHE_DIR` across restarts to reuse selections.
The API is ready after `Application startup complete`.
Before directing traffic, confirm
in rank 0's log `Using B12X for attention` and `Using V2 Model Runner`, `RoCEnante all-reduce is
live` and the reported `GPU KV cache size`, then send one text, one image and
one audio request to `mimo-v2.6-flash` on port 8000 and check the answers.
Reasoning output uses the `mimo` reasoning parser; tool calls use the `mimo`
tool-call parser.

Stop application traffic, then stop rank 0 before rank 1:

```bash
docker stop -t 60 mimo-v26-flash-rl-tp2-r0
```

Do not restart one rank beneath live collectives; stop both, then start
rank 1 before rank 0 again. The launcher removes any existing container that
carries the rank's container name before it creates the rank's container.

## Launcher switches

The launcher reads serving tunables from its process environment
(`KV_BYTES`, `SPEC_TOKENS`, `CG_CAP`, `ROCE_AR`, `MM`, `EXTRA_ENV`
and others listed in the [runtime README](../../runtime/mimo-v26-flash/README.md)).
A changed value is research-only and does not inherit this profile's record.
The 64-row capture ceiling covers C8 with DFlash5 (48 query rows).
C16 may exceed this full-graph ceiling; admission is not a graph-coverage guarantee.

## Evidence and limitations

No TP2 results have been collected for this B12X source composition.
The [TP4 record](../../performance/records/mimo-v26-flash/b12x-tp4-20260923.md)
establishes bounded text serving on four Sparks only. The
[historical Triton pair record](../../performance/records/mimo-v26-flash/tp2-pair-20260922.md)
describes different attention kernels, draft KV dtype and graph coverage;
its capacity and performance must not be attributed to this profile.

- Image, video and audio limits are configured but media inference has not
  been rechecked on this source composition.
- Safetensors is required for the recorded setup. FastSafetensors produced
  corrupted text in the tested MiMo configuration.
- The derived image bypasses its base image's package-hash attestation.
  Launcher checks verify declared source labels, not runtime file integrity.
- No soak, accuracy suite, SparkCache composition or independent reproduction.
