# MiMo-V2.6-Flash-RL with DFlash and SIRCL on four Sparks

Profile: `mimo-v26-flash-rl-tp4`. Status: **Development (implemented, not qualified)**.
This profile uses pinned Karmic Kraken and B12X sources for native global,
sliding-window and DFlash draft attention, with BF16 target and draft KV.
TP4 text serving and bounded performance tests passed on one site.
Media requests and sustained workloads remain unqualified.

Inspect the selected defaults with
`python scripts/profiles.py resolve mimo-v26-flash-rl-tp4`.

The [runtime contract](../../runtime/mimo-v26-flash/b12x-image.json) pins the
base image and both source revisions. The
[runtime guide](../../runtime/mimo-v26-flash/README.md) describes the image
build. The [serving recipe](recipe.json) records the serving settings.

| Setting | Value |
|---|---|
| Parallelism | TP4/DCP1; switchless 0-1-2-3-0 cycle, both ports and both PCIe domains per rank |
| Context / sequences / batch | 262,144 / 16 / 8,192 |
| KV | 20 GiB per rank shared by BF16 target and draft caches |
| Attention | Native B12X target global/SWA attention and B12X noncausal draft attention |
| Loading / speculation | Safetensors / DFlash, 5 drafted tokens |
| Runner | V2; `VLLM_USE_V2_MODEL_RUNNER=1` |
| Media | 3 images, 1 video (16 frames), 1 audio per prompt |
| Collectives | SIRCL custom mode (graph-only and fused prefill sessions) with RoCEnante virtual diagonals; PyNCCL over four rails otherwise |
| Capture | Full and piecewise cudagraphs up to 64 tokens |

## Prepare the four hosts

This profile runs on the fabric the GLM-5.3 managed-mesh profile installs.
Complete the hardware, driver and safety prerequisites of the
[managed-mesh quickstart](../../docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md)
through its fabric installation: the physical cycle, the ConnectX-7 driver
configuration for hardware forwarding, the site and fabric descriptions and
the managed fabric service that applies routes, neighbor tables and the
hardware-offloaded relay rule (EtherType 0x88b5) on every rail. The launcher
refuses to start with SIRCL enabled unless such a rule is present on an
interface. Do not start that profile's model service; only its fabric
service is needed, and it must be active on all four ranks.

Stop other model workloads on the four ranks first. The management network
carries torch rendezvous and NCCL/Gloo bootstrap between the four
management addresses; scope the rank-zero API port to its intended clients.

## Image and checkpoint

Build the derived image on an ARM64 Spark from the repository root:

```bash
bash runtime/mimo-v26-flash/build-image.sh
IMAGE=sparkring:mimo-b12x-6afb999-4f3028
```

This image is not published to GHCR. Follow the
[build and distribution steps](../../runtime/mimo-v26-flash/README.md#build-and-distribute)
to load the same image on all four ranks, then set `IMAGE` in each private
rank environment to its local image ID. The launcher's source-label check
rejects the unmodified base image.

Download the checkpoint at revision
`5711b268169967567844e1e560e8a3966da959b1` into the same absolute directory
on every rank (about 166 GB):

```bash
hf download XiaomiMiMo/MiMo-V2.6-Flash-RL \
  --revision 5711b268169967567844e1e560e8a3966da959b1 \
  --local-dir /srv/models/MiMo-V2.6-Flash-RL
```

Verify `config.json` (SHA-256
`61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621`) and
`model.safetensors.index.json` (SHA-256
`09d9b96a77ed9765fa4e02a6a45f92797702eef432da426efc6b45c1131b1812`) on every
rank and that no `*.incomplete` files remain; the launcher refuses a
directory that contains any. Revisions before `b2674c72` ship
`dflash/config.json` with a trailing comma; the launcher detects the invalid
file and mounts a corrected copy.

## Configure each rank

Two private files per rank. First the launcher environment:

```bash
cp runtime/mimo-v26-flash/ring.env.example /srv/private/mimo-tp4-rank.env
cp runtime/mimo-v26-flash/sircl-rank.env.example /srv/private/mimo-tp4-sircl.env
chmod 0600 /srv/private/mimo-tp4-rank.env /srv/private/mimo-tp4-sircl.env
```

In the launcher environment, `RANK` is 0 on the serving rank and 1 to 3 on
the followers; `HOST_IP` and `MGMT_IFNAME` are the rank's own management
address and interface; `MASTER_ADDR` is rank 0's management address on every
rank; `NCCL_IB_HCA_LIST` names the four RoCE device functions with port
numbers; `MODEL_DIR` and `CACHE_DIR` are the checkpoint and a separate
writable compile-cache directory; `SIRCL_ENV_FILE` is the absolute path of
the second file.

The SIRCL file carries the rank's transport assignments: primary and
secondary peers, devices and GIDs for the two rail pairs, control ports and
the session switches. Their values follow the managed-mesh fabric plan
(slot 0 reaches rank XOR 1, slot 1 reaches rank XOR 3; odd ranks swap the
two devices), documented in the mesh profile's
[SIRCL environment template](../../runtime/glm53-flash-jj-r8-gb10/sircl-fused.env.example).
The `SPARK_TP4_*` and `VLLM_SPARK_*` assignments of the mesh profile's
rendered rank environment are the same values and may be copied. The file is
in Docker `--env-file` syntax: no quotes, no shell expansion.

Validate the inputs without starting anything:

```bash
python scripts/launch.py mimo-v26-flash-rl-tp4 --check /srv/private/mimo-tp4-rank.env
```

The check confirms both environment files, checkpoint files, source labels,
image presence and that a relay rule exists. It does not test the fabric.

## Start, check and stop

Start ranks 3, 2 and 1, then rank 0:

```bash
python scripts/launch.py --execute mimo-v26-flash-rl-tp4 --run /srv/private/mimo-tp4-rank.env
docker logs --follow mimo-v26-flash-rl-tp4-r0
```

Cold startup includes safetensors loading, B12X selection, kernel preparation
and graph capture. Retain `CACHE_DIR` across restarts to reuse selections.
The API is ready after `Application startup complete`.
Before directing traffic, confirm on every rank `SIRCL capability vote
accepted: physical_ranks=4`, `Spark TP4 graph-only session ready` and
`Spark TP4 fused prefill session ready: ... rails=2 exposure=fused`, and on
rank 0 `Using B12X for attention` and `Using V2 Model Runner`, the reported `GPU KV cache
size` and `Application startup complete`. Then send one text and one image
request to `mimo-v2.6-flash` on port 8020 and check the answers. Reasoning
output uses the `mimo` reasoning parser; tool calls use the `mimo` tool-call
parser.

Stop application traffic, then stop rank 0 before the followers:

```bash
docker stop -t 60 mimo-v26-flash-rl-tp4-r0
```

Do not restart one rank beneath live collectives; stop all four, then start
the followers before rank 0 again. The launcher removes any existing
container that carries the rank's container name before it creates the
rank's container.

## Launcher switches

`SIRCL=0` selects PyNCCL over the four rails. Tunables (`KV_BYTES`,
`SPEC_TOKENS`, `CG_CAP`, `MM`, `LINEAR_BACKEND`, `EXTRA_ENV`) are listed in
the [runtime guide](../../runtime/mimo-v26-flash/README.md).
The measured configuration uses SIRCL and a 20 GiB KV reservation per rank.

## Evidence and limitations

The [B12X TP4 record](../../performance/records/mimo-v26-flash/b12x-tp4-20260923.md)
reports approximately 4.0K/3.3K prefill tok/s at 8K/128K, C1 decode
54.5/47.2 tok/s and repeated C8 measurements of 162–199/142–149 tok/s.
Speculative acceptance differs across runs; these measurements do not
establish a universal decode speedup. The engine reported 2,189,381 KV tokens.

- Image, video and audio limits are configured but media inference has not
  been rechecked on this source composition.
- Safetensors is required for the recorded setup. FastSafetensors produced
  corrupted text in the tested MiMo configuration.
- The derived image bypasses its base image's package-hash attestation.
  Launcher checks verify declared source labels, not runtime file integrity.
- No soak, accuracy suite, SparkCache composition or independent reproduction.
