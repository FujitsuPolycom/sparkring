# MiMo-V2.6-Flash-RL with DFlash and SIRCL on four Sparks

Profile: `mimo-v26-flash-rl-tp4`. Status: **Development** (implemented, not
qualified). This four-rank configuration serves
`XiaomiMiMo/MiMo-V2.6-Flash-RL` with its bundled DFlash draft, text, image,
video and audio inputs, and SIRCL collectives over the hardware-forwarded
managed mesh: the graph-only session for captured width-4096 decode
collectives and the fused bidirectional prefill session on both rail pairs,
with RoCEnante virtual diagonals over the relay rule. Its evidence is a
single-site, single-day record; see [limitations](#evidence-and-limitations).

Inspect the selected defaults with
`python scripts/profiles.py resolve mimo-v26-flash-rl-tp4`.

The [runtime contract](../../runtime/mimo-v26-flash/image.json) records the
published image, the vLLM build and every overlay file's digest and mount
target; the [runtime README](../../runtime/mimo-v26-flash/README.md) explains
why the overlay exists. The [serving recipe](recipe.json) summarizes model,
topology and serving settings.

| Setting | Value |
|---|---|
| Parallelism | TP4/DCP1; switchless 0-1-2-3-0 cycle, both ports and both PCIe domains per rank |
| Context / sequences / batch | 262,144 / 16 / 8,192 |
| KV | 20 GiB bf16 target cache per rank; fp8 draft cache |
| Attention | DiffKV Triton kernel with the PR 839 split-KV verification dispatch |
| Speculation | DFlash, 5 drafted tokens |
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

Pull the published image on every rank and keep its immutable reference:

```bash
IMAGE=ghcr.io/fujitsupolycom/sparkring@sha256:26c366af994cf42e38e4596db4d611342a3466fd8ca49d6037237d04249e5132
docker pull --platform linux/arm64 "$IMAGE"
```

Pin `IMAGE` by image ID in the rank environment when a local tag resolves to
different builds on different ranks.

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

The check confirms both environment files, checkpoint files, overlay files,
image presence and that a relay rule exists. It does not test the fabric.

## Start, check and stop

Start ranks 3, 2 and 1, then rank 0:

```bash
python scripts/launch.py --execute mimo-v26-flash-rl-tp4 --run /srv/private/mimo-tp4-rank.env
docker logs --follow mimo-v26-flash-rl-tp4-r0
```

Startup takes about 10 minutes on the reference cycle. Every follower prints
`AssertionError: collective_rpc should not be called on follower node`
during KV sizing and continues; that line is not a failure in this build.
Before directing traffic, confirm on every rank `SIRCL capability vote
accepted: physical_ranks=4`, `Spark TP4 graph-only session ready` and
`Spark TP4 fused prefill session ready: ... rails=2 exposure=fused`, and on
rank 0 `Using TRITON_ATTN_DIFFKV for attention`, the reported `GPU KV cache
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

`SIRCL=0` serves the same settings on PyNCCL over the four rails without the
mesh relay; that configuration measured lower single-stream and C8
throughput and fits a 30 GiB KV reservation. Other tunables (`KV_BYTES`,
`SPEC_TOKENS`, `CG_CAP`, `MM`, `PAD_V`, `LINEAR_BACKEND`, `EXTRA_ENV`) are
listed in the [runtime README](../../runtime/mimo-v26-flash/README.md). A
changed value is research-only and does not inherit this profile's record.
A 30 GiB reservation with SIRCL enabled exhausted one rank's device memory
during warmup. The RoCEnante one-shot all-reduce is a pair transport: on
this cycle its queue pairs to non-adjacent ranks do not connect.

## Evidence and limitations

The [ring record](../../performance/records/mimo-v26-flash/tp4-ring-20260922.md)
holds the measured configurations: single stream 63.4 tok/s, C8 aggregate
184.0 tok/s, 48K cold prefill 3,721 tok/s with 55.7 tok/s decode, 120K cold
prefill 2,901 tok/s with 46.6 tok/s decode, repetition tests clean, image
check passed, 2,189,381 KV tokens; the PyNCCL alternative measured 57.8 /
151.1 / 3,826 and 53.5 / 2,933 and 48.8 with 3,284,072 KV tokens.

- Single runs on one day; the noise band is about 3 percent single-stream
  and 5 percent at C8. 120K decode with SIRCL is about 5 percent below PyNCCL.
- The overlay bypasses the image's attestation entrypoint; a rebuilt image
  that absorbs the recorded overlay files would remove that bypass.
- Video and audio inputs are enabled but only an image request was checked
  on this topology.
- No soak, accuracy suite, SparkCache composition or independent reproduction.
- The mesh relay is installed and owned by the GLM-5.3 managed-mesh fabric
  service; this profile does not install, verify or repair it beyond
  checking that a relay rule exists.
