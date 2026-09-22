# MiMo-V2.6-Flash-RL with DFlash on two Sparks

Profile: `mimo-v26-flash-rl-tp2`. Status: **Development** (implemented, not
qualified). This two-rank configuration serves
`XiaomiMiMo/MiMo-V2.6-Flash-RL` with its bundled DFlash draft, text, image,
video and audio inputs, and the RoCEnante one-shot all-reduce over one direct
cable. Its evidence is a single-site, single-day record; see
[limitations](#evidence-and-limitations).

Inspect the selected defaults with
`python scripts/profiles.py resolve mimo-v26-flash-rl-tp2`.

The [runtime contract](../../runtime/mimo-v26-flash/image.json) records the
published image, the vLLM build and every overlay file's digest and mount
target; the [runtime README](../../runtime/mimo-v26-flash/README.md) explains
why the overlay exists. The [serving recipe](recipe.json) summarizes model,
topology and serving settings.

| Setting | Value |
|---|---|
| Parallelism | TP2/DCP1; one direct cable, both Socket Direct functions |
| Context / sequences / batch | 262,144 / 16 / 8,192 |
| KV | 12 GiB bf16 target cache per rank (16 GiB text-only); fp8 draft cache |
| Attention | Padded-V symmetric Triton kernel with split-KV verification dispatch |
| Speculation | DFlash, 5 drafted tokens |
| Media | 3 images, 1 video (16 frames), 1 audio per prompt |
| Collectives | RoCEnante one-shot all-reduce up to 2 MiB; NCCL above |
| Capture | Full and piecewise cudagraphs up to 32 tokens |

## Prepare the two hosts

Complete the two-Spark steps in
[PREREQUISITES.md](../../docs/operations/prerequisites.md): one direct cable,
persistent addresses on both RoCE device functions, a verified RoCEv2/IPv4 GID
on each, `/dev/infiniband` available to containers and the NVIDIA persistence
daemon running. The management network carries torch rendezvous and NCCL/Gloo
bootstrap between the two management addresses. Scope the rank-zero API port
to its intended clients. Stop other model workloads first: the weights alone
take 81 GiB of each rank's unified memory.

Host memory bounds this profile. With the 12 GiB reservation about 10 GiB of
host memory stays available on a 128 GB Spark; a host out-of-memory killer
tuned tighter than that (the reference pair runs earlyoom with a 4 percent
SIGTERM threshold) ends the serving processes or the persistence daemon.

## Image and checkpoint

Pull the published image on both ranks and keep its immutable reference:

```bash
IMAGE=ghcr.io/fujitsupolycom/sparkring@sha256:26c366af994cf42e38e4596db4d611342a3466fd8ca49d6037237d04249e5132
docker pull --platform linux/arm64 "$IMAGE"
```

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
`IMAGE` is the reference above or its local image ID.

Validate the inputs without starting anything:

```bash
python scripts/launch.py mimo-v26-flash-rl-tp2 --check /srv/private/mimo-tp2-rank.env
```

The check confirms the environment file, checkpoint files, overlay files and
image presence. It does not test the fabric.

## Start, check and stop

Start rank 1, then rank 0:

```bash
python scripts/launch.py --execute mimo-v26-flash-rl-tp2 --run /srv/private/mimo-tp2-rank.env
docker logs --follow mimo-v26-flash-rl-tp2-r0
```

Startup takes about 14 minutes on the reference pair: weight loading,
b12x expert preparation, encoder profiling and cudagraph capture. The API is
ready after `Application startup complete`. Before directing traffic, confirm
in rank 0's log `Using TRITON_ATTN for attention`, `RoCEnante all-reduce is
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
(`KV_BYTES`, `SPEC_TOKENS`, `CG_CAP`, `ROCE_AR`, `MM`, `PAD_V`, `EXTRA_ENV`
and others listed in the [runtime README](../../runtime/mimo-v26-flash/README.md)).
A changed value is research-only and does not inherit this profile's record.
Raising `CG_CAP` to 64 lets an eight-request decode step (48 query rows with
the 5-token draft) run as one full cudagraph; that setting is the four-Spark
profile's default and is unmeasured on the pair.

## Evidence and limitations

The [pair record](../../performance/records/mimo-v26-flash/tp2-pair-20260922.md)
holds the measured configurations: single stream 36.0 tok/s, C8 aggregate
91.3 tok/s, 48K cold prefill 2,183 tok/s with 33.7 tok/s decode, 120K cold
prefill 1,488 tok/s with 29.7 tok/s decode, repetition tests clean, image and
audio checks passed, 547,326 KV tokens.

- Single runs on one day; the noise band is about 3 percent single-stream
  and 5 percent at C8.
- The overlay bypasses the image's attestation entrypoint; a rebuilt image
  that absorbs the recorded overlay files would remove that bypass.
- Video input is enabled but was not exercised.
- No soak, accuracy suite, SparkCache composition or independent reproduction.
