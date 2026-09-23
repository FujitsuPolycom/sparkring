# MiMo-V2.6-Flash-RL runtime

The [TP2 profile](../../profiles/mimo-v26-flash-rl-tp2/README.md) and
[TP4 profile](../../profiles/mimo-v26-flash-rl-tp4/README.md) use native B12X
attention for MiMo global and sliding-window layers and its DFlash draft.
Packed Q/K-192, V-128 pages keep the unequal head dimensions without padding V.
TP4 has bounded text-serving evidence; TP2 is **research-only** pending hardware testing.

## Build and distribute

The [image contract](b12x-image.json) pins the published SparkRing native base,
LIL vLLM revision `6afb99982576a7a2eb53d667189e859629e22739` and B12X revision
`4f3028b19c1d8290dc72b6f483aba40de23eae5a`. These Python packages include
native MiMo attention selection, packed DiffKV handling and noncausal DFlash
attention. Native CUDA extensions and SparkRing transport remain from the base.
Changing either source pin requires compatibility testing.

On an ARM64 Spark with Git, Python 3 and Docker, run from the repository root:

```bash
bash runtime/mimo-v26-flash/build-image.sh
IMAGE=sparkring:mimo-b12x-6afb999-4f3028
docker image inspect --format '{{.Id}}' "$IMAGE"
docker save "$IMAGE" -o /var/tmp/mimo-b12x-image.tar
```

Copy the archive to each serving rank through the site's transfer network,
then run on each recipient:

```bash
docker load -i /var/tmp/mimo-b12x-image.tar
docker image inspect --format '{{.Id}}' sparkring:mimo-b12x-6afb999-4f3028
```

All ranks must report the same image ID. Put that ID in each rank's private
`IMAGE` assignment. No prebuilt image for this composition is published.
The builder retains its temporary source context and prints the path.
The measured TP4 image used the same source revisions over a child of the
published base containing verification-generated caches; a build directly
from the public base has not been benchmarked.

[check_image.py](check_image.py) checks ARM64 architecture and pinned source
labels before either launcher removes a serving container. The check rejects
the unmodified base and wrong source revisions. Labels establish declared
provenance; they do not attest package bytes. The CLI is invoked through a
login shell to activate the base's CUDA compatibility driver because inherited
package attestation does not cover the replaced Python packages.

## Serving defaults

| Setting | TP2 | TP4 |
|---|---|---|
| Target / draft attention | B12X / B12X | B12X / B12X |
| Target / draft KV | BF16 / BF16 | BF16 / BF16 |
| Loader / runner | Safetensors / V2 | Safetensors / V2 |
| Context / sequences / batch | 262,144 / 16 / 8,192 | 262,144 / 16 / 8,192 |
| KV reservation per rank | 12 GiB; 16 GiB text-only | 20 GiB |
| DFlash tokens / graph ceiling | 5 / 64 | 5 / 64 |
| Linear backend / MoE backend | B12X / B12X | auto / B12X |
| Collectives | RoCEnante pair, NCCL fallback | SIRCL over managed mesh, NCCL fallback |
| Media limits | 3 images, 1 video (16 frames), 1 audio | Same |

TP2 memory use with BF16 draft KV and 64-row capture is unmeasured.
TP4 keeps automatic linear-kernel selection because its global-attention
QKV slice is 3392 wide, which is not a multiple of 128. The 64-row graph
ceiling covers C8/DFlash5; 16 admitted requests can exceed that ceiling.

Safetensors is the recorded loader. FastSafetensors produced corrupted text
in the tested MiMo setup. FP8 target KV and FP8 B12X draft KV are not qualified
by these profiles. `VLLM_PLUGINS=b12x_loader` loads the plugin but does not
change the explicit `--load-format safetensors` selection.

## Launch controls

Use [pair.env.example](pair.env.example) or [ring.env.example](ring.env.example)
and [sircl-rank.env.example](sircl-rank.env.example). The launchers accept
`--check RANK_ENV_FILE` or `--run RANK_ENV_FILE`. Follow the topology's
profile for fabric setup, startup ordering, log checks and stopping.

`MM=1` enables the listed media limits, `MM=image` enables only images,
and `MM=0` selects text-only serving. `KV_BYTES`, `SPEC_TOKENS`, `CG_CAP`,
`MAX_MODEL_LEN`, `MAX_NUM_SEQS` and `MAX_BATCHED` override serving limits.
For an explicit text-only TP2 KV override also set `KV_BYTES_EXPLICIT=1`.
`ROCE_AR` controls pair collectives; `SIRCL` controls ring collectives.
`EXTRA_ENV` adds comma-separated container environment assignments.
AOT compilation is disabled when media is enabled.

The maintained launchers do not mount the retained `overlay/` or `patches/`
files; those belong to the historical Triton measurements. The only optional
model-file mount repairs invalid JSON in older `dflash/config.json` files
using [fix_dflash_config.py](fix_dflash_config.py).
