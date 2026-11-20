# DeepSeek V4.1 row-owned prefill

**Research-only.** The plugin targets the B12X engine in installer image
`dev-20261013-dcp2prefill-status038-sircl-tp6`. Its CPU tests compare
row-owned tensors with the image's full-row forward and check the source
pins. See [the model profile](../../../profiles/deepseek-v41-flash-tp4/profile.json)
for the serving configuration.

Each tensor-parallel rank keeps the residual and mixing coefficients for
its own prefill rows. The plugin gathers normalized attention and feed-forward
inputs, then reduce-scatters their partial outputs. At the compressed encoder
boundary, it prepares global KV from full inputs and gathers the residual state
for the replay rows.

Before that boundary, the installed experiment scores each attention indexer's
tiles on their row owners when the tile boundaries align. It gathers the selected
top-k indices in token order before tensor-parallel attention. The candidate
source layer scores the full batch so later layers can reuse its candidate table.

Engram tables contribute each lookup element from one vocabulary shard. With
a replicated Engram projection, reduce-scatter assembles the lookup rows
directly on their token owners. Each owner projects and mixes just its rows.
With a column-parallel projection, the input rows still need to match on all
ranks for the output gather. The plugin preserves the full-row Engram
reduction and projection for that configuration.

## Install on a four-Spark SIRCL path

Review the deployment on a four-Spark path with a recorded SIRCL fabric.

```sh
sudo sparkring install --profile deepseek-v41-flash-tp4 --checkpoint 1017 \
  --image dev-20261013-dcp2prefill-status038-sircl-tp6 --transport sircl --nccl never \
  --row-owned-prefill --plan
```

Run the command with `--yes` in place of `--plan` to apply the plan. The
installer stages its retained source on each rank and mounts the plugin
read-only. The setting selects eager prefill, full CUDA graphs for uniform
decode, disk-resident Engram tables with replicated projection, 8,192-token
prefill chunks, DSpark5 with adaptive verification, and a 262,144-token
context limit. Use `--context-length`
for a smaller limit.

The installer enables compact routing, full-head local-query attention,
narrow KV gathers, and packed compressor projections. Full-head prefill
holds additional weights per rank. The SIRCL hardware trial used manual
containers. The installer plan has offline coverage. [SIRCL trial](../../../performance/records/deepseek-v41-flash/row-owned-sircl-path4-20261119.md)

### Configure a development container

Mount this directory, including its plugin metadata, at `/opt/ds41_row_shard`
in every serving container. Prepend that path to `PYTHONPATH` and append
`ds41_row_shard` to the existing `VLLM_PLUGINS` list. Set
`DS41_ROW_SHARD_PREFILL=1` on every rank.
Set `DS41_ROW_SHARD_INDEXER=1` on every rank to distribute indexer scoring.
Set `DS41_ROW_SHARD_COMPACT_ROUTES=1` to run the gate and modular router
on local rows and gather the selected expert IDs and FP32 weights.
Set `DS41_ROW_SHARD_LOCAL_QUERIES=1` to assemble the full-head Q and output
weights once at load and prepare separate prefill plans. Each query owner
uses complete local KV caches. Set `DS41_ROW_SHARD_KV_LATENTS=1` to gather
only KV latents on eligible layers. With both flags set,
`DS41_ROW_SHARD_COMPRESSOR_LATENTS=1` also packs and gathers the
index-source layers' projections without changing their dtype bits.
Set `DS41_ROW_SHARD_ENGRAM_BANDS=1` to test feature-banded Engram lookups.
The unbanded build had the highest medians on the measured NCCL cycle.

The TP4 profile normally uses a column-parallel Engram projection. To test
the reduce-scatter path, set `"projection_tp": false` in the profile's
`--engram-config` while keeping its disk-table settings. Replicating the
projection changes device memory use. Record KV capacity and prefill
throughput for that exact configuration.

The plugin verifies the source hashes before installing its hooks. Every TP
rank agrees on the flags and eligibility before changing collectives. The
default minimum is 512 owned rows per rank
(`DS41_ROW_SHARD_MIN_OWNED_ROWS`). Eligible forwards use eager execution,
have an unpadded token count divisible by the TP size, and include a prefill
request. Pure decode, captured graphs, and smaller or padded tails use the
image's full-row path.

The first eight eligible forwards log `DS41_ROW_SHARD` with the rank, owned
rows and collective counts. With replicated Engram projection, `engram_rs=2`
reports that both Engram layers delivered lookup rows to their owners.
The `indexer_gathers` counter reports how many indexers distributed scoring.

## Verify the source pins and tensor layout

From the repository root, point `SPARKRING_DS41_IMAGE_SOURCES` at a directory
containing the extracted `vllm/` source tree and run:

```bash
python -m pytest integrations/vllm/ds41_row_shard/tests -q
```

The CPU suite checks full-row equivalence on simulated TP2, TP4 and TP8 ranks,
masked rows, collective counts, and pinned image source files. Live tests
must establish output and cache correctness, actual branch admission,
time to first token, decode behavior and usable KV capacity.

## Reference implementation

The signed Ring 47 Hyperflare export `1431adf0` reports an 8,192-token
prefill chunk and RoCEnante proxy ABI 14. Its decoder, attention, and Engram
source files match Hyperflare #330. Its variable-length gather takes the
rank-local view before the opaque call, while #330 takes the view inside
that call. Both pass the selected rows to RoCEnante.

| Change | SparkRing port |
|---|---|
| [Hyperflare #282](https://github.com/NVIDIA/hyperflare-inference/pull/282) | The existing plugin reduce-scatters attention and MoE outputs and retains row-local mHC state. |
| [Hyperflare #283](https://github.com/NVIDIA/hyperflare-inference/pull/283) | Separate B12X prefill plans run full-head attention on owned query rows. Narrow gathers provide complete KV and owned-row builders write the attention metadata. |
| [Hyperflare #284](https://github.com/NVIDIA/hyperflare-inference/pull/284) | Compact routing runs the gate and native selector on each owner's rows and gathers selected expert IDs and FP32 route weights. |
| [Hyperflare #330](https://github.com/NVIDIA/hyperflare-inference/pull/330) | The installer image stages vocabulary-sharded lookup values. SparkRing reduce-scatters these before the replicated Engram projection and mixes the result into the owned residual. Hyperflare's banded reduction operates on projected partials in its own layout. |
| [FlashInfer #106](https://github.com/local-inference-lab/flashinfer/pull/106) and [#107](https://github.com/local-inference-lab/flashinfer/pull/107) | Both heads diverge from the export's FlashInfer revision `4fd836dc`. The NCCL cycle uses a different transport. On open chains, SparkRing's SIRCL subset gather has its own forwarding schedule and publishes the flag after the payload on the lane's queue pair. |
