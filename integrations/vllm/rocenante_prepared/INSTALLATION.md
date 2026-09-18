# Prepared transport image integration

Status: **implemented assets; bounded two/four-rank GPU checks qualified** in the
[image-specific record](../../../performance/records/transport/rocenante-prepared-35cf12b2-20260918.md).
This is an explicit image-extension operation, not a replacement of retained
repository assets or an instruction to modify a running deployment.

## Exact installation boundaries

| Input / destination | Required identity |
|---|---|
| Existing `/opt/sparkring/transports/sparkring_transport_selector.py` | SHA256 `c820cf956d216f0f1e83632121ff3a04c2d743f6f6e2663f1551b27f1cdda50c` |
| Replacement [selector](sparkring_transport_selector.py), same installed path | SHA256 `256a3a4d189d97ccf87f5cb72011bd38ef1c63c27bf335ddc6e78f090fc745b7` |
| Retained `/opt/venv/lib/python3.12/site-packages/sparkring_transport.pth` | SHA256 `c4865202d229c0501368f6f31f810d97dda25e50c0afc12cc4edd4bb8a5695a6`; unchanged |
| [Prepared bundle manifest](manifest.json) | SHA256 `e8577c447a69ac75253758a0964791e862ccca1ad13168e7addefa6ec96369c9` |

Stage with [package.py](package.py), then install its exact contents at
`/opt/sparkring/transports/tp2-rocenante-adaptive-prepared/`. Before writing the
selector or bundle, verify all six absolute `image_source_preimages` entries in
the manifest against the installed B12X package. The image-extension receipt
must retain those checks and the source/destination file inventory. Do not change
the legacy bundle or overwrite either manifest with another identity.

Select on **every** participating rank:

```text
SPARKRING_TRANSPORT_PROFILE=tp2-rocenante-adaptive-prepared
SPARKRING_TRANSPORT_MANIFEST_SHA256=e8577c447a69ac75253758a0964791e862ccca1ad13168e7addefa6ec96369c9
```

The selector preserves the legacy profile on the R37 B12X runtime. On a runtime
containing B12X's prepared API it explicitly refuses that legacy profile and
names the prepared profile instead; it never silently substitutes transport
code. Unknown profiles, source drift, incomplete inventories and late conflicting
imports remain startup failures.

## Two- and four-rank capability

The profile identifier retains a `tp2` prefix, but its communicator supports
world sizes 2 through 16. The preserved peer-path mapping and native kernels
include four-rank communication. This does **not** qualify every topology or
replace the separate TP4 weighted-mesh bundle. Use the existing Qwen profile's
per-rank HCA mapping and host/network setup; do not invent a common map for all
ranks or treat four HCA functions as four selected paths.

## Bounded real-hardware probe

[probe.py](probe.py) requires a reserved idle stack, the installed candidate image,
the exact selected manifest and unchanged profile NIC settings. It loads no
model and changes no network configuration. Run on every rank with a private
rendezvous port; `NODES` is 2 or 4, `RANK` is that node's rank, and `MASTER` is
the coordinated rendezvous address:

```bash
python -m torch.distributed.run --nnodes "$NODES" --nproc-per-node 1 \
  --node-rank "$RANK" --master-addr "$MASTER" --master-port "$PORT" \
  -- /qualification/probe.py --run --output-dir /qualification/results
```

The probe checks FP16/BF16/FP32 reductions against NCCL and identical outputs
across ranks; direct and padded gathers; consecutive misaligned gathers with
independent outputs; four frozen-kernel graph replays with alternating grid
sizes, stable output addresses and no replay allocations; selected HCA traffic;
and proxy health. All ranks must report `passed` in `rank-N.json`. The controller
must enforce an outer timeout and retain logs/image identity with those records.
Run TP4 first when qualifying the Qwen TP4 launch, then repeat on TP2.

These tests do not measure serving throughput or qualify the model/cache path.
Startup/generation and source-matched SparkCache restore checks follow them.
