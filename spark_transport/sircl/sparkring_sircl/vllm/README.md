# SIRCL vLLM adapter

The adapter puts SIRCL ring sessions in front of every collective of every
vLLM process group. A ring session is one group's instance of
`sparkring_sircl.oneshot.AllReduce`, this package's RDMA collective session;
it reaches a Spark that shares no cable with the caller through the hardware
relays of the Sparks between them. A tensor-parallel (TP) group can therefore
serve on Sparks that do not all share cables, for example four consecutive
Sparks of an eight-Spark ring, whose ranks 0 and 3 are two relays apart.
NCCL cannot connect such ranks: it picks devices and addresses without the
relay table, and its queue-pair setup times out.

The adapter is a vLLM platform plugin and a vLLM general plugin, both named
`sircl`. Both do nothing unless `SIRCL_MODE=custom`; when `VLLM_PLUGINS` is
set, it must also list `sircl`. NCCL is off by default (`SIRCL_NCCL=never`):
SIRCL carries every collective of every multi-rank group, and NCCL runs only
when the operator opts in (section [NCCL on groups whose Sparks do not all
share cables](#nccl-on-groups-whose-sparks-do-not-all-share-cables)).

Status: the [component status table](../../STATUS.md#component-status), rows
"vLLM adapter". [`SURVEY.md`](SURVEY.md) lists the collectives vLLM issues
per model and phase and where the adapter carries each one.
[`RUNBOOK.md`](RUNBOOK.md) serves SparkRing profiles on groups of a ring with
the serve launcher (`python -m sparkring_sircl.vllm.serve`,
[`serve/`](serve/cli.py)) and adds SIRCL to containers another launcher
starts (`bundle`). [`STATUS.md`](STATUS.md) lists the adapter's limitations
and the checks before relying on a layout.

## Enable it

1. Install the `sparkring-sircl` distribution in the serving image. It
   registers `sircl` in `vllm.platform_plugins`
   (`sparkring_sircl.vllm.platform:activate`) and `vllm.general_plugins`
   (`sparkring_sircl.vllm.plugin:register`). The serve launcher instead
   mounts the package tree with a generated
   `sparkring_sircl-<version>.dist-info` on `PYTHONPATH`, which makes the same
   entry points visible without changing the image.
2. Set these variables on every rank ([`settings.py`](settings.py)). A
   malformed value raises `SettingError` naming the variable.

| Variable | Meaning | Default |
|---|---|---|
| `SIRCL_MODE` | `custom` enables the adapter in every vLLM process; `disabled` leaves vLLM unchanged | `disabled` |
| `SIRCL_FABRIC` | physical fabric of the Sparks the instance runs on: `ring:N`, `path:N`, `pair` or `pair:2` (two cables) | none; required with `SIRCL_MODE=custom` |
| `SIRCL_RANK_POSITIONS` | fabric position of every global rank, comma-separated in rank order | `0,1,...,world-1` |
| `SIRCL_GROUPS` | vLLM group kinds that get a SIRCL session: `tp`, `dcp` | `tp,dcp` |
| `SIRCL_NCCL` | `never`: no NCCL collective on any multi-rank group; `auto`: NCCL only where the group's cabling allows it; `topology` is another name for `auto` | `never` |
| `SIRCL_LARGE_ALLREDUCE` | all-reduces above the dispatch ceiling: `auto` (NCCL where it may run and no graph is captured, else SIRCL), `sircl` (always SIRCL) or `nccl` (NCCL; refused where NCCL may not run) | `auto` |
| `SIRCL_RELAY_PER_PEER_BYTES` | per-peer bytes of one gather or scatter op on a group with relayed lanes, a multiple of 16 | a measured drop-free value, else 75 % of a 512 KiB hairpin queue over the busiest relay's load |
| `SIRCL_RECEIPT_DIR` | directory for one JSON receipt per rank and group | unset |
| `SIRCL_SESSION_MODULE` | module that provides the session class `AllReduce`, `API_VERSION` and `is_supported`; a comma-separated list is tried in order | `sparkring_sircl.oneshot` |
| `SIRCL_P2P_GROUPS` | group kinds that get point-to-point channels: `pp`, and `tp` and `dcp` groups with a session; `none` disables them | `pp,tp,dcp` |
| `SIRCL_P2P_MODULE` | module that provides the channel class `PointToPoint` | `sparkring_sircl.p2p` |
| `SIRCL_VLLM_SHIMS` | pinned shims the general plugin installs at registration, besides those the communicator installs (section [Shims](#shims)) | none |
| `SIRCL_FUSED_NORM` | `1`: vLLM's post-all-reduce RMSNorm helper runs as one fused SIRCL kernel where the result is bit-identical; `0`: vLLM's all-reduce and RMSNorm | `0` |
| `VLLM_PLUGINS` | vLLM's plugin filter; platform plugins pass it too, so when set it must name `sircl` | unset |

The ring session reads its own variables ([`sparkring_sircl/env.py`](../env.py)).
The TP slot ([`tp_slot.py`](tp_slot.py)) reads the all-reduce capacity,
dispatch ceiling and all-gather capacity (`SIRCL_ALLREDUCE_CAPACITY_BYTES`,
`SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES`, `SIRCL_ALLGATHER_MAX_BYTES`, each
131072 bytes by default); decode context parallel (DCP) sessions read
`GLM_DCP_RDMA_*` ([`dcp_collectives.py`](dcp_collectives.py)).

- `SIRCL_FABRIC` describes the whole instance; the adapter derives each
  session's layout and route map from it and passes both explicitly. Leave
  the session's `SIRCL_LAYOUT` unset in vLLM processes.
- `SIRCL_PEER_ROUTES` is optional. When set, it must equal the map
  `SIRCL_FABRIC` derives; a mismatch on any rank fails setup on every rank.
- Route maps name the RDMA devices by their DGX OS names, `rocep1s0f0` and
  `roceP2p1s0f0` on port 0, `rocep1s0f1` and `roceP2p1s0f1` on port 1,
  unless `SIRCL_FABRIC_DOCUMENT` names a fabric document
  (`sparkring-fabric/v1`) whose port functions give other names
  ([`fabric.py`](fabric.py) takes them from `sparkring_sircl.routes`). Every
  Spark must name its functions alike, and every rank of an instance must see
  the same document.

| Placement | Settings |
|---|---|
| TP4 on Sparks 0-3 of an eight-Spark ring | `SIRCL_FABRIC=ring:8` |
| TP4 on Sparks 4-7, a second instance | `SIRCL_FABRIC=ring:8 SIRCL_RANK_POSITIONS=4,5,6,7` |
| TP2 on Sparks 6-7 | `SIRCL_FABRIC=ring:8 SIRCL_RANK_POSITIONS=6,7` |
| TP8 on the whole ring | `SIRCL_FABRIC=ring:8` |
| TP8 on the whole ring, NCCL's ring algorithm for eager messages above the dispatch ceiling | `SIRCL_FABRIC=ring:8 SIRCL_NCCL=auto NCCL_ALGO=Ring NCCL_SKIP_TREE_CONNECT=1`, with an NCCL that runs its ring between cabled neighbours (each rank's `NCCL_IB_HCA` naming the devices that face them; [patched NCCL](../../../nccl/README.md)) |

## Hooks

The adapter changes no vLLM file. [`hooks.py`](hooks.py) records each hook
with its anchor lines in the pinned vLLM builds, and a CPU test finds every
applicable anchor in every tree it is given.

| Hook | Kind | What it carries |
|---|---|---|
| Platform plugin [`platform.py`](platform.py): returns [`SirclCudaPlatform`](cuda_platform.py), a subclass of vLLM's `CudaPlatform` whose `get_device_communicator_cls` names [`SirclCudaCommunicator`](communicator.py) | official | the device communicator of every group |
| `SirclCudaCommunicator`, a `CudaCommunicator` subclass | official | `all_reduce`, `all_reduce_in_place`, `all_gather`, `all_gatherv`, `reduce_scatter`, `reduce_scatterv`, `gather`, `broadcast`, `send`, `recv`, `batch_isend_irecv`, and an added `all_to_all_single` |
| General plugin [`plugin.py`](plugin.py) | official | before any process group exists: the NCCL environment check, the platform check (vLLM falls back to its own CUDA platform when a platform plugin fails to import), the placement check, the NCCL tripwire and the shims |
| Tripwire carrier ([`guard.py`](guard.py), `GroupAdapter.carry_torch`) | wraps `torch.distributed` | direct `torch.distributed` calls on a group with a session: `all_reduce` (sum, max, min), `broadcast`, `all_gather`, `all_gather_into_tensor`, `reduce_scatter_tensor` (sum) and `all_to_all_single` (equal splits), for example online quantization's weight `amax` reductions and the DCP chunked-context KV gather |
| `b12x_ar_comm` attribute of the TP communicator | attribute vLLM reads | the capture context and post-step health check of every session in the process |
| Pinned shims ([`shims.py`](shims.py)) | pinned by file hash | collectives no official hook reaches (section [Shims](#shims)) |

## Shims

A shim replaces or wraps one piece of vLLM that no official hook reaches. It
installs only when every vLLM file it touches has the SHA-256 recorded for a
pinned build ([`pins.py`](pins.py)); otherwise it raises `ShimRefused` and
replaces nothing (`worker_regimes` logs a warning instead). An installed shim
calls vLLM's original whenever SIRCL does not own the group.

[`shims.json`](shims.json) is the catalog: per shim its purpose, models, the
vLLM definitions it wraps or calls, files, anchors, verified builds, how it is
enabled and disabled, what happens without it, measurements with their
conditions, and what it needs from the session. [`catalog.py`](catalog.py)
generates it (`python -m sparkring_sircl.vllm.catalog --write`), and a CPU
test compares the two.

| Shim | Serves | Installed when | Turned off by |
|---|---|---|---|
| `dcp_all_to_all` | the DCP output combine (`dcp_a2a_lse_reduce`), which calls `torch.distributed.all_to_all_single` directly: GLM-5.3 and other models on vLLM's DeepSeek-V3.2 code | the communicator builds a DCP group of two or more ranks with a session | `SIRCL_GROUPS=tp` |
| `dcp_b12x_transport` | the same groups with the B12X attention backend and `VLLM_USE_B12X_DCP_A2A=1`: vLLM gets no B12X PCIe transport (CUDA IPC, one machine only) for a group SIRCL owns | with `dcp_all_to_all` | `SIRCL_GROUPS=tp` |
| `fused_allreduce_rms_norm` | vLLM's eager all-reduce + residual add + RMSNorm helper | `SIRCL_FUSED_NORM=1` (`bundle --fused-norm on`) | `SIRCL_FUSED_NORM=0`, the default |
| `mhc_prefill_shard` | GLM-5.3-Flash's mHC prefill row ownership, which reduce-scatters and all-gathers on PyNccl directly: vLLM's ownership code gets a PyNccl stand-in that runs SIRCL's plans | a TP group without PyNccl gets a session while `VLLM_GLM53_MHC_PREFILL_SHARD` is set | `--mhc-prefill-shard off` (`VLLM_GLM53_MHC_PREFILL_SHARD=0`) |
| `qwen_hc_prefill_shard` | Qwen3.8's hyper-connection prefill row ownership, through the same stand-in | a TP group without PyNccl gets a session while `VLLM_QWEN3_8_HC_PREFILL_MODE=shard` | `--env VLLM_QWEN3_8_HC_PREFILL_MODE=off` |
| `roce_slot` | vLLM's RoCE all-reduce slot of every TP group builds SIRCL's slot class | the general plugin sees `VLLM_ENABLE_ROCE_ALLREDUCE=1` | `VLLM_ENABLE_ROCE_ALLREDUCE=0`, which both launchers set |
| `worker_regimes` | the worker's warm-up, memory profiling, sleep, wake-up, weight loads and profiler run in the sessions' startup flag-wait regime; the warm-up's return arms the serving regime | the first group that gets a session | no switch |

With NCCL off every multi-rank group is built without PyNccl, so the two
prefill row-ownership shims serve pairs as well as four-Spark groups.
`SIRCL_VLLM_SHIMS` names shims the general plugin installs at registration
whatever the conditions above. A launch that needs `dcp_all_to_all`,
`dcp_b12x_transport`, `mhc_prefill_shard` or `qwen_hc_prefill_shard` on a
vLLM where the shim refuses fails setup on every rank and names the setting
that avoids it; without `worker_regimes` the sessions keep the startup
flag-wait limit while serving.

| Pinned build (`pins.SUPPORTED`) | Source |
|---|---|
| `lil-image-aba309e4610c` | the Local Inference Lab vLLM fork installed in serving image `aba309e4610c` (`0.1.dev21553+gab86b7073.d20261001`) |
| `lil-karmic-kraken-beta-4a87c588` | `local-inference-lab/vllm` `integration/karmic-kraken-beta` at `4a87c588`; it has no mHC or Qwen3.8 row-ownership module, so `mhc_prefill_shard` and `qwen_hc_prefill_shard` are not verified there |
| `sparkring-kraken-beta-20261007-bc9ea774` | SparkRing's vLLM branch `sparkring/kraken-beta-20261007` at `bc9ea774`: the `aba309e4610c` build merged with `integration/karmic-kraken-beta`, with a source overlay |

`python -m sparkring_sircl.vllm.serve shims` prints the catalog; with
`--vllm-tree PATH` (a `vllm` package or a checkout) each shim's status there,
with `--probe FILE` the status in the vLLM a saved `SIRCL-SERVE-PROBE` line
saw, and with `--json` the records. `stage` and `bundle --stage` print the
statuses in every Spark's image.

| Status | Meaning | Installs |
|---|---|---|
| `verified` | the files the shim touches match a pinned build | yes, when its condition holds |
| `applicable-unverified` | its files, the definitions it wraps or calls and its hook anchors are present, but no pinned build matches the files | no: it refuses until a matching build is pinned |
| `absent` | a file, definition or anchor it relies on is missing | no |

To verify the shims of another vLLM build:

1. Take the `applicable-unverified` shims of `shims --vllm-tree PATH` and
   compare each one's catalog `code` definitions and anchor lines with the
   pinned build the tree derives from. Verify a shim only when every
   definition it wraps or calls is unchanged and every anchor is present; a
   shim whose code changed needs an adapted shim and a review.
2. Add a `VllmBuild` to `pins.SUPPORTED` with the hashes
   `python -m sparkring_sircl.vllm.pins PATH` prints (`None` for a missing
   file), its source commit, and a comment naming the hook files that differ
   and why the shims' code is unchanged.
3. Add the build to the `builds` of every hook in [`hooks.py`](hooks.py) whose
   code it has, run `python -m sparkring_sircl.vllm.catalog --write`, and run
   the tests with `SIRCL_TEST_VLLM_ROOTS` naming the tree.

## NCCL on groups whose Sparks do not all share cables

[`fabric.py`](fabric.py) classifies every group from `SIRCL_FABRIC` and the
rank positions. Under `SIRCL_NCCL=never`, the default, every multi-rank group
is treated as `none`, whatever its cabling. Under `SIRCL_NCCL=auto` the
cabling decides:

| Policy | Cabling | NCCL may run |
|---|---|---|
| `all` | every pair of ranks shares a cable: a pair, a triangle | anything |
| `ring` | consecutive ranks share cables, last to first included: a whole ring in rank order | its ring algorithm only (all-reduce, all-gather, reduce-scatter, broadcast, and point-to-point between cabled ranks), with `NCCL_ALGO=Ring` and `NCCL_SKIP_TREE_CONNECT=1`; without both the group is `none` |
| `none` | some consecutive pair shares no cable: four Sparks of a larger ring, a DCP4 path inside TP8, strided subgroups | nothing |

The adapter enforces the policy in four places:

1. **Construction.** For a `none` group, `SirclCudaCommunicator` builds its
   parent class with `VLLM_DISABLE_PYNCCL=1` set for that call only, because
   PyNccl's constructor runs a warm-up all-reduce over the whole group. It
   refuses when vLLM has already cached its environment.
2. **Lazy NCCL.** torch creates a group's NCCL communicators on first use,
   and nothing uses those of a `none` group. The general plugin refuses
   `VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1` with `NCCL_RUNTIME_CONNECT=0`, which
   would connect every communicator, the world group's ring included, at
   creation. With `SIRCL_NCCL` unset or `never`, it refuses
   `VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1` alone, which creates an NCCL
   communicator over every rank at startup. With NCCL off the
   serve launcher's `plan` also refuses `--load-format instanttensor` and
   `--enable-eplb` ([Serving without NCCL](RUNBOOK.md#serving-without-nccl)).
3. **Dispatch.** Every plan that names NCCL is checked against the group's
   policy before the call ([`guard.py`](guard.py), `GuardedGroup.check`).
4. **Tripwire.** The `torch.distributed` collective functions are wrapped: a
   call on an NCCL group (registered, or classified on first use such as the
   default group) that NCCL may not run raises `NcclAcrossRelayError` naming
   the group, operation, uncabled pair and remedies. On a group with a
   session the group's carrier runs the calls the hook table lists instead.
   Calls from C++ inside compiled graphs are not covered; the passes that
   make them are refused at setup.

The communicator also refuses at setup the settings whose collectives bypass
it:

| Group | Refused |
|---|---|
| `none` TP group | `enable_batch_sharded_sampling`, `fuse_gemm_comms`, `fuse_allreduce_rms` and all-to-all backends other than the naive one; without a session also `VLLM_GLM53_MHC_PREFILL_SHARD`, `VLLM_QWEN3_8_HC_PREFILL_MODE=shard` and adaptive verification (with a session the tripwire's carrier runs its broadcast). The serve launcher's `plan` refuses the same recipe settings before any container starts |
| `none` DCP group | `VLLM_USE_DIRECT_DCP_A2A`, `VLLM_USE_DIRECT_DCP_Q_GATHER`, `VLLM_USE_DIRECT_DCP_KV_GATHER` (peer GPU memory maps work within one machine only), `VLLM_MLA_PREFILL_DCP_OVERLAP=1` (PyNccl directly) and prefill context parallelism above 1 |
| every group with a session | vLLM's micro-batching (`--enable-dbo`, or `--ubatch-size` above 1): each micro-batch's forward runs in its own thread, and a session's operations must be issued in one order on every rank |

## Dispatch

[`planner.py`](planner.py) decides each call from facts every rank shares:
the group's policy, the session's agreed limits (capacity C, dispatch ceiling
D, gather capacity G, prepared dtypes, relay-safe op size), the call's dtype,
shape, size and dimension, and whether a CUDA graph is being captured; never
from pointers. [`executor.py`](executor.py) issues the session ops on the
caller's stream.

| Collective | Within one op | Larger, or a shape one op does not take |
|---|---|---|
| all-reduce of a prepared dtype (BF16; FP16 and FP32 too on a `none` group) | one op up to D | eager, where NCCL may all-reduce and `SIRCL_LARGE_ALLREDUCE` is not `sircl`: NCCL; otherwise the session's `all_reduce_large` (method `large`), which chooses its pieces and schedule; a session without it gets pieces of at most D from the host (method `chunked`) |
| all-reduce, other dtypes | — | NCCL where it may run; otherwise all-gather, then a rank-ordered local sum (method `gather_sum`) |
| all-gather, any dimension | one op up to the gather op size (G, capped by the relay-safe per-peer size) | eager, where NCCL may run: NCCL; otherwise `all_gather_large` (method `large`), or rows of the `[outer, inner]` view or column tiles of a row; bool, complex and FP8 move as bytes. A column gather the session would run on its ring or chain as a dimension-0 gather runs as that gather plus one local copy ([Column gathers](#column-gathers)) |
| reduce-scatter | the session's reduce-scatter where prepared: the whole message in one call when the session states its op size, else strided calls | eager, where NCCL may run: NCCL; otherwise all-reduce, then this rank's chunk |
| all-gatherv, reduce-scatterv with uneven sizes | — | padded all-gather; all-reduce, then this rank's rows |
| broadcast, gather | — | eager, where NCCL may run: NCCL; otherwise all-gather of the bytes, then the source's copy or the concatenation on the destination |
| all-to-all | the session's all-to-all where prepared (DCP groups) | eager on an `all` group: NCCL; otherwise all-gather, then this rank's chunks |
| send, recv, isend, irecv, batch_isend_irecv | the group's point-to-point channels, eager (method `direct` or `relayed`) | without a channel: NCCL between cabled ranks of an `all` or `ring` group, else refused |

A session with a measured tuning table (`SIRCL_TUNING_TABLE`,
[`../tuning.py`](../tuning.py)) chooses its own algorithm, schedule and
piece per op, and applies the session settings the table records (link
slots, link slot, chain slot, large-message piece) where its environment
leaves them unset. A table chooses only among SIRCL's settings: its marks of
where NCCL measured faster are measurements, and the adapter routes no call
to NCCL by them in any `SIRCL_NCCL` mode or `SIRCL_LARGE_ALLREDUCE` value: the
planner takes no table input (`sessionapi.tuned_backend` reads a mark; no plan
uses it). Where the opt-in `SIRCL_NCCL=auto` lets NCCL run, the rules above
decide what it carries.
Composed plans give every rank the same bits; a chained large all-reduce may
differ from the rank-ordered sum in the last place, the same on every rank.

## Column gathers

A column gather is an all-gather along a dimension with more than one row in
front of it, such as a column-split projection's output gathered along its
last dimension (GLM-5.3 at TP8 gathers `[8192, 328]` BF16 per rank at an
8,192-token prefill chunk). Each rank's shard lands in one piece per row of
the output, while the session's ring and chain all-gathers carry one
contiguous piece per rank, so `all_gather_large` along that dimension runs as
tiled ops.

[`executor.py`](executor.py) (`ColumnGather`, `column_gather_route`) runs such
a call as one dimension-0 all-gather of the shard's bytes into a staging
buffer laid out as `[world, *shard]`, then one local copy into the requested
layout, when the plan sends the call to `all_gather_large` and the session
would run a dimension-0 gather of the same bytes on its ring or chain
(`gather_uses_ring` or `gather_uses_chain`: its gather schedule, minimums,
tuning table and capture mode). These facts are the same on every rank. The
copy moves each row in the widest integer view that divides it, so the
output is bit-identical to the tiled gather's for every dtype.

- **Switch.** `SIRCL_COLUMN_GATHER`: `1` (default) or `0`, which keeps
  `all_gather_large` along the dimension. It is read when a group is set up
  and must be the same on every rank; the bundle sets it with
  `--column-gather on|off`.
- **Memory.** Outside CUDA graph capture each session keeps one staging
  buffer, grown to the largest `world * shard` bytes it carried (41 MiB for
  the GLM-5.3 shard above at TP8). A captured call takes its staging from the
  graph's memory pool.
- **Capture.** No host synchronization and no compilation: the ring and chain
  launchers are compiled when the group is prepared.
- **Receipt.** `column_gather=on|off` in the receipt line;
  `column_gather_detail` in the JSON receipt names the calls each route
  (`ring`, `chain`) carried and the staging bytes kept.
- **Ring harness.** With `--eager-path adapter`, the harness's column cases
  go through the same executor; the summary's adapter methods read
  `large+column-ring` or `large+column-chain` where a call was staged.

## Point-to-point

vLLM's pipeline parallelism (PP) passes activations between stages with
`torch.distributed.isend` and `irecv` and broadcasts the sampled tokens; the
device communicator offers `send`, `recv` and `batch_isend_irecv`. SIRCL
carries them on point-to-point channels (`sparkring_sircl.p2p`): each ordered
pair of a group's ranks is a first-in first-out channel of RDMA writes into
pinned host slots, paced by the receiver's credits; a pair that shares no
cable uses relayed lanes.

- **Groups.** Every PP group of two or more ranks, and every TP or DCP group
  with its own session, gets channels when `SIRCL_P2P_GROUPS` names its kind;
  a group with the same global ranks in the same order shares them. A PP
  group routes over the fabric of every rank of the instance.
- **Windows.** [`p2p.py`](p2p.py) plans every group's channels, the same on
  every rank: sessions' forward windows first, then PP lanes, then session
  groups' channels, within 75 % of every relay hairpin queue. A pair left
  without one chunk (`SIRCL_P2P_CHUNK_BYTES`, 32 KiB) of window has no
  channel; the receipt names the queue.
- **Dispatch.** Point-to-point calls, from the communicator or as direct
  `torch.distributed` calls, go to the channels first, whatever NCCL may do;
  a broadcast on a group without a session runs as sends from the source. A
  PP group NCCL may not run fails setup on every rank when its channels
  cannot be built.
- **Semantics.** Tag-free (a nonzero tag is not carried). The n-th receive
  toward a peer takes the n-th message that peer sent; a different byte count
  fails the channels on both ranks. A rank issues the sends a peer waits for
  before the receives that wait on that peer, as vLLM's PP order does.
  Calls under CUDA graph capture are refused.

Evidence: CPU tests on emulated ranks
([`tests/test_vllm_p2p.py`](../../tests/test_vllm_p2p.py)), the native
layer's simulator and binding tests, and the GPU emulation of the channels
(`python -m sparkring_sircl.testing.p2p_emulation`). The channels have not
run on a ring or served PP.

## Sessions

The TP group's session is the slot class [`tp_slot.py`](tp_slot.py)
(`SirclRingAllReduce`): the instance vLLM built in its RoCE slot, or one the
communicator builds with the route map and layout `SIRCL_FABRIC` derives. On
a `none` group it also prepares FP16 and FP32 all-reduce, because nothing
else can carry them.

DCP groups use [`dcp_collectives.py`](dcp_collectives.py)
(`SirclDcpCollectives`), which routes over the TP group's cables, uses
relay-safe op sizes and requires the session's scatter collectives (without
them a DCP group fails setup on every rank). At TP8 with DCP 4, the two DCP
groups (ranks 0-3 and 4-7) hold their own sessions at the same time. The
schedule and link variables (`settings.TP_SESSION_VARIABLES`, such as
`SIRCL_LARGE_SCHEDULE` and the `SIRCL_CHAIN_*` and `SIRCL_LINK_*` sizes)
apply to the TP session only; a DCP session keeps the session defaults,
because chain and ring ops need every Spark of the session's fabric to host
one of its ranks.

A group on a `none` placement whose global ranks equal, in order, those of a
group with its own session shares that session (voted; the TP group's is
preferred). vLLM builds an expert-parallel (EP) group for every
mixture-of-experts model; at data parallelism 1 its ranks are the TP group's,
so it shares the TP session (receipt `session=shared:tp:0`), and online
weight quantization's `amax` reductions at startup run there. Each rank
issues both groups' collectives from one thread in program order. Any other
group on a `none` placement without a session refuses every collective.

## Fused all-reduce + residual add + RMSNorm

Status: research-only ([component status](../../STATUS.md#component-status));
off by default.

vLLM's DeepSeek-V3.2 code, GLM-5.3 among its models, hands every attention
and MLP output un-reduced, with the residual stream and the next RMSNorm, to
the eager helper `fused_allreduce_rms_norm`: two calls per decoder layer. On
GB10 that helper is the TP all-reduce followed by vLLM's RMSNorm. With
`SIRCL_FUSED_NORM=1`, [`norm_fusion.py`](norm_fusion.py) binds the fused
kernels of `sparkring_sircl.fused_norm` to the TP group's session, and the
`fused_allreduce_rms_norm` shim runs a qualifying call as one collective:
vLLM's plain `RMSNorm`, BF16 contiguous rows of the model's hidden size, 1 to
the GPU's multiprocessor count of rows (48 on GB10), within the dispatch
ceiling. Any other call runs the helper unchanged.

The result is bit-identical to SIRCL's all-reduce followed by vLLM's
`vllm_c` `fused_add_rms_norm` kernel, the CUDA default without inductor
compilation. Setup refuses the switch on every rank where vLLM would use
another provider (`native` under inductor compilation, or
`VLLM_BATCH_INVARIANT=1`) and wherever the kernels cannot bind. The receipt
shows `fused_norm=on` and counts fused calls as
`all_reduce/sircl/fused_rms_norm`. Bit-identity is checked in GPU emulation
on an RTX 5090 against a NumPy model of vLLM's kernel, not on GB10.

## CUDA graph capture

- Every dtype a group may all-reduce under capture is prepared when the group
  is built. When any session schedule is not `pieces`, every prepare call
  passes `links=True`, so the chain and ring collectives compile then and
  never inside a capture or while peers wait.
- Captured calls stay on SIRCL. The executor never creates or switches
  streams and never synchronizes; the session enforces one stream per capture.
- vLLM enters only the TP, PP and DP groups' capture contexts. The TP slot's
  `capture` is chained to every session of the process
  (`adapter.capture_all`), so DCP sessions enter capture mode too.

## Fail-stop and health

Setup is collective: route checks, settings, construction and preparation
are voted, and a failure on any rank raises on every rank. After setup, a
refused plan raises `SirclDispatchError` on every rank alike; a timeout or
progress-thread error poisons the session, and the worker's post-step
`check_health` checks every session through the TP slot
(`adapter.check_all_health`) and raises for a poisoned one. Nothing is
retried on another backend.

A session waits for a late peer at most the limit of its flag-wait regime:
`startup` (`SIRCL_STARTUP_WAIT_S`, 600 s) or `serving`
(`SIRCL_SERVING_WAIT_S`, 20 s). Sessions start in the startup regime. The
`worker_regimes` shim runs the worker's warm-up, profiling, sleep, wake-up and
weight loads in the startup regime (`adapter.startup_all`) and arms the
serving regime when the warm-up returns; the first post-step check after
that (the engine's first real step) puts every session in the serving regime.

## Receipts

Every rank logs one line per group when the group is ready, for example
(wrapped here; `...` marks session-dependent values):

```text
SIRCL receipt group=tp:0 global_rank=3 rank=3 world=4 layout=ring:8 fabric=path:0-1-2-3
positions=0,1,2,3 nccl=none pynccl=skipped session=ring lanes=2 hcas=... capacity=...
dispatch=... oneshot_max=... gather=... op_per_peer=262144 large_piece=... gather_piece=...
schedules=... chain_min=... ring_min=... links=... tuning=- mhc=off fused_norm=off
p2p=... wait=startup:600s vllm=... state=ready
```

With `SIRCL_RECEIPT_DIR`, the same record is written as
`rank<global>-<group>.json` (schema `sircl-vllm-receipt/v1`,
[`receipt.py`](receipt.py)). The JSON adds `nccl_mode` (the mode the adapter
resolved, `never` or `auto`; `topology` is read as `auto`) and `nccl_rule`
(`NCCL: opt-in only (auto); tables choose among SIRCL options`), the plan
counters (calls per collective, backend and method), `fused_norm_detail`, the directories the
process imported `vllm` and `b12x` from (which show whether a source overlay
serves), the session's `stats()`, and `p2p_detail` (channels, relayed peers,
and pairs without a channel with the reason). The worker's post-step check
rewrites it: with fresh statistics after a step that added a row or changed
the wait regime and at least every 60 s, with the counts alone at most once a
second, and when the group closes. A receipt without any `nccl` row shows
that NCCL carried nothing on that group. `SirclCudaCommunicator.sircl_report()`
returns the same record at any time.

## Other SparkRing vLLM integrations

| Integration | With SIRCL |
|---|---|
| SparkRing's four-rank all-reduce and vocabulary adapters ([`integrations/vllm/`](../../../../integrations/vllm/README.md), installed from `sitecustomize`) | composed on a four-Spark ring: signatures the four-rank all-reduce adapter admits use its sessions, when its source matches the revision [`tp4.py`](tp4.py) pins by hash; refused on any other four-rank TP placement while enabled. The serve launcher and `bundle` turn every four-rank hook off with `SPARK_TP4_ENABLED=0` and empty `VLLM_SPARK_TP4_MODE` and `VLLM_SPARK_TP4_VOCAB_MODE` |
| SparkRing's `prepared` transport (RoCEnante) of the published installer images, which runs in vLLM's RoCE all-reduce slot (`VLLM_ENABLE_ROCE_ALLREDUCE=1`, selected by `SPARKRING_TRANSPORT_PROFILE`) | refused: a TP group whose RoCE slot holds another transport fails setup, because one RDMA transport serves a group. The launchers set `VLLM_ENABLE_ROCE_ALLREDUCE=0` and empty `SPARKRING_TRANSPORT_PROFILE` |
| The GLM-5.3 collective-routing overlay in `integrations/vllm/`, which wraps `CudaCommunicator.all_reduce` | refused at setup: it and SIRCL ring sessions cannot serve one run |

Registration is order-independent: the tests install each integration before
and after SIRCL's plugin and get the same outcome.
