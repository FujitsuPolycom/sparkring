# SIRCL ring sessions

SIRCL, SparkRing's Switchless Inference RDMA Collective Layer, carries
tensor-parallel and decode-context-parallel (DCP) collectives between DGX
Sparks over RoCE without a switch. This package, `sparkring-sircl` 0.3.1
(path `spark_transport/sircl`, import name `sparkring_sircl`), holds its ring
sessions: collectives for groups of 2 to 8 Sparks cabled as a ring, a path of
consecutive Sparks, a pair or a triangle, for several independent groups on
one ring, and for subgroups such as DCP groups. Members of a group that share
no cable reach each other through the ConnectX-7 of every Spark between them,
which relays their packets in hardware.

Every component's status and evidence are in one table:
[STATUS.md, Component status](STATUS.md#component-status).

## How it works

### Sessions and lanes

A session (`sparkring_sircl.oneshot.AllReduce`) is one rank's membership in
one group's collective instance. It holds:

- an arena: pinned host memory that the GB10 GPU addresses at its host
  pointer and that every opened RDMA device registers with `ibv_reg_mr`.
  Peers write into it and kernels read it in place;
- a command ring of 32-bit words in the arena, through which kernels hand
  ops to the progress thread;
- one reliable-connected queue pair per lane, where a lane joins one local
  RDMA device to one RDMA device of a peer;
- a native progress thread that posts the RDMA writes.

Every peer is reached over one or two lanes. A rank's route map names the
local device of each lane: `SIRCL_PEER_ROUTES` (or `peer_routes=`), written
`<peer>=<device>[/<device>],...`. `sparkring_sircl.routes` derives route maps
from a layout, which names the group's cables and every rank's fabric
position:

| Layout | Meaning |
|---|---|
| `ring:8` | an eight-Spark ring, rank `r` at position `r` |
| `ring:8:0,4` | positions 0 and 4 of that ring: a subgroup on its parent's fabric |
| `ring:3` | a triangle |
| `path:0-3` | the path of the Sparks at positions 0 to 3 |
| `cables=0.port0-1.port1;positions=0,1` | explicit cables and positions |

```python
from sparkring_sircl import routes

group = routes.derive_routes(routes.Layout.parse("path:0-3"), lanes=2)
group.route_text(0)   # '1=rocep1s0f0/roceP2p1s0f0,2=rocep1s0f0/roceP2p1s0f0,3=rocep1s0f0/roceP2p1s0f0'
```

Lanes follow the shortest paths over the group's own cables, lane 0 on a
primary PCIe function and lane 1 on a secondary one. Device names are the DGX
OS names `rocep1s0f0`, `roceP2p1s0f0` (port 0) and `rocep1s0f1`,
`roceP2p1s0f1` (port 1). `SIRCL_FABRIC_DOCUMENT` names a fabric document
(schema `sparkring-fabric/v1`) whose port functions give other names; every
Spark must name its functions alike (`routes.load_roles`). Independent groups
on one ring must share no cable and no Spark (`routes.isolation_problems`).

A payload is striped over a peer's lanes. Each lane posts its stripe as one
RDMA write and then, on the same queue pair, a 4-byte write of the op's
sequence number into the lane's flag line in the peer's arena. A flag that
shows the sequence proves its stripe landed; kernels wait on these flags.

### Relays and forward windows

Members without a shared cable reach each other through relays: per Spark,
origin routes, destination tags and NIC relay filters that forward a lane's
packets out of the other port on the same function class. Sessions read host
networking and never configure it. The relay plan installer `sircl-fabric`
(`sparkring_sircl.fabric`; [RUNBOOK.md, Relay plan installer](RUNBOOK.md#relay-plan-installer))
derives each plan from `sparkring_sircl.routes`, so installed relays and
session lanes agree. A lane crosses at most `SIRCL_MAX_RELAYS` relays
(default 3).

A relay forwards through a hairpin queue (512 KiB per egress function and
direction, `SIRCL_HAIRPIN_QUEUE_BYTES`) that cannot pause its sender. With a
layout, every lane through a relay gets a forward window: the progress thread
posts its stripe in signaled chunks (`SIRCL_FORWARD_CHUNK_BYTES`, 32 KiB) and
keeps at most the window unacknowledged. A window is the largest multiple of
the chunk that is at most `SIRCL_FORWARD_WINDOW_BYTES` (128 KiB) and, for
every queue the lane crosses, at most 75 % of the queue divided among the
lanes that share it. Bytes leave a window when their write completes or,
with `SIRCL_FORWARD_PROOF=1` (default), as soon as a flag proves them
delivered: a peer's kernel starts an op only after it finished every earlier
op, so the peer's flag of op F proves it received the arena writes of every
op before F. A session without a layout does not know which lanes cross
relays and has no windows, so give a layout whenever lanes cross relays.

### Setup agreement

Construction is collective over `exchange_group`, a CPU process group of the
session's ranks (gloo in vLLM):

1. Each rank validates its route map against the layout, opens its devices,
   resolves GID indices and builds a setup record and a connection record.
2. One all-gather exchanges them (`sparkring_sircl.agreement`). Setup fails
   on every rank, with one message naming each failing rank and reason, when
   any rank failed locally, when a shared setting differs from rank 0's, when
   lane counts differ, or when lanes do not pair. Shared settings include the
   capacities, kernel geometry, wait limits, schedules, minimums, link pieces,
   staggers, the layout and the tuning table's hash; devices, GID indices and
   route maps are rank-local.
3. The ranks connect their lanes, prove every lane with one small write
   (`lane_check_ms`, default 2,000), exchange verdicts and start their
   progress threads.

### Progress thread and kernels

The progress thread is plain C (`oneshot/_roce_proxy.c`, libibverbs and
pthreads). It watches the command ring's doorbell, posts each lane's stripes
and flags, forwards chain and ring traffic between neighbours without the
kernel, and records failures in the command ring. `sparkring_sircl.build`
compiles it into the build cache as `roce_proxy-<16 hex digits of the
source's SHA-256>.so`, so each source revision has its own library; the
binding `oneshot/_proxy.py` checks the native ABI version (9, the wire
contract peers compare) and the library's local feature identity
(`roce_local_features`: bit 0, `roce_destroy` returns the number of verbs
calls that failed; bit 1, link op word bit 24, a rank's own items as flags
only) when it loads the library. It refuses a library whose features lack
`REQUIRED_FEATURES` (bit 0), whether the source build, a path or
`SIRCL_NATIVE_LIBRARY` names it, such as an earlier build; the point-to-point
binding checks `p2p_local_features` the same way. `SIRCL_PROGRESS_CPU` or
`progress_cpu=` pins the thread.

Kernels are written in the CUTLASS CuTe DSL and compiled per dtype, group
size, rank, lane count and geometry. Every op is one kernel launch that
stages the input, rings the doorbell, waits for every peer's lane flags
and reduces or gathers in place. `prepare()` compiles every launcher a session
can need; a CUDA graph capture that reaches an unprepared launcher raises.
The DSL's disk cache (`CUTE_DSL_CACHE_DIR`) lets later processes load compiled
kernels. Eager launches of the one-shot, two-shot and all-gather kernels pass
a prebuilt argument block, checked byte for byte against the DSL's own
argument conversion when the launcher is built (`SIRCL_FAST_LAUNCH=0` keeps
the DSL's call). In those kernels block 0 polls the peers' flags in host
memory and hands arrival to the other blocks through device memory
(`SIRCL_FLAG_POLLERS=one-block`; `every-block` polls from every block).

### Flag waits and fail-stop

A kernel waits for a peer's flag at most the session's wait limit, measured
on the GPU clock. The limit is a command-ring word, so a change applies to
eager calls and graph replays from the following launch on.

| Regime | Limit | Variable | Use |
|---|---|---|---|
| startup | 600 s | `SIRCL_STARTUP_WAIT_S` | compilation, warm-up and graph capture, while peers may lag for minutes |
| serving | 20 s | `SIRCL_SERVING_WAIT_S` | steady serving, where a longer lag means a failed peer |

A session starts in the startup regime. `enter_serving()` selects the serving
regime; `enter_startup()` and `with session.startup():` select startup again.

Fail-stop: a wait beyond the limit, or a progress-thread failure, poisons the
session. `check_health()` then raises, naming the rank, peer, lane, limit and
regime; later launches do nothing, and the failed op's outputs are not to be
trusted. A session never retries a collective on another backend.

### Exactness

The one-shot and two-shot all-reduce, two-shot pieces and scatter ops store
the float32 sum of every rank's values in rank order, rounded once: every
rank stores identical bits, whatever the piece size. Chain and ring
schedules add along the chain or around the ring and round at every hop: the
bits are deterministic for a size and order and identical on every rank, and
can differ in the last place from the rank-ordered sum.
`sparkring_sircl.references` reproduces every schedule bit for bit.
All-gathers and all-to-alls copy bytes unchanged. Per-algorithm rules:
[STATUS.md, Algorithms and exactness](STATUS.md#algorithms-and-exactness).

## Install and build

```bash
pip install ./spark_transport/sircl             # no dependencies
pip install './spark_transport/sircl[kernels]'  # adds cuda-python and nvidia-cutlass-dsl
sircl-prepare                                   # builds the native session library into the build cache
python -m sparkring_sircl.p2p.build             # builds the point-to-point library
```

- Dependencies: none. Route planning, the native build, the relay plan
  installer and the CPU simulator use only the standard library. Torch, CUDA
  Python and the CuTe DSL come from the serving image; the `kernels` extra
  (`cuda-python>=12.6`, `nvidia-cutlass-dsl>=4.2`) installs the latter two
  where no image provides them, and the `test` extra installs pytest and
  numpy. Importing `sparkring_sircl` imports neither torch nor vLLM;
  `sparkring_sircl.oneshot` loads torch, CUDA Python and the DSL on first
  access to the session class.
- Native libraries: compiled with the host C compiler (`CC`, else `gcc`,
  `cc`, `clang`) and the libibverbs development headers as
  `-O2 -std=gnu11 -shared -fPIC`, linked against `ibverbs`, `pthread` and
  `dl`. The cache is `SIRCL_BUILD_CACHE_DIR`, else `<XDG cache home>/sircl/roce`.
  A library cached for the same source digest loads without compiling;
  without one, session setup compiles it, so deployments run `sircl-prepare`
  at image build or host preparation. `sircl-prepare --print-path` prints the
  library's path and `--force` rebuilds. `SIRCL_NATIVE_LIBRARY` and
  `SIRCL_P2P_NATIVE_LIBRARY` load prebuilt libraries instead.
- RoCE GID indices: `sparkring_sircl.roce_gid` imports SparkRing's resolver,
  `integrations/vllm/spark_roce_gid.py`, which selects the RoCE v2 GID entry
  that carries the device's IPv4 fabric address. A repository checkout
  loads it from that path. The serve launcher's and the ring harness's
  staged trees carry it as `spark_roce_gid.py` at the top of the tree, beside
  `sparkring_sircl`. A copy installed elsewhere needs a directory on
  `sys.path` that provides `spark_roce_gid` (for example
  `PYTHONPATH=integrations/vllm`); without one, importing the session raises
  `ImportError`. `SIRCL_GID_INDEX` (fallback `NCCL_IB_GID_INDEX`) skips
  resolution and uses one index for every device.

## Public API

### Sessions: `sparkring_sircl.oneshot`

```python
import torch
from sparkring_sircl.oneshot import AllReduce

session = AllReduce(
    exchange_group=cpu_group,            # gloo process group of the session's ranks
    device=torch.device("cuda", 0),
    max_size=2 << 20,                    # all-reduce capacity
    max_gather_bytes=2 << 20,            # all-gather shard capacity
    layout="path:0-3",                   # else SIRCL_LAYOUT
    progress_cpu="19",                   # else SIRCL_PROGRESS_CPU
)                                        # route map: peer_routes= or SIRCL_PEER_ROUTES
session.prepare((torch.bfloat16,), padded_gather=True)   # before any CUDA graph capture
out = session.all_reduce(x)                       # up to max_size bytes
big = session.all_reduce_large(hidden)            # any size
logits = session.all_gather_large(shard, dim=-1)  # any size, any dimension
session.enter_serving()
session.check_health()
session.close()                          # collective; None after a healthy close ("Close")
```

Module exports: `AllReduce` (also `RoceOneshotAllReduce`), `API_VERSION` (1),
`SUPPORTED_DTYPES` (float16, bfloat16, float32), `SUPPORTED_WORLD_SIZES`
(2 to 16; Spark fabrics hold 2 to 8), `DEFAULT_MAX_SIZE` (2 MiB),
`DEFAULT_MAX_GATHER_BYTES` (16 MiB), `ALGORITHMS`, `ALGORITHM_CHOICES`,
`SCATTER_MODES`, `MAX_LANES` (2), `MAX_DEVICES` (4),
`is_supported(device=None)` (an integrated GPU and an active RDMA device),
`discover_hcas(gid_index=None)` and `default_gid_index()`.

Constructor keywords beyond the example, each defaulting to the variable in
parentheses ([Environment](#environment)): `peer_routes`
(`SIRCL_PEER_ROUTES`), `hca_names` (`SIRCL_DEVICES`, else the route map's
devices), `gid_index` (`SIRCL_GID_INDEX`), `threads` (`SIRCL_THREADS`),
`blocks` (`SIRCL_BLOCKS`), `algorithm` (`SIRCL_ALLREDUCE_ALGORITHM`),
`post_order` (`SIRCL_POST_ORDER`), `dispatch_limit_bytes`
(`SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES`), `spin_limit` (`SIRCL_SPIN_LIMIT`),
`startup_wait_s` and `serving_wait_s`, plus `lane_check_ms` (2,000). The
class methods `from_exchange_group` and `from_process_group` build a session
from a group and a capacity.

Contract: every rank issues the same collectives with the same sizes in the
same order; a session runs one collective at a time, in launch order across
streams.

| Collective | Accepts |
|---|---|
| `all_reduce(inp, *, out=None, stream=None, algorithm=None)` | contiguous float16, bfloat16 or float32, a multiple of 16 bytes up to `max_size`, as one one-shot or two-shot launch; `auto` runs one-shot up to `oneshot_max_bytes`, two-shot above. A tuning table's or built-in plan's ring or chain choice for the size belongs to `all_reduce_large` and is neither applied nor counted here |
| `all_reduce_large(inp, *, out=None, stream=None)` | float16, bfloat16 or float32 of any size; `large_reduce_plan(nbytes)` names its ops, the same on every rank whatever each rank's pointer alignment ([Pointer alignment](#pointer-alignment)); a tail below 16 bytes travels zero-padded |
| `all_gather(inp, *, dim=-1, out=None, stream=None)` | any plain dtype along dimension 0 or the last, shards up to `max_gather_bytes`; unaligned shapes take a padded path |
| `all_gather_large(inp, *, dim=-1, out=None, stream=None)` | any dense tensor, any dimension, any size; shapes that are not 16-byte rows on 16-byte-aligned tensors need `prepare(padded_gather=True)` before a capture |
| `reduce_scatter(inp, *, out=None, stream=None, chunk_bytes=None, src_stride_bytes=None)` | `W` chunks, contiguous or strided; stores chunk `rank` of the sum |
| `all_to_all(inp, out, *, stream=None, chunk_bytes=None, src_stride_bytes=None, dst_stride_bytes=None)` | sends chunk `p` to rank `p`; stores rank `s`'s chunk at `s * dst_stride_bytes` |

| Group | Methods |
|---|---|
| eligibility (the same answer on every rank) | `should_allreduce`, `should_all_gather`, `should_reduce_scatter`, `should_all_to_all` |
| decisions (`mode=` `eager` or `graph`, the call's mode when omitted) | `select_algorithm(nbytes)`, `large_reduce_plan(nbytes)`, `large_reduce_staging(nbytes, input_aligned, output_aligned)`, `gather_uses_chain(inp, dim)`, `gather_uses_ring(inp, dim)`, `scatter_uses_chain(inp)`, `scatter_uses_ring(inp)` |
| preparation and capture | `prepare(dtypes=(torch.bfloat16,), *, padded_gather=False, algorithms=None, scatter=False, links=False)` (`scatter`: the reduce-scatter kernels; `links`: every chain and ring kernel, for schedules changed at run time); `capture(stream=None)`, a context around a CUDA graph capture |
| health and regimes | `check_health()`, `poisoned`, `enter_startup()`, `enter_serving()`, `startup()`, `wait_limit_s` |
| close | `close(*, abort=False)`, collective over the exchange group ([Close](#close)); `close_result`, the result a later call returns |
| settings changed between ops, the same call on every rank | `set_chain_chunk_bytes(n)`, `set_link_chunk_bytes(n, collective=None)`, `link_chunk_for(collective)`, `set_ring_stagger(d)`, `set_ring_gather_stagger(d)`, `set_chain_min_bytes(n, collective=None)`, `set_ring_min_bytes(n, collective=None)`, `chain_min_for(collective)`, `ring_min_for(collective)` |
| rank-local grid caps (a CUDA graph keeps its capture's grid) | `blocks`, `set_blocks(n)`, `large_blocks`, `set_large_blocks(n)` |
| tuning tables | `tuning_facts()`, `tuned_choice(collective, nbytes, mode=None)`, `tuned_backend(collective, nbytes, mode=None)`, `untuned()` |
| diagnostics | `stats()`, `call_profile(reset=False)`, `event_trace_records(reset=True)` |

`stats()` reports the agreed limits, layout, lane check, posting order, wait
regime, flag-poll rate, schedule settings, `chain_available`,
`link_available`, `ring_available` with `ring_problem`, forward windows, the
tuning table's decisions and the native counters.

### Close

`close(*, abort=False)` is collective over the exchange group: every rank
calls it, and ranks close the sessions and point-to-point channel sets that
share a group in the same order, because the group pairs collectives by the
order in which each rank issues them. A close refuses further work,
synchronizes the device and holds two teardown rounds
(`sparkring_sircl.teardown`), each one all-gather of a 512-byte note per rank
on the exchange group, run in a daemon thread and waited for at most the
flag-wait limit plus 5 s (`teardown.SLACK_S`):

1. round 1 carries the rank's own failure (a flag wait that timed out, a
   stopped progress thread, a failed device synchronization) while every
   progress thread still runs, so every write a rank owes its peers is posted
   while their queue pairs exist;
2. the rank stops its progress thread;
3. round 2, held only when round 1 arrived, completes once every rank has
   stopped, so no write targets the queue pairs and arena destroyed next.

The result, kept as `close_result` and returned by later calls, is None after
a healthy close, else this rank's failure, the first failed peer's note
(`rank <i>: ...`) or the round that did not complete; the session logs it as a
warning. A round that did not complete marks the exchange group unusable for
teardown rounds in the process, and later closes on it stop and destroy
without rounds. `roce_destroy` returns the number of verbs calls that failed,
and the binding refuses a native library whose local feature identity does
not promise that count ([Progress thread and kernels](#progress-thread-and-kernels));
when it is not zero, or after a failed device or stream synchronization, the
registered arena stays allocated for the rest of the process
(`teardown.RETAINED`), since a queue pair, a registration or a kernel may
still use it. An exception inside the teardown is part of the result; the
native context is still stopped and destroyed exactly once, and a destroy that
raised keeps the arena. An interrupted close records its failure, keeps the
context and the arena and re-raises. `abort=True` holds no round; setup
failures and garbage collection close that way.

A note names its close (the object's kind, its ordinal among the rank's
objects of the group, the round), so a round in which a rank closes another
object, or another round, does not complete. An abandoned round stays pending
on the group and pairs with the next collective another user issues there
(vLLM's own collectives on its CPU group), which needs a failure first (a peer
later than the round's limit, or a rank that closed without rounds). A group
whose round did not complete must not carry further sessions, channel sets or
collectives: recreate it, or end the processes that share it. The rounds order
submissions, not completions: a write a progress thread submitted before it
stopped (a final credit) may still be in flight when a peer destroys its queue
pairs, which revokes it before the arena's memory region is deregistered.

### Algorithms and schedules

Within the capacity, `all_reduce` uses one of two algorithms:

- one-shot: every rank writes its whole message to every peer and sums every
  copy;
- two-shot: rank `p` receives chunk `p` from every peer, sums it, and sends
  the result to every peer, so each rank sends `2 (W - 1) / W` of the
  message. It needs multi-phase posting (`multi_phase`: at most two lanes per
  peer, slots below 2^30 bytes).

`SIRCL_ALLREDUCE_ALGORITHM=swing` is refused: this build has no Swing kernel
for sessions.

Large collectives run under one of four schedules:

| Schedule | `all_reduce_large` | `all_gather_large` | `reduce_scatter` |
|---|---|---|---|
| `pieces` | two-shot ops of `large_piece_bytes` | row or column tiles of `gather_piece_bytes` | scatter ops of at most `large_piece_bytes` |
| `chain` | one pipelined op: half the message reduces along the chain each way, results travel back | each shard travels toward both chain ends | partials of each owner's chunk travel toward it from both ends |
| `ring` | a ring reduce-scatter, then a ring all-gather of the results | finished pieces forwarded around the ring | partials passed around the ring, each rank adding its values |
| `auto` | `chain` from the chain minimum when a chain exists, else `pieces` | as for the all-reduce | as for the all-reduce |

`SIRCL_LARGE_SCHEDULE`, `SIRCL_GATHER_SCHEDULE` and `SIRCL_SCATTER_SCHEDULE`
select them (defaults `auto`, `auto`, `pieces`). `ring` runs a ring op from
the ring minimum on and behaves as `auto` below it; `chain` runs every
eligible call as a chain op. Minimums count each collective's own size: the
all-reduce's message, the all-gather's output and the reduce-scatter's input
([Dispatch settings](#dispatch-settings)). The decision follows sizes and
agreed settings only, so every rank takes the same one.

- Chain: needs a layout whose every Spark hosts a rank, ranks forming a chain
  of cable neighbours joined by direct lanes (a path, or a ring used as a
  chain from rank 0 with its closing cable unused; `routes.chain_order`), and
  at least 64 threads per block. `chain_available` covers the all-reduce and
  `link_available` the all-gather and reduce-scatter. A chain all-gather also
  needs a shard that lands in one piece of the output (`dim` is the first
  dimension above size 1) and is a multiple of 16 bytes.
- Ring: closes the chain by its last rank's lanes to its first, over the
  closing cable of a cycle or through relays on a path. On a path each relay
  hairpin queue may carry at most one ring lane (`routes.ring_window`).
  `ring_available` says whether the ring runs and `ring_problem` why not;
  naming a `ring` schedule for a group whose ring cannot run fails setup.
- Ring staggers: a relayed partial (`SIRCL_RING_STAGGER`, `D`) or forwarded
  piece (`SIRCL_RING_GATHER_STAGGER`, `D3`) leaves `D` rounds after its input
  arrived, so a link keeps several items in flight. Staggers change timing,
  never bits.

### Pointer alignment

A collective's ops follow its shared arguments only: the message size, shape,
dtype and the session's agreed settings. A rank's pointer alignment is a fact
of its own memory and changes no op, so every rank runs the same ring, chain,
scatter or transport ops whatever its tensors' alignment. The ring and chain
kernels and the scatter ops need 16-byte aligned pointers, so a rank whose
input or output is not 16-byte aligned runs the same ops on aligned working
buffers (`oneshot/_aligned.py`: a copy of the input, an output copied back) in
`all_reduce_large`, `all_gather_large`, `reduce_scatter` and `all_to_all`;
the one-shot, two-shot and all-gather transport ops stage their own pieces.
The working buffers are fresh allocations, from the graph's private pool
inside a CUDA graph capture, so a captured op replays its staging copies.
`large_reduce_plan(nbytes)` gives the ops, the same on every rank, and
`large_reduce_staging(nbytes, input_aligned, output_aligned)` adds whether
this rank stages its input and its output.

### Tuning tables

A tuning table (`sparkring_sircl.tuning`, schema `sircl-tuning-table/v2`;
`v1` tables, which name no blocks and no conditions, are read too) holds the ring harness's measurements of every candidate on one group shape
and build, and the decisions derived from them: per collective (`all_reduce`, `all_gather`,
`reduce_scatter`, `all_to_all`) and mode (`eager`, `graph`), size intervals,
each with the fastest SIRCL choice (algorithm or schedule, grid cap, piece,
staggers and, for a chain or ring schedule, the thread blocks per role of the
kernel that runs it: `blocks`, 1 to 64) and whether NCCL measured faster
there. A v2 table also records how its measurements were taken
(`conditions`: `rotate_buffers`, the input and output windows each case cycled
through). At a measured size the
fastest candidate is the one with the shortest period of back-to-back calls
(the ring harness's `period_us`: per rank the median over consecutive calls
of their mean time, the slowest rank's); between measured sizes, each
candidate's cost model `a + b * bytes + c * items`, fitted to its own
measurements, decides. No table ships with the package.

- Key: the group shape (`pair`, `path:<n>`, `cycle:<n>`, or
  `strided:<kind>:<fabric size>:<offsets>`), size, lane count, most relays on
  a lane, hashes of the native and kernel sources and the SIRCL version.
- `SIRCL_TUNING_TABLE` takes comma-separated paths. A session (it needs a
  layout) takes the table whose key matches `tuning_facts()`, so one process
  can name a table per group shape; two differing matches are refused, and
  unmatched tables are listed in `stats()["tuning"]["unmatched"]`. The chosen
  table's hash joins the setup agreement.
- Each op applies the table's choice for its collective, per-rank size and
  mode, then restores the session's settings; a choice's `blocks` run that op
  at those blocks per role unless `SIRCL_GATHER_LINK_BLOCKS`,
  `SIRCL_SCATTER_LINK_BLOCKS`, `SIRCL_REDUCE_LINK_BLOCKS`, `SIRCL_LINK_BLOCKS`
  or `SIRCL_CHAIN_BLOCKS` sets the collective's. A choice the session cannot run
  (among them an all-reduce algorithm for a message above the capacity) is
  counted as unusable, and the rules decide. An all-reduce algorithm's
  decisions end at the largest message it was measured at; larger messages
  follow the rules. Ops inside `with session.untuned():` follow the session's
  own settings.
- Settings: a table records the session settings its choices ran under and
  need (`settings`, `tuning.SETTINGS`): the link slots
  (`SIRCL_LINK_SLOTS`) and a link slot that holds the largest chosen link
  piece (`SIRCL_LINK_SLOT_BYTES`) with a link schedule among its choices, a
  chain slot (`SIRCL_CHAIN_SLOT_BYTES`) with a chain all-reduce, and the
  large-message piece (`SIRCL_LARGE_PIECE_BYTES`) with two-shot pieces. A
  session that takes the table applies each one its environment leaves
  unset; a value the environment sets wins, and `stats()["tuning"]["settings"]`
  names both. The ring harness and the serve launcher refuse settings of
  their own below a table's (`tuning.settings_conflicts`).
- A table chooses only among SIRCL's settings. `tuned_backend()` reports
  where NCCL measured faster; these marks are measurements, and the vLLM
  adapter routes no call to NCCL by them in any `SIRCL_NCCL` mode. Sessions
  never call NCCL.
- Built-in plans: a group shape with a built-in plan (`tuning.BUILTIN_PLANS`)
  has decisions without a table. A session that names no matching table takes
  the plan, written as a table whose key is the session's own
  (`tuning.builtin_document`; `stats()["tuning"]["path"]` is
  `builtin:<shape>`), for every collective whose schedule variable is unset.
  A configured link piece keeps its value, a schedule a caller sets at run
  time takes the collective back from the plan, and `SIRCL_BUILTIN_PLAN=0`
  turns the plan off. The cabled pair's plan, measured with the ring harness
  at 1 block per role ([STATUS.md](STATUS.md#large-messages-on-a-cabled-pair)):
  the ring all-reduce in 256 KiB pieces from 3 MiB messages, the ring
  all-gather in 256 KiB pieces from 2 MiB shards and in 512 KiB pieces from
  16 MiB, the ring reduce-scatter in 256 KiB pieces from 8 MiB inputs and in
  512 KiB pieces from 64 MiB; below those sizes the rules decide. On two ranks
  the ring adds the same two values as the two-shot op and scatter ops,
  rounded once, so the plan changes no result bit. The cycle of eight's plan,
  measured the same way: the ring all-reduce from the first message above the
  two-shot capacity (2 MiB) in 128 KiB pieces, in 256 KiB pieces from 4 MiB
  and in 512 KiB pieces from 16 MiB; at 2 MiB the two-shot op and the ring tie
  and the rules keep two-shot. On eight ranks the ring adds each element's
  values in ring order, rounding at every hop, where the two-shot op adds them
  in rank order and the chain in chain order: every rank's result is the same,
  but its bits differ from those of the rules' schedules, which a tuning table
  or `SIRCL_LARGE_SCHEDULE` keeps. A session with a plan has a link area and
  compiles the ring launchers in `prepare()`.
- Eager and graph modes may choose differently at one size, so an eager and
  a captured all-reduce of the same input can differ in the last place.
- `python -m sparkring_sircl.ring tune` measures tables and `tune-table`
  rebuilds them from a run's results ([RUNBOOK.md](RUNBOOK.md));
  `tuning.render` prints one.

### Point-to-point channels: `sparkring_sircl.p2p`

`PointToPoint(exchange_group=, device=, peer_routes=None, layout=None,
channels=None, ...)` is one rank's channel context for a group of 2 to 16
ranks, separate from the group's collective session. Construction is
collective: the ranks agree on every shared setting, prove every lane and
start a progress thread of their own (`p2p/_p2p_proxy.c`).

| Method | Behaviour |
|---|---|
| `isend(tensor, peer)`, `irecv(tensor, peer)` | return a `P2PWork` at once; the transfer runs on the channel's own CUDA stream once the calling stream reaches the call; `work.wait()` makes the calling stream wait, not the host |
| `send(tensor, peer)`, `recv(tensor, peer)` | as above, waited for at once |
| `batch_isend_irecv([(kind, tensor, peer), ...])` | issues a batch, sends first |
| `has_channel(peer)`, `channel_problem(peer)` | whether a pair has a channel, and why not |
| `enter_startup()`, `enter_serving()`, `startup()`, `check_health()`, `stats()` | as for sessions |
| `close(*, abort=False)`, `close_result` | as a session's ([Close](#close)), after the channel set waited for its streams' work; `p2p_destroy` returns the number of verbs calls that failed |

- Every ordered pair of `channels` (default every pair) is first-in
  first-out: the n-th receive from a peer takes the n-th message that peer
  sent, and must name its byte count; a different count stops the group.
  Any dtype and shape on the context's device, outside CUDA graph capture
  only.
- A message travels as items of one slot (`SIRCL_P2P_SLOTS` slots of
  `SIRCL_P2P_SLOT_BYTES`, default 8 of 512 KiB, per direction), paced by the
  receiver's credits; a sender runs at most two rounds of slots ahead.
- Every direction of a pair has its own CUDA stream. Beyond the GPU's
  hardware queues (`CUDA_DEVICE_MAX_CONNECTIONS`), a waiting receive also
  holds back later work on its queue, so issue the sends peers wait for
  before the receives that wait on them.
- Relayed lanes post 32 KiB chunks within forward windows of at most
  `SIRCL_P2P_WINDOW_BYTES` (128 KiB). The windows through one relay queue,
  together with the sessions' windows, stay within 75 % of it
  (`p2p.budget`); a pair left without room for one chunk has no channel.
- Fail-stop: a timeout, a size mismatch or a failed write poisons the context,
  and the progress thread sends an abort notice naming the origin rank to
  every peer, so the whole group stops.

### CPU placement: `sparkring_sircl.cpus`

A GB10 has ten Cortex-X925 performance cores and ten Cortex-A725 efficiency
cores. Eager collectives are bound by the launching thread, and a progress
thread on an efficiency core, or sharing a core with a spinning launching
thread, delays the whole group. `plan(policy="performance", allowed=None)`
returns a `Placement`: launching threads on all but the last fastest-class
CPU, the progress thread on the last one. `apply(placement)` pins the calling
thread, which threads it creates later inherit.

```python
from sparkring_sircl import cpus

placement = cpus.plan()
cpus.apply(placement)                    # before torch creates its threads
session = AllReduce(..., progress_cpu=placement.progress_cpu_list)
```

### Other modules

| Module | Contents |
|---|---|
| `routes` | `Fabric`, `Layout`, `derive_lanes`, `derive_routes`, `parse_peer_routes`, `format_peer_routes`, `validate_route_map`, `check_complementary`, `relay_load`, `relay_queues`, `forward_windows`, `ring_window`, `chain_order`, `isolation_problems`, `load_roles`, `RouteError` |
| `agreement` | `agreement_failures(statuses, layout)`: the setup verdict |
| `teardown` | the teardown rounds of a close ([Close](#close)): `ordered_close`, `close_native`, `tensor_exchange`, `unusable`, `RETAINED` |
| `build` | `build()`, `library_path()`, `cache_dir()` and `sircl-prepare` |
| `roce_gid` | the resolver's interface, `resolve_device_gid_index(device)` among it, and `source_file()` |
| `groups` | `tp_groups`, `dcp_groups`: vLLM's rank layout without importing vLLM |
| `posting` | posting orders of the progress thread (`rank`, `ring-farthest`, `farthest`) |
| `latency_model` | `oneshot_limit`: the default one-shot limit of a layout |
| `protocol`, `pieces`, `scatter_plan` | wire-protocol arithmetic and piece plans shared by kernels and progress thread (torch-free) |
| `references` | host references of every schedule, bit for bit (`torch` passed as an argument) |
| `bounds` | lower time bounds of a schedule from a Spark's host-interface and cable rates |
| `callprofile` | the eager call profile (`SIRCL_CALL_PROFILE`) |
| `env` | every session variable; `python -m sparkring_sircl.env` prints them |
| `fused_norm` | all-reduce fused with the residual add and RMSNorm on a session |
| `ring` | the standalone ring harness `sircl-ring`, which runs collectives on chosen Sparks without vLLM ([RUNBOOK.md](RUNBOOK.md)) |
| `fabric` | the relay plan installer `sircl-fabric` (`plan`, `facts`, `show`, `diff`, `up`, `down`, `marker`); sessions never import it |
| `testing` | the verbs stand-in, CPU simulators, relay-host simulator and GPU emulations |
| `vllm` | the vLLM adapter ([its README](sparkring_sircl/vllm/README.md)) |

Two diagnostics are off by default. `SIRCL_CALL_PROFILE=<calls>` keeps host
timestamps of the most recent eager calls; `call_profile()` returns per
operation, path and size the median microseconds of each stage
(`SIRCL_CALL_PROFILE_GPU=1` adds device time, `SIRCL_CALL_PROFILE_FILE=<prefix>`
writes `<prefix>.rank<rank>.json`). `SIRCL_EVENT_TRACE=<records>` records
every chunk and item of chain and link ops in the progress thread and the
kernels, and in a ring kernel when each block began (`KERNEL_START`) and when
block 0 rang the link doorbell (`KERNEL_BELL`); `event_trace_records()`
returns them on the host clock, and
`python -m sparkring_sircl.ring trace` turns a run's records into stage times.

## Dispatch settings

Defaults and rules of the session settings. Every setting below joins the
setup agreement, so ranks with different values fail setup together, except
the posting order, the forward proof and the event trace, which are
rank-local. A setter that changes an agreed setting between ops takes the
same call on every rank. A tuning table's choice for an op applies its own
algorithm, schedule, piece, stagger, grid cap and minimums to that op.

| Setting | Default | Rule |
|---|---|---|
| all-reduce capacity `C` (`max_size`) | 2 MiB | multiple of 16; the largest `all_reduce` message |
| dispatch ceiling (`SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES`) | `C` | multiple of 16 from 16 to `C`; `should_allreduce` accepts up to it, `all_reduce` up to `C` |
| all-gather capacity `G` (`max_gather_bytes`) | 16 MiB | multiple of 16; 0 disables all-gathers; also the tile size of `all_gather_large` |
| algorithm (`SIRCL_ALLREDUCE_ALGORITHM`, `SIRCL_LARGE_ALGORITHM`) | `auto`, `twoshot` | `auto` runs one-shot up to the one-shot limit and the large algorithm above it; `swing` is refused |
| one-shot limit (`SIRCL_ONESHOT_MAX_BYTES`) | with a layout and two-shot: the latency model's limit (28,672 on a ring of eight posting farthest first, 73,728 on a path of four), at most 131,072 and `C`; else 131,072 | `stats()["oneshot_max_source"]` names the source |
| posting order (`SIRCL_POST_ORDER`) | `farthest` with a layout, else `rank` | rank-local; sets when each lane's write starts, never what it carries |
| launch grid (`SIRCL_THREADS`, `SIRCL_BLOCKS`, `SIRCL_LARGE_BLOCKS`, `SIRCL_PACKS_PER_THREAD`) | 512, 8, 32, 2 | a launch takes the smallest power of two of blocks with at most `SIRCL_THREADS × SIRCL_PACKS_PER_THREAD` 16-byte packs per block, capped by `SIRCL_BLOCKS` for one-shot ops and plain all-gathers and by `SIRCL_LARGE_BLOCKS` for two-shot ops, tiles and scatter ops; `set_blocks` and `set_large_blocks` change the caps rank-locally |
| flag pollers (`SIRCL_FLAG_POLLERS`) | `one-block` | `one-block`: block 0 polls and hands arrival to the other blocks; `every-block`: every block polls |
| forward window, chunk, hairpin queue (`SIRCL_FORWARD_WINDOW_BYTES`, `SIRCL_FORWARD_CHUNK_BYTES`, `SIRCL_HAIRPIN_QUEUE_BYTES`) | 131,072; 32,768; 524,288 | windows only on lanes through relays, from the layout; a window holds at most 60 chunks; window 0 turns windows off and shrinks pieces to the relay-safe size |
| forward proof (`SIRCL_FORWARD_PROOF`) | 1 | 1: bytes the op order proves delivered leave the window at once; 0: only completions free it |
| large piece (`SIRCL_LARGE_PIECE_BYTES`) | a tuning table's, else the larger of 4 MiB and `C` | op size of `all_reduce_large` pieces and scatter ops; the arena's slots hold it |
| schedules (`SIRCL_LARGE_SCHEDULE`, `SIRCL_GATHER_SCHEDULE`, `SIRCL_SCATTER_SCHEDULE`) | `auto`, `auto`, `pieces` | `auto`, `chain`, `ring` or `pieces` ([Algorithms and schedules](#algorithms-and-schedules)); naming `chain` or `ring` where it cannot run fails setup |
| chain minimums (`SIRCL_CHAIN_MIN_BYTES`) | all-reduce 8 MiB, all-gather output 8 MiB, reduce-scatter input 4 MiB | `auto` runs chain ops from them; the variable sets one size for all three |
| ring minimums (`SIRCL_RING_MIN_BYTES`) | all-reduce 4 MiB, all-gather output 8 MiB, reduce-scatter input 4 MiB | `ring` runs ring ops from them; the variable sets one size for all three |
| chain geometry (`SIRCL_CHAIN_CHUNK_BYTES`, `SIRCL_CHAIN_SLOT_BYTES`, `SIRCL_CHAIN_SLOTS`, `SIRCL_CHAIN_BLOCKS`, `SIRCL_CHAIN_UNROLL`) | 512 KiB, 1 MiB, 4, 4, 4 | chunk a multiple of 16 up to the slot; slot a multiple of 4,096; 2 to 32 slots; unroll 1 to 8 packs per thread per pass; the chain area (4 streams × slots × slot bytes, twice) adds 32 MiB of pinned memory to a session on a chain |
| link geometry (`SIRCL_LINK_CHUNK_BYTES`, `SIRCL_LINK_SLOT_BYTES`, `SIRCL_LINK_SLOTS`, `SIRCL_LINK_BLOCKS`, `SIRCL_LINK_UNROLL`) | 512 KiB, 512 KiB, `2 W` slots and at least 8 (16 on the cycle of eight, 8 on a path of four; `protocol.default_link_slots`), by group shape and link kernel ([link blocks](#link-blocks)), 4 | a tuning table's slot count and slot apply where these are unset; without `SIRCL_LINK_SLOT_BYTES` the slot holds the largest configured piece rounded up to 4 KiB, up to 1 MiB, or the table's slot when larger; the link area (4 links × slots × slot bytes, twice) adds 32 MiB to a session with 8 slots of 512 KiB, 64 MiB with 16 |
| link blocks per collective (`SIRCL_GATHER_LINK_BLOCKS`, `SIRCL_SCATTER_LINK_BLOCKS`, `SIRCL_REDUCE_LINK_BLOCKS`) | by group shape and link kernel: 1 for the ring all-reduce, all-gather and reduce-scatter on a cabled pair and on the cycle of eight and for the ring all-reduce and all-gather on a path of four, 4 for every other kernel and shape (`protocol.LINK_BLOCKS_BY_SHAPE`) | blocks per role of one collective's link kernels under both its schedules, 1 to 64; `SIRCL_LINK_BLOCKS` sets every collective's without one; `stats()["link_blocks"]` names each kernel's |
| link piece per collective (`SIRCL_GATHER_LINK_CHUNK_BYTES`, `SIRCL_SCATTER_LINK_CHUNK_BYTES`, `SIRCL_REDUCE_LINK_CHUNK_BYTES`) | the link piece | chain and ring all-gathers, chain and ring reduce-scatters, ring all-reduces; multiples of 16 up to the link slot |
| ring staggers (`SIRCL_RING_STAGGER`, `SIRCL_RING_GATHER_STAGGER`) | `auto`: 1 when the link slots hold it, else 0 | 0 to 4 rounds; a stagger `D` needs `D (W - 1) + 2` link slots: 5 on a path of four, 9 on a ring of eight, so `auto` gives 1 on both with their default slots (8 and 16) |
| event trace (`SIRCL_EVENT_TRACE`) | 0 (off) | records kept on each side, native and kernel; traced kernels compile apart from untraced ones |
| tuning tables (`SIRCL_TUNING_TABLE`) | none | [Tuning tables](#tuning-tables); the same table on every rank |
| built-in plan (`SIRCL_BUILTIN_PLAN`) | 1 | 0 turns off the built-in plan of a group shape that has one (a cabled pair, the cycle of eight; [Tuning tables](#tuning-tables)) |
| flag-wait limits (`SIRCL_STARTUP_WAIT_S`, `SIRCL_SERVING_WAIT_S`) | 600 s, 20 s | GPU-clock seconds, up to 4,294. `SIRCL_SPIN_LIMIT` (20,000,000 polls) bounds a wait only when no time limit is set |

### Link blocks

A link kernel runs one collective under one schedule: the ring all-reduce,
all-gather and reduce-scatter (`ring_reduce`, `ring_gather`,
`ring_scatter`) and the chain all-gather and reduce-scatter
(`chain_gather`, `chain_scatter`); the chain all-reduce runs on the chain
kernel (`SIRCL_CHAIN_BLOCKS`). Each of a link kernel's roles runs on
`link_blocks_for(collective, schedule)` blocks, which take the role's items
in turn. A role's blocks share the GPU's path to pinned host memory, so more
blocks split the same bandwidth into slower passes and lengthen every item's
time from arrival to departure; the defaults are 1 block where the ring
harness measured that faster and 4 elsewhere (table above). A session's
blocks join the setup agreement. A tuning table's chain or ring choice may run
one op at other blocks (its `blocks`; `set_op_blocks` for a caller that fixes
an op's), except for a collective whose blocks the environment sets: every
link kernel and the chain kernel find a launch's last block as the arrival
that completes its grid on the kernel's tail word and return that word to 0,
so consecutive launches may use different grids.

## Environment

`python -m sparkring_sircl.env` prints every session variable with its
default and meaning; `python -m sparkring_sircl.p2p.settings` prints the
channel variables. Variables are read when a session or channel context is
built. [Dispatch settings](#dispatch-settings) lists the variables that tune
dispatch, the flag-wait limits and the tuning tables; the others a deployment
sets are:

| Variable | Default | Meaning |
|---|---|---|
| `SIRCL_PEER_ROUTES` | required | route map `<peer>=<device>[/<device>],...` |
| `SIRCL_LAYOUT` | unset (no layout checks) | `ring:<n>[:<positions>]`, `path:<a>-<b>[:<positions>]` or `cables=...;positions=...` |
| `SIRCL_FABRIC_DOCUMENT` | the DGX OS device names | fabric document (`sparkring-fabric/v1`) naming every function's RDMA device and interface |
| `SIRCL_DEVICES` | the route map's devices | RDMA devices to open, 1 to 4 |
| `SIRCL_GID_INDEX` | resolved per device | one GID index for every device (fallback `NCCL_IB_GID_INDEX`) |
| `SIRCL_TRAFFIC_CLASS` | 0 | DSCP/ECN byte of every queue pair (fallback `NCCL_IB_TC`) |
| `SIRCL_MAX_RELAYS` | 3 | most relays on any lane |
| `SIRCL_PROGRESS_CPU` | unpinned | CPU list of the progress thread, for example `9` or `5-9,15-19` |
| `SIRCL_BUILD_CACHE_DIR` | `<XDG cache home>/sircl/roce` | native build cache |
| `SIRCL_NATIVE_LIBRARY`, `SIRCL_P2P_NATIVE_LIBRARY` | unset | prebuilt native libraries |
| `SIRCL_P2P_SLOTS`, `SIRCL_P2P_SLOT_BYTES`, `SIRCL_P2P_WINDOW_BYTES`, `SIRCL_P2P_CHUNK_BYTES`, `SIRCL_P2P_PROGRESS_CPU` | 8, 512 KiB, 128 KiB, 32 KiB, unpinned | point-to-point slots, windows and progress CPU (also `SIRCL_P2P_BLOCKS`, `SIRCL_P2P_THREADS`, `SIRCL_P2P_UNROLL`) |

Diagnostics: `SIRCL_CALL_PROFILE`, `SIRCL_CALL_PROFILE_GPU`,
`SIRCL_CALL_PROFILE_FILE`, `SIRCL_EVENT_TRACE`, `SIRCL_FAST_LAUNCH`. Refused
at setup: `SIRCL_POST_MODE` other than `verbs`, `SIRCL_TRACE` other than
unset or 0, `SIRCL_TOPOLOGY` other than `direct`. `SIRCL_FUSED_NORM_MAX_ROWS`
sets the rows per fused all-reduce and RMSNorm launch. `SIRCL_COLUMN_GATHER`
(default `1`) is read by the vLLM adapter's executor, not by sessions: it
carries column gathers as a dimension-0 ring or chain all-gather plus one
local copy ([adapter README](sparkring_sircl/vllm/README.md#column-gathers)).

## Build and test

From the repository root:

```bash
pip install pytest numpy                  # or the package's `test` extra
python -m pytest spark_transport/sircl -q
```

- The CPU tests import the package from the source tree and need pytest and
  numpy.
- Tests of the native layers (proxy, point-to-point, ring-link, scatter,
  Swing and fused-norm simulators, the bindings, posting) compile
  `_roce_proxy.c` and `_p2p_proxy.c` against an in-memory verbs stand-in
  (`sparkring_sircl/testing/fake_verbs`) and need a GCC-compatible compiler
  on a POSIX host; elsewhere they skip.
- The vLLM adapter tests and one ring-harness test need torch and skip
  without it. Inside a SparkRing checkout, tests also check that GID
  resolution uses `integrations/vllm/spark_roce_gid.py`.

The GPU emulation runs every rank of a group as a thread of one process on
one GPU, with the real session class and kernels over the simulator build of
the native library, and checks every collective, eager and captured, against
the host references:

```bash
CUTE_DSL_ARCH=sm_120a python -m sparkring_sircl.testing.gpu_emulation --layout path:0-3 --lanes 2
CUTE_DSL_ARCH=sm_120a python -m sparkring_sircl.testing.gpu_emulation --layout ring:8 --lanes 2 --path-latency 20000,10000,12000,150000
CUTE_DSL_ARCH=sm_120a python -m sparkring_sircl.testing.p2p_emulation --layout ring:8 --lanes 2
```

It needs CUDA, torch with CUDA, CUDA Python, the CuTe DSL, a host C compiler
and a GPU that addresses pinned host memory at its host pointer.
`CUTE_DSL_ARCH` names the GPU architecture when the DSL cannot detect it.
`--column-gather-only` prepares the sessions and runs only the column-gather
checks (`sparkring_sircl/testing/column_gather_checks.py`): the vLLM adapter's
staged dimension-0 gather against `all_gather_large` along the column
dimension, bit for bit, eager and captured.

Every rank of an emulated group shares the one GPU, so a group of several
ranks (`testing.gpu_emulation.EmulatedGroup`, also when another tool builds
it) sizes two of the GPU's resources unless the caller set them, and the
`settings` check names what it set: `CUDA_DEVICE_MAX_CONNECTIONS=32` before
the CUDA context exists, so a kernel waiting for a peer does not hold back
commands queued behind it on a shared hardware queue (a context created
before the group is reported in a warning); and the large-message grid cap
`SIRCL_LARGE_BLOCKS` at one block per multiprocessor for every rank's grid
(`testing.kernel_gpu_checks.emulation_large_blocks`), because every block of
a collective spins until every rank has staged, so every rank's grid must be
resident at once. On an RTX 5090 (170 multiprocessors) eight ranks get 16
blocks each. Every check places its inputs on the device from the calling
thread before the rank threads start, since a rank thread blocked in a copy
from pageable host memory can stall the verbs stand-in's delivery thread. On
Sparks every rank has its own GPU and keeps the defaults.
`--path-latency BASE_NS,RELAY_NS[,BYTES_PER_US[,ACK_DELAY_NS]]` delays each
write by a base time plus a time per relay, limits each queue pair's rate and
delays completions, so relayed lanes wait on their forward windows.

The CPU tests and the GPU emulation check the protocol, the native layer and
the kernels' arithmetic. They do not qualify kernels on GB10 or RDMA on a
fabric; that evidence comes from the ring harness on Sparks
([RUNBOOK.md](RUNBOOK.md), [STATUS.md](STATUS.md#component-status)).

## Limitations

- Relay plans do not persist: `sircl-fabric up` installs a plan and
  `sircl-fabric down` removes it, and nothing restores a plan after a reboot.
- The SparkRing installer does not configure ring sessions, relay plans or
  tuning tables, and has no `sparkring fabric tune` command; tables come from
  the ring harness's `tune` command.
- Further limits of layouts, relays and serving:
  [STATUS.md, Limitations](STATUS.md#limitations).

## Related documents

- [RUNBOOK.md](RUNBOOK.md): the ring harness procedure and the relay plan
  installer.
- [sparkring_sircl/vllm/README.md](sparkring_sircl/vllm/README.md): the vLLM
  adapter. It loads sessions from `sparkring_sircl.oneshot`
  (`SIRCL_SESSION_MODULE`), leaves NCCL off unless `SIRCL_NCCL` opts in, and
  logs a receipt per rank and group stating what each collective will use
  (also as JSON with `SIRCL_RECEIPT_DIR`).
- [STATUS.md](STATUS.md): component status, supported layouts, exactness,
  measured performance and limitations.
- [PROVENANCE.md](PROVENANCE.md): the origin of every file.
- [docs/architecture/sircl.md](../../docs/architecture/sircl.md): SIRCL in
  SparkRing's architecture.
