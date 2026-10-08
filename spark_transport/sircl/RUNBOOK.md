# SIRCL ring harness runbook

The ring harness runs SIRCL ring sessions on chosen DGX Sparks of a cabled
ring for correctness and timing, without a model: one container per rank from
the serving image, GPU 0, host networking. Per case it compares every output
bit for bit with a host reference, times eager calls and CUDA graph replays,
reads every Spark's RDMA and Ethernet error counters before and after, and
writes JSON results and a summary table.

- `python -m sparkring_sircl.ring` (console script `sircl-ring`): the harness;
- `python -m sparkring_sircl.ring.p2p`: the
  [point-to-point cases](#point-to-point-cases);
- `python -m sparkring_sircl.fabric` (console script `sircl-fabric`): the
  [relay plan installer](#relay-plan-installer).

The harness does not serve a model and never configures networking. Serving
through the vLLM adapter has its own runbook,
[`sparkring_sircl/vllm/RUNBOOK.md`](sparkring_sircl/vllm/RUNBOOK.md); component
statuses and measured results are in [`STATUS.md`](STATUS.md#component-status).

## Safety classes

| Class | Meaning |
|---|---|
| OFFLINE | reads and writes only the operator's machine |
| READ-ONLY REMOTE | runs read-only commands on the Sparks over SSH |
| MUTATES HOST | writes files under the site's `remote_dir`, starts or removes containers, or changes host networking objects on the Sparks |
| STOPS SERVING | can interrupt a running model |

| Command | Class |
|---|---|
| `plan`, `run --print`, `tune --print`, `summarize`, `trace`, `tune-table` | OFFLINE |
| `preflight` | READ-ONLY REMOTE |
| `stage` | MUTATES HOST: files under `remote_dir`, one short build container per Spark |
| `run`, `tune` | MUTATES HOST: one container per rank on GPU 0. Refuses while any other container runs on a used Spark; `--force` runs beside it (STOPS SERVING: both then share GPU 0 and the fabric) |
| `cleanup` | MUTATES HOST: removes containers labelled `sircl-ring` only |
| point-to-point `plan`, `run --print`, `summarize` | OFFLINE |
| point-to-point `run` | as `run` |
| relay plan installer | [Safety classes of the installer](#safety-classes-of-the-installer) |

## Inputs

### Site file

The harness and the relay plan installer read one site file (schema
`sircl-ring-site/v1`). Copy
[`site.example.json`](sparkring_sircl/ring/site.example.json), replace every
placeholder and keep the copy out of version control.

| Field | Meaning |
|---|---|
| `image` | serving image ID or tag, present on every Spark |
| `lan_interface` | wired-LAN interface on every Spark; the control exchange (setup records, verdicts, barriers) runs over it, never over fabric addresses |
| `control_port` | free TCP port (1024 to 65000) on the first Spark of each configuration, reachable over the wired LAN |
| `remote_dir` | absolute working directory on every Spark (default `/tmp/sircl-ring`) |
| `ring` | 2 to 16 Sparks in cabling order: entry `i`'s port 0 is cabled to entry `i + 1`'s port 1, wrapping around. Each entry: `name` (the Spark's host name, for example `spark-a`), `ssh` (`user@address`, for example `operator@192.0.2.10`), `lan_address` (wired-LAN IPv4 address) and an optional `docker` |
| `gid_index` | optional RoCE GID index for every device; without it every rank resolves each device's RoCE v2 IPv4 entry |
| `docker` | optional command that runs Docker on the Sparks (default `docker`); a ring entry's `docker` overrides it. 1 to 8 words of letters, digits and `_.@:/+=-`, the last one `docker` or a path ending in `/docker`, for example `sudo -n docker` for an SSH user outside the docker group with passwordless sudo |

### Operator machine and Sparks

- Operator's machine: Python 3.10 or later, an OpenSSH client with key access
  to every Spark in batch mode, and a checkout of this repository. Run every
  command from `spark_transport/sircl`: `stage` ships the package with
  SparkRing's RoCE GID resolver (`integrations/vllm/spark_roce_gid.py`), which
  it finds in the checkout.
- Every SSH user has Docker access on its Spark, directly or through the
  site's `docker` command. Containers run as root, so files they write under
  `remote_dir` belong to root.
- The serving image holds a C compiler and the libibverbs development
  headers; `stage` builds the native library with them.
- Every Spark has GPU 0 and the four ConnectX-7 RDMA devices under their DGX
  OS names (`rocep1s0f0`, `roceP2p1s0f0`, `rocep1s0f1`, `roceP2p1s0f1`), port
  state ACTIVE. The preflight and the NCCL baseline name exactly these.

### Measurement prerequisites

- **Relay plan.** A lane between group members that share no cable crosses
  the ConnectX-7 of every member between them; those Sparks need the relay
  plan. Every configuration except `pairs` and the point-to-point `pair` has
  such lanes. Install it with the [relay plan installer](#relay-plan-installer):
  on a ring of eight, layout `ring8` carries the lanes of every built-in
  configuration; `2xTP4` carries those of `path4` and `two-tp4` but not the
  whole-ring lanes of `dcp4`. The preflight checks every lane's origin route.
- **Hairpin setting.** A relay forwards through a hardware hairpin queue that
  cannot pause its sender. Every ConnectX function of a relaying Spark needs
  the mlx5 devlink driverinit parameters `hairpin_queue_size` 8192 and
  `hairpin_num_queues` 4 in effect (they apply after a function restart) and
  hardware TC offload. Forward windows assume this 512 KiB queue
  (`SIRCL_HAIRPIN_QUEUE_BYTES`); with the driver's default the relays drop
  frames and the error counters move. Neither the harness nor the installer
  sets or checks it. SparkRing applies it on four-Spark rings
  (`sudo sparkring hairpin`, [commands](../../docs/operations/commands.md#hairpin));
  `sparkring node hairpin status` prints it on a Spark with the SparkRing
  package. A function restart removes the routes and filters on that
  function, so apply the hairpin setting before the relay plan.
- **Idle Sparks.** No other container runs on a used Spark; stop serving
  there first.
- **CPU placement.** `--cpu-policy performance` (default) pins every rank to
  the Spark's Cortex-X925 performance cores before torch loads and gives the
  native progress thread one that no launching thread uses; `none` leaves
  placement to the scheduler. `rank-N.json` records the CPUs (`placement`).

## Configurations

`--config` names one configuration or a comma-separated list; `--config all`
runs `pairs`, `path4`, `two-tp4` and `ring` (`pairs` and `ring` on a ring of
fewer than eight). A failing configuration stops the list.
`--groups '0-3;4-7' --name NAME` defines other groups of consecutive Sparks
or the whole ring.

| Configuration | Groups (ring positions) | Cases | Checks |
|---|---|---|---|
| `pairs` | 0-1, 2-3, 4-5, 6-7 at once | small cases | lanes over one cable, no relays |
| `path4` | 0-3 (ring of 5 or more) | small cases | lanes through up to 2 relays |
| `two-tp4` | 0-3 and 4-7 at once (ring of 8 or more) | small cases | two relayed groups side by side, each inside its own Sparks |
| `ring` (`ring8` on a ring of eight) | every Spark | small cases | lanes through up to 3 relays, relay load factor 3 on a ring of eight |
| `path4-large` | 0-3 | all-reduce of 1, 8, 32 and 64 MiB; reduce-scatter of 8, 32 and 64 MiB per rank and all-gather of 4 and 16 MiB shards, both `[rows, 4096]` BF16 along dimension 0; each as pieces, chain and ring, chain chunks of 256 KiB to 1 MiB, link pieces of 256 and 512 KiB | large-message schedules on a path of four |
| `two-tp4-large` | 0-3 and 4-7 at once | the `path4-large` cases | the same on two paths at once |
| `ring-large` (`ring8-large`) | every Spark | all-reduce of 8, 32, 64 and 96 MiB as pieces, chain (the cycle used as a chain) and ring | large all-reduces on the whole ring |
| `dcp4` | 0-3 and 4-7, plus a session over all eight (ring of exactly 8) | the decode context parallel (DCP) exchanges of GLM-5.3 at tensor parallel size 8 with DCP groups of four (vLLM's `a2a` combine): all-to-all of `[4 * rows, 8 * 514]` BF16, query all-gather of `[rows, 8 * 576]` BF16, indexer all-gather of 16 KiB rows, and at decode the all-reduce of `[rows, 6144]` BF16 on the session over all eight; decode rows 1 to 128 eager and in graph replay, a prefill chunk of 8,192 rows eager | DCP groups beside the tensor-parallel session that spans them |
| `ring-swing` (`ring8-swing`) | every Spark (ring size a power of two) | Swing all-reduce of 1 MiB, 1 MiB + 16 B, 1.5 MiB and 2 MiB beside the default all-reduce of the same sizes | Swing against its reference (BF16 rounding after every step); `rank-N.json` names whether the session or the host functions of `oneshot/_swing_ops.py` ran it (`session.swing.through`) |
| `ring-latency` (`ring8-latency`) | every Spark | one-shot and two-shot all-reduce of 4 to 64 KiB, 96 and 128 KiB, once per posting order (`rank`, `ring-farthest`) | the one-shot to two-shot crossover (`SIRCL_ONESHOT_MAX_BYTES`) and the posting order, beside the latency model |
| `path4-latency` | 0-3 | the `ring-latency` cases with the orders `rank` and `farthest` | the same on a path of four |
| `path4-crossover` | 0-3 | all-reduce, all-gather and reduce-scatter of 256 KiB to 4 MiB, each as ring, chain and pieces | the sizes from which chain and ring are faster (`SIRCL_CHAIN_MIN_BYTES`, `SIRCL_RING_MIN_BYTES`) |

**Small cases.** BF16 all-reduces and all-gather shards of 16 B to 128 KiB
in powers of two plus sizes at and 16 bytes either side of size limits, and
five all-gather shapes along the last dimension (odd ones included); the plan
lists every size. Per size: 3 checked calls, 300 timed eager calls, 1,000
timed graph replays.

**Large cases** (`--large`). All-reduces of 256 KiB to 96 MiB (the two-shot
`all_reduce` up to the 2 MiB capacity, `all_reduce_large` above it) and
`all_gather_large` shards of 1, 4 and 16 MiB; each case checks one call and
times 20. With `--transport-only`, CPU-staged one-shot ops up to the capacity.

**Exactness.** The host reference follows the summation order of the op that
ran: the rank-ordered float32 sum rounded once to BF16 (one-shot, two-shot,
pieces, scatter ops), the chain or ring order with one rounding per hop, the
Swing reference, or the concatenation. A large case also runs twice on the
same inputs; a bit that differs between the calls counts as a mismatch.

### Options

| Option | Effect |
|---|---|
| `--transport-only` | CPU-staged one-shot ops through the native layer only, no CUDA kernel: checks routes, relays, lane checks and posting on the NICs (not with `dcp4`, `ring-swing`, the `-latency` configurations or `tune`) |
| `--large`, `--large-capacity`, `--large-iterations` | add the large cases; session capacity (2 MiB); timed calls (20) |
| `--large-sizes`, `--large-gather-sizes`, `--chain-gather-sizes`, `--reduce-scatter-sizes` | byte sizes in place of a configuration's large all-reduces, `all_gather_large` shards, link all-gathers and reduce-scatters |
| `--large-schedules`, `--gather-schedules`, `--scatter-schedules` | schedules swept, from `auto`, `pieces`, `chain` and `ring` |
| `--large-piece`, `--chain-chunks`, `--link-chunks`, `--gather-link-chunks`, `--scatter-link-chunks`, `--reduce-link-chunks` | op size of the two-shot pieces; chain chunks and link pieces swept (multiples of 16 bytes up to the slot; larger link pieces need `--session-env SIRCL_LINK_SLOT_BYTES=<bytes>`). A sweep option that applies to no case of a configuration fails the plan |
| `--session-env SIRCL_RING_STAGGER=<s>`, `--session-env SIRCL_RING_GATHER_STAGGER=<s>` | ring stagger (0 to 4) of the partials and of the forwarded pieces; stagger `s` needs `s * (W - 1) + 2` link slots (`SIRCL_LINK_SLOTS`, default `2 W` and at least 8: 16 on the cycle of eight) |
| `--large-blocks 4,8,16,32` | every two-shot and large-message case once per launch grid cap |
| `--baseline nccl`, `--nccl-library`, `--nccl-env NAME=VALUE` | NCCL's rows beside SIRCL's ([NCCL baseline](#nccl-baseline)) |
| `--tuning-table PATH` | sessions decide from a measured table ([Tuning tables](#tuning-tables)) |
| `--session-env NAME=VALUE` | any documented `SIRCL_*` variable on every rank (`python -m sparkring_sircl.env` lists them); variables the harness sets from its own options are refused |
| `--session-env SIRCL_EVENT_TRACE=<records>`, `--eager-profile`, `--eager-path adapter` | event trace and eager call profile ([Diagnostics](#diagnostics)) |
| `--cpu-policy`, `--forward-window BYTES` | `performance` (default) or `none`; `SIRCL_FORWARD_WINDOW_BYTES` of every session (`0`: no forward windows) |
| `--startup-wait`, `--serving-wait` | flag-wait limits in seconds during setup and warm-up (300) and during the timed cases (20); a stopped rank ends its peers within the serving limit |
| `--dcp-decode-rows`, `--dcp-prefill-rows`, `--swing-sizes`, `--latency-sizes`, `--post-orders` | row counts of `dcp4` (or `none`), sizes of `ring-swing` and the `-latency` configurations, posting orders from `rank`, `ring-farthest` and `farthest` |
| `--relay-us`, `--post-us`, `--write-us` | latency model: one-way latency per relay (0.75 us), posting time per lane where the session does not measure it (0.3 us), direct write (0, excluded) |
| `--host-send-gbps`, `--host-recv-gbps`, `--host-cap-gbps`, `--cable-gbps` | rates of the bound printed beside each large row: host interface send (24 GB/s) and receive (26.8 GB/s), one rate for both, cable direction (24 GB/s) |
| `--correctness-iterations`, `--eager-iterations`, `--graph-iterations`, `--spin-limit` | call counts per size and the spin limit |
| `--run-id`, `--output`, `--timeout`, `--force`, `--print` | run name (default the start time and the launching process's id), result folder (`sircl-ring-results`), wait limit (1,800 s), run beside other containers, print the plan only |

### NCCL baseline

`--baseline nccl` adds NCCL's all-reduce and all-gather (dimension 0) of
every case's size and mode, timed the same way, and a `vs NCCL` column
(NCCL's median over SIRCL's; above 1, SIRCL is faster). NCCL runs on pairs
and whole cycles only. Each container starts with NCCL's environment for this
fabric (`ring/nccl.py`: the image's NCCL preloaded from
`/opt/sparkring/toolchain/nccl/lib/libnccl.so.2`, InfiniBand transport on all
four RDMA functions, the site's GID index or 3, the ring algorithm on the
switchless ring, four channels); `--nccl-env` overrides any variable, and the
plan prints the effective environment. A rank stops the run when NCCL's log
(`nccl-rank-<N>.log` in the run directory on the Spark) does not show
`NET/IB` after the warm-up all-reduce.
On a pair, an 8 KiB eager NCCL all-reduce slower than 100 us marks the
group's NCCL rows `[NCCL degraded]`. NCCL's all-reduce rows are checked
within the rounding of `W` BF16 steps, its all-gathers bit for bit.

## Procedure

```bash
cd spark_transport/sircl
SITE=/path/to/site.json
python -m sparkring_sircl.ring plan      --site "$SITE" --config all                   # 1. OFFLINE
python -m sparkring_sircl.ring preflight --site "$SITE" --config all                   # 2. READ-ONLY REMOTE
python -m sparkring_sircl.ring stage     --site "$SITE" --config all                   # 3. MUTATES HOST
python -m sparkring_sircl.ring run       --site "$SITE" --config path4 --print         # 4. OFFLINE
python -m sparkring_sircl.ring run       --site "$SITE" --config all --transport-only  # 5. MUTATES HOST
python -m sparkring_sircl.ring run       --site "$SITE" --config all                   # 5. MUTATES HOST
python -m sparkring_sircl.ring cleanup   --site "$SITE"                                # 6. MUTATES HOST
python -m sparkring_sircl.ring summarize --results sircl-ring-results/<run id>         # 7. OFFLINE
```

1. **Plan.** Per configuration: groups, every rank's `SIRCL_PEER_ROUTES`,
   every relayed lane with the Sparks it crosses and its forward window, the
   relay load factor, the containers with each Spark's Docker command, sizes.
2. **Preflight.** Per used Spark: SSH reachable; Docker reachable; image
   present; a GPU listed; four RDMA devices ACTIVE; the wired-LAN address
   equal to the site file's and outside every fabric subnet; no container
   running; and `ip route get` sends every lane's destination over the lane's
   own device. Expected: `preflight passed`, else every blocker and exit 1.
3. **Stage.** Copies the sources to `<remote_dir>/src/<source digest>` and
   builds the native library inside the image into `<remote_dir>/build-cache`.
   Expected per Spark: `built`, the library path, torch and CUDA versions.
   Stage again after any package change; `run` refuses an unstaged digest.
4. **Dry run.** `run --print` prints the launch plan and contacts nothing.
5. **Run.** One configuration at a time, smallest first, `--transport-only`
   before the GPU cases, then the large and specialized ones as needed. Each
   run starts the containers, waits, collects the results, removes the
   containers and prints the summary table.
6. **Cleanup.** Expected: `0 harness containers left` per Spark. Files under
   `remote_dir` stay; remove that directory by hand.
7. **Summarize.** Prints the table again for a configuration's or a run's
   folder.

**Fail-stop.** A failing rank writes its error and exits; the launcher waits
up to 60 s for the other ranks, then removes every container. A hanging rank
is ended by its watchdog after 1,500 s. A run that prints
`harness containers may remain` needs `cleanup`.

**Exit codes.** `run` and `tune` exit 0 when every rank of every
configuration exited 0, 1 otherwise, 2 on an invalid site file or plan; a
rank exits 1 on an error, 2 on a bit mismatch, 3 when its watchdog ends it.
The summary's status (`PASSED` or `FAILED`) also fails a configuration whose
error counters moved; `summarize` and `trace` exit 0 only when every
configuration passed.

### Results

`sircl-ring-results/<run id>/<configuration>/` holds `plan.json` and
`plan.txt`; per rank `rank-N.json` (`sircl-ring-rank-result/v1`: placement,
session settings, every case, native counters, NCCL details, tuning
decisions, the error of a failed rank) and `rank-N.log`; `result.json`
(`sircl-ring-configuration-result/v1`: merged cases, problems, warnings,
status); `summary.txt`; and with `tune`, `tuning-group<index>.json`.

Summary columns: `group`, `collective`, `mode` (`eager` or `graph`), `bytes`,
`shape`, `exact` (`yes` or `NO`), `p50 us`, `p90 us`, `p99 us` of the slowest
rank (per call the maximum over the group's ranks, then the percentile),
`busbw GB/s` at the slowest median (bus factor `2 (W - 1) / W` for
all-reduce, `(W - 1) / W` for all-gather), `vs NCCL` with the baseline, and
every error counter that moved: each RDMA device's sysfs `hw_counters` and
`counters` and its interface's Ethernet statistics (`rx_out_of_buffer`,
hairpin drops). Bracketed notes give the algorithm or schedule, posting
order, bound and fraction of it reached (`sparkring_sircl.bounds`), latency
model, grid cap and forward-window waits. Lines after the table: `crossover`
(the `SIRCL_ONESHOT_MAX_BYTES` below the first size at which two-shot is
faster), `posting orders`, `tuning, group G`, eager profiles, `warning:` and
`problem:`.

### Tuning tables

A tuning table (`sircl-tuning-table/v1`, format in
[`README.md`](README.md#tuning-tables)) holds the measured times of every
candidate on one group shape and the decisions derived from them:

```bash
python -m sparkring_sircl.ring tune --site "$SITE" --config ring8 --baseline nccl           # MUTATES HOST
python -m sparkring_sircl.ring tune --site "$SITE" --groups 0-1 --name pair --baseline nccl  # MUTATES HOST
python -m sparkring_sircl.ring tune --site "$SITE" --groups 0-3 --name path4                # MUTATES HOST
python -m sparkring_sircl.ring tune-table --results sircl-ring-results/<run id>             # OFFLINE
python -m sparkring_sircl.ring run --site "$SITE" --config ring8 --large \
    --tuning-table sircl-ring-results/<run id>/ring8/tuning-group0.json                     # MUTATES HOST
```

- `tune` runs a configuration's groups like `run` and times every candidate
  of the all-reduce, all-gather, reduce-scatter and all-to-all at every
  per-rank size from 4 KiB to 128 MiB in powers of two (`--quick`: every
  fourth), eager and in graph replay: one-shot and two-shot at each grid cap
  (`--tune-grids`); from 256 KiB, pieces, tiles and scatter ops, the chain at
  each piece (`--tune-pieces`), the ring at each piece and stagger
  (`--tune-staggers`); Swing where the session offers it; NCCL with
  `--baseline nccl`. From 4 MiB, a candidate 1.5 times slower than the
  fastest at two sizes in a row stops (`--tune-prune-from`, `--tune-prune`).
- `tune` sessions take link slots that hold the largest piece swept, as many
  as the session's default (`2 W`, at least 8: 16 on the cycle of eight) or as
  every stagger swept needs, whichever is more, unless `--session-env` sets
  them.
- Each group's table (exact cases only) is printed with its hash and written
  to `tuning-group<index>.json`; `tune-table` rebuilds them from a folder.
  Each table records the session settings its choices ran under and need
  ([README.md, Tuning tables](README.md#tuning-tables)); `tune-table` prints
  them and, per collective and mode, the tune cases, the exact and inexact
  ones, and where the table decides nothing.
- `run --tuning-table PATH` (one per group shape) writes the tables beside
  `plan.json` on every Spark and sets `SIRCL_TUNING_TABLE`; each group's
  sessions take the table's settings where `--session-env` leaves them unset.
  The plan refuses a table no group matches and a `--session-env` value below
  a setting of a group's table; rows that force a schedule, piece, stagger or
  grid cap run with the table suspended. The summary names, per group, the
  settings the sessions ran under beside the table's. The serve launcher
  stages tables the same way (`--tuning-table`).
- A table's key names the group shape, size, lanes, relays, native and
  kernel source hashes, package version and image: run `tune` again after a
  change to any of them.

### Diagnostics

```bash
python -m sparkring_sircl.ring run --site "$SITE" --config path4-large --session-env SIRCL_EVENT_TRACE=65536
python -m sparkring_sircl.ring trace --results sircl-ring-results/<run id>/path4-large
python -m sparkring_sircl.ring run --site "$SITE" --groups 0-1 --name pair --eager-profile --eager-path adapter
```

- **Event trace.** Each rank runs one extra call of every large chain or ring
  case with the event trace; `trace` prints per case, rank and stream the
  median and 90th percentile of the stages `kernel`, `notice`, `credit`,
  `wire` and `gap`. Traced timings do not compare with untraced ones: the
  traced chain kernel is compiled separately and the clock is read per event.
- **Eager profile.** Eager rows of `all_reduce`, `all_gather`,
  `all_reduce_large` and `all_gather_large` record the sessions' call stages;
  the summary prints the first and the slowest rank's stage medians.
  `--eager-path adapter` runs those calls through the vLLM adapter's planner
  and executor; `--session-env SIRCL_CALL_PROFILE_GPU=1` adds device times.

## Point-to-point cases

`python -m sparkring_sircl.ring.p2p` runs SIRCL's point-to-point channels on
one group: the whole ring (`--config ring`, default), `--config path4`,
`--config pair` or consecutive Sparks (`--positions 2-5`). It shares the
harness's site file, staged sources (`stage --config ring`), container label
and `cleanup`.

```bash
python -m sparkring_sircl.ring.p2p plan --site "$SITE"                                  # OFFLINE
python -m sparkring_sircl.ring.p2p run --site "$SITE"                                   # MUTATES HOST
python -m sparkring_sircl.ring.p2p summarize --results sircl-ring-results/<run id>/p2p  # OFFLINE
```

- Cases: `pair A-B` (checked messages both ways, ping-pong latency as half a
  round trip, burst bandwidth over `--bandwidth-bytes`, default 256 MiB) and
  `shift d` (every rank sends to the rank `d` ahead at once). Defaults on the
  ring of eight: pairs 0-1, 0-2, 0-3, 0-4 (zero to three relays) and shifts
  1, 2, 4. `--cases` and `--sizes` (default 4 KiB to 64 MiB in powers of
  four) choose others. `run` refuses beside other containers unless `--force`.
- `sircl-ring-results/<run id>/p2p` holds the plan, `p2p-rank-N.json`
  (`sircl-ring-p2p-rank-result/v1`, with error-counter increases),
  `p2p-result.json` and `p2p-summary.txt` (per case and size: hops, checked
  messages, one-way p50/p99 latency, GB/s). A pass is status `passed` and no
  counter increase in any `p2p-rank-N.json` (the status omits counters).

## Relay plan installer

The relay plan installer (`python -m sparkring_sircl.fabric`, console script
`sircl-fabric`) installs, compares and removes, over SSH, the relay plan that
lanes between group members without a shared cable need. It derives each
plan from `sparkring_sircl.routes`, from which SIRCL sessions and the harness
derive their lanes, and checks that every SIRCL lane is carried.

### What a Spark holds

The plan carries every shortest path between two members of a group on both
function classes. Per member:

- **origin routes**: a /32 route, scope link, over the lane's local network
  device with its address as source, and a permanent neighbour entry holding
  the adjacent Spark's MAC on that cable;
- **marker rules**: one marker process per RDMA device (source
  `sparkring_sircl/fabric/mesh_marker.c`) whose mlx5 RDMA-TX rules rewrite
  the EtherType of RoCE packets to each routed destination into
  `0x88b4 + k`, `k` relays left; tags follow the destination address, never
  the UDP port, and every queue pair uses flow label 0;
- **relay filters**: on the ingress of the device that receives a relayed
  path, one hardware flower filter per tag that sets the downstream Spark's
  MAC, rewrites the tag to the one for `k - 1` (`0x0800` for the last relay)
  and redirects out of the other port;
- **a record**, `/run/sparkring-fabric/relay-state.json`: group, layout,
  relay egress, plan digest and the ingress qdiscs the installer added.

Objects exist only for same-group destinations and on group members; every
origin route, followed through the plan, reaches its destination without
touching a Spark outside the group. The plan prints the SIRCL lanes through
each relay's 512 KiB hairpin queue, the load factor `f` and the per-peer op
bound `0.75 * 512 KiB / f` (131,072 bytes on the ring of eight, 393,216 on a
path of four).

### Layouts

| Layout | Groups (ring positions) | Relay egress | Objects per Spark |
|---|---|---|---|
| `ring8` (`ring<N>` on other ring sizes) | every Spark: each reaches all four addresses of every Spark it shares no cable with | same function | 12 routes, 12 relay filters, 4 markers |
| `2xTP4` | 0-3 and 4-7 | sibling function | ends: 4 routes, 2 markers; Sparks 1, 2, 5, 6: 2 routes, 6 relay filters, 2 markers |
| `4xTP2` | 0-1, 2-3, 4-5, 6-7 | none needed | the record only |
| `--groups '0-3;4-7'`, `'7,0,1,2'` or `'7-2'` (`--name` names it) | consecutive Sparks, wrap-around included | sibling function | relay filters on inner members only |

A relay forwards out of its other port through the function of the lane's
class (`same`) or the sibling Socket Direct function on that port
(`sibling`); either way the frame carries the downstream Spark's MAC of the
lane's class. `--relay-egress` overrides the default. `--group` acts on one
group (index, label such as `path:4-5-6-7`, or members such as `4-7`) and
contacts no other group's Sparks. `--max-relays` (default 3, the qualified
limit) caps the relays per lane.

### Ownership

The installer changes only objects that carry its marks: routes and
permanent neighbours with routing protocol 82 (`proto 82`); flower filters at
preferences 11 to 17 of chain 0 on the ingress of the four fabric network
devices (preference `10 + k`, handle `k`); processes of the marker executable
(`/var/tmp/ring8-mesh-marker`, process name `ring8-mesh-mark`; `--marker`
sets another path); an ingress qdisc it added (removed by `down` once no
filter remains); and the record. It never touches NetworkManager
connections, addresses, links, or other routes, neighbours, filters, qdiscs
or processes. Marker logs (`/tmp/ring8-mesh-marker-<device>.log`) stay after
`down`.

Relay routes and neighbours without protocol 82 show as `unowned`;
`up --apply` and `down --apply` refuse a Spark that holds any unless
`--adopt` is given (`diff` shows what it would do). With `--adopt`, `up`
marks those equal to the plan in place (only the protocol changes) and
removes the others, and `down` removes them. `up` and `down` refuse a Spark
whose record names another group; to switch layouts, run `down` first.

### Safety classes of the installer

| Command | Class |
|---|---|
| `plan` | OFFLINE |
| `facts`, `show`, `diff` | READ-ONLY REMOTE (no `sudo`) |
| `up`, `down` without `--apply` (the default) | READ-ONLY REMOTE: a dry run that prints every change and the exact script per Spark |
| `up --apply`, `down --apply` | MUTATES HOST: routes, neighbours, qdiscs, filters, marker processes and the record on the selected groups' Sparks. STOPS SERVING for a model whose lanes use objects that change |
| `marker` | READ-ONLY REMOTE; with `--apply` MUTATES HOST (compiles the marker source on the Spark and installs the executable) |

`--apply` refuses on any blocker, runs each Spark's script
(`set -euo pipefail`; it first checks the host name), then reads
the Sparks again and verifies them against the plan and with `ip route get`
of every lane destination. A changed relay filter or marker is re-created,
briefly interrupting the lanes that use it; unchanged objects stay untouched.

### Installer procedure

Each ring entry's `name` must equal the Spark's host name
(`--skip-hostname-check` overrides), and its SSH user needs passwordless
`sudo -n` for `--apply`. `marker --apply` compiles on the Spark and needs
`cc` with the libibverbs and libmlx5 development headers there. In Git Bash
on Windows, set `MSYS_NO_PATHCONV=1` before passing absolute paths such as
`--marker`.

```bash
cd spark_transport/sircl
python -m sparkring_sircl.fabric facts  --site "$SITE" --output facts.json                # 1. READ-ONLY REMOTE
python -m sparkring_sircl.fabric plan   --site "$SITE" --layout ring8 --facts facts.json  # 1. OFFLINE
python -m sparkring_sircl.fabric show   --site "$SITE"                                    # 2. READ-ONLY REMOTE
python -m sparkring_sircl.fabric marker --site "$SITE" --apply                            # 3. MUTATES HOST
python -m sparkring_sircl.fabric diff   --site "$SITE" --layout ring8                     # 4. READ-ONLY REMOTE
python -m sparkring_sircl.fabric up     --site "$SITE" --layout ring8                     # 5. READ-ONLY REMOTE
python -m sparkring_sircl.fabric up     --site "$SITE" --layout ring8 --apply             # 6. MUTATES HOST
python -m sparkring_sircl.ring preflight --site "$SITE" --config ring                     # 7. READ-ONLY REMOTE
python -m sparkring_sircl.fabric down   --site "$SITE" --layout ring8 --apply             # removal, MUTATES HOST
```

1. **Plan.** `facts` writes the Sparks' fabric addresses and MACs to a facts
   file (`sparkring-fabric-facts/v1`); without `--facts`, `plan` shows
   placeholders.
2. **Show.** Per Spark: its group record (or `no group record`), fabric
   addresses, the marker executable's digest, and every relay object
   labelled with its group or `unowned`; `--raw` adds the raw `ip -j` and
   `tc -j` output (iproute2 5.x and 6.x spellings are parsed).
3. **Marker.** Builds the marker executable where it is missing; a missing
   executable blocks `up`.
4. **Compare.** `diff` prints the changes per Spark, a summary line such as
   `ring8 up: no host changes on 8 Spark(s)`, and the lane route check.
5. **Dry run.** Prints every change and script. A group whose cables do not
   join their two links in one subnet is refused: the site's ring order
   differs from the cabling.
6. **Apply.** Stop serving on the affected Sparks first. Expected: `exit 0`
   per Spark, then
   `verified: 8 Spark(s) hold the up state of cycle:0-1-2-3-4-5-6-7`.
7. **Check** with the harness preflight ([Procedure](#procedure), step 2).

Exit codes: 0 done (`diff`: no host changes); 1 `diff` found host changes,
`show` or `marker` could not reach a Spark, or applying or verifying failed;
2 invalid input; 3 a blocker stopped a dry run or `--apply`, `diff` found a
group that cannot be planned, an unreachable Spark or a lane destination
routed over another device, or `facts` found a problem.

### Limitations

- Nothing restores a relay plan after a reboot: its objects and the record
  are runtime state (`/run` is cleared at boot), and a ConnectX function
  restart removes the routes and filters on that function. After either, run
  `up --apply` again once the hairpin setting is in effect.
- The installer does not set or check the hairpin setting
  ([Measurement prerequisites](#measurement-prerequisites)).
- SparkRing's installer (`sparkring install`, `sparkring setup`) does not
  run the relay plan installer; install relay plans with `sircl-fabric`.

## Related documents

- [`README.md`](README.md): sessions, schedules, environment variables,
  tuning tables, build and tests.
- [`STATUS.md`](STATUS.md#component-status): the status of every component;
  [`STATUS.md`](STATUS.md#measured-performance): measured results and their
  conditions.
- [`sparkring_sircl/vllm/RUNBOOK.md`](sparkring_sircl/vllm/RUNBOOK.md):
  serving through the vLLM adapter on Sparks of a ring.
