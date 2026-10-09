# SIRCL vLLM adapter runbook: serving SparkRing profiles on a ring

`python -m sparkring_sircl.vllm.serve` ([`serve/cli.py`](serve/cli.py))
serves a SparkRing serving profile on consecutive Sparks of a switchless ring,
with SIRCL's vLLM adapter in front of every collective. It runs on the
operator's machine and reaches each Spark over SSH with the ring harness's
site file and SSH helpers. It creates its own containers, one per rank, named
`sircl-serve-<run>-r<rank>`. It is not SparkRing's installer: it never
changes an installer deployment, and the installer does not manage its
containers.

Status: serving through the adapter is **research-only**
([package status table](../../STATUS.md#component-status)). The checks to run
on a ring before relying on a layout are in [`STATUS.md`](STATUS.md).

| Group | `--positions` | Cabling | NCCL |
|---|---|---|---|
| four consecutive Sparks (TP4), a path | `0-3`, `4-7`, `7-2` (the same as `7,0,1,2`) | ranks 0 and 3 share no cable; their lanes cross the hardware relays of the two middle Sparks | never runs |
| two neighbouring Sparks (TP2), a pair | `0-1` | one cable | runs only with `--nccl auto` |
| several disjoint groups, one instance each | `--groups '0-1;2-3;4-5;6-7'` | as above, per group | as above |

## What the launcher changes

The launcher reads the profile from a SparkRing checkout and stops on any
disagreement between its sources ([`serve/profile.py`](serve/profile.py)):
`profile.json` and the release's `installer-image.json`, the serving
configuration (`config.json`), the per-rank Compose files (every key must be
one the launcher translates), `SHA256SUMS` with the checkpoint manifest,
`runtime/common/loader-seccomp.json` and `profiles/thinking.json`. It runs
each rank's container as its Compose file describes it, with these changes
([`serve/plan.py`](serve/plan.py)):

| Setting | Value | Purpose |
|---|---|---|
| `PYTHONPATH` | `/sircl/src`, the staged package tree (read-only); with `--overlay`, `/opt/sparkring-overlay:/sircl/src` | vLLM finds the `sircl` platform and general plugins through the tree's generated `sparkring_sircl-<version>.dist-info`; nothing is installed into the image |
| `VLLM_PLUGINS` | the profile's list plus `sircl` | load SIRCL's plugins |
| `SIRCL_MODE=custom`, `SIRCL_FABRIC=ring:<size>`, `SIRCL_RANK_POSITIONS`, `SIRCL_GROUPS=tp` | from `--positions` | place the group on the ring |
| `SIRCL_NCCL` | `never` (default); `auto` with `--nccl auto` (`--nccl topology` is the same) | where NCCL may run (section "Serving without NCCL") |
| `SIRCL_LARGE_ALLREDUCE` | `--large-allreduce auto` (default), `sircl` or `nccl` | where eager all-reduces above the dispatch ceiling run on a group NCCL may use; `nccl` is refused where NCCL may not all-reduce |
| `SIRCL_SESSION_MODULE=sparkring_sircl.oneshot`, `SIRCL_NATIVE_LIBRARY`, `SIRCL_BUILD_CACHE_DIR` | `/sircl/build-cache/roce_proxy-<hash>.so` (read-only mount) | the sessions and the native library `stage` built from the staged source |
| `SIRCL_ALLREDUCE_CAPACITY_BYTES`, `SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES`, `SIRCL_ALLGATHER_MAX_BYTES` | `--capacity` 131,072, `--dispatch` (the capacity), `--gather` 155,648 | one-op limits; larger all-reduces and all-gathers run in the session's large-message ops (or on NCCL, where it may run) |
| `SIRCL_GID_INDEX` | `--gid-index`, else the site's, else the profile's `NCCL_IB_GID_INDEX`, else 3 | RoCE GID index |
| `SIRCL_STARTUP_WAIT_S`, `SIRCL_SERVING_WAIT_S` | `--startup-wait` 600, `--serving-wait` 20 | how long a session waits for a late peer: through setup, warm-up, graph capture, profiling, sleep and wake-up; then from the first step after warm-up |
| `SIRCL_RECEIPT_DIR=/sircl/run/receipts` | per-run mount | every rank writes one JSON receipt per group |
| `SPARK_TP4_ENABLED=0`, `VLLM_SPARK_TP4_MODE` and `VLLM_SPARK_TP4_VOCAB_MODE` empty | | the four-rank adapter stays off: every four-rank startup hook of the repository's vLLM integration ([`integrations/vllm`](../../../../integrations/vllm/README.md)), which serves four-Spark cycles only |
| `VLLM_ENABLE_ROCE_ALLREDUCE=0`, `SPARKRING_TRANSPORT_PROFILE` and `SPARKRING_TRANSPORT_MANIFEST_SHA256` empty | | the installer image's `prepared` transport (RoCEnante) stays off; SIRCL is the one RDMA transport |
| `NCCL_IB_HCA` | only where NCCL runs (a pair with `--nccl auto`): the devices facing the partner, port 0's functions on the lower position and port 1's on the higher | the profile's value names port 0 on both ranks, the cabling of two Sparks connected port to port |
| `VLLM_HOST_IP`, `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`, `--master-addr` | from the site file | each Spark's wired-LAN address and interface; rank 0's address is the master |
| `--port` (also in the rank-0 health check), `--master-port` | the instance's ports | API and torch.distributed rendezvous |
| container name and labels | `sircl-serve-<run>-r<rank>`, `sircl-serve=<run>`, no `io.sparkring.*` label | `stop` finds exactly these containers |

Options that set a variable only when given (otherwise the session's default
or the profile's value applies; `plan` shows which):

| Option | Variable | Meaning |
|---|---|---|
| `--oneshot-max BYTES` | `SIRCL_ONESHOT_MAX_BYTES` | largest all-reduce the sessions run one-shot: a multiple of 16 up to the dispatch ceiling, 0 for two-shot only. Unset, the latency model's limit for the layout: 73,728 B on a path of four, 131,072 B on a pair, 28,672 B on the ring of eight |
| `--large-blocks N` | `SIRCL_LARGE_BLOCKS` | grid cap of two-shot and large-message launches, a power of two from 1 to 1,024 (default 32) |
| `--large-schedule`, `--gather-schedule`, `--scatter-schedule` | `SIRCL_LARGE_SCHEDULE`, `SIRCL_GATHER_SCHEDULE`, `SIRCL_SCATTER_SCHEDULE` | `auto`, `chain`, `ring` or `pieces` (defaults `auto`, `auto`, `pieces`; `auto` never selects the ring) |
| `--chain-min BYTES`, `--ring-min BYTES` | `SIRCL_CHAIN_MIN_BYTES`, `SIRCL_RING_MIN_BYTES` | smallest collective run as one chain or ring op, one size for all three collectives (defaults for all-reduce, all-gather, reduce-scatter: chain 8, 8, 4 MiB; ring 4, 8, 4 MiB) |
| `--link-slot`, `--link-chunk`, `--gather-link-chunk`, `--scatter-link-chunk`, `--reduce-link-chunk`, `--link-slots`, `--ring-gather-stagger` | `SIRCL_LINK_SLOT_BYTES`, `SIRCL_LINK_CHUNK_BYTES`, `SIRCL_{GATHER,SCATTER,REDUCE}_LINK_CHUNK_BYTES`, `SIRCL_LINK_SLOTS`, `SIRCL_RING_GATHER_STAGGER` | slot, pieces, slots per link and ring all-gather stagger of the chain and ring collectives; `plan` refuses a piece above the slot and a stagger the slots cannot hold |
| `--tuning-table PATH` | `SIRCL_TUNING_TABLE` | section "Measured tuning tables" |
| `--spin-limit N` | `SIRCL_SPIN_LIMIT` | poll budget of waits without a time limit; time-limited flag waits ignore it |
| `--nccl-debug` (implied by `--require-no-nccl`) | `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT` | NCCL logs every communicator it creates |
| `--mhc-prefill-shard off` | `VLLM_GLM53_MHC_PREFILL_SHARD=0` | section "mHC prefill sharding" |
| `--b12x-cache-dir CONTAINERPATH` | `B12X_COMPILE_CACHE_DIR` | where B12X compiles its kernels; must lie outside the read-only mounts |
| `--reasoning-effort LEVEL` | rank 0's `--default-chat-template-kwargs '{"reasoning_effort":"LEVEL"}'` | effort of requests that name none; LEVEL must be a level `profiles/thinking.json` records for the checkpoint (GLM-5.3-Flash: `low`, `high`, `max`); SparkRing's installer sets it the same way ([install reference](../../../../docs/operations/install-reference.md#thinking)) |
| `--env KEY=VALUE` | any other variable, every rank | replaces the profile's value (the plan shows `<profile's value> -> <value> (--env)`); refuses every `SIRCL_*` variable, naming its option, and the other variables the launcher sets: `PYTHONPATH`, `VLLM_PLUGINS`, `B12X_COMPILE_CACHE_DIR`, the addresses and interfaces, `NCCL_IB_HCA`, `VLLM_GLM53_MHC_PREFILL_SHARD` and the transport switches |

The launcher leaves `SIRCL_POST_ORDER` unset, so each session posts its lanes
to the peers with the most relays first. Every other variable, every `B12X_*`
and `CUTE_*` variable included, keeps the profile's value; B12X keys its
compiled kernels on the `B12X_*` values, so a warm cache keeps serving.

`plan` refuses, naming what to change: micro-batching (`--enable-dbo`,
`--ubatch-size` above 1); where NCCL may not run, the `fuse_gemm_comms` and
`fuse_allreduce_rms` compilation passes, `--enable-batch-sharded-sampling`
and expert-parallel all-to-all backends other than `naive` and
`allgather_reducescatter`; with NCCL off, the NCCL-free settings of section
"Serving without NCCL"; a profile that sets a variable the launcher owns
(`SIRCL_LAYOUT`, `SIRCL_PEER_ROUTES`, `VLLM_DISABLE_PYNCCL` and the
algorithm overrides); and session settings a session would refuse at setup.

### Files on the Sparks

Under the site's `remote_dir`:

| Path | Contents | Lifecycle |
|---|---|---|
| `serve/src/<digest>/` | the package tree, its dist-info and `spark_roce_gid.py`; `<digest>` covers every file | written once by `stage`; a directory without its completion marker is reported for the operator to inspect and remove |
| `build-cache/` | the native collective and point-to-point libraries, built in the serving image and named by source hash; shared with the ring harness | written once per source hash |
| `serve/loader-seccomp-<sha16>.json` | the profile's seccomp policy | written once |
| `serve/runs/<run>/receipts/`, `serve/runs/<run>/tuning/` | the ranks' receipts; staged tuning tables | per run |
| `serve/cache/<profile>/` | vLLM, B12X, Triton and CuTe DSL caches (the default `--cache-path`) | cache |

## Safety classes

| Class | Meaning | Commands |
|---|---|---|
| OFFLINE | reads and writes only the operator's machine | `plan`, `start --print`, `bundle`, `shims` |
| READ-ONLY REMOTE | read-only commands on the Sparks over SSH | `preflight`, `wait`, `status`, `logs`, `collect` (writes local files), `check` (also sends inference requests to these runs' servers), `bundle-check` |
| MUTATES HOST | writes under `remote_dir` or the cache directory, or starts containers | `stage` (files and one short-lived build container per Spark), `start` (one serving container per rank), `bundle --stage` |
| STOPS SERVING | removes containers | `stop` (these runs' containers); `stop --all-runs` (every `sircl-serve` container on every Spark of the site) |

`start` refuses while any other container runs on a used Spark;
`--allow-running NAME` lets one that holds no GPU memory stay.

## Inputs

- **Site file** (`sircl-ring-site/v1`): the ring harness's site file
  ([package runbook](../../RUNBOOK.md)); each Spark's `docker` field is the
  command that runs Docker there, for example `"sudo -n docker"`. Two keys of
  the launcher's own ([`serve/sitefile.py`](serve/sitefile.py)): `model_path`
  in a ring entry, that Spark's checkpoint directory; and `sudo` (top level or
  ring entry), the prefix of host-side file operations outside containers.
  Without `sudo`, a Spark whose `docker` starts with `sudo` uses those words
  (`"sudo -n"`); `"sudo": ""` suits a Spark whose sudo rules allow only
  Docker, whose model and cache directories must then be readable by the SSH
  user.

  ```json
  {"name": "spark-c", "ssh": "user@192.0.2.12", "lan_address": "192.0.2.12", "docker": "sudo -n docker",
   "model_path": "/srv/sparkring/<cluster>/checkpoints/local-inference-lab--GLM-5.3-Flash-NVFP4-Spark/a608241037e4c2565356bff7ca293f2133888f88"}
  ```

- **Repository checkout** (`--repository`) holding the profile (`--profile`,
  default `glm53-flash-nvfp4-spark-tp4`).
- **Groups, ports, run IDs.** `--positions` names one group (default: the
  first N Sparks), `--groups` several. Instance k listens on `--api-port` + k
  (default 8017) and rendezvouses on `--master-port` + k (default: the
  profile's). Run IDs are `tp<N>-<first>-<last>`; `--run-id` names one, or
  prefixes several.
- **Model directory**: `--model-path N=PATH` for Spark N, else the site
  entry's `model_path`, else `--model-path PATH`. The launcher never downloads
  weights; SparkRing's installer keeps checkpoints at
  `/srv/sparkring/<cluster>/checkpoints/<owner>--<name>/<revision>`.
- **Cache directory** (`--cache-path PATH` or `N=PATH`, default
  `<remote_dir>/serve/cache/<profile>`). An empty cache compiles every B12X
  kernel at startup; a copy of the installer's cluster cache
  (`/srv/sparkring/<cluster>/cache`) reuses kernels compiled for the same
  image, checkpoint and `B12X_*` values.

Prerequisites: the relay plan of every group whose lanes cross relays is
installed ([Relay plan installer](../../RUNBOOK.md#relay-plan-installer));
the profile's installer image is on every used Spark; the operator's machine
has Python 3.10 or later with PyYAML and an OpenSSH client with batch-mode
key access to every Spark. In Git Bash on Windows, set `MSYS_NO_PATHCONV=1`.

## Procedure

Run every command from `spark_transport/sircl`, each with the same arguments:

```bash
SITE=/path/to/site.json            # sircl-ring-site/v1 with model_path entries, kept out of version control
REPO=/path/to/sparkring            # repository checkout
SERVE="python -m sparkring_sircl.vllm.serve"
ARGS="--site $SITE --repository $REPO --positions 0-3"
```

### 1. Plan (OFFLINE)

```bash
$SERVE plan $ARGS
$SERVE start --print $ARGS
```

`plan` prints the group and its NCCL policy (`path:0-1-2-3 with NCCL policy
none (SIRCL_NCCL=never); every collective runs on SIRCL`), route maps and
relays, SIRCL sizes and sessions, every environment change with its reason,
the carrier of prefill row ownership, the process groups vLLM builds and
their carriers, and each rank's model directory, host file-operation prefix
and `docker run` command. `plan --json` prints the record
(`sircl-serve-plan/v1`). `start --print` adds every remote command of `stage`
and `start`, and contacts nothing.

### 2. Preflight (READ-ONLY REMOTE)

```bash
$SERVE preflight $ARGS
```

It runs the ring harness's checks with the serving image (Docker, image, GPU,
LAN address, the four RDMA devices ACTIVE, every lane routed over its own
device; `--no-fabric` skips them), then on each rank's Spark: every pinned
checkpoint file at its manifest size and the SHA-256 of `config.json` and
`model.safetensors.index.json` equal to the pins; available memory at least
`--gpu-memory-utilization` of the total; running containers; the cache
directory and its free space; staging state; free ports and `curl` on rank-0
Sparks; overlay directories. Expected: `preflight passed`; otherwise
`BLOCKER:` lines and exit code 1.

### 3. Stage (MUTATES HOST)

```bash
$SERVE stage $ARGS
```

On each Spark it writes the package tree and seccomp policy once, then runs
one short-lived container of the serving image (`--network none`) that builds
the native libraries and probes what serving will import
([`serve/probe.py`](serve/probe.py)). Expected, per Spark:

```text
run tp4-0-3 rank 0 (spark-a): package tree <digest> staged, seccomp policy written, library roce_proxy-<hash>.so built, sparkring_sircl from /sircl/src/sparkring_sircl/__init__.py, vllm from <image vLLM>/__init__.py, b12x from <image B12X>/__init__.py
run tp4-0-3 rank 0 (spark-a): vLLM at <image vLLM> matches pinned build lil-image-aba309e4610c
run tp4-0-3 rank 0 (spark-a): shims: <every catalogued shim's status>
stage complete
```

Shim statuses are `verified`, `applicable-unverified` or `absent`
([Shims](README.md#shims); `$SERVE shims` prints the catalog). A blocker
names another `sparkring_sircl` earlier on the module path, a second `sircl`
entry point, a missing library, an overlay that does not supply `vllm` and
`b12x`, or a needed shim whose vLLM files match no pinned build.

### 4. Start and wait (MUTATES HOST)

```bash
$SERVE start $ARGS --wait --timeout 3600
```

`start` checks every used Spark (no running container, no container of these
runs, staged files present, ports free, overlay directories) and prints
`nothing was started` on any `BLOCKER:`. It creates the cache and run
directories, writes tuning tables and starts every rank; a failed `docker
run` removes what these runs started. `--wait` polls until each instance's
`/v1/models` lists the model, reports every 60 s, and stops waiting on an
instance whose log names a SIRCL setup or dispatch error, a refused shim, a
poisoned session or a failed `sircl` plugin import. Expected:

```text
API ready after <seconds> s: run tp4-0-3 serves GLM-5.3-Flash-NVFP4-Spark-TP4
run tp4-0-3 rank 0 (spark-a): <n> receipt line(s), <n> registration line(s), <n> platform activation line(s)
  SIRCL receipt group=tp:0 global_rank=0 rank=0 world=4 layout=ring:8 fabric=path:0-1-2-3 positions=0,1,2,3 nccl=none pynccl=skipped session=ring <...> mhc=sircl fused_norm=off <...> wait=startup:600s vllm=<image vLLM> state=ready
  SIRCL receipt group=ep:0 global_rank=0 rank=0 world=4 <...> nccl=none pynccl=skipped session=shared:tp:0 <...> state=ready
```

A rank whose container exits is reported with its last 40 log lines; the
containers stay unless `--stop-on-failure` is given. A rank without a
`group=tp` receipt line fails the run (the `sircl` plugins did not run in its
worker). `wait` alone resumes waiting.

### 5. Check (READ-ONLY REMOTE plus inference requests)

```bash
$SERVE check $ARGS --long-prompt 16384
```

Three greedy chat requests per instance with the profile's request settings
([`serve/checks.py`](serve/checks.py)); the first two are the repository's
acceptance checks:

| Name | Prompt | Passes when the reply |
|---|---|---|
| count | `Count from 1 to 20, comma separated. Output only the numbers.` | contains `1, 2, ..., 20` |
| arithmetic | `What is 17*23? Reply with the number only.` | is exactly `391` |
| capital | `What is the capital of France? Reply with one word.` | is the word `Paris` |

`--long-prompt 16384` also times one prompt of about 16,384 tokens and, for
the GLM-5.3-Flash profiles' checkpoint, counts its session ops from rank 0's
receipt. `check` fails on a rank without a `ready` `tp:0` receipt; on a group
with `nccl=none`, PyNccl built or a decision row on `nccl`; any `refuse` row;
a `tp:0` group without an all-reduce on SIRCL; and, when given, an overlay,
`--large-blocks` cap or tuning table the receipts do not confirm. Expected:
`passed` per prompt, the decision rows per rank and group (for example
`all_reduce/sircl/direct`, `all_reduce/sircl/large`,
`reduce_scatter/sircl/scatter`) and `run <run>: check passed`. After the first
request the receipts show `wait=serving:20s`.

### 6. Status, logs, collect and stop

```bash
$SERVE status $ARGS                        # READ-ONLY REMOTE
$SERVE logs $ARGS --rank 3 --tail 200      # READ-ONLY REMOTE
$SERVE stop $ARGS                          # STOPS SERVING
```

`stop` saves every rank's log and receipts and the plan record to
`sircl-serve-results/<run>/` (`--output`; `--no-collect` skips it), removes
the containers labelled `sircl-serve=<run>` and prints how many remain
(expected `0`); `collect` saves without stopping. `stop --all-runs` removes
every `sircl-serve` container of the site without collecting. Neither touches
another container or any file under `remote_dir`.

### Other groups

```bash
$SERVE start --site $SITE --repository $REPO --positions 4-7 --wait   # beside a run on 0-3
$SERVE start --site $SITE --repository $REPO --positions 7-2 --wait   # across the ring's 7-0 cable
PAIRS="--site $SITE --repository $REPO --profile glm53-flash-nvfp4-spark-tp2 --groups 0-1;2-3;4-5;6-7"
$SERVE stage $PAIRS && $SERVE start $PAIRS --wait && $SERVE check $PAIRS
```

The four pairs serve on ports 8017-8020, every collective on SIRCL. With
`--nccl auto`, NCCL carries eager all-reduces above the dispatch ceiling over
each pair's cable (receipts `nccl=all pynccl=built`) unless
`--large-allreduce sircl` is given.

## Another checkpoint or vLLM build

A profile's image can serve a checkpoint the profile does not pin, through a
vLLM and B12X from a source overlay. Each option changes only what it names:

- `--overlay HOSTDIR` (or `N=HOSTDIR`) mounts the directory read-only at
  `/opt/sparkring-overlay`, first on `PYTHONPATH`, so its `vllm` and `b12x`
  replace the image's. It holds `vllm/__init__.py`, `b12x/__init__.py` and
  `OVERLAY.json`; `preflight` requires one tree on every Spark (the first of
  `tree_sha256`, `tree_hash`, `tree_digest`, `tree`, `sha256`, `digest` in
  `OVERLAY.json`, else its SHA-256) and reports top-level `*.dist-info` and
  `*.egg-info` entries, which change the versions the image's entrypoint reads.
- Version-pinned shims install only where the vLLM files they wrap match a
  pinned build (`$SERVE shims --vllm-tree PATH` shows each one's status).
  `stage` refuses an overlay that cannot carry the group's prefill row
  ownership; `--mhc-prefill-shard off` (GLM-5.3-Flash) or `--env
  VLLM_QWEN3_8_HC_PREFILL_MODE=off` (Qwen3.8) serves without it.
- `--vllm-arg FLAG=VALUE` (or `FLAG` for a switch) replaces or adds a vLLM
  argument on every rank, `--drop-vllm-arg FLAG` removes one, and
  `--speculative-set KEY=VALUE` sets one field of `--speculative-config`.
  Ports, addresses, ranks, parallel sizes, the executor backend, the served
  model name, `--model` and `--default-chat-template-kwargs` are refused.
- `--checkpoint-id REPOSITORY@REVISION` names the checkpoint the model
  directories hold; `preflight` then compares the copies across Sparks.
  `--thinking-behaviour NAME` names its behaviour in `profiles/thinking.json`.
- `--b12x-cache-dir /cache/<directory>` keeps the overlay's B12X kernels apart
  from those the image's B12X compiled.

```bash
CSF="--profile glm53-flash-nvfp4-spark-tp4 --positions 0-3
  --model-path /srv/sparkring/<cluster>/checkpoints/<owner>--<name>/<revision>
  --overlay /srv/sparkring/overlays/<overlay> --b12x-cache-dir /cache/b12x-<overlay>
  --vllm-arg --quantization=nvfp4_csf --vllm-arg --load-format=nvfp4_csf
  --speculative-set moe_backend=marlin
  --checkpoint-id local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd33efa73c3bb7c95b74eee10cad34f9be
  --thinking-behaviour glm53-flash-template --reasoning-effort high"
$SERVE plan --site $SITE --repository $REPO $CSF
```

## Session scope of the schedule and link options

The schedule, minimum and link options set the tensor-parallel session only;
SIRCL's adapter builds decode-context-parallel sessions with those variables
removed (`settings.TP_SESSION_VARIABLES`). `plan` and `bundle --text` print
each session's schedules, chain order, ring availability, link sizes and
minimums, and refuse what a session would refuse at setup on every rank: a
`chain` or `ring` schedule on a session whose ranks do not occupy its whole
fabric, a `chain` whose ranks form no chain of cable neighbours, a `ring`
that cannot run (a relay queue would carry two ring lanes), and link geometry
the session rejects.

## Measured tuning tables

A tuning table (schema `sircl-tuning-table/v1`, made by the ring harness's
`tune` command) holds, per collective, size and mode (eager or CUDA graph
replay), the fastest SIRCL choice and whether NCCL measured faster. Its key
names the group shape (`pair`, `path:<n>`, `cycle:<n>` or a strided shape),
ranks, lanes, relays, source hashes and SIRCL version, so a table from
another shape or build does not apply; a decode-context-parallel group of
four consecutive Sparks has the key of `path:4`. With `--tuning-table PATH`
(repeatable), `plan` and `bundle` print which table each session takes (a
session no table matches uses its rules) and refuse a malformed table, a
table no session takes and two tables matching one session. `start` and
`bundle --stage` write the tables to `<run directory>/tuning/` on every Spark
before any container starts.

- A table also records the session settings its choices ran under and need
  (link slots, link slot, chain slot, large-message piece). Every session
  that takes it applies the ones the launcher leaves unset, and the plan
  lists them under the table. A launcher option below a setting of the table
  the tensor-parallel session takes (`--link-slots`, `--link-slot`) is
  refused; an equal or larger value is kept. The plan's session checks count
  the table's link slots.
- A table chooses only among SIRCL's algorithms, schedules, pieces, grids and
  the settings they need. Its NCCL marks are measurements and route no call,
  under every `--nccl` value. The plan lists them in the table's decisions
  (`(NCCL faster)`) and says so per session: `the table chooses among SIRCL
  options only: its NCCL marks are measurements and route no call; the rules
  decide what NCCL carries here` where the group's policy lets NCCL run, and
  `SIRCL carries every size: NCCL may not run on this group` under
  `--nccl never` or on a path.
- Receipts name the table (`tuning=<hash>`); `check` and `bundle-check` fail
  when it differs from the plan's match. The receipts also show the table's
  settings beside the session's own values.

## Serving without NCCL

`serve`, `bundle` and `bundle-check` share one rule (`--nccl`),
`NCCL: opt-in only (auto); tables choose among SIRCL options`. The plan text
(and `bundle --text`) states it with the launch's mode; the plan and bundle
JSON hold it as `nccl_rule` beside `nccl_mode`, and every receipt holds both.
NCCL runs only when the operator opts in:

| Mode | Pair | Path | Whole ring (bundle) |
|---|---|---|---|
| `never` (default) | none | none | none |
| `auto` (`topology` is the same) | every collective (`all`) | none | NCCL's ring algorithm (`ring`), with `NCCL_ALGO=Ring` and `NCCL_SKIP_TREE_CONNECT=1` |

Under `never`, SIRCL carries every collective of tensor-, expert- and
decode-context-parallel serving. For tensor parallelism over every rank:

| Group | Carrier |
|---|---|
| `world` | no device communicator; its gloo group carries control messages; SIRCL's tripwire refuses NCCL calls on its device group and the default group |
| `tp` | SIRCL's communicator and the group's session |
| `ep` (mixture-of-experts models) | shares `tp`'s session, which also carries its direct `torch.distributed` calls (online quantization's `amax` all-reduces) |
| `dcp` | one rank per group at decode-context parallelism 1; otherwise a session of its own (`--session-groups tp,dcp`), refused at startup without one |
| `pp`, `dp`, `pcp` | one rank per group: no collective |

On a group with a session, the direct calls `all_reduce` (sum, max, min),
`broadcast`, `all_gather`, `all_gather_into_tensor`, `reduce_scatter_tensor`
(sum) and `all_to_all_single` (equal splits) run on the session; other direct
calls are refused, and PyNccl is not built. Under `auto` on the whole ring,
NCCL's ring carries `ep`, while the decode-context-parallel groups, paths of
four, keep their own sessions.

`--require-no-nccl` asks for serving without NCCL and proves it:

```bash
$SERVE start $ARGS --require-no-nccl --wait
$SERVE check $ARGS --require-no-nccl
```

- `plan`, `start` and `bundle` refuse a group NCCL may run on and the three
  settings that create NCCL communicators outside SIRCL's groups
  (`--load-format instanttensor`, `--enable-eplb`,
  `VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1`), and set `--nccl-debug`.
- `check` and `bundle-check` fail unless every group of every rank has a
  receipt with `nccl=none`, `pynccl=skipped` and no decision row on NCCL, and
  no container log holds an NCCL communicator line (`Init START`,
  `Init COMPLETE`, `ncclCommInitRank`, `ncclCommSplit`, `NCCL version`, or
  vLLM's `vLLM is using nccl`).
- Other NCCL lines (`NCCL INFO`, `NCCL WARN`), which NCCL prints when it reads
  its settings, are reported as library activity without a communicator.

The paths the tripwire cannot see (the GEMM + reduce-scatter fusion pass,
EPLB, the InstantTensor loader) are reached only through refused settings;
the log scan covers any communicator they would create ([`SURVEY.md`](SURVEY.md)).

## mHC prefill sharding

GLM-5.3-Flash mixes several residual streams in every decoder layer
(hyper-connections, "mHC"). With `VLLM_GLM53_MHC_PREFILL_SHARD=1`, the
profiles' setting, vLLM splits that mixing among the tensor-parallel ranks in
every eager forward of a full prefill chunk (8,192 rows) and exchanges rows
with reduce-scatters and all-gathers sent straight to PyNccl: at TP4, 90
reduce-scatters of `[8192, 4096]` BF16 and 90 all-gathers of `[2048, 4096]`
shards per chunk.

Where NCCL may not run (every group under the default `--nccl never`), the
`mhc_prefill_shard` shim ([`mhc.py`](mhc.py)) lends vLLM's row-ownership
code a stand-in for PyNccl: the session's reduce-scatter on the whole message
(else an all-reduce and this rank's rows) and a byte-copying all-gather
(receipts: `mhc=sircl`, `reduce_scatter/sircl/scatter`). On a pair under
`--nccl auto`, vLLM's own PyNccl path runs (`mhc=pynccl`). The shim is pinned
to both vLLM files; on another build the group's setup fails, and
`--mhc-prefill-shard off` serves with full rows.

## Qwen3.8 hyper-connection prefill row ownership

Qwen3.8-Flash-Next keeps several hyper-connection streams per token. With
`VLLM_QWEN3_8_HC_PREFILL_MODE=shard`, the setting of SparkRing's Qwen3.8
profiles, vLLM gives each tensor-parallel rank `rows / W` rows of the
multi-stream state in every eager pure prefill of at least 1,024 rows (a
multiple of `W`, at TP2 or TP4) and exchanges block boundaries with
reduce-scatters and all-gathers sent straight to PyNccl.

Where NCCL may not run (pairs and four-Spark groups under the default
`--nccl never`), the `qwen_hc_prefill_shard` shim ([`qwen_hc.py`](qwen_hc.py))
carries them as for mHC (receipts: `mhc=sircl`); on a pair under `--nccl
auto`, vLLM's own PyNccl path runs. The shim is pinned to both vLLM files of
the image's build; on another build the group's setup fails and names `--env
VLLM_QWEN3_8_HC_PREFILL_MODE=off`, which keeps full rows.

## Decode-context parallelism in profile serving

`--dcp-size N` (every serve command; default 1) serves a profile with vLLM's
decode-context parallelism N, where N divides the profile's tensor
parallelism. The launcher sets `--decode-context-parallel-size N` on every
rank (`--vllm-arg` may not set it) and `SIRCL_GROUPS=tp,dcp`, so SIRCL's
communicator builds a session for every decode-context-parallel group of N
consecutive ranks, with the session defaults and the tuning table that
matches its shape, as `bundle --session-groups tp,dcp --dcp-size N` does.

- `plan` lists the DCP sessions and groups and their NCCL policy. It refuses
  an N that does not divide the tensor parallelism, a checkpoint outside
  `plan.DCP_MODELS` (below) and an attention backend other than B12X
  (`--vllm-arg --attention-backend=B12X` names it where the recipe does not).
- `stage` requires the image's vLLM to match the pinned builds of the
  `dcp_all_to_all` and `dcp_b12x_transport` shims; `check` requires every
  rank's decode-context-parallel receipt with a session. With NCCL off, the
  default, the DCP collectives run on the DCP sessions, and
  `--require-no-nccl` proves that no NCCL communicator exists.

The checkpoints of `plan.DCP_MODELS` use multi-head latent attention with
DeepSeek-V3.2's sparse indexer; every other checkpoint is refused with
`--dcp-size` above 1:

| Model | Checkpoints | Conditions beyond a divisor of the tensor parallelism, a session per DCP group and B12X attention |
|---|---|---|
| GLM-5.3-Flash | `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`, `local-inference-lab/GLM-5.3-Flash-NVFP4`, `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD`, `nvidia/GLM-5.3-Flash-NVFP4` | the KV-cache interleave and mHC sizes below |
| GLM-5.3 | `local-inference-lab/GLM-5.3-NVFP4` | none; the recipe's KV-cache interleave stays as it is |

GLM-5.3 at TP8 with DCP 4 plans on Sparks 0-7 with the decode-context-parallel
sessions of `bundle --positions 0-7 --session-groups tp,dcp --dcp-size 4`:
ranks 0-3 and 4-7, two paths of four that NCCL may not use (CPU tests on a
synthetic profile; not run on the ring through this launcher). The
repository's catalog lists no GLM-5.3 profile; its research profile
`glm53-nvfp4-tp8` (`profiles/research-catalog.json`) sets decode-context
parallelism 4 in its recipe, which this launcher's profile reader refuses
(the launcher sets it with `--dcp-size`). In image `aba309e4610c`'s vLLM,
GLM-5.3's B12X attention needs a source change, which a deployment's own
vLLM plugins must supply ([`SURVEY.md`](SURVEY.md), section 5); the plan
cannot check it, and a vLLM without it stops at startup.

GLM-5.3-Flash adds two startup conditions under decode-context parallelism,
which the launcher checks:

- The KV cache is interleaved over the DCP ranks in blocks of
  `--cp-kv-cache-interleave-size` tokens, a multiple of 4. The launcher sets 4
  where the recipe leaves the flag unset (`plan.DCP_INTERLEAVE`) and refuses
  a value that is not a multiple of 4.
- mHC prefill row ownership (`VLLM_GLM53_MHC_PREFILL_SHARD=1`, the profiles'
  value) starts only at the tensor and decode-context parallelism a pinned
  vLLM build admits (`pins.MHC_ADMITS`; a build not listed there admits
  TP2/DCP1, TP4/DCP1, TP4/DCP2 and TP4/DCP4). With mHC sharding on and DCP
  above 1, `plan` refuses sizes no pinned build admits, and `stage` refuses a
  Spark whose vLLM's mHC files match a build that does not admit them.
  `--mhc-prefill-shard off` serves without row ownership.

GLM-5.3-Flash at TP2 with DCP 2 on the cabled pair of Sparks 0-1, NCCL off
and mHC prefill sharding off (`CKPT` and `GLM` as in
[Installer profiles on the ring of eight](#installer-profiles-on-the-ring-of-eight)).
Each Spark holds two sessions, the tensor-parallel one and the DCP one:

```bash
DCP="--site $SITE --repository $REPO --profile glm53-flash-nvfp4-spark-tp2 --positions 0-1 --model-path $CKPT/$GLM --dcp-size 2 --mhc-prefill-shard off --require-no-nccl"
$SERVE plan $DCP && $SERVE preflight $DCP && $SERVE stage $DCP
$SERVE start $DCP --wait --timeout 3600
$SERVE check $DCP --long-prompt 16384
$SERVE stop $DCP
```

Decode-context parallelism in profile serving is implemented and CPU-tested;
it has not served on a ring.

## Adding SIRCL to another launcher

A launcher that keeps its own image, model, plugins and vLLM arguments takes
SIRCL's part from `bundle` ([`serve/bundle.py`](serve/bundle.py)). Global
rank `r` runs on the Spark at the bundle's position `r`.

```bash
$SERVE bundle --site $SITE --positions 0-7 --run-id tp8 > bundle.json                    # OFFLINE
$SERVE bundle --site $SITE --positions 0-7 --run-id tp8 --image $IMAGE --stage > bundle.json  # MUTATES HOST
# the launcher starts global rank r on Spark r with that rank's docker_args
$SERVE bundle-check --site $SITE --positions 0-7 --run-id tp8 --container 'NAME-{rank}'   # READ-ONLY REMOTE
```

`bundle.json` (`sircl-vllm-bundle/v1`) lists per rank the three mounts
(staged tree, build cache, run directory), the environment, and `docker_args`,
the `--mount` and `--env` arguments ready to add. The launcher merges two
values with its own: `pythonpath_prepend` goes before its `PYTHONPATH` (with
`:`), and `vllm_plugins_add` (`sircl`) joins its `VLLM_PLUGINS` when it sets
one. `requirements` list what the container needs: host network, the GPU and
RDMA devices, unlimited locked memory, no other transport in vLLM's RoCE slot,
no `SIRCL_*` variable from the launcher's own environment, and, where NCCL may
not run, the three NCCL-free settings; SIRCL adds no seccomp policy or
capability. `--stage` also stages the tree, run directories and tuning
tables and builds the native libraries in `--image` (default: the site's) on
every Spark, with progress on standard error.

`bundle` takes the size, schedule, link, tuning, NCCL, overlay, vLLM argument,
`--b12x-cache-dir` and `--env` options of `start`, and:

- `--session-groups tp,dcp` gives decode-context-parallel groups a session;
  `--dcp-size N` (vLLM's decode-context parallelism) plans those sessions and
  their tuning tables and is refused above 1 without `tp,dcp`. What such a
  group requires of vLLM's settings is in [`README.md`](README.md).
- `--fused-norm on` (research-only) runs vLLM's post-all-reduce RMSNorm helper
  as one fused SIRCL collective; setup refuses it unless vLLM's RMSNorm runs
  the `vllm_c` provider of `fused_add_rms_norm`.
- `--column-gather on|off` sets `SIRCL_COLUMN_GATHER` on every rank; without
  the option the variable stays unset and the adapter's default (on)
  applies ([Column gathers](README.md#column-gathers)).
- `--reasoning-effort LEVEL` with `--repository` and `--checkpoint
  REPOSITORY@REVISION` (or `--thinking-behaviour NAME`) gives global rank 0
  the `--default-chat-template-kwargs` value to add.
- `--profile ID` with `--repository` takes the SIRCL switches that SparkRing
  profile pins in its serving configuration's environment
  (`plan.PROFILE_VARIABLES`: `SIRCL_FUSED_NORM` and `SIRCL_COLUMN_GATHER`) as
  the defaults of `--fused-norm` and `--column-gather`, and refuses an option
  given with another value. SparkRing's installer and its serving A/B runner
  read the same switches from the profile, and `start` and `plan` carry them
  in the profile's environment. A profile that sets any other `SIRCL_*`
  variable is refused: session sizes come from the options and tuning
  tables. The bundle's `profile` field records the profile and its switches.
- `--text` prints the group map, sessions and carriers instead of the JSON.

`bundle-check` takes the bundle's `--nccl`, `--session-groups`, `--dcp-size`,
`--tuning-table`, `--overlay`, `--large-blocks` and `--require-no-nccl` and
applies the checks of `check` to the named containers, assuming tensor
parallelism over every rank. A pipeline-parallel launch (research-only) goes
through `bundle`; its receipts show `group=pp:<i>` with `send`/`recv` rows on
`sircl`.

## Installer profiles on the ring of eight

The repository's catalog (`profiles/catalog.json`) lists 35 profiles. `plan`
on Sparks 0-1 for tensor parallelism 2 and on Sparks 0-3 for 4 gives the
results below;
`test_every_catalog_profile_plans_on_the_ring_or_names_what_blocks_it`
([`tests/test_vllm_serve.py`](../../tests/test_vllm_serve.py)) repeats it
against the repository holding the package, or the checkout
`SPARKRING_REPOSITORY` names. Test one profile at a time.

| Placement | Under `--nccl never` (default) | Under `--nccl auto` |
|---|---|---|
| Sparks 0-1, a pair (TP2) | `none`: SIRCL carries every collective, prefill row ownership through the shims | `all`: SIRCL carries collectives up to the dispatch ceiling and every captured one; NCCL carries larger eager all-reduces, vLLM's PyNccl row ownership and direct `torch.distributed` calls |
| Sparks 0-3, a path (TP4) | `none`: ranks 0 and 3 share no cable | `none` |

Profiles that plan, all on installer image `sha256:aba309e4610c`, whose vLLM
is the adapter's pinned build `lil-image-aba309e4610c`:

| Profile | TP, Sparks | Checkpoint per Spark | Prefill row ownership under the default |
|---|---|---|---|
| `glm53-flash-nvfp4-spark-tp4` | 4, 0-3 | 174.8 GiB | mHC, `mhc_prefill_shard` shim |
| `glm53-flash-nvfp4-spark-tp2` | 2, 0-1 | 174.8 GiB | mHC, `mhc_prefill_shard` shim (vLLM's PyNccl under `--nccl auto`) |
| `qwen38-flash-next-qad-tp4` | 4, 0-3 | 102.6 GiB | hyper-connection `shard`, `qwen_hc_prefill_shard` shim |
| `swift15-qwen38-flash-next-tp4` | 4, 0-3 | 173.7 GiB | hyper-connection `shard`, `qwen_hc_prefill_shard` shim |
| `qwen38-flash-next-tp2` | 2, 0-1 | 102.6 GiB | hyper-connection `shard`, `qwen_hc_prefill_shard` shim (vLLM's PyNccl under `--nccl auto`) |
| `swift15-qwen38-flash-next-tp2` | 2, 0-1 | 173.7 GiB | hyper-connection `shard`, `qwen_hc_prefill_shard` shim (vLLM's PyNccl under `--nccl auto`) |
| `mimo-v26-flash-mopd-tp4` | 4, 0-3 | 165.6 GiB | none |
| `mimo-v26-flash-mopd-tp2` | 2, 0-1 | 165.6 GiB | none |
| `deepseek-v41-flash-tp4` | 4, 0-3 | 475.3 GiB | none; adaptive verification broadcasts a confidence snapshot every step, which the tripwire's carrier runs on the session ([`SURVEY.md`](SURVEY.md)) |

Profiles that do not plan, with the blocker each names:

| Blocker | Profiles |
|---|---|
| `recipe` profile that "runs from its own launcher" (named in the refusal); SIRCL's part comes from `bundle` | `deepseek-v4-flash-0731`, `deepseek-v4-flash-0731-pair`, `deepseek-v4-flash-vision-exp-tp4`, `deepseek-v41-flash-cycle`, `deepseek-v41-flash-sglang-cycle` (serves with SGLang, which the vLLM adapter does not reach), `glm52-exl3-r7-3.5bpw`, `glm53-flash-nvfp4-dflash2-bf16-tp4`, `glm53-mtp3-cache-checkpoints-tp4`, `glm53-spark-mtp3-managed-mesh-tp4`, `qwen38-27b-exl3-k5k6`, `qwen38-27b-exl3-k5k6-pair`, `sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1`, `sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1`, `sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4`, `sparkcache-glm53-flash-nvfp4-dflash2-bf16-sparkcache-tp4` |
| `release-profile` profile that "runs from its own launcher" | `glm53-flash-spark-tp2-dcp1`, `glm53-flash-spark-tp2-dcp1-nocache`, `glm53-flash-spark-tp2-dcp1-sparkcache`, `glm53-flash-spark-tp4-dcp1`, `glm53-flash-spark-tp4-dcp1-nocache`, `glm53-flash-spark-tp4-dcp1-sparkcache`, `glm53-flash-spark-tp4-dcp4`, `glm53-flash-spark-tp4-dcp4-sparkcache` |
| the release "has no installer image lock" (`runtime/releases/shared-2026.09.3/installer-image.json`) | `qwen38-flash-next-tp2-sparkcache`, `qwen38-flash-next-qad-tp4-sparkcache` |
| qualified on a "switched fabric" | `glm53-flash-spark-tp4-switched` |

Start commands (`CKPT` is the installer's checkpoint root,
`/srv/sparkring/<cluster>/checkpoints`):

```bash
GLM=local-inference-lab--GLM-5.3-Flash-NVFP4-Spark/a608241037e4c2565356bff7ca293f2133888f88
QWEN=local-inference-lab--Qwen3.8-Flash-Next-NVFP4/60215d26cf5e42c2db6128774032d57fc62678da
SWIFT=ukisai--Swift-1.5-Qwen3.8-Flash-Next-NVFP4/3ff0520224f264a2d0ac4ab56ece8f2f13aadb38
MIMO=XiaomiMiMo--MiMo-V2.6-Flash-MOPD/2479e2d0029eca9a34cc7e7f55a121925f81908e
DSV41=deepseek-ai--DeepSeek-V4.1-Flash/dba1be0a40aa45a94ad051997016db3960a90277
# for example; *-tp4 profiles take --positions 0-3, *-tp2 profiles --positions 0-1
ARGS="--site $SITE --repository $REPO --profile qwen38-flash-next-qad-tp4 --positions 0-3 --model-path $CKPT/$QWEN"
$SERVE plan $ARGS && $SERVE preflight $ARGS && $SERVE stage $ARGS
$SERVE start $ARGS --wait --timeout 3600 && $SERVE check $ARGS --long-prompt 16384
$SERVE stop $ARGS
```

## Measured results

Conditions: eight Sparks cabled as one ring, installer image `aba309e4610c`,
the groups' relay plans installed, one run per row. "Ready" is the time from
`start` to the API serving the model; an empty kernel cache compiles every
B12X kernel first. Throughput: [package status](../../STATUS.md#measured-performance).

| Profile, Sparks | NCCL | Ready | Result |
|---|---|---|---|
| `glm53-flash-nvfp4-spark-tp4`, 0-3; also 0-3 and 4-7 at once | none (paths) | 174 s on 0-3, installer caches copied | three prompts passed; every rank's `tp:0` receipt `nccl=none pynccl=skipped`, no `nccl` decision row |
| `qwen38-flash-next-qad-tp4`, 0-3 | none | 528 s, empty cache | functional checks passed; hyper-connection rows on SIRCL |
| `deepseek-v41-flash-tp4`, 0-3 | none | 373 s, cache from the installer | functional checks passed |
| `mimo-v26-flash-mopd-tp4`, 7-0-1-2 | none | 383 s, empty cache | functional checks passed |
| `swift15-qwen38-flash-next-tp4`, 3-6 | none | 499 s, empty cache | functional checks passed; hyper-connection rows on SIRCL |
| `glm53-flash-nvfp4-spark-tp2`, 0-1 | `--nccl auto` | 497 s, empty cache | functional checks passed |
| `glm53-flash-nvfp4-spark-tp2` serving `GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` with the overlay and argument edits of section "Another checkpoint or vLLM build", 4-5 | `--nccl never --require-no-nccl` | 544 s | check passed: every group `nccl=none pynccl=skipped`, no NCCL communicator line in either log; mHC rows on the session (90 reduce-scatters per chunk) |
| GLM-5.3 at TP8 with decode-context parallelism 4, started by a launcher outside this package with the bundle's additions: `--session-groups tp,dcp`, `--nccl never --large-allreduce sircl --require-no-nccl`, 1 MiB capacity and dispatch ceiling, `--fused-norm on` | none | 211 s | three smoke prompts passed; `bundle-check --require-no-nccl` passed with no NCCL line in any of the eight logs |

## Limitations

The launcher serves tensor parallelism, with decode-context parallelism
through `--dcp-size` for the GLM-5.3-Flash checkpoints and GLM-5.3, one rank and one
instance per Spark; it never downloads weights, configures networking or restores relay
plans after a Spark reboots. The adapter's limitations are in
[`STATUS.md`](STATUS.md).
