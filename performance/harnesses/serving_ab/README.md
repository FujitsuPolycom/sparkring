# Serving A/B runner

The serving A/B runner serves one installer profile on chosen DGX Sparks of a
cabled ring under several collective transports, on the same image, model and
vLLM settings, and compares them. It starts every arm fresh in a chosen order,
checks before measuring what carried the collectives, measures decode, time to
first token and output agreement from the operator's machine, and writes JSON
results and tables. Its [run receipts](#run-receipts) record what every arm
resolved and how it was measured, and refuse comparisons that differ in more
than the declared change. Status: **implemented** (CPU tests in
[`test_serving_ab.py`](test_serving_ab.py) and [`test_receipt.py`](test_receipt.py));
the containers it renders follow the installer's own container for each
profile, but runs through it are research measurements, not serving
qualification.

```bash
python -m performance.harnesses.serving_ab plan --site SITE --profile PROFILE --positions 2-3 \
  --image-lock LOCK --order W:S+,W:N,S+,N,N,S+ --metrics phase1 --run-id RUN --output DIR
python -m performance.harnesses.serving_ab run  ...same options...
python -m performance.harnesses.serving_ab.report DIR S+1 N1 N2 S+2
```

Run it from the repository root with Python 3.11 or later on Linux (the decode
benchmark needs a POSIX terminal module). `SITE` is the ring harness's site
file (`sircl-ring-site/v1`, [RUNBOOK.md, Site file](../../../spark_transport/sircl/RUNBOOK.md#site-file)):
it names every Spark's SSH target, LAN address and Docker command, so no host
detail is part of this package. `SERVING_AB_SSH` names another OpenSSH client
(for example the Windows one, `/mnt/c/Windows/System32/OpenSSH/ssh.exe`, from
WSL when the keys live on the Windows side).

## Safety classes

| Command | Class |
|---|---|
| `plan` | READ-ONLY REMOTE: `sudo -n` reads on the Sparks to find the checkpoint (none with `--model-path` for every position) |
| `run` | MUTATES HOST: the loader seccomp policy under `<remote_dir>/serving-ab/source`, the SIRCL arms' staged tree, native library, tuning tables and run directories under the site's `remote_dir`; one cache directory per arm; containers labelled `serving-ab=<run>`; before every start, `sync` and `drop_caches` on every chosen Spark (frees the page, dentry and inode caches; changes no setting). Loads the GPUs and the fabric of the chosen Sparks |
| `report` | OFFLINE |
| `receipt collect`, `receipt preflight`, `receipt window` | READ-ONLY REMOTE: through Node A, `sudo -n sparkring status --json`, the lock owner files, and on the deployment's Sparks `docker inspect`, `docker exec ... env` (serving variables only), `docker logs`, the SIRCL receipt files and `nvidia-smi`; `GET /metrics` and `GET /v1/models` on the API. No generation request, install, restart or lock |
| `receipt diff`, `receipt check`, `receipt method` | OFFLINE |

## Arms

Every rank's base is the installer's container for the profile: the profile
adapter's specification (`runtime/common/toolchain_profiles.py`), adapted to the
image lock (`--image-lock`) as the installer runs it on a host
(`runtime/common/compose.py`, `installer_container`: the toolchain entrypoint,
CUDA 13.4 and NCCL 2.32.3 library paths, the runtime-status plugin,
image-scoped caches, the B12X weight loader's io_uring seccomp policy and health
timing; without the per-rank runtime binding, which only the installer writes).
Research-only profiles without Compose exports render the same way.
Every arm then applies the same site substitutions: its own container name and
labels, the Spark's checkpoint copy, the arm's own cache directory (compile and
JIT caches; `S` and `S+` share one), the rank's LAN address, and `NCCL_DEBUG=INFO`,
`NCCL_DEBUG_SUBSYS=INIT` (logging, so every NCCL communicator shows in the
logs). Each arm differs from the others only in its transport part
([`spec.py`](spec.py)):

| Arm | Transport | Part added to the base |
|---|---|---|
| `S` | SIRCL ring sessions on every collective, NCCL off | what `python -m sparkring_sircl.vllm.serve bundle --nccl never --require-no-nccl` gives the rank: three mounts, the SIRCL environment, `PYTHONPATH` and `sircl` in `VLLM_PLUGINS`; `--capacity`, `--dispatch` and `--tuning-table` pass to the bundle, which also takes the SIRCL switches the profile pins in its environment (`bundle --repository` this checkout `--profile PROFILE`: fused norm and column gathers, as the installer applies them; a profile that pins `SIRCL_FUSED_NORM=1`, such as `glm53-nvfp4-tp8` or `deepseek-v41-flash-tp4`, therefore runs the fused path in `S` as in `S+`, because the bundle refuses `--fused-norm off` against the pin, and the plan prints that as an arm note); `--sircl-env KEY=VALUE` adds a variable at the container level; `--roce-slot` keeps `VLLM_ENABLE_ROCE_ALLREDUCE=1`, so SIRCL's `roce_slot` shim takes vLLM's RoCE slot |
| `S+` | `S` with SIRCL's performance paths that are off by default | `bundle --fused-norm on` (`SIRCL_FUSED_NORM=1`: all-reduce, residual add and RMSNorm in one kernel, research-only, bit-identical by design; it applies only to models on vLLM's DeepSeek-V3.2 code) |
| `N` | vLLM's own communicator and PyNccl over the image's NCCL | `VLLM_ENABLE_ROCE_ALLREDUCE=0`, no RoCEnante bundle (`SPARKRING_TRANSPORT_PROFILE` empty), `SPARK_TP4_ENABLED=0`, and the NCCL variables `bundle --nccl auto` gives the rank: the RDMA devices facing its peers (`NCCL_IB_HCA`) and, on a whole cycle, `NCCL_ALGO=Ring` with `NCCL_SKIP_TREE_CONNECT=1`; `sircl` removed from `VLLM_PLUGINS`, so a profile that lists SIRCL's plugin does not start its communicator. The profile's other NCCL settings stay. The auto bundle takes the plan's decode-context parallelism. Refused where NCCL may not run: a group whose consecutive ranks do not all share a cable, such as four Sparks of an eight-Spark ring, and a decode-context-parallel group of that kind, such as each group of four consecutive Sparks at DCP 4 on all eight |
| `P` | the profile's prepared transport | the base unchanged. Refused unless the placement is one the prepared transport runs: a whole pair site, a whole four-Spark ring or one of its halves |

`--dcp-size N` serves decode-context parallelism N in every arm, by the serve
launcher's own rules (`sparkring_sircl/vllm/serve/plan.py`): the size must
divide the tensor parallelism and the checkpoint and attention backend must run
it (`dcp_problem`); the SIRCL arms' bundles take `--session-groups tp,dcp`; the
model's `--cp-kv-cache-interleave-size` is added only where the recipe leaves it
unset (`dcp_interleave`, 4 for GLM-5.3-Flash); and where no pinned vLLM build
admits mHC prefill row ownership at the sizes (`mhc_dcp_problem`),
`VLLM_GLM53_MHC_PREFILL_SHARD=0`. `plan` prints each such change as a deviation
from the profile. Without the option every arm runs the profile's own size.

`--checkpoint NAME` serves another checkpoint of the profile's table with its
pinned settings, as the installer's `--checkpoint` does; `--model-of PROFILE`
serves another profile's pinned model with this profile's settings. Both are
printed as deviations from the profile, and the checkpoint search uses the
model chosen.

`--set-arg FLAG=VALUE` sets a vLLM argument and `--set-env KEY=VALUE` a
container variable in every arm's command; `VALUE` given as `@PATH` is read from
that file (for JSON such as `--speculative-config` or `--hf-overrides`), and
`--set-arg FLAG` alone adds a bare flag such as `--async-scheduling`. Each
is applied to every rank's base before the arms' parts and printed as a
deviation from the profile with the profile's own value; `plan.json` keeps the
full values under `overrides`.

`--mount HOST_PATH=CONTAINER_PATH` (research only) binds a host directory
read-only into every rank's container, for a second checkpoint such as a
separate speculative drafter that the profile's container does not mount. Both
paths are absolute; `/models/target`, `/cache` and `/sircl` and paths below them
are refused, because the runner and the SIRCL bundle mount them. `plan` checks
that the host path exists on every chosen Spark, prints each mount as a
deviation and keeps it under `overrides.mounts`.

`plan` prints every rank's command of every arm and, for a SIRCL arm against
`N`, the difference of rank 0's commands: the environment variables, mounts and
any other token that differs. It also prints a `WARNING:` line, and `plan.json`
keeps it under `warnings`, for each check of the enhancement catalog
(`checks` in `performance/enhancements.json`) that an arm's rank-0 command
meets, such as MXFP8 or NVFP4 dense linears without `--linear-backend`.

## Memory before every start

vLLM checks at startup that the free memory covers `--gpu-memory-utilization`
of the total, and on GB10's unified memory the page cache of the previous
start's checkpoint reads and its teardown count against that free memory. Before
every start the runner therefore requires that no container of the run is left
on any chosen Spark, drops the caches there (`sync; echo 3 >
/proc/sys/vm/drop_caches` under `sudo -n`), and polls `MemAvailable` until two
readings 5 s apart differ by less than 256 MiB on every Spark, for at most
180 s. The campaign log and the start's `memory-before.json` record each
Spark's settled available memory beside the share vLLM asks for, and a Spark
below that share refuses the start, since vLLM would refuse it at startup.

## Checks before measuring

[`verify.py`](verify.py) reads every rank's container log and, for the SIRCL
arms, its receipts, and stops the run before measuring when an arm does not run
what it should:

- `S`, `S+`: every rank's tensor-parallel receipt reads `nccl=none` and
  `pynccl=skipped`, no log shows an NCCL communicator, the sircl plugin was
  loaded; the installed shims are listed. With decode-context parallelism
  every rank also needs a `dcp:` receipt with a SIRCL session and the
  `dcp_all_to_all` shim. `S+` needs `fused_norm` on, and the
  receipts taken after the measurement count the fused calls (decision rows
  `all_reduce` / `sircl` / `fused_rms_norm`).
- `N`: every rank shows `Init COMPLETE` for an NCCL communicator and vLLM's
  `PYNCCL` all-reduce backend; no rank loaded the sircl plugin.

Each start's `evidence.json` holds these results with every rank's shims,
all-reduce backends and receipt lines.

## Metrics

[`measure.py`](measure.py) runs from the operator's machine against rank 0's API:

| Set | Decode | TTFT | Outputs |
|---|---|---|---|
| `warmup` | contexts 0 and 32k, 1 and 8 streams, 10 s | 8k and 32k, once | fingerprints, prompt logprobs |
| `phase1` | contexts 0 and 32k, 1, 2, 4 and 8 streams, 30 s cells after 10 s | 8k and 32k, three samples | as above |
| `phase2` | as `phase1` | 2k, 8k, 32k and 128k, three samples | as above |
| `phase1-16k` | as `phase1`, at contexts 0, 16k and 32k | 8k, 16k and 32k, three samples | as above |
| `c16` | contexts 0 and 16k, 1, 2, 4, 8 and 16 streams, 30 s cells | 8k, 16k and 32k, three samples | as above |
| `none` | none: the start becomes ready, passes its checks and is profiled or stopped | none | none |

The decode step follows a named measurement method (`--method`,
[`receipt/method.py`](receipt/method.py)):

| Method | Decode |
|---|---|
| `serving-ab-temp0` (default) | the set's contexts and streams, temperature 0, at most 1,024 output tokens, prefill skipped |
| `owner-temp1-matrix` | contexts 8k, 16k, 32k, 64k and 128k, 1, 2, 4 and 8 streams, 20 s cells after a 5 s warm-up, temperature 1, at most 2,048 output tokens, the integrated prefill scout and the coding-peak probe at temperature 1, whatever the set; needs llm-inference-bench 0.7.6 in `--bench-dir` |

`--temperature` and `--max-tokens` replace the method's values. Each start's
`method.json` and `measure.json` name the method, or `custom` when an override
leaves no named method. Warm-up starts keep the default method.

Decode runs llm-inference-bench's `llm_decode_bench.py` (`--bench-dir`) through
[`bench_run.py`](bench_run.py), which turns off its self-update check. TTFT runs
[`prefill_probe.py`](../validation/prefill_probe.py) (unique prefixes; every
sample reports its cached tokens). Fingerprints ([`fingerprint.py`](fingerprint.py))
are four greedy 96-token answers compared token by token; prompt logprobs run
[`logprob_probe.py`](../../records/qwen38-flash-next/decode-ab-20260925/logprob_probe.py).
Restarts of one arm need not give identical tokens, so output agreement is
reported against the same-arm restarts, not as a pass or fail.

## Profiling

`--profile-label LABEL` captures vLLM's torch profiler after that start's
measurement: one untimed 32-token request, `/start_profile`, one one-stream completion of
`--profile-tokens` tokens (default 60), `/stop_profile`. The containers need
the profiler enabled with its folder under the cache mount, for example
`--set-arg=--profiler-config={"profiler":"torch","torch_profiler_dir":"/cache/torch-profile"}`
(or `VLLM_TORCH_PROFILER_DIR` in a vLLM that reads it), so the traces land in the
arm's cache directory; rank 0's Spark's traces are copied into the start's
`torch-profile-r0.tar`, with `profile.json` (the request's usage, the wall
time and the trace folder's listing).

## Outputs

`DIR/plan.json`; per start `DIR/<label>/` with the containers' logs and
`docker inspect`, `resolved.json` (the start's run receipt, built from those
logs and the containers' environment; see [Run receipts](#run-receipts)),
`memory-before.json`, `ready.json`, `evidence.json`, `method.json`, `measure.json`, `decode.json`,
`prefill.jsonl`, `fingerprint.json`, `logprobs.json` (and `fused-after.json` for
`S+`); after the run `DIR/summary.json` and `DIR/tables.txt`
([`report.py`](report.py)): each start's method, receipt digest, key resolved
settings, declared change and check verdict; decode engine steps/s per cell and arm with ranges
and ratios to `N`, per-start tok/s, steps/s and acceptance, TTFT per prompt
length, and pairwise fingerprint and logprob agreement. Labels are `W-<arm>`
for warm-ups and `<arm><n>` for measured starts.

The report shows each start's layout (`path-4 in cycle-8`: four consecutive
Sparks of a ring of eight, which is not a ring of four, `cycle-4 in cycle-4`)
and refuses a ratio, prints `REFUSED` in its cell and states the reason above
the tables, when the two arms' decode results record different measurement
settings (tool version, temperature, token limit, contexts, streams, cell
duration, warm-up, prefill mode, coding peak and its temperature), when a start
of either arm holds a `check.json` whose check failed, or when the arms ran in
different layouts. `python -m performance.harnesses.serving_ab.report DIR
LABEL ... --allow-layout-mix` compares across layouts on purpose.

## Run receipts

A run receipt proves what a measured arm served and how it was measured. An
arm counts against its control only when the two differ in the declared change
and nothing else, and when both were measured by the same method.
`python -m performance.harnesses.serving_ab.receipt` ([`receipt/`](receipt/__init__.py))
runs from the repository root; `NODE_A` is Node A's SSH target, from which
every Spark is reached as root.

To measure an arm against its control:

1. **Pre-flight**, before installing or measuring:

   ```bash
   python -m performance.harnesses.serving_ab.receipt preflight --node-a NODE_A --positions 6,7 --owner CAMPAIGN
   ```

   It refuses while `/metrics` shows a running or waiting request in any of
   five readings 2 s apart, or while another owner reserves a requested Spark:
   `/tmp/ring8-hw.lock` (every Spark), `/tmp/ring8-hw-<first>-<last>.lock` and
   `/tmp/ring8-sets/<position>` on Node A. An owner file `NAME EPOCH` older than
   2,700 s is abandoned. `--reservations hwlock,setlock,command:CMD` adds a
   reservation tool that prints `[{"positions": [...], "owner": "..."}]`.
2. **Collect** the arm once it is ready, and the control the same way:

   ```bash
   python -m performance.harnesses.serving_ab.receipt collect --node-a NODE_A --positions 6,7 --output ARM_DIR
   ```

   It writes `ARM_DIR/resolved.json` and the raw reads under `ARM_DIR/raw/`.
3. **Declare** the arm's change as setting keys of `resolved.json` and the
   values the arm must hold, in `ARM_DIR/arm.json`:

   ```json
   {"schema": "serving-ab-arm/v1", "arm": "WD", "control": "W",
    "changes": {"speculative.kv_cache_dtype": "auto"}}
   ```

   A value may be `{"from": CONTROL_VALUE, "to": ARM_VALUE}`. A key ending in
   `.*` with the value `null` declares every difference under that prefix.
4. **Check** the arm against its control:

   ```bash
   python -m performance.harnesses.serving_ab.receipt check ARM_DIR CONTROL_DIR --declare ARM_DIR/arm.json --output ARM_DIR/check.json
   ```

   It exits 1 and lists every setting that differs without a declaration and
   every declared change that did not take effect: the arm lacks the value, or
   the control already holds it. A different layout (group shape or positions,
   fabric, fabric digest, SIRCL tuning state) is such a difference, named as a
   layout difference; declare `"layout.*": null` to compare across layouts on
   purpose. Warnings name a declared change that leaves
   the effective value unchanged, such as a draft KV dtype of `auto` against a
   control whose draft already inherits a BF16 target KV.
5. **Measure** with a named method, and record the window afterwards:

   ```bash
   python -m performance.harnesses.serving_ab.receipt window --node-a NODE_A --positions 6,7 \
     --since T0 --client CLIENT_ADDRESS --max-concurrency 8 --expect-tool-starts 1 --output ARM_DIR/window.json
   python -m performance.harnesses.serving_ab.receipt method RESULT.json --expect owner-temp1-matrix
   ```

   `window` counts rank 0's access-log requests from `T0` by source address,
   method and path. It fails when another address sent anything but monitoring
   reads (`GET /metrics`, `/health`, `/v1/models` and similar; `--strict` fails
   those too); with `--expect-tool-starts`, when the client's address started
   llm-inference-bench (its `GET /version`) more often than the campaign did;
   and with `--max-concurrency`, when vLLM's engine line reports more running
   plus waiting requests than the benchmark sends. `method` fails unless the
   result's recorded settings are the method's.

`diff A B` lists every difference of two receipts. Every command exits 0 when
its verdict holds, 1 when it does not and 2 on a read or usage error, so a
campaign script stops on the first refusal:

```bash
set -e
R="python -m performance.harnesses.serving_ab.receipt"
$R preflight --node-a NODE_A --positions 6,7 --owner CAMPAIGN
# ... install the arm, wait until it is ready ...
$R collect --node-a NODE_A --positions 6,7 --output runs/WD1
$R check runs/WD1 runs/W1 --declare arms/WD.json --output runs/WD1/check.json
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
# ... run the benchmark with the declared method into runs/WD1/result.json ...
$R window --node-a NODE_A --positions 6,7 --since "$T0" --client CLIENT_ADDRESS --max-concurrency 8 \
  --expect-tool-starts 1 --output runs/WD1/window.json
$R method runs/WD1/result.json --expect owner-temp1-matrix
```

What `resolved.json` holds, per rank and for the deployment (rank 0):

- `settings`, which `check` compares: the layout (`layout.group` such as
  `pair`, `path-4` or `cycle-4`, `layout.positions`, `layout.fabric` such as
  `cycle-8`, `layout.fabric_digest` and `layout.sircl_tuning`, from
  `sparkring status --json`, else from the SIRCL receipt line), the target KV
  dtype and block size, the
  speculative configuration including the draft's `kv_cache_dtype`, the
  engine's resolved `compilation_config` (mode, CUDA graph mode and sizes,
  custom ops, pass configuration), async scheduling, `max_num_seqs`,
  `max_num_batched_tokens`, the prefill schedule interval, the Engram
  configuration, quantization, load format, MoE, linear and attention
  backends, every other explicit vLLM argument, the vLLM process's `VLLM_*`,
  `B12X_*`, `SIRCL_*`, `OMP_*`, `NCCL_*` and related variables, the SIRCL
  session settings, the image ID and the deployment's profile, checkpoint and
  transport. Addresses, ports, rank numbers, device and interface names and
  cache paths are left out, so receipts of different Sparks compare.
- `derived`: the effective target and draft KV dtypes (`auto` is the model
  dtype; a draft without `kv_cache_dtype` inherits the target's).
- `observed`: KV capacity, the resolved attention block size, the kernels
  chosen, custom fusions, variables vLLM set itself, the SparkRing startup audit
  lines and GPU clocks. `diff` and `check` report them; they never fail a check.
- `sources`: where each value comes from, and `digest`, the SHA-256 of
  `settings`; `rank_mismatches` lists settings that worker ranks read the same
  way and resolve differently.

Limitations:

- vLLM logs no resolved async-scheduling value. Unset, it is inferred from
  vLLM's rule: on for EAGLE, MTP, DFlash, DSpark and draft-model speculation
  and without speculation, with the `mp` or `uni` executor; `sources` says
  `inferred`.
- A captured line cut short is parsed as far as it goes; a setting after the
  cut is `not recorded`, not vLLM's default.
- vLLM turns on breakable CUDA graphs for some architectures
  (`VLLM_USE_BREAKABLE_CUDAGRAPH`, an `observed` value), which sets the
  compilation mode to `NONE` whatever `--compilation-config` asks;
  `compilation.mode` shows the resolved mode.
- An access log names a client by address, so two clients on one machine look
  like one. The pre-flight keeps a second client out. Inside the window,
  `--expect-tool-starts` catches a second instance of llm-inference-bench that
  starts there, and `--max-concurrency` catches one only while the sum of both
  exceeds the benchmark's streams, at vLLM's 10 s logging interval; another
  kind of client on the same machine is not seen.
- The serving A/B runner's own `resolved.json` reads the containers'
  environment from `docker inspect`, not the vLLM process's, and takes the
  layout from the SIRCL receipt lines, or from its positions on the site's ring
  for an arm without them.

## Locking

[`setlock.py`](setlock.py). By default a run takes the set lock of its
positions (`--lock set`, owner `--lock-name`, default `serve-<positions>`): one
directory per position under `/tmp/ring8-sets/` on the lock host
(`--lock-host`, site position 0 by default), all created atomically or none,
beside the ring's global lock `/tmp/ring8-hw.lock` of `hwlock.sh`. A set lock
waits while the global lock is held, and two runs whose positions overlap
exclude each other. Holders refresh their time every 300 s; a lock older than
2,700 s is abandoned. `--lock global-held` takes none, for a run started under
`bash hwlock.sh OWNER python -m performance.harnesses.serving_ab run ...`.

Limitation: `hwlock.sh` does not yet wait for set locks, so a global-lock holder
can start beside a running set lock. Making it wait while any
`/tmp/ring8-sets/<position>` exists, under the same staleness rule, closes that.
