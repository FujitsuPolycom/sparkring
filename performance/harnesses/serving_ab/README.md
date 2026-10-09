# Serving A/B runner

The serving A/B runner serves one installer profile on chosen DGX Sparks of a
cabled ring under several collective transports, on the same image, model and
vLLM settings, and compares them. It starts every arm fresh in a chosen order,
checks before measuring what carried the collectives, measures decode, time to
first token and output agreement from the operator's machine, and writes JSON
results and tables. Status: **implemented** (CPU tests in
[`test_serving_ab.py`](test_serving_ab.py)); the containers it renders follow the
installer's own container for each profile, but runs through it are research
measurements, not serving qualification.

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
| `run` | MUTATES HOST: the loader seccomp policy under `<remote_dir>/serving-ab/source`, the SIRCL arms' staged tree, native library, tuning tables and run directories under the site's `remote_dir`; one cache directory per arm; containers labelled `serving-ab=<run>`. Loads the GPUs and the fabric of the chosen Sparks |
| `report` | OFFLINE |

## Arms

Every rank's base is the installer's container for the profile: the profile
adapter's specification (`runtime/common/qwen_flash_next.py`), adapted to the
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
| `S` | SIRCL ring sessions on every collective, NCCL off | what `python -m sparkring_sircl.vllm.serve bundle --nccl never --require-no-nccl` gives the rank: three mounts, the SIRCL environment, `PYTHONPATH` and `sircl` in `VLLM_PLUGINS`; `--capacity`, `--dispatch` and `--tuning-table` pass to the bundle; `--sircl-env KEY=VALUE` adds a variable at the container level; `--roce-slot` keeps `VLLM_ENABLE_ROCE_ALLREDUCE=1`, so SIRCL's `roce_slot` shim takes vLLM's RoCE slot |
| `S+` | `S` with SIRCL's performance paths that are off by default | `bundle --fused-norm on` (`SIRCL_FUSED_NORM=1`: all-reduce, residual add and RMSNorm in one kernel, research-only, bit-identical by design; it applies only to models on vLLM's DeepSeek-V3.2 code) |
| `N` | vLLM's own communicator and PyNccl over the image's NCCL | `VLLM_ENABLE_ROCE_ALLREDUCE=0`, no RoCEnante bundle (`SPARKRING_TRANSPORT_PROFILE` empty), `SPARK_TP4_ENABLED=0`, and the NCCL variables `bundle --nccl auto` gives the rank: the RDMA devices facing its peers (`NCCL_IB_HCA`) and, on a whole cycle, `NCCL_ALGO=Ring` with `NCCL_SKIP_TREE_CONNECT=1`. The profile's other NCCL settings stay. Refused where NCCL may not run: a group whose consecutive ranks do not all share a cable, such as four Sparks of an eight-Spark ring |
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

`plan` prints every rank's command of every arm and, for a SIRCL arm against
`N`, the difference of rank 0's commands: the environment variables, mounts and
any other token that differs.

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
| `phase1` | contexts 0 and 32k, 1, 2, 4 and 8 streams, 30 s cells after 10 s, temperature 0, at most 1,024 tokens | 8k and 32k, three samples | as above |
| `phase2` | as `phase1` | 2k, 8k, 32k and 128k, three samples | as above |

Decode runs llm-inference-bench's `llm_decode_bench.py` (`--bench-dir`) through
[`bench_run.py`](bench_run.py), which turns off its self-update check. TTFT runs
[`prefill_probe.py`](../validation/prefill_probe.py) (unique prefixes; every
sample reports its cached tokens). Fingerprints ([`fingerprint.py`](fingerprint.py))
are four greedy 96-token answers compared token by token; prompt logprobs run
[`logprob_probe.py`](../../records/qwen38-flash-next/decode-ab-20260925/logprob_probe.py).
Restarts of one arm need not give identical tokens, so output agreement is
reported against the same-arm restarts, not as a pass or fail.

## Outputs

`DIR/plan.json`; per start `DIR/<label>/` with the containers' logs and
`docker inspect`, `ready.json`, `evidence.json`, `measure.json`, `decode.json`,
`prefill.jsonl`, `fingerprint.json`, `logprobs.json` (and `fused-after.json` for
`S+`); after the run `DIR/summary.json` and `DIR/tables.txt`
([`report.py`](report.py)): decode engine steps/s per cell and arm with ranges
and ratios to `N`, per-start tok/s, steps/s and acceptance, TTFT per prompt
length, and pairwise fingerprint and logprob agreement. Labels are `W-<arm>`
for warm-ups and `<arm><n>` for measured starts.

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
