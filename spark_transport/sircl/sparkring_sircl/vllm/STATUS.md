# SIRCL vLLM adapter: limitations and ring checks

The adapter's status is in the package's status table,
[Component status](../../STATUS.md#component-status); serving through the
adapter is research-only. Design: [`README.md`](README.md). Collectives vLLM
issues: [`SURVEY.md`](SURVEY.md). Serving procedure: [`RUNBOOK.md`](RUNBOOK.md).

## Limitations

- **Pinned shims.** A shim installs only where the vLLM files it wraps match a
  pinned build (`python -m sparkring_sircl.vllm.serve shims`). A group that
  needs an unmatched shim fails setup on every rank; prefill row ownership
  then serves with `--mhc-prefill-shard off` or `--env
  VLLM_QWEN3_8_HC_PREFILL_MODE=off`. Without the `worker_regimes` shim,
  sessions keep the startup flag-wait limit while serving.
- **Calls outside the communicator.** Where NCCL may not run, a direct
  `torch.distributed` call runs only in the forms the tripwire's carrier
  supports and only on a group with a session; any other stops every rank
  with `NcclAcrossRelayError`, naming the group, the operation and the
  remedies. Models and recipes other than the catalog profiles that plan may
  issue such calls.
- **Refused settings.** Micro-batching on every group; where NCCL may not
  run, the `fuse_gemm_comms` and `fuse_allreduce_rms` passes, batch-sharded
  sampling and all-to-all backends other than `naive` and
  `allgather_reducescatter`; with NCCL off, `--load-format instanttensor`,
  `--enable-eplb` and `VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1`.
- **Launch shapes.** The serve launcher serves tensor parallelism only, from
  `serving-profile` profiles. Pipeline parallelism runs only through `bundle`
  (research-only); every group map assumes tensor parallelism over all ranks.
- **Fused all-reduce + RMSNorm** (research-only) needs vLLM's RMSNorm on the
  `vllm_c` provider; its bit-identity is checked in GPU emulation, not on GB10.
- **`SIRCL_*` variables** set outside the launcher or the bundle reach the
  tensor-parallel session unchecked.
- **Not part of the package:** restoring relay plans after a Spark reboots
  (the relay plan installer installs and removes them; nothing restores them
  at boot), SparkRing installer integration, and `sparkring fabric tune`
  (tuning tables come from the ring harness's `tune` command).

## Checks before relying on a layout

For each layout, profile and image ([`RUNBOOK.md`](RUNBOOK.md); relay plans:
[package runbook](../../RUNBOOK.md#relay-plan-installer)):

- [ ] `python -m sparkring_sircl.fabric diff --site "$SITE" --groups <group>`
      reports no host changes and every lane routed over its own device.
- [ ] The ring harness passes for the layout: bit-exact, no RDMA error
      counter change.
- [ ] `preflight` passes; `stage` reports a pinned vLLM build, every needed
      shim `verified` and no blocker.
- [ ] `start --wait` reaches API ready with a `group=tp` receipt in state
      `ready` on every rank (`nccl=none pynccl=skipped` where NCCL may not run).
- [ ] `check --long-prompt 16384` passes; after it the receipts show
      `wait=serving:<limit>s`.
- [ ] Without NCCL: `check --require-no-nccl` (or `bundle-check
      --require-no-nccl`) passes.
- [ ] The RDMA error counters (`hw_counters` under
      `/sys/class/infiniband/<device>/ports/1`, and `rx_out_of_buffer`) stay
      unchanged while serving, on every group's devices.
- [ ] Fused norm: the same tokens at temperature 0 with `--fused-norm on` and
      `off`.
