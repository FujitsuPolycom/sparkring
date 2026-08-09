# SparkRing EXL3 + LMCache troubleshooting

This page covers operational issues observed during EXL3 3.25-bpw plus
LMCache CS512 deployment on four directly cabled DGX Sparks. For
configuration errors (exit 3), see the
[acceptance runbook](EXL3_ACCEPTANCE_RUNBOOK.md) troubleshooting section first.

## Long double 315.78-GiB verification

The EXL3 bootstrap verifies all 81 model shards plus seven runtime metadata
files. First-time adoption hashes all 88 execution inputs, which can take
several minutes without consuming GPU time. The model weight bytes total
339,069,245,936 (≈315.78 GiB). This is a long sequential operation; it
is not stuck if:

- the process is still running and producing hash output;
- no shard has failed the expected SHA-256;
- the total shard count is progressing toward 81.

If a shard fails verification, the bootstrap aborts with the shard index
and expected hash. Re-copy the proven artifact; do not weaken the hashes.

## Page-cache pressure during model load

Loading 315.78 GiB of model weights on each rank can exhaust the Linux
page cache, causing slow subsequent operations or OOM conditions in
unrelated processes. Symptoms:

- `free -h` shows near-zero cached memory during model load;
- `dmesg` reports memory pressure or OOM-killer invocations;
- SSH sessions become sluggish or drop during load.

Mitigation:

- Ensure adequate free NVMe space for the checkpoint plus build headroom
  (see [PREREQUISITES.md](PREREQUISITES.md));
- Do not run other memory-intensive workloads on the Sparks during
  model load or graph capture;
- If page-cache pressure causes a host OOM, the LMCache server or engine
  container may be killed. Check `docker inspect` for `OOMKilled` and
  use the startup evidence classifier to distinguish this from a
  recoverable OOM.

## Recoverable OOM versus fatal signals

The EXL3 profile sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`.
A `CUDA out of memory` line during startup is classified as **recoverable**
if:

- the container stays running;
- a subsequent progress line appears (e.g. `KV cache allocated`,
  `Engine is ready`);
- `docker inspect` shows `RestartCount = 0` and `OOMKilled = false`.

It is a **fatal** condition if:

- the container was OOM-killed (`State.OOMKilled = true`);
- the container restarted (`RestartCount > 0`);
- an Xid GPU error appears in the logs;
- SSH connection to the rank failed during startup;
- NCCL/RoCE/fabric bootstrap failed;
- no progress line appears after the OOM message.

The startup evidence classifier distinguishes these automatically:

```bash
python scripts/sparkring_startup_evidence.py \
  --engine-log engine-r0.log --engine-log engine-r1.log \
  --engine-log engine-r2.log --engine-log engine-r3.log \
  classify
```

NVIDIA errors are **never** globally ignored. Each signature is classified
individually with provenance (line number, pattern label, SHA-256).

## Headless Wi-Fi management

If the management network is Wi-Fi, SSH sessions may drop during
high-throughput operations (model fanout, graph capture). Symptoms:

- intermittent SSH timeouts during bootstrap or launch;
- `ssh: connect to host ... port 22: Connection timed out` in logs;
- the launcher reports a rank as failed even though the container is
  still running.

Mitigation:

- Use wired management where possible (see
  [PREREQUISITES.md](PREREQUISITES.md));
- Ensure the Wi-Fi signal is stable and the SSH client has
  `ServerAliveInterval` and `ServerAliveCountMax` configured;
- If a rank is still running after a transient SSH failure, use the
  launcher `status` command to verify container state without restarting.

## Rollback failures

The LMCache launcher rollback removes only containers with the exact
`org.sparkring.exl3-profile` label. A foreign container with the same
name but a different label causes exit 73 and is intentionally not
removed. To resolve:

1. Identify the foreign container: `docker inspect <name> --format '{{json .Config.Labels}}'`
2. Confirm it is safe to remove manually
3. Remove it: `docker rm --force <name>`
4. Re-run `verify-rollback`

If rollback itself fails during a start attempt, the captured stop and
verification artifacts identify the exact remaining container. Do not
bypass the label guard; the refusal is a safety result.

## PowerShell log tailing

On a Windows operator machine, you can tail remote logs with PowerShell:

```powershell
# Tail a single rank's engine log
ssh -o BatchMode=yes user@rank0 "docker logs -f glm52-sparkring-exl3-lmcache-cs512-r0"

# Tail all four ranks in parallel (PowerShell 7+)
$ranks = @('rank0', 'rank1', 'rank2', 'rank3')
foreach ($r in $ranks) {
    Start-Job -Name "log-$r" -ScriptBlock {
        param($target, $name)
        ssh -o BatchMode=yes $target "docker logs -f $name 2>&1"
    } -ArgumentList $r, "glm52-sparkring-exl3-lmcache-cs512-r$($r[-1])"
}
Receive-Job -Keep *
```

For the LMCache server logs, replace the container name with
`glm52-sparkring-lmcache-cs512-server-r<N>`.

## LMCache CS512 geometry verification

Before a live acceptance run, verify the LMCache CS512 block-256
geometry offline:

```bash
python scripts/exl3_cache_geometry_gate.py verify
```

The `verify` command reports two categories:

- **Verified checks** (configuration facts read from the recipe):
  chunk_size=512, parent_chunk_size=256, lazy L1, LRU eviction,
  one-server-per-rank topology, SparkCache disabled
  (SPARK_CONTEXT_CACHE_ENABLE=0), and APC enabled
  (--enable-prefix-caching). These have `passed` verdicts.
- **Planned live gates** (require a live cluster, never `passed`):
  boundary token counts (511/512/513/1024/1025), DCP consensus
  evidence (object counts > 0 on all ranks plus TTFT ratio warm < cold;
  /status does not expose per-rank hit counters), and capacity metrics
  from /status (eviction_count is not exposed by the current schema).

A plan for the full geometry + timing suite:

```bash
python scripts/exl3_cache_geometry_gate.py plan
```

This discloses the C1/C2/C4/C8 and 16K/64K cold/warm timing cells that
require a live cluster, delegated to `exl3_cache_acceptance.py` and the
acceptance gate's performance matrix.

## Native APC and engine restart

The EXL3 profile enables vLLM's native prefix cache (APC) via
`--enable-prefix-caching` while disabling SparkCache
(SPARK_CONTEXT_CACHE_ENABLE=0). These are distinct cache layers:

- **SparkCache** (`sparkcache/`) is disabled to avoid interfering with
  LMCache attribution.
- **APC** (vLLM `--enable-prefix-caching`) is enabled and operates
  in-engine. It is not cleared by LMCache server restart.

To clear native APC while retaining LMCache objects, the live procedure
must restart engines (which clears in-engine APC state) while keeping
LMCache servers alive. The live cache acceptance gate's engine-only
restart phase exercises this boundary. Do not assume an LMCache server
restart alone clears APC — it does not.

## Readiness timeout is not KV exhaustion

The default automatic 60-second readiness allowance can be too short
for the unique-context C2/C4/C8 cells on the EXL3 profile. A readiness
timeout is not proof of KV exhaustion. To resolve:

- Reject the cell;
- Increase only the cell-warmup timeout to 300 seconds;
- Rerun.

A valid result requires exact effective concurrency and zero request
errors. Do not quote suppressed or partially populated measurements.

## Prohibited claims

- Do not describe EXL3+LMCache as "accepted" — it is live-validated,
  not accepted. NF3 remains the accepted deterministic alternative.
- Do not claim repeat determinism without immutable evidence (a committed
  expected-output file with a reviewed SHA-256).
- Do not quote reference-lane throughput numbers as public-lane results.
- Do not claim LMCache persistence (NVMe durability) — the CS512 recipe
  uses RAM-only L1 storage; persistence is server-resident reuse across
  engine restart, not disk durability.
- Do not merge evidence between lanes or configurations.
