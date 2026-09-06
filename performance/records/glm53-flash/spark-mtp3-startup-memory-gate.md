# Managed model startup memory-gate evidence

Status: qualified for the bounded refusal and startup cases described here.
The [machine-readable record](spark-mtp3-startup-memory-gate-20260905.json),
schema `sparkring-managed-startup-memory-validation/v1`, preserves the
measurements and source hashes collected on 2026-09-05.

## Conditions

Four NVIDIA DGX Spark GB10 hosts used 4,096-byte kernel pages. The served model
was GLM-5.3-Flash-NVFP4-Spark with TP4/DCP4, native MTP3, SparkCache, and the
hardware-forwarded mesh. Its image ID was
`sha256:3b4768e5ba31cadcc882dffa06d7b667af44abdf157d5c11b7ac7fe962e80c43`.
The JSON record's `source_hashes` identifies the tested coordinator, host
memory controls, launcher, and profile inputs.

The startup gate requires at least 96 GiB of available RAM and 200 equivalent
free 32 MiB blocks in the kernel's Normal memory zones on every rank. Model
processes were stopped before the guarded memory-preparation measurements.

## Measurement

Available memory comes from `/proc/meminfo`'s `MemAvailable` field. Contiguous
capacity is calculated from `/proc/buddyinfo`: each block of at least 32 MiB
contributes its size divided by 32 MiB. The record retains per-rank snapshots,
repair outcomes, and each coordinated startup command's exit status.

`readiness_elapsed_seconds` is retained as a reported observation. The record
does not identify its clock or exact timing boundaries, so that field does
not support a startup-speed comparison. The qualification claim depends on
gate outcomes, readiness completion, and native collective checks.

## Result

After one guarded reclamation/compaction pass, ranks 0–3 had 80, 175, 185, and
57 equivalent 32 MiB blocks. Every rank remained below the 200-block threshold
and returned `reboot-required`. No automatic reboot occurred.

Four operator-authorized reboots were recorded. Subsequent block counts were
3,671, 3,685, 3,687, and 3,683. During coordinated startup, all ranks passed
the stopped-model, idle-host, memory-preparation, memory-check, and peer-readiness
checks. Memory preparation reported `reclaimed: false` because the measured
memory already met both thresholds. All model units started successfully;
four-rank model readiness and native collective checks at 4, 20, 28, and 64
token rows passed.

Local CPU validation reports 324 tests passed and three platform-dependent
skips. These tests use mocked host interfaces and do not access cluster hosts.

## Conclusion

The tested managed controls refused model startup when contiguous-memory
capacity remained insufficient and allowed startup when every rank met the
thresholds. The adequate-memory path skipped reclamation. These results
qualify those bounded startup behaviors for the recorded source and image.

## Limitations

The record does not qualify automatic reboot, compaction beneath active model
work, unattended recovery, or a serving-speed improvement. Direct Docker
startup bypasses the managed authorization lock. The raw JSON remains an
immutable record of the hardware run; this description adds interpretation
without changing its measurements.
