# Automatic collection during CUDA autotuning

Status: implemented; GPU startup qualification is required.

The source transformation in [stream_gate_gc.py](stream_gate_gc.py) changes
`b12x/preparation/_measurement.py`. B12X holds a CUDA stream behind a host-owned
flag while submitting timed kernel calls. A Python cyclic finalizer that unloads
a CUDA library can wait for that stream, preventing the same host thread from
releasing the flag.

The transformation disables automatic cyclic garbage collection before queuing
the stream wait. It releases the flag before restoring collection, including
when submission raises an exception. A process-wide count protected by a lock
keeps collection disabled until every overlapping gate has released its flag.
Collection remains disabled if it was disabled before the first gate entered.
Kernel selection, timing events and asynchronous submission remain enabled.

This does not prevent explicit `gc.collect()` calls, reference-count finalizers,
or unrelated code changing the process-wide GC setting during a gate. Callers
must not perform those operations while CUDA work is held. The transformation
does not retain compiled libraries permanently or suppress library unloading.

[CPU regression tests](test_stream_gate_gc.py) execute the transformed gate
with a fake CUDA driver and real Python cyclic finalizers. The unpatched gate
permits finalization before its flag is released; the transformed gate defers
automatic finalization until afterward. The tests also cover exceptions,
disabled collection, nesting, and overlapping host threads. Set
`SPARKRING_B12X_SOURCE_ROOT` to an unpatched B12X checkout to exercise the complete
build input instead of the embedded method fixture. Its source file,
`b12x/preparation/_measurement.py`, has SHA-256
`e45ca21666e5e056f13a4ecc71faaba6ceac2d165090a2dce4366b6c92f28ac0`.
CPU checks do not prove CUDA startup completion or exclude other sources of
driver synchronization.

## Bounded CUDA check

On a reserved GPU, run [check_stream_gate_gc.py](check_stream_gate_gc.py) inside
the selected image. It loads one no-op kernel and uses the actual CuTe module
destructor. A native watchdog releases the gate if unloading blocks; each arm
also has a process deadline. No model or network download is needed.

```bash
python check_stream_gate_gc.py --mode cycle-installed
```

The installed-guard check requires successful library unloading after the gate
opens, without watchdog intervention. On an unpatched control image, `--mode all`
compares direct destruction, automatic cyclic collection, and scoped collection
deferral. Direct-reference destruction remains outside the guard's contract.
This is a correctness check, not a throughput benchmark. Full-model startup and
restart qualification are separate requirements.
