"""Bounded probe of CuTe library finalization during a held CUDA stream gate.

Run inside the serving image on an explicitly reserved GPU. This loads one
no-op PTX kernel, not a model. A native watchdog releases the mapped host flag
after two seconds even if the CUDA unload call holds Python's GIL. Every arm
runs in its own process; a parent timeout is a second containment boundary.

The automatic-GC arms retain the installed CudaDialectJitModule destructor and
the installed B12X _StreamGate. The only counterfactual is disabling cyclic GC
before opening the gate and restoring it after the host releases the gate.
An explicit reference-count finalization arm establishes whether unloading a
CUDA library actually waits for the gated stream in this driver/image pair.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


WATCHDOG_C = r"""
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <time.h>
struct watchdog {
    pthread_t thread;
    _Atomic uint32_t *flag;
    uint32_t target;
    int ticks;
    _Atomic int cancel;
    _Atomic int fired;
};
static void *run(void *raw) {
    struct watchdog *w = raw;
    struct timespec delay = {.tv_sec = 0, .tv_nsec = 1000000};
    for (int i = 0; i < w->ticks; ++i) {
        if (atomic_load(&w->cancel)) return NULL;
        nanosleep(&delay, NULL);
    }
    if (!atomic_load(&w->cancel)) {
        atomic_store(&w->fired, 1);
        atomic_store_explicit(w->flag, w->target, memory_order_release);
    }
    return NULL;
}
void *start_watchdog(uintptr_t flag, uint32_t target, int ticks) {
    struct watchdog *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->flag = (_Atomic uint32_t *)flag;
    w->target = target;
    w->ticks = ticks;
    if (pthread_create(&w->thread, NULL, run, w)) {free(w); return NULL;}
    return w;
}
int finish_watchdog(void *raw) {
    struct watchdog *w = raw;
    atomic_store(&w->cancel, 1);
    pthread_join(w->thread, NULL);
    int fired = atomic_load(&w->fired);
    free(w);
    return fired;
}
"""

PTX = b""".version 8.0
.target sm_80
.address_size 64
.visible .entry gate_probe_noop() {
    ret;
}
\x00"""


def emit(event: str, **values) -> None:
    print(
        json.dumps(
            dict(
                event=event,
                utc=datetime.now(timezone.utc).isoformat(),
                monotonic=time.monotonic(),
                **values,
            ),
            sort_keys=True,
        ),
        flush=True,
    )


@contextmanager
def suspend_cyclic_gc():
    """Keep automatic cyclic finalizers outside a host-controlled GPU wait."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def load_watchdog(directory: Path):
    binary = directory / "watchdog.so"
    subprocess.run(
        [
            "cc",
            "-x",
            "c",
            "-",
            "-shared",
            "-fPIC",
            "-pthread",
            "-O2",
            "-std=c11",
            "-o",
            str(binary),
        ],
        input=WATCHDOG_C,
        text=True,
        check=True,
        timeout=30,
    )
    library = ctypes.CDLL(str(binary))
    library.start_watchdog.argtypes = (ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int)
    library.start_watchdog.restype = ctypes.c_void_p
    library.finish_watchdog.argtypes = (ctypes.c_void_p,)
    library.finish_watchdog.restype = ctypes.c_int
    return library


def checked(result):
    status, *values = result
    if int(status) != 0:
        raise RuntimeError(f"CUDA probe call failed: {status}")
    return values[0] if values else None


def run_arm(mode: str, delay_ms: int) -> dict:
    import torch
    from cuda.bindings import driver, runtime
    from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitModule
    from b12x.preparation._measurement import _StreamGate

    torch.cuda.init()
    stream = torch.cuda.Stream()
    gate = _StreamGate()
    started = time.monotonic()
    finalizations = []
    unload_statuses = []
    old_thresholds = gc.get_threshold()
    originally_enabled = gc.isenabled()
    actual_unload = runtime.cudaLibraryUnload

    def checked_unload(library):
        result = actual_unload(library)
        unload_statuses.append(int(result[0]))
        return result

    runtime.cudaLibraryUnload = checked_unload

    class ObservedModule(CudaDialectJitModule):
        def unload(self):
            begin = time.monotonic()
            held = gate.flag.value < gate.sequence
            emit(
                "unload-enter",
                mode=mode,
                held=held,
                flag=gate.flag.value,
                target=gate.sequence,
                gc_enabled=gc.isenabled(),
            )
            super().unload()
            elapsed = time.monotonic() - begin
            finalizations.append(dict(held=held, seconds=elapsed))
            emit("unload-exit", mode=mode, seconds=elapsed)

    with tempfile.TemporaryDirectory(prefix="sparkring-cuda-gate-probe-") as directory:
        watchdog = load_watchdog(Path(directory))
        code = ctypes.create_string_buffer(PTX)
        library = checked(
            driver.cuLibraryLoadData(
                ctypes.addressof(code),
                [],
                [],
                0,
                [],
                [],
                0,
            )
        )
        kernel = checked(driver.cuLibraryGetKernel(library, b"gate_probe_noop"))
        function = checked(driver.cuKernelGetFunction(kernel))
        checked(
            driver.cuLaunchKernel(
                function, 1, 1, 1, 1, 1, 1, 0, stream.cuda_stream, 0, 0
            )
        )
        stream.synchronize()
        gc.collect()
        gc.disable()
        # Create the victim after collection so it remains in generation zero;
        # automatic generation-two collection has additional long-lived heuristics.
        module = ObservedModule(None, None, None, [runtime.cudaLibrary_t(int(library))])
        assert gc.is_tracked(module)
        assert any(value is module for value in gc.get_objects(0))
        if mode != "direct-held":
            module._probe_cycle = module
            del module
        # Enabling GC does not itself collect. A high threshold keeps setup and
        # context-manager allocation from reclaiming the deliberately cyclic
        # module before the CUDA wait has actually entered the stream.
        gc.set_threshold(1_000_000, 1_000_000, 1_000_000)
        gc.enable()
        thread = watchdog.start_watchdog(int(gate.pointer), gate.sequence + 1, delay_ms)
        if not thread:
            raise RuntimeError("native watchdog could not start")
        watchdog_fired = False
        try:
            guard = suspend_cyclic_gc() if mode == "cycle-guarded" else nullcontext()
            with guard:
                with gate.hold(stream):
                    emit(
                        "gate-held",
                        mode=mode,
                        flag=gate.flag.value,
                        target=gate.sequence,
                        gc_enabled=gc.isenabled(),
                    )
                    if mode == "direct-held":
                        del module
                    else:
                        # Automatic collection is triggered by tracked allocation,
                        # not gc.collect(); disabling GC must suppress this arm.
                        gc.set_threshold(1, 1, 1)
                        retained = [[index] for index in range(1024)]
                        del retained
            emit("gate-released", mode=mode, flag=gate.flag.value, target=gate.sequence)
            gc.set_threshold(*old_thresholds)
            gc.collect()
            stream.synchronize()
        finally:
            gate.flag.value = gate.sequence
            watchdog_fired = bool(watchdog.finish_watchdog(thread))
            gc.set_threshold(*old_thresholds)
            if originally_enabled:
                gc.enable()
            else:
                gc.disable()
            gate.close()
            runtime.cudaLibraryUnload = actual_unload
        report = dict(
            mode=mode,
            watchdog_fired=watchdog_fired,
            finalizations=finalizations,
            unload_statuses=unload_statuses,
            seconds=time.monotonic() - started,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            measurement_source=__import__("inspect").getfile(_StreamGate),
            measurement_sha256=hashlib.sha256(
                Path(__import__("inspect").getfile(_StreamGate)).read_bytes()
            ).hexdigest(),
        )
        if len(finalizations) != 1 or unload_statuses != [0]:
            raise AssertionError(f"expected one actual CuTe module finalizer: {report}")
        item = finalizations[0]
        expected_stall = mode in ("direct-held", "cycle-held")
        report["prediction_confirmed"] = (
            item["held"] == expected_stall
            and watchdog_fired == expected_stall
            and (
                item["seconds"] >= delay_ms / 1000 * 0.7
                if expected_stall
                else item["seconds"] < delay_ms / 1000 * 0.3
            )
        )
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "direct-held",
            "cycle-held",
            "cycle-guarded",
            "cycle-installed",
        ),
        default="cycle-installed",
    )
    parser.add_argument("--watchdog-ms", type=int, default=2000)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 500 <= args.watchdog_ms <= 5000:
        parser.error("watchdog must be between 500 and 5000 ms")
    if args.child:
        if args.mode == "all":
            parser.error("Child execution requires one arm")
        result = run_arm(args.mode, args.watchdog_ms)
        emit("result", **result)
        return 0 if result["prediction_confirmed"] else 2
    results = []
    arms = (
        ("direct-held", "cycle-held", "cycle-guarded")
        if args.mode == "all"
        else (args.mode,)
    )
    for arm in arms:
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                arm,
                "--watchdog-ms",
                str(args.watchdog_ms),
                "--child",
            ],
            text=True,
            capture_output=True,
            timeout=90,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        print(process.stdout, end="", flush=True)
        print(process.stderr, end="", file=sys.stderr, flush=True)
        results.append(dict(mode=arm, returncode=process.returncode))
    emit("summary", arms=results)
    return 0 if all(item["returncode"] == 0 for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
