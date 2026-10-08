"""GPU checks of the fused all-reduce + residual add + RMSNorm on an emulated group (test support).

Every rank of one group runs as a thread of one process on one CUDA device
over the simulator build of the native layer
(:class:`sparkring_sircl.testing.gpu_emulation.EmulatedGroup`). The checks bind
the fused kernels to every rank's session (``sparkring_sircl.fused_norm.bind``)
and compare every rank's normalized rows and new residual bit for bit with
``fused_norm._reference``, whose reciprocal square root is the GPU's own
``rsqrt.approx.f32`` (computed by a one-instruction kernel on the reference's
operands), so the comparison is exact:

- one-shot and two-shot messages of 1 row up to the rows every rank's CTAs can
  keep resident on the shared GPU;
- ranks running the fused kernel and ranks running the plain all-reduce of the
  same algorithm in one op (the plain ranks hold the rank-ordered sum);
- the post-all-reduce RMSNorm interface ``allreduce_add_rms_norm`` (the
  session's own algorithm choice) and vLLM's in-place interface
  ``try_fused_add_rms_norm``;
- a decline: rows above ``max_rows`` return None on every rank and post no op;
- the fused op captured in a CUDA graph and replayed with new inputs.

``python -m sparkring_sircl.testing.fused_norm_gpu_checks [--layout ring:8] [--lanes 2] [--hidden 6144]``
prints one line per check. Requirements as for ``kernel_gpu_checks``.
"""

# Annotations stay evaluated (no postponed evaluation): the module defines a
# CuTe DSL kernel.

import argparse
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import routes as routes_mod
from . import native_build

Check = tuple[str, bool, str]


def _rsqrt_kernel():
    """A compiled ``run(src_address, dst_address, groups)``: ``rsqrt.approx.f32`` of ``4 * groups`` floats."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass import Int32, Int64

    from ..fused_norm._ptx import ld_global_v4, rsqrt_approx, st_global_v4
    from ..oneshot._compile import compile_launcher, current_cuda_stream
    from ..oneshot._cute_intrinsics import f32_as_u32, u32_as_f32

    class RsqrtApprox:
        @cute.jit
        def __call__(self, src: Int64, dst: Int64, groups: Int32, stream: cuda.CUstream) -> None:
            self.kernel(src, dst, groups).launch(grid=(1, 1, 1), block=[128, 1, 1], stream=stream)

        @cute.kernel
        def kernel(self, src: Int64, dst: Int64, groups: Int32) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            index = Int32(tidx)
            while index < groups:
                words = ld_global_v4(src + Int64(index) * Int64(16))
                out = [f32_as_u32(rsqrt_approx(u32_as_f32(word))) for word in words]
                st_global_v4(dst + Int64(index) * Int64(16), out[0], out[1], out[2], out[3])
                index += Int32(128)

    compiled = compile_launcher(RsqrtApprox(), 16, 16, 1, current_cuda_stream(), name="rsqrt.approx probe",
                                cache_key=("rsqrt-approx",))

    def run(src: int, dst: int, groups: int) -> None:
        compiled(int(src), int(dst), int(groups), current_cuda_stream())

    return run


def gpu_rsqrt(run, device) -> Callable[[np.ndarray], np.ndarray]:
    """``rsqrt`` for ``fused_norm._reference``: the GPU's ``rsqrt.approx.f32`` of every value."""
    import torch

    def rsqrt(values: np.ndarray) -> np.ndarray:
        flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        groups = -(-flat.size // 4)
        padded = np.ones(groups * 4, dtype=np.float32)
        padded[:flat.size] = flat
        src = torch.from_numpy(padded).to(device)
        dst = torch.empty_like(src)
        run(src.data_ptr(), dst.data_ptr(), groups)
        torch.cuda.synchronize(device)
        return dst.cpu().numpy()[:flat.size].reshape(np.shape(values))

    return rsqrt


def _bf16_numpy(torch, tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16).copy()


def _bf16_tensor(torch, array: np.ndarray, device):
    return torch.from_numpy(np.ascontiguousarray(array).view(np.int16)).view(torch.bfloat16).to(device)


class FusedChecks:
    def __init__(self, group: Any, fused: Sequence[Any], hidden: int, rsqrt) -> None:
        import torch

        self.torch = torch
        self.group = group
        self.fused = fused
        self.hidden = hidden
        self.rsqrt = rsqrt
        self.world = group.world
        self.rng = np.random.default_rng(17)
        self.eps = 1e-6

    def _case(self, rows: int):
        from ..fused_norm import _reference as ref

        partials = [ref.random_bf16(self.rng, (rows, self.hidden)) for _ in range(self.world)]
        residual = ref.random_bf16(self.rng, (rows, self.hidden))
        weight = ref.random_bf16(self.rng, (self.hidden,), scale=0.8)
        normed, z = ref.fused_add_rms_norm(partials, residual, weight, self.eps, rsqrt=self.rsqrt)
        return partials, residual, weight, normed, z, ref.rank_order_sum(partials)

    def fused_op(self, algorithm: str | None, rows: int, plain: Sequence[int] = (), interface: str = "call") -> Check:
        """One fused op; ``interface`` is ``call`` (explicit ``algorithm``), ``allreduce_add_rms_norm`` or
        ``in_place`` (``try_fused_add_rms_norm``), the last two with the session's own algorithm."""
        torch = self.torch
        partials, residual, weight, normed, z, reduced = self._case(rows)
        if algorithm is None:
            algorithm = self.fused[0].select_algorithm(rows * self.hidden * 2)
        label = (f"fused {algorithm} {rows} row(s) of {self.hidden}" + (f", plain ranks {list(plain)}" if plain else "")
                 + {"call": "", "allreduce_add_rms_norm": ", allreduce_add_rms_norm",
                    "in_place": ", in place (try_fused_add_rms_norm)"}[interface])

        def operation(rank: int, session: Any):
            x = _bf16_tensor(torch, partials[rank], session.device)
            if rank in plain:
                return session.all_reduce(x, algorithm=algorithm), None
            r = _bf16_tensor(torch, residual, session.device)
            w = _bf16_tensor(torch, weight, session.device)
            if interface == "in_place":
                if not self.fused[rank].try_fused_add_rms_norm(x, r, w, self.eps):
                    raise RuntimeError("try_fused_add_rms_norm declined")
                return x, r
            if interface == "allreduce_add_rms_norm":
                result = self.fused[rank].allreduce_add_rms_norm(x, r, w, self.eps)
                if result is None:
                    raise RuntimeError("allreduce_add_rms_norm declined")
                return result
            out, new_residual = self.fused[rank](x, r, w, self.eps, algorithm=algorithm)
            return out, new_residual

        try:
            results = self.group.each(operation)
        except Exception as error:  # noqa: BLE001
            return label, False, f"{type(error).__name__}: {error}"
        wrong = []
        for rank, (out, new_residual) in enumerate(results):
            if rank in plain:
                ok = _bf16_numpy(torch, out).tobytes() == reduced.tobytes()
            else:
                ok = (_bf16_numpy(torch, out).tobytes() == normed.tobytes()
                      and _bf16_numpy(torch, new_residual).tobytes() == z.tobytes())
            if not ok:
                wrong.append(rank)
        return label, not wrong, f"ranks {wrong} differ" if wrong else ""

    def declined(self, rows: int) -> Check:
        """``allreduce_add_rms_norm`` of ``rows`` rows above ``max_rows`` returns None everywhere and posts no op."""
        torch = self.torch
        label = f"decline of {rows} rows (max_rows {self.fused[0].max_rows})"

        def operation(rank: int, session: Any):
            before = session.stats().get("ops_posted")
            x = torch.zeros((rows, self.hidden), dtype=torch.bfloat16, device=session.device)
            result = self.fused[rank].allreduce_add_rms_norm(x, torch.zeros_like(x),
                                                             torch.ones(self.hidden, dtype=torch.bfloat16,
                                                                        device=session.device), self.eps)
            return result is None, session.stats().get("ops_posted") == before

        try:
            results = self.group.each(operation)
        except Exception as error:  # noqa: BLE001
            return label, False, f"{type(error).__name__}: {error}"
        wrong = [rank for rank, (none, still) in enumerate(results) if not (none and still)]
        return label, not wrong, f"ranks {wrong} fused or posted" if wrong else "None on every rank, no op posted"

    def graph(self, algorithm: str, rows: int, seeds: int = 2) -> Check:
        from .gpu_emulation import wait_stream

        torch = self.torch
        group = self.group
        state: list[dict] = [dict() for _ in range(self.world)]
        label = f"graph replay fused {algorithm} {rows} row(s)"
        try:
            for rank, session in enumerate(group.sessions):
                with torch.cuda.stream(group.streams[rank]):
                    x = torch.zeros((rows, self.hidden), dtype=torch.bfloat16, device=session.device)
                    r = torch.zeros_like(x)
                    w = torch.zeros(self.hidden, dtype=torch.bfloat16, device=session.device)
                    graph = torch.cuda.CUDAGraph()
                    with session.capture():
                        with torch.cuda.graph(graph, stream=group.streams[rank]):
                            out, _ = self.fused[rank](x, r, w, self.eps, algorithm=algorithm)
                    wait_stream(group.streams[rank])
                state[rank].update(x=x, r=r, w=w, out=out, graph=graph)
            for _ in range(seeds):
                partials, residual, weight, normed, z, _ = self._case(rows)

                def replay(rank: int, session: Any):
                    state[rank]["x"].copy_(_bf16_tensor(torch, partials[rank], session.device))
                    state[rank]["r"].copy_(_bf16_tensor(torch, residual, session.device))
                    state[rank]["w"].copy_(_bf16_tensor(torch, weight, session.device))
                    state[rank]["graph"].replay()

                group.each(replay)
                for rank in range(self.world):
                    if (_bf16_numpy(torch, state[rank]["out"]).tobytes() != normed.tobytes()
                            or _bf16_numpy(torch, state[rank]["r"]).tobytes() != z.tobytes()):
                        return label, False, f"rank {rank} differs"
        except Exception as error:  # noqa: BLE001
            return label, False, f"{type(error).__name__}: {error}"
        return label, True, f"{seeds} replays"


def run_checks(layout_text: str = "ring:8", lanes: int = 2, *, hidden: int = 6144,
               library: str | os.PathLike | None = None,
               report: Callable[[Check], None] | None = None) -> list[Check]:
    import torch

    from ..fused_norm import bind
    from .gpu_emulation import EmulatedGroup, wait_stream

    if library is None:
        build = Path(os.environ.get("SIRCL_TEST_BUILD_DIR", Path.cwd() / ".build" / "sim"))
        library = native_build.build_shared_library(build)
    multiprocessors = torch.cuda.get_device_properties(0).multi_processor_count
    world = routes_mod.Layout.parse(layout_text).world
    # Every CTA of every rank must be resident at once on the one shared GPU.
    max_rows = max(1, min(48, multiprocessors // world))
    capacity = max(1 << 20, -(-max_rows * hidden * 2 // 4096) * 4096)
    group = EmulatedGroup(layout_text, lanes, max_size=capacity, max_gather_bytes=0, library=library)
    checks: list[Check] = []

    def add(check: Check) -> None:
        checks.append(check)
        if report is not None:
            report(check)

    try:
        started = time.perf_counter()
        fused = []
        for rank, session in enumerate(group.sessions):
            with torch.cuda.stream(group.streams[rank]):
                session.prepare((torch.bfloat16,))
                fused.append(bind(session, hidden=hidden, max_rows=max_rows))
        group.load_modules((torch.bfloat16,))
        for rank, session in enumerate(group.sessions):
            with torch.cuda.stream(group.streams[rank]):
                poison = session._counter_layout.poison_word
                session._counters[poison] = 1
                x = torch.zeros((1, hidden), dtype=torch.bfloat16, device=session.device)
                for algorithm in ("oneshot", "twoshot"):
                    fused[rank](x, torch.zeros_like(x), torch.zeros(hidden, dtype=torch.bfloat16,
                                                                    device=session.device), 1e-6,
                                algorithm=algorithm)
                wait_stream(group.streams[rank])
                session._counters[poison] = 0
                wait_stream(group.streams[rank])
        rsqrt = gpu_rsqrt(_rsqrt_kernel(), group.sessions[0].device)
        add(("prepare", True, f"{time.perf_counter() - started:.1f} s; up to {max_rows} rows"))
        suite = FusedChecks(group, fused, hidden, rsqrt)
        for algorithm in ("oneshot", "twoshot"):
            for rows in sorted({1, 2, 3, max_rows}):
                add(suite.fused_op(algorithm, rows))
            add(suite.fused_op(algorithm, min(3, max_rows), plain=tuple(range(1, world, 2))))
            add(suite.graph(algorithm, min(4, max_rows)))
        for rows in sorted({1, 2, 3, max_rows}):
            add(suite.fused_op(None, rows, interface="allreduce_add_rms_norm"))
        add(suite.fused_op(None, 2, interface="in_place"))
        add(suite.declined(max_rows + 1))
        healthy = [not session.poisoned for session in group.sessions]
        add(("health", all(healthy), "" if all(healthy) else f"healthy ranks {healthy}"))
    finally:
        group.close()
    return checks


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", default="ring:8")
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=6144)
    args = parser.parse_args(argv)

    def show(check: Check) -> None:
        name, ok, detail = check
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)

    checks = run_checks(args.layout, args.lanes, hidden=args.hidden, report=show)
    failed = sum(1 for _, ok, _ in checks if not ok)
    print(f"{len(checks)} checks, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
