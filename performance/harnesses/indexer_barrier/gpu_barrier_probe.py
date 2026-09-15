"""Bounded GB10 publication probe using the installed B12X barrier helper.

Each kernel executes one group barrier, so blocks cannot disagree about entering
a subsequent barrier. A delayed publishing warp creates the adversarial schedule.
The entry-sync variant adds block-level sync_threads before group arrival so all
warps publish before the block leader announces readiness.
"""
import hashlib
import inspect
import json

import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int32, Int64
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
import b12x.attention.dsa_indexer.fused_indexer as indexer
from b12x._lib.intrinsics import (
    get_ptr_as_int64, red_add_global_i32, ld_global_acquire_i32,
)


@dsl_user_op
def bounded_delay(cycles: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [Int64(cycles).ir_value(loc=loc, ip=ip)],
        "{ .reg .u64 start, now, elapsed; .reg .pred p; "
        "mov.u64 start, %clock64; delay_loop: nanosleep.u32 1000; "
        "mov.u64 now, %clock64; sub.u64 elapsed, now, start; "
        "setp.lt.u64 p, elapsed, $0; @p bra delay_loop; }",
        "l", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


class Probe:
    def __init__(self, fixed):
        self.fixed = fixed

    @cute.jit
    def __call__(self, state: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
        self.kernel(state, output).launch(
            grid=(3, 1, 1), block=(1024, 1, 1), cooperative=True, stream=stream,
        )

    @cute.kernel
    def kernel(self, state: cute.Tensor, output: cute.Tensor):
        block, _, _ = cute.arch.block_idx()
        tx, _, _ = cute.arch.thread_idx()
        if block == Int32(0):
            if tx == Int32(128):
                red_add_global_i32(get_ptr_as_int64(state, Int32(0)), Int32(512))
        if block == Int32(1):
            if tx == Int32(128):
                bounded_delay(Int64(20000000))
                red_add_global_i32(get_ptr_as_int64(state, Int32(1)), Int32(1))
        if cutlass.const_expr(self.fixed):
            cute.arch.sync_threads()
        indexer._fused_group_barrier(state, Int32(0), Int32(0), Int32(3), tx)
        if tx == Int32(0):
            low = ld_global_acquire_i32(get_ptr_as_int64(state, Int32(0)))
            high = ld_global_acquire_i32(get_ptr_as_int64(state, Int32(1)))
            output[block] = low + high


def main():
    source = inspect.getsourcefile(indexer)
    props = torch.cuda.get_device_properties(0)
    print(json.dumps({"gpu": props.name, "sm": [props.major, props.minor],
                      "torch": torch.__version__, "source": source,
                      "source_sha256": hashlib.sha256(open(source, "rb").read()).hexdigest()},
                     sort_keys=True), flush=True)
    state = torch.zeros(indexer._COOP_STATE_WORDS, device="cuda", dtype=torch.int32)
    output = torch.zeros(3, device="cuda", dtype=torch.int32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    state_arg, output_arg = from_dlpack(state), from_dlpack(output)
    results = []
    for fixed in (False, True):
        fn = cute.compile(Probe(fixed), state_arg, output_arg, stream)
        samples = []
        for _ in range(10):
            state.zero_()
            output.zero_()
            fn(state_arg, output_arg, stream)
            torch.cuda.synchronize()
            samples.append(output.cpu().tolist())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state.zero_()
            output.zero_()
            capture_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            fn(state_arg, output_arg, capture_stream)
        graph_samples = []
        for _ in range(10):
            graph.replay()
            torch.cuda.synchronize()
            graph_samples.append(output.cpu().tolist())
        result = {"entry_sync": fixed, "eager": samples, "graph": graph_samples,
                  "incomplete_reads": sum(any(x != 513 for x in row)
                                          for row in samples + graph_samples)}
        print(json.dumps(result), flush=True)
        results.append(result)
    if results[0]["incomplete_reads"] <= 0:
        raise RuntimeError("baseline did not expose the race")
    if results[1]["incomplete_reads"] != 0:
        raise RuntimeError("entry barrier did not fix publication")
    print("PASS: actual B12X helper exposes premature publication; entry sync prevents it", flush=True)


if __name__ == "__main__":
    main()
