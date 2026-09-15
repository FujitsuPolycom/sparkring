"""CPU replay of host validation/comparison bodies; CUDA kernels are not run."""
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

import pytest

HERE = Path(__file__).resolve().parent
INCLUDE = HERE.parents[1] / "include"


def _run_cpp(tmp_path, source, *, syntax_only=False):
    path = tmp_path / "host.cpp"
    path.write_text(source)
    binary = tmp_path / "host-test"
    if os.name == "nt":
        def linux(p):
            drive, tail = os.path.splitdrive(str(p))
            return "/mnt/" + drive[0].lower() + "/" + tail.lstrip("\\/").replace("\\", "/")
        available = subprocess.run(["bash", "-lc", "command -v g++"], capture_output=True, timeout=30)
        if available.returncode:
            pytest.skip("WSL g++ unavailable")
        command = " ".join(shlex.quote(v) for v in ["g++", "-std=c++17", "-I", linux(tmp_path), "-I", linux(HERE), "-I", linux(INCLUDE), linux(path), *( ["-fsyntax-only"] if syntax_only else ["-o", linux(binary)] )])
        built = subprocess.run(["bash", "-lc", command], text=True, capture_output=True, timeout=30)
        assert built.returncode == 0, built.stderr
        if syntax_only:
            return
        result = subprocess.run(["bash", "-lc", shlex.quote(linux(binary))], capture_output=True, text=True, timeout=30)
    else:
        compiler = shutil.which("g++")
        if compiler is None:
            pytest.skip("g++ unavailable")
        subprocess.run([compiler, "-std=c++17", "-I", str(tmp_path), "-I", str(HERE), "-I", str(INCLUDE), str(path), *(["-fsyntax-only"] if syntax_only else ["-o", str(binary)])], check=True, timeout=30)
        if syntax_only:
            return
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_smoke_comparator_rejects_nonfinite_results(tmp_path):
    source = (HERE / "bidirectional_bulk_cuda_smoke_test.cu").read_text()
    begin = source.index("struct OracleComparison")
    end = source.index("\n}  // namespace", begin)
    body = source[begin:end]
    # Replace only the BF16 representation and GPU-to-host copy interface.
    # The actual numerical comparison loops execute unchanged on host arrays.
    prelude = '''
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
using __nv_bfloat16 = float;
float __bfloat162float(float x) { return x; }
namespace spark_transport { constexpr unsigned kTp4PrefillRankCount = 4; }
using RankPointers = std::array<std::uint8_t*, 4>;
constexpr int cudaMemcpyDeviceToHost = 0;
int cudaMemcpy(void* a, const void* b, std::size_t n, int) { std::memcpy(a,b,n); return 0; }
void check_cuda(int, const char*) {}
'''
    main = '''
int main() {
  std::vector<float> values{1.0f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()};
  RankPointers outputs{};
  outputs.fill(reinterpret_cast<std::uint8_t*>(values.data()));
  std::vector<float> expected(3,1.0f), observed(3);
  if (compare_to_rounded_fp32(outputs,expected,observed,0.0625,0.02).mismatches != 8) return 1;
  values.assign(3,1.0f);
  if (compare_to_rounded_fp32(outputs,expected,observed,0.0625,0.02).mismatches != 0) return 2;
}
'''
    _run_cpp(tmp_path, prelude + body + main)


def test_host_launch_rejects_mismatched_descriptor_geometry(tmp_path):
    source = (HERE / "bidirectional_bulk_kernels.cu").read_text()
    begin = source.index("cudaError_t launch_bidirectional_stage_initial(")
    body = source[begin:source.rindex("}  // namespace")]
    # Launch syntax alone becomes ordinary no-op function calls. Preserve host
    # argument checks exactly; no device code, streams or CUDA API is executed.
    body = re.sub(r"<<<.*?>>>", "", body)
    prelude = '''
#include "bidirectional_bulk_abi.hpp"
using namespace spark_transport::tiled_prefill_research;
using cudaError_t = int;
using cudaStream_t = void*;
constexpr int cudaErrorInvalidValue = 1;
int launches = 0;
int cudaGetLastError() { return 0; }
unsigned launch_blocks(unsigned q) { return bidirectional_bulk_query_rows_supported(q) ? 1 : 0; }
bool launch_arguments_valid(const void* a, const void* b) { return a && b; }
template<class... T> void stage_initial_kernel(T...) { ++launches; }
template<class... T> void reduce_forward_kernel(T...) { ++launches; }
template<class... T> void reduce_finalize_seed_gather_kernel(T...) { ++launches; }
template<class... T> void gather_forward_kernel(T...) { ++launches; }
template<class... T> void gather_finish_kernel(T...) { ++launches; }
'''
    calls = ["launch_bidirectional_stage_initial(p,p,d,nullptr,q)",
             "launch_bidirectional_reduce_forward(p,p,p,d,nullptr,q)",
             "launch_bidirectional_reduce_finalize_seed_gather(p,p,p,p,d,nullptr,q)",
             "launch_bidirectional_gather_forward(p,p,p,d,nullptr,q)",
             "launch_bidirectional_gather_finish(p,p,d,nullptr,q)"]
    main = 'int main() { std::uint8_t byte{}; auto* p=&byte; BidirectionalBulkDescriptor d{}; d.query_rows=8192; unsigned q=1024;\n'
    main += "\n".join(f"if ({call} != cudaErrorInvalidValue) return 1;" for call in calls)
    main += 'if (launches != 0) return 2; q=8192;\n'
    main += "\n".join(f"if ({call} != 0) return 3;" for call in calls)
    main += 'return launches == 5 ? 0 : 4; }'
    _run_cpp(tmp_path, prelude + body + main)


def test_generic_smoke_host_code_resolves_its_declared_constants(tmp_path):
    # Only CUDA host API declarations are supplied. Compile the unmodified smoke
    # translation unit for C++ name/type checking without linking or execution.
    (tmp_path / "cuda_runtime.h").write_text("""
#pragma once
#include <cstddef>
using cudaError_t = int;
using cudaStream_t = void*;
constexpr int cudaSuccess=0, cudaStreamNonBlocking=1;
constexpr int cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2, cudaMemcpyDeviceToDevice=3;
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaMalloc(void**,std::size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*,unsigned);
cudaError_t cudaMemsetAsync(void*,int,std::size_t,cudaStream_t);
cudaError_t cudaMemcpyAsync(void*,const void*,std::size_t,int,cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamDestroy(cudaStream_t);
""")
    _run_cpp(tmp_path, '#include "tiled_cuda_smoke_test.cu"\n', syntax_only=True)
