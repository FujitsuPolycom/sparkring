"""CPU failure injection for ownership recorded before verbs submission."""
from pathlib import Path
import subprocess

from spark_transport.tests.test_probe_entrypoint_options import compile_cpu

ROOT = Path(__file__).resolve().parents[1]


def test_posting_failure_never_forgets_successful_payload(tmp_path):
    command = compile_cpu(tmp_path, r'''
#include "../experiments/tiled_prefill/fused_prefill_posting.hpp"
#include <cassert>
#include <deque>
#include <new>
#include <stdexcept>
struct Record { unsigned id; unsigned wqe_span; };
struct Pending {
  std::deque<Record> records{{99, 1}};
  bool reject{};
  void push_back(const Record& item) {
    if (reject) throw std::bad_alloc();
    records.push_back(item);
  }
  void pop_back() { records.pop_back(); }
};
int main() {
  using spark_transport::tiled_prefill_research::detail::post_reserved_work;
  for (int failure = 0; failure < 4; ++failure) {
    Pending pending;
    unsigned outstanding=1, posted=0;
    pending.reject = failure == 3;
    bool rejected=false;
    try {
      post_reserved_work(pending, outstanding, Record{7, 2}, [&] {
        assert(outstanding == 3 && pending.records.back().id == 7);
        if (failure == 1) throw std::runtime_error("payload not posted");
        ++posted;
      }, [&] {
        assert(outstanding == 3 && posted == 1);
        if (failure == 2) throw std::runtime_error("doorbell not posted");
        ++posted;
      });
    } catch (const std::exception&) { rejected=true; }
    assert(rejected == (failure != 0));
    assert(outstanding == ((failure == 1 || failure == 3) ? 1U : 3U));
    assert(pending.records.front().id == 99);
    assert(posted == (failure == 0 ? 2U : failure == 2 ? 1U : 0U));
  }
  for (bool failure : {false, true}) {
    Pending pending;
    unsigned outstanding=1;
    try {
      post_reserved_work(pending, outstanding, Record{8,1}, [&] {
        assert(outstanding == 2 && pending.records.back().id == 8);
        if (failure) throw std::runtime_error("credit not posted");
      }, [] {});
    } catch (const std::runtime_error&) { assert(failure); }
    assert(outstanding == (failure ? 1U : 2U));
  }
}
''')
    subprocess.run(command, check=True, capture_output=True, timeout=5)


def test_cuda_cta_arrival_follows_all_thread_payload_fences():
    source = (ROOT / "experiments/tiled_prefill/fused_prefill_kernels.cu").read_text()
    barrier = source.split("__device__ bool flow_barrier(", 1)[1].split("__device__ Bf16Packet", 1)[0]
    assert barrier.index("__syncthreads();") < barrier.index("if (threadIdx.x == 0)") < barrier.index("atomicAdd(")
    assert source.count("if (!flow_barrier(descriptor,") == 2


def test_proxy_routes_both_posting_paths_through_reserved_ownership():
    source = (ROOT / "experiments/tiled_prefill/fused_prefill_verbs_proxy.cpp").read_text()
    assert source.count("detail::post_reserved_work(") == 2
    pair = source.split("void post_exchange(", 1)[1].split("bool try_post_credit", 1)[0]
    assert pair.index("doorbell_id = next_work_id") < pair.index("detail::post_reserved_work") < pair.index("payload_id, false") < pair.index("doorbell_id, true")
