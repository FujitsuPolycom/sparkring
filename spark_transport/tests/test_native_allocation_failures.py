"""CPU failure injection into actual native constructor and queue source blocks."""
from pathlib import Path
import subprocess

from test_probe_entrypoint_options import compile_cpu

SRC = Path(__file__).resolve().parents[1] / "src"


def run(tmp_path, source):
    command = compile_cpu(tmp_path, source)
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_memory_buffer_constructor_unwind(tmp_path):
    text = (SRC / "memory_buffer.cu").read_text()
    def block(name):
        start = text.index(f"  {name}(std::size_t bytes")
        end = text.index("  void* host_data() const override", start)
        return text[start:end].replace(" override", "")
    support = r"""
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
int live=0, failure=0;
constexpr int cudaHostAllocMapped=1,cudaHostAllocWriteCombined=2,cudaMemAttachGlobal=1;
void check_cuda(int r,const char*){if(r)throw std::runtime_error("injected");}
int cudaHostAlloc(void**p,size_t n,unsigned){*p=malloc(n);++live;return 0;}
int cudaHostGetDevicePointer(void**p,void*host,int){*p=host;return failure==1;}
int cudaFreeHost(void*p){free(p);--live;return 0;}
int cudaMalloc(void**p,size_t n){*p=malloc(n);++live;return 0;}
int cudaMallocManaged(void**p,size_t n,int){return cudaMalloc(p,n);}
int cudaMemset(void*,int,size_t){return failure==2;}
int cudaDeviceSynchronize(){return failure==3;}
int cudaFree(void*p){free(p);--live;return 0;}
"""
    source = support + "class CudaMappedBuffer {public:\n" + block("CudaMappedBuffer")
    source += "private:void*host_{};void*device_{};size_t bytes_{};bool write_combined_{};};\n"
    source += "class CudaAllocationBuffer {public:\n" + block("CudaAllocationBuffer")
    source += "private:void*data_{};size_t bytes_{};bool managed_{};};\n"
    source += r"""
int main(){
 for(failure=0;failure<=3;++failure){
  try{CudaMappedBuffer x(16,false);}catch(const std::runtime_error&){}
  if(live)return 1;
  for(bool managed:{false,true}){
   try{CudaAllocationBuffer x(16,managed);}catch(const std::runtime_error&){}
   if(live)return 2;
  }
 }
}
"""
    run(tmp_path, source)


def test_verbs_bounds_reject_unsigned_wrap_before_posting(tmp_path):
    text = (SRC / "verbs_endpoint.cpp").read_text()
    start = text.index("  if (local_offset", text.index("void VerbsEndpoint::write("))
    end = text.index("  ibv_sge scatter_gather", start)
    source = r"""
#include <cstddef>
#include <cstdint>
#include <stdexcept>
struct Buffer {std::size_t size()const{return 16;}} buffer_;
struct Remote {std::size_t buffer_bytes=16;} remote_;
void validate(std::size_t local_offset,std::size_t remote_offset,std::size_t bytes){
""" + text[start:end] + r"""
}
int main(){
 validate(0,0,16); validate(16,16,0);
 for(int which=0;which<2;++which){
  try{validate(which?0:SIZE_MAX,which?SIZE_MAX:0,1);return 1;}
  catch(const std::out_of_range&){}
 }
 try{validate(1,0,16);return 2;}catch(const std::out_of_range&){}
}
"""
    run(tmp_path, source)


def test_fused_proxy_failed_queue_allocation_retains_idle_slot(tmp_path):
    text = (SRC / "tp4_fused_prefill_session.cpp").read_text()
    start = text.index("class ProxyWorker {")
    end = text.index("\n}  // namespace", start)
    body = text[start:end].replace("std::deque<Work>", "FailingDeque<Work>")
    # The thread assignment is in the body, after every member constructor.
    assert ": proxy_(proxy) {" in body
    assert "thread_ = std::thread" in body
    support = r"""
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
constexpr std::uint32_t kOperationSlots=2;
bool fail_push=true;
template<class T>struct FailingDeque:std::deque<T>{
 void push_back(const T&x){if(fail_push)throw std::bad_alloc();std::deque<T>::push_back(x);}
};
struct Tp4FusedPrefillHealthStatus{bool healthy,poisoned,proxy_thread_running;std::uint64_t submitted_sequence,completed_sequence;};
namespace research {struct FusedPrefillVerbsProxy{int run_operation(std::uint64_t,std::uint64_t,std::uint32_t){return 0;}};}
"""
    run(tmp_path, support + body + r"""
int main(){research::FusedPrefillVerbsProxy proxy;ProxyWorker worker(proxy);
 try{worker.enqueue(1,8,0);return 1;}catch(const std::bad_alloc&){}
 worker.wait_idle();
 fail_push=false;worker.enqueue(2,8,0);worker.wait_idle();
 return worker.health_status().completed_sequence==2?0:2;
}
""")


def test_fused_cuda_partial_construction_releases_each_acquired_resource(tmp_path):
    text = (SRC / "tp4_fused_prefill_session.cpp").read_text()
    start = text.index("    try {", text.index("worker_ = std::make_unique<ProxyWorker>"))
    end = text.index("\n  ~Impl()", start)
    constructor = text[start:end]  # Includes the constructor's closing brace.
    start = text.index("  void release_cuda_resources() noexcept")
    end = text.index("\n  Tp4BidirectionalPrefillOptions options_;", start)
    cleanup = text[start:end]
    support = r"""
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <set>
constexpr int kOperationSlots=2,cudaEventDisableTiming=1;
using cudaEvent_t=void*;
int calls=0,fail_at=0;std::set<void*> live;
int inject(){return ++calls==fail_at;}
void check_cuda(int r,const char*){if(r)throw std::runtime_error("injected");}
template<class T>int cudaMalloc(T**p,std::size_t n){if(inject())return 1;*p=static_cast<T*>(malloc(n));live.insert(*p);return 0;}
int cudaMemset(void*,int,std::size_t){return inject();}
int cudaEventCreateWithFlags(void**p,int){return cudaMalloc(p,1);}
int cudaFree(void*p){if(live.erase(p)!=1)abort();free(p);return 0;}
int cudaEventDestroy(void*p){return cudaFree(p);}
int cudaEventSynchronize(void*){return 0;}
struct Channel{void barrier(){check_cuda(inject(),"barrier");}};
class Impl {public:Impl(){
"""
    fields = r"""
~Impl(){release_cuda_resources();}
private:
std::uint64_t*device_sync_{};std::uint64_t*device_descriptors_{};
std::array<cudaEvent_t,2>kernel_done_{};std::array<bool,2>kernel_done_armed_{};
std::array<std::unique_ptr<Channel>,1> channels_{std::make_unique<Channel>()};
"""
    run(tmp_path, support + constructor + fields + cleanup + r"""
};
int main(){for(fail_at=0;fail_at<=6;++fail_at){calls=0;try{Impl x;}catch(const std::runtime_error&){}if(!live.empty())return fail_at+1;}}
""")



def test_tensor_worker_partial_state_allocation_is_released(tmp_path):
    text = (SRC / "gpu_tp4_tensor.cu").read_text()
    start = text.index("GpuTp4TensorWorker::GpuTp4TensorWorker(")
    end = text.index("void GpuTp4TensorWorker::enqueue(", start)
    constructors = text[start:end]
    support = r"""
#include "spark_transport/gpu_tp4_tensor.hpp"
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
struct __nv_bfloat16 {std::uint16_t value;};
int calls=0,fail_at=0;std::set<void*>live;
constexpr int cudaDevAttrCooperativeLaunch=1;
void check_cuda(int r,const char*){if(r)throw std::runtime_error("injected");}
int cudaGetDevice(int*p){*p=0;return 0;}
int cudaDeviceGetAttribute(int*p,int,int){*p=1;return 0;}
int cudaMalloc(void**p,std::size_t n){if(++calls==fail_at)return 1;*p=malloc(n);live.insert(*p);return 0;}
int cudaMemset(void*,int,std::size_t){return ++calls==fail_at;}
int cudaFree(void*p){if(live.erase(p)!=1)abort();free(p);return 0;}
namespace spark_transport {
struct DirectGraphOperationState {std::uint64_t epoch,sequence;std::uint32_t lock,reserved;};
struct SplitGraphOperationState {std::uint64_t sequence,doorbell;std::size_t bytes,slot;};
struct StripedGraphOperationState {std::uint64_t sequence,doorbell;std::size_t bytes,stripe,generation;};
"""
    run(tmp_path, support + constructors + r"""
}
int main(){using namespace spark_transport;
 ExchangeBufferLayout layout{};
 for(bool direct:{false,true})for(int selector=0;selector<3;++selector){
  for(fail_at=0;fail_at<=3;++fail_at){calls=0;
   try{GpuTp4TensorWorker worker(40*8192,8192,(void*)1,layout,(void*)2,layout,
       Tp4AllreduceProtocol::kTwoSlotDeferredAck,
       static_cast<Tp4GraphKernelStrategy>(selector),Tp4AllreduceSchedule::kSequential,direct);}
   catch(const std::runtime_error&){}
   if(!live.empty())return 1;
  }
 }
}
""")
