// Occupancy estimate for a synthetic eight-flow, six-stage control skeleton.
// This is not the fused-prefill kernel: its register/shared-memory usage and
// occupancy do not qualify that kernel's cooperative launch. No kernel launches.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

struct alignas(64) FusedFlowState {
  std::uint64_t operation_base;
  std::uint64_t producer_token;
  std::uint64_t consumer_token;
  std::uint32_t stage_arrivals[6];
  std::uint32_t stage_senses[6];
};
static_assert(sizeof(FusedFlowState) == 128);

struct FusedKernelState {
  FusedFlowState flows[8];
  std::uint64_t fatal_sequence;
};

template <int CtasPerFlow>
__global__ void fused_skeleton(FusedKernelState* state) {
  __shared__ std::uint64_t control[CtasPerFlow];
  const int flow = blockIdx.x / CtasPerFlow;
  const int lane = blockIdx.x % CtasPerFlow;
  if (threadIdx.x == 0 && flow < 8) {
    control[lane] = state->flows[flow].producer_token;
    if (control[lane] == UINT64_MAX) state->fatal_sequence = control[lane];
  }
}

template <int CtasPerFlow>
bool report(int sms) {
  int active{};
  const auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, fused_skeleton<CtasPerFlow>, 256, 0);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "Occupancy query failed: %s\n", cudaGetErrorString(status));
    return false;
  }
  const int requested = 8 * CtasPerFlow;
  const int capacity = active * sms;
  std::printf("FUSED_OCCUPANCY ctas_per_flow=%d requested=%d "
              "active_blocks_per_sm=%d sms=%d capacity=%d coresident=%d "
              "state_bytes=%zu\n",
              CtasPerFlow, requested, active, sms, capacity,
              capacity >= requested, sizeof(FusedKernelState));
  return true;
}

int main() {
  int sms{};
  int concurrent{};
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentKernels, 0) !=
          cudaSuccess) return 1;
  std::printf("FUSED_DEVICE sms=%d concurrent_kernels=%d\n", sms, concurrent);
  return report<1>(sms) && report<2>(sms) && report<4>(sms) ? 0 : 1;
}
