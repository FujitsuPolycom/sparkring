#pragma once

#include "fused_prefill_abi.hpp"

#include <cuda_runtime.h>

namespace spark_transport::tiled_prefill_research {

// Launches exactly 8 flows x 4 CTAs cooperatively. The function refuses to
// launch unless all 32 CTAs can be simultaneously resident: the per-flow
// barriers intentionally make oversubscription unsafe.
cudaError_t launch_fused_prefill_q8192_n4(
    const FusedPrefillDescriptor* device_descriptors,
    cudaStream_t stream);

}  // namespace spark_transport::tiled_prefill_research
