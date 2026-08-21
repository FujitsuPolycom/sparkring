#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kTargetPhase = 0;
constexpr int kMetadataColumns = 5;
constexpr int kRequestSlotColumn = 0;
constexpr int kRoundColumn = 1;
constexpr int kWidthColumn = 2;
constexpr int kPhaseColumn = 3;
constexpr int kFlagsColumn = 4;
constexpr int kControlRequestSlot = 0;
constexpr int kControlPhase = 1;
constexpr int kControlActiveSlot = 2;
constexpr int kControlArmed = 3;
constexpr int kCounterClaimed = 0;
constexpr int kCounterCompleted = 1;
constexpr int kCounterOverflow = 2;
constexpr int kCounterWrongPhase = 3;
constexpr int kCounterOrphanLayer = 4;
constexpr int kCounterDuplicateLayer = 5;
constexpr int kCounterIncompleteRound = 6;
constexpr int kCounterInvalidExpert = 7;
constexpr int kCounterRejectionOrder = 8;
constexpr int kCounterRejectionValue = 9;
constexpr std::uint64_t kInvalidExpertFlag = 1;

template <typename Input>
__global__ void record_routes_kernel(
    const Input* topk_ids, std::int16_t* routes, std::int64_t* metadata,
    std::int64_t* layer_masks, std::int64_t* stream_control,
    std::int64_t* request_rounds, std::int64_t* counters,
    std::int64_t capacity_rounds, int layer_index, int width, int stream_slot,
    int num_layers, int max_width, int topk, int num_experts,
    int max_request_slots) {
  auto* counters_u64 = reinterpret_cast<unsigned long long*>(counters);
  auto* request_rounds_u64 =
      reinterpret_cast<unsigned long long*>(request_rounds);
  auto* masks_u64 = reinterpret_cast<unsigned long long*>(layer_masks);
  auto* metadata_u64 = reinterpret_cast<unsigned long long*>(metadata);
  __shared__ long long capture_slot;
  __shared__ int invalid_expert;

  if (threadIdx.x == 0) {
    invalid_expert = 0;
    if (stream_control[stream_slot * 4 + kControlArmed] != 1) {
      capture_slot = -1;
    } else {
      const std::int64_t request_slot =
          stream_control[stream_slot * 4 + kControlRequestSlot];
      const std::int64_t phase =
          stream_control[stream_slot * 4 + kControlPhase];
      if (phase != kTargetPhase) {
        atomicAdd(counters_u64 + kCounterWrongPhase, 1ULL);
        stream_control[stream_slot * 4 + kControlActiveSlot] = -1;
        capture_slot = -1;
      } else if (request_slot < 0 || request_slot >= max_request_slots) {
        atomicAdd(counters_u64 + kCounterOrphanLayer, 1ULL);
        stream_control[stream_slot * 4 + kControlActiveSlot] = -1;
        capture_slot = -1;
      } else if (layer_index == 0) {
        const auto claimed =
            atomicAdd(counters_u64 + kCounterClaimed, 1ULL);
        if (claimed >= static_cast<unsigned long long>(capacity_rounds)) {
          atomicAdd(counters_u64 + kCounterOverflow, 1ULL);
          stream_control[stream_slot * 4 + kControlActiveSlot] = -1;
          capture_slot = -1;
        } else {
          capture_slot = static_cast<long long>(claimed);
          stream_control[stream_slot * 4 + kControlActiveSlot] = capture_slot;
          const auto logical_round =
              atomicAdd(request_rounds_u64 + request_slot, 1ULL);
          const auto metadata_base = capture_slot * kMetadataColumns;
          metadata[metadata_base + kRequestSlotColumn] = request_slot;
          metadata_u64[metadata_base + kRoundColumn] = logical_round;
          metadata[metadata_base + kWidthColumn] = width;
          metadata[metadata_base + kPhaseColumn] = phase;
          metadata[metadata_base + kFlagsColumn] = 0;
          layer_masks[capture_slot * 2] = 0;
          layer_masks[capture_slot * 2 + 1] = 0;
        }
      } else {
        capture_slot =
            stream_control[stream_slot * 4 + kControlActiveSlot];
        if (capture_slot < 0 ||
            capture_slot >= static_cast<long long>(capacity_rounds)) {
          atomicAdd(counters_u64 + kCounterOrphanLayer, 1ULL);
          capture_slot = -1;
        }
      }
    }
  }
  __syncthreads();
  if (capture_slot < 0) {
    return;
  }

  const int elements = width * topk;
  for (int index = threadIdx.x; index < elements; index += blockDim.x) {
    const auto expert = static_cast<long long>(topk_ids[index]);
    if (expert < 0 || expert >= num_experts) {
      atomicExch(&invalid_expert, 1);
    }
  }
  __syncthreads();
  if (invalid_expert != 0) {
    if (threadIdx.x == 0) {
      atomicAdd(counters_u64 + kCounterInvalidExpert, 1ULL);
      atomicOr(metadata_u64 + capture_slot * kMetadataColumns + kFlagsColumn,
               kInvalidExpertFlag);
      stream_control[stream_slot * 4 + kControlActiveSlot] = -1;
    }
    return;
  }

  const auto route_base =
      ((capture_slot * num_layers + layer_index) * max_width) * topk;
  for (int index = threadIdx.x; index < elements; index += blockDim.x) {
    routes[route_base + index] =
        static_cast<std::int16_t>(topk_ids[index]);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const int mask_word = layer_index / 64;
    const auto bit = 1ULL << (layer_index % 64);
    const auto prior =
        atomicOr(masks_u64 + capture_slot * 2 + mask_word, bit);
    if ((prior & bit) != 0) {
      atomicAdd(counters_u64 + kCounterDuplicateLayer, 1ULL);
    }
    if (layer_index == num_layers - 1) {
      __threadfence();
      const auto expected_low = ~0ULL;
      const auto high_bits = num_layers - 64;
      const auto expected_high =
          high_bits == 64 ? ~0ULL : ((1ULL << high_bits) - 1ULL);
      const auto observed_low = masks_u64[capture_slot * 2];
      const auto observed_high = masks_u64[capture_slot * 2 + 1];
      const auto flags =
          metadata_u64[capture_slot * kMetadataColumns + kFlagsColumn];
      if (observed_low == expected_low &&
          observed_high == expected_high && flags == 0) {
        atomicAdd(counters_u64 + kCounterCompleted, 1ULL);
      } else {
        atomicAdd(counters_u64 + kCounterIncompleteRound, 1ULL);
      }
      stream_control[stream_slot * 4 + kControlActiveSlot] = -1;
    }
  }
}

__global__ void record_rejection_kernel(
    const std::int32_t* num_sampled, const std::int32_t* num_rejected,
    std::int32_t* rejected_tokens, const std::int64_t* metadata,
    const std::int64_t* stream_control, std::int64_t* counters,
    std::int64_t capacity_rounds, int stream_slot) {
  if (stream_control[stream_slot * 4 + kControlArmed] != 1) {
    return;
  }

  auto* counters_u64 = reinterpret_cast<unsigned long long*>(counters);
  const auto claimed = counters[kCounterClaimed];
  const auto completed = counters[kCounterCompleted];
  if (claimed <= 0 || claimed > capacity_rounds || completed != claimed ||
      stream_control[stream_slot * 4 + kControlActiveSlot] != -1) {
    atomicAdd(counters_u64 + kCounterRejectionOrder, 1ULL);
    return;
  }

  const auto capture_slot = claimed - 1;
  const auto metadata_base = capture_slot * kMetadataColumns;
  const auto request_slot =
      stream_control[stream_slot * 4 + kControlRequestSlot];
  const auto width = metadata[metadata_base + kWidthColumn];
  if (metadata[metadata_base + kRequestSlotColumn] != request_slot ||
      metadata[metadata_base + kPhaseColumn] != kTargetPhase ||
      (width != 5 && width != 6)) {
    atomicAdd(counters_u64 + kCounterRejectionOrder, 1ULL);
    return;
  }

  const auto sampled = static_cast<std::int64_t>(num_sampled[0]);
  const auto rejected = static_cast<std::int64_t>(num_rejected[0]);
  if (sampled < 1 || rejected < 0 || rejected > width - 1 ||
      sampled + rejected != width) {
    atomicAdd(counters_u64 + kCounterRejectionValue, 1ULL);
    return;
  }
  if (atomicCAS(rejected_tokens + capture_slot, -1,
                static_cast<std::int32_t>(rejected)) != -1) {
    atomicAdd(counters_u64 + kCounterRejectionOrder, 1ULL);
  }
}

void record_rejection_cuda(
    const at::Tensor& num_sampled, const at::Tensor& num_rejected,
    at::Tensor& rejected_tokens, const at::Tensor& metadata,
    const at::Tensor& stream_control, at::Tensor& counters,
    std::int64_t stream_slot) {
  TORCH_CHECK(num_sampled.is_cuda() && num_rejected.is_cuda(),
              "sampler counts must be CUDA tensors");
  TORCH_CHECK(rejected_tokens.is_cuda() && metadata.is_cuda() &&
                  stream_control.is_cuda() && counters.is_cuda(),
              "rejection capture tensors must be CUDA");
  TORCH_CHECK(num_sampled.device() == rejected_tokens.device() &&
                  num_rejected.device() == rejected_tokens.device() &&
                  metadata.device() == rejected_tokens.device() &&
                  stream_control.device() == rejected_tokens.device() &&
                  counters.device() == rejected_tokens.device(),
              "rejection capture tensors must share a device");
  TORCH_CHECK(num_sampled.is_contiguous() && num_rejected.is_contiguous(),
              "sampler counts must be contiguous");
  TORCH_CHECK(num_sampled.scalar_type() == at::kInt &&
                  num_rejected.scalar_type() == at::kInt,
              "sampler counts must be int32");
  TORCH_CHECK(num_sampled.dim() == 1 && num_sampled.numel() == 1,
              "rejection capture requires exactly one sampled request");
  TORCH_CHECK(num_rejected.dim() == 1 && num_rejected.numel() == 1,
              "rejection capture requires exactly one rejected request");
  TORCH_CHECK(rejected_tokens.dim() == 1 &&
                  rejected_tokens.scalar_type() == at::kInt,
              "rejected-token arena must be one-dimensional int32");
  TORCH_CHECK(metadata.dim() == 2 &&
                  metadata.size(0) == rejected_tokens.size(0) &&
                  metadata.size(1) == kMetadataColumns &&
                  metadata.scalar_type() == at::kLong,
              "rejection metadata arena is malformed");
  TORCH_CHECK(stream_control.dim() == 2 &&
                  stream_control.size(1) == 4 &&
                  stream_control.scalar_type() == at::kLong,
              "rejection stream-control table is malformed");
  TORCH_CHECK(counters.dim() == 1 && counters.size(0) == 10 &&
                  counters.scalar_type() == at::kLong,
              "rejection counter table is malformed");
  TORCH_CHECK(stream_slot >= 0 && stream_slot < stream_control.size(0),
              "rejection stream slot is outside control table");

  const c10::cuda::CUDAGuard guard(num_rejected.device());
  const auto stream =
      at::cuda::getCurrentCUDAStream(num_rejected.device().index());
  record_rejection_kernel<<<1, 1, 0, stream>>>(
      num_sampled.data_ptr<std::int32_t>(),
      num_rejected.data_ptr<std::int32_t>(),
      rejected_tokens.data_ptr<std::int32_t>(),
      metadata.data_ptr<std::int64_t>(),
      stream_control.data_ptr<std::int64_t>(),
      counters.data_ptr<std::int64_t>(), rejected_tokens.size(0),
      static_cast<int>(stream_slot));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void record_cuda(
    const at::Tensor& topk_ids, at::Tensor& routes, at::Tensor& metadata,
    at::Tensor& layer_masks, at::Tensor& stream_control,
    at::Tensor& request_rounds, at::Tensor& counters, std::int64_t layer_index,
    std::int64_t width, std::int64_t stream_slot, std::int64_t num_layers,
    std::int64_t num_experts) {
  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be CUDA");
  TORCH_CHECK(routes.is_cuda() && metadata.is_cuda() &&
                  layer_masks.is_cuda() && stream_control.is_cuda() &&
                  request_rounds.is_cuda() && counters.is_cuda(),
              "all capture tensors must be CUDA");
  TORCH_CHECK(topk_ids.device() == routes.device() &&
                  metadata.device() == routes.device() &&
                  layer_masks.device() == routes.device() &&
                  stream_control.device() == routes.device() &&
                  request_rounds.device() == routes.device() &&
                  counters.device() == routes.device(),
              "all capture tensors must share a device");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt ||
                  topk_ids.scalar_type() == at::kLong,
              "topk_ids must be int32 or int64");
  TORCH_CHECK(routes.scalar_type() == at::kShort, "routes must be int16");
  TORCH_CHECK(metadata.scalar_type() == at::kLong &&
                  layer_masks.scalar_type() == at::kLong &&
                  stream_control.scalar_type() == at::kLong &&
                  request_rounds.scalar_type() == at::kLong &&
                  counters.scalar_type() == at::kLong,
              "capture metadata tensors must be int64");
  TORCH_CHECK(width == 5 || width == 6, "only Q5/Q6 are supported");
  TORCH_CHECK(num_layers == 75, "GLM-5.2 requires 75 routed layers");
  TORCH_CHECK(num_experts == 256, "GLM-5.2 requires 256 experts");
  TORCH_CHECK(layer_index >= 0 && layer_index < num_layers,
              "layer index outside routed-layer range");
  TORCH_CHECK(stream_slot >= 0 && stream_slot < stream_control.size(0),
              "stream slot outside control table");
  TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.size(0) == width &&
                  topk_ids.size(1) == 8,
              "topk_ids must have exact [width, 8] shape");
  TORCH_CHECK(routes.dim() == 4 &&
                  routes.size(1) == num_layers &&
                  routes.size(2) == 6 && routes.size(3) == 8,
              "malformed fixed route arena");
  TORCH_CHECK(metadata.dim() == 2 &&
                  metadata.size(0) == routes.size(0) &&
                  metadata.size(1) == kMetadataColumns,
              "malformed metadata arena");
  TORCH_CHECK(layer_masks.dim() == 2 &&
                  layer_masks.size(0) == routes.size(0) &&
                  layer_masks.size(1) == 2,
              "malformed layer-mask arena");
  TORCH_CHECK(stream_control.dim() == 2 && stream_control.size(1) == 4,
              "malformed stream-control table");
  TORCH_CHECK(request_rounds.dim() == 1,
              "malformed request-round table");
  TORCH_CHECK(counters.dim() == 1 && counters.size(0) == 10,
              "malformed counter table");

  const c10::cuda::CUDAGuard guard(topk_ids.device());
  const auto stream = at::cuda::getCurrentCUDAStream(topk_ids.device().index());
  constexpr int threads = 64;
  if (topk_ids.scalar_type() == at::kInt) {
    record_routes_kernel<<<1, threads, 0, stream>>>(
        topk_ids.data_ptr<std::int32_t>(), routes.data_ptr<std::int16_t>(),
        metadata.data_ptr<std::int64_t>(),
        layer_masks.data_ptr<std::int64_t>(),
        stream_control.data_ptr<std::int64_t>(),
        request_rounds.data_ptr<std::int64_t>(),
        counters.data_ptr<std::int64_t>(), routes.size(0),
        static_cast<int>(layer_index), static_cast<int>(width),
        static_cast<int>(stream_slot), static_cast<int>(num_layers),
        static_cast<int>(routes.size(2)), static_cast<int>(routes.size(3)),
        static_cast<int>(num_experts),
        static_cast<int>(request_rounds.size(0)));
  } else {
    record_routes_kernel<<<1, threads, 0, stream>>>(
        topk_ids.data_ptr<std::int64_t>(), routes.data_ptr<std::int16_t>(),
        metadata.data_ptr<std::int64_t>(),
        layer_masks.data_ptr<std::int64_t>(),
        stream_control.data_ptr<std::int64_t>(),
        request_rounds.data_ptr<std::int64_t>(),
        counters.data_ptr<std::int64_t>(), routes.size(0),
        static_cast<int>(layer_index), static_cast<int>(width),
        static_cast<int>(stream_slot), static_cast<int>(num_layers),
        static_cast<int>(routes.size(2)), static_cast<int>(routes.size(3)),
        static_cast<int>(num_experts),
        static_cast<int>(request_rounds.size(0)));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY(sparkring_target_route_capture, library) {
  library.def(
      "record(Tensor topk_ids, Tensor(a!) routes, Tensor(b!) metadata, "
      "Tensor(c!) layer_masks, Tensor(d!) stream_control, "
      "Tensor(e!) request_rounds, Tensor(f!) counters, int layer_index, "
      "int width, int stream_slot, int num_layers, int num_experts) -> ()");
  library.def(
      "record_rejection(Tensor num_sampled, Tensor num_rejected, "
      "Tensor(a!) rejected_tokens, Tensor metadata, Tensor stream_control, "
      "Tensor(b!) counters, int stream_slot) -> ()");
}

TORCH_LIBRARY_IMPL(sparkring_target_route_capture, CUDA, library) {
  library.impl("record", &record_cuda);
  library.impl("record_rejection", &record_rejection_cuda);
}
