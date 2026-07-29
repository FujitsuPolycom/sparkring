#ifndef SPARK_CACHE_SNAPSHOT_RING_HPP_
#define SPARK_CACHE_SNAPSHOT_RING_HPP_

#include "spark_cache_snapshot.h"

#include <array>
#include <cstdint>
#include <string>

namespace spark_cache::snapshot {

struct Slot {
  SparkCacheSnapshotSlotState state = SPARK_CACHE_SNAPSHOT_SLOT_FREE;
  std::uint64_t generation = 0;
  std::uint64_t context_sequence = 0;
  std::uint64_t used_bytes = 0;
  bool discard = false;
};

/*
 * Portable oracle for the native CUDA implementation. It contains no CUDA
 * calls or allocation and is intentionally small enough for exhaustive tests.
 */
class RingState {
 public:
  explicit RingState(std::uint32_t slot_count) : slot_count_(slot_count) {}

  bool valid() const {
    return slot_count_ >= SPARK_CACHE_SNAPSHOT_MIN_SLOTS &&
           slot_count_ <= SPARK_CACHE_SNAPSHOT_MAX_SLOTS;
  }

  SparkCacheSnapshotStatus reserve(
      std::uint64_t context_sequence,
      std::uint64_t used_bytes,
      SparkCacheSnapshotTicket* output) {
    if (!valid() || output == nullptr || context_sequence == 0 ||
        used_bytes == 0) {
      return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
    }
    for (std::uint32_t index = 0; index < slot_count_; ++index) {
      auto& slot = slots_[index];
      if (slot.state != SPARK_CACHE_SNAPSHOT_SLOT_FREE) {
        continue;
      }
      slot.generation += 1;
      if (slot.generation == 0) {
        slot.generation = 1;
      }
      slot.context_sequence = context_sequence;
      slot.used_bytes = used_bytes;
      slot.discard = false;
      slot.state = SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING;
      output->generation = slot.generation;
      output->slot_index = index;
      output->reserved = 0;
      return SPARK_CACHE_SNAPSHOT_OK;
    }
    return SPARK_CACHE_SNAPSHOT_WOULD_BLOCK;
  }

  SparkCacheSnapshotStatus complete(
      const SparkCacheSnapshotTicket& ticket) {
    auto* slot = resolve(ticket);
    if (slot == nullptr) {
      return SPARK_CACHE_SNAPSHOT_DROPPED;
    }
    if (slot->state != SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING) {
      return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
    }
    if (slot->discard) {
      reset(slot);
      return SPARK_CACHE_SNAPSHOT_DROPPED;
    }
    slot->state = SPARK_CACHE_SNAPSHOT_SLOT_READY;
    return SPARK_CACHE_SNAPSHOT_OK;
  }

  SparkCacheSnapshotStatus claim(
      const SparkCacheSnapshotTicket& ticket) {
    auto* slot = resolve(ticket);
    if (slot == nullptr || slot->discard) {
      return SPARK_CACHE_SNAPSHOT_DROPPED;
    }
    if (slot->state == SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING) {
      return SPARK_CACHE_SNAPSHOT_NOT_READY;
    }
    if (slot->state != SPARK_CACHE_SNAPSHOT_SLOT_READY) {
      return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
    }
    slot->state = SPARK_CACHE_SNAPSHOT_SLOT_WRITING;
    return SPARK_CACHE_SNAPSHOT_OK;
  }

  SparkCacheSnapshotStatus release(
      const SparkCacheSnapshotTicket& ticket) {
    auto* slot = resolve(ticket);
    if (slot == nullptr) {
      return SPARK_CACHE_SNAPSHOT_DROPPED;
    }
    if (slot->state != SPARK_CACHE_SNAPSHOT_SLOT_READY &&
        slot->state != SPARK_CACHE_SNAPSHOT_SLOT_WRITING) {
      return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
    }
    reset(slot);
    return SPARK_CACHE_SNAPSHOT_OK;
  }

  std::uint32_t abandon(std::uint64_t context_sequence) {
    std::uint32_t affected = 0;
    for (std::uint32_t index = 0; index < slot_count_; ++index) {
      auto& slot = slots_[index];
      if (slot.state == SPARK_CACHE_SNAPSHOT_SLOT_FREE ||
          slot.context_sequence != context_sequence) {
        continue;
      }
      affected += 1;
      if (slot.state == SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING ||
          slot.state == SPARK_CACHE_SNAPSHOT_SLOT_WRITING) {
        slot.discard = true;
      } else {
        reset(&slot);
      }
    }
    return affected;
  }

  std::uint32_t abandon_all() {
    std::uint32_t affected = 0;
    for (std::uint32_t index = 0; index < slot_count_; ++index) {
      auto& slot = slots_[index];
      if (slot.state == SPARK_CACHE_SNAPSHOT_SLOT_FREE) {
        continue;
      }
      affected += 1;
      if (slot.state == SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING ||
          slot.state == SPARK_CACHE_SNAPSHOT_SLOT_WRITING) {
        slot.discard = true;
      } else {
        reset(&slot);
      }
    }
    return affected;
  }

  bool has_writing() const {
    for (std::uint32_t index = 0; index < slot_count_; ++index) {
      if (slots_[index].state == SPARK_CACHE_SNAPSHOT_SLOT_WRITING) {
        return true;
      }
    }
    return false;
  }

  /*
   * The CUDA owner calls this only after its completion event reports success.
   * It cannot free active or writer-owned bytes.
   */
  SparkCacheSnapshotStatus reap_discarded(std::uint32_t index) {
    if (index >= slot_count_) {
      return SPARK_CACHE_SNAPSHOT_INVALID_ARGUMENT;
    }
    auto& slot = slots_[index];
    if (slot.state == SPARK_CACHE_SNAPSHOT_SLOT_FREE) {
      return SPARK_CACHE_SNAPSHOT_DROPPED;
    }
    if (slot.state != SPARK_CACHE_SNAPSHOT_SLOT_GPU_FILLING ||
        !slot.discard) {
      return SPARK_CACHE_SNAPSHOT_INVALID_STATE;
    }
    reset(&slot);
    return SPARK_CACHE_SNAPSHOT_OK;
  }

  const Slot* inspect(std::uint32_t index) const {
    if (index >= slot_count_) {
      return nullptr;
    }
    return &slots_[index];
  }

 private:
  Slot* resolve(const SparkCacheSnapshotTicket& ticket) {
    if (ticket.slot_index >= slot_count_) {
      return nullptr;
    }
    auto& slot = slots_[ticket.slot_index];
    if (slot.state == SPARK_CACHE_SNAPSHOT_SLOT_FREE ||
        slot.generation != ticket.generation) {
      return nullptr;
    }
    return &slot;
  }

  static void reset(Slot* slot) {
    const auto generation = slot->generation;
    *slot = Slot{};
    slot->generation = generation;
  }

  std::uint32_t slot_count_ = 0;
  std::array<Slot, SPARK_CACHE_SNAPSHOT_MAX_SLOTS> slots_{};
};

bool validate_config(
    const SparkCacheSnapshotConfig& config,
    std::string* detail);

bool validate_sources(
    const SparkCacheSnapshotSource* sources,
    std::uint32_t source_count,
    std::uint32_t max_sources,
    std::string* detail);

bool calculate_payload_layout(
    const SparkCacheSnapshotSource* sources,
    std::uint32_t source_count,
    std::uint32_t row_count,
    std::uint64_t slot_bytes,
    SparkCacheSnapshotReadyView* output,
    std::string* detail);

}  // namespace spark_cache::snapshot

#endif  // SPARK_CACHE_SNAPSHOT_RING_HPP_
