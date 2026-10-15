/* Size limits of the link and chain kernels' arguments, and the plan of a large reduction that keeps to them.
 * Header-only, so tests/test_link_plan.c checks the code the engine runs. */
#ifndef SCCL_LINK_PLAN_H
#define SCCL_LINK_PLAN_H

#include <stdint.h>

enum { SCCL_PLAN_PACK = 16 };

/* The link, chain and point-to-point kernels take pack counts (16-byte units) as int32_t. */
static inline int sccl_packs_fit(uint64_t bytes) { return bytes / SCCL_PLAN_PACK <= (uint64_t)INT32_MAX; }

enum { SCCL_LARGE_PIECES = 0, SCCL_LARGE_RING = 1, SCCL_LARGE_CHAIN = 2 };

/* The link op of a large all-reduce or reduce whose 16-byte-aligned body is `body` bytes, over `world` ranks
 * (SIRCL's pieces.reduce_plan). `ring_schedule`: the ring schedule applies to this size; `link_on`: the link
 * pack is loaded; `chain`: the chain schedule, or auto from the chain minimum, with the chain pack loaded.
 * Under the ring schedule: one ring op over the largest prefix of `world` equal chunks of whole packs, each
 * chunk below 2^31 bytes, else none. Otherwise: one chain op over the whole body when its pack count fits,
 * else none. None means the body runs in pieces. *bytes is the link op's size (0 for none). */
static inline int sccl_large_reduce_plan(uint64_t body, uint64_t world, int ring_schedule, int link_on, int chain,
                                         uint64_t *bytes) {
  *bytes = 0;
  if (!body || world < 2) return SCCL_LARGE_PIECES;
  if (ring_schedule) {
    uint64_t ring_bytes = body / (SCCL_PLAN_PACK * world) * SCCL_PLAN_PACK * world;
    if (!link_on || !ring_bytes || ring_bytes / world >= (1ull << 31)) return SCCL_LARGE_PIECES;
    *bytes = ring_bytes;
    return SCCL_LARGE_RING;
  }
  if (!chain || !sccl_packs_fit(body)) return SCCL_LARGE_PIECES;
  *bytes = body;
  return SCCL_LARGE_CHAIN;
}

#endif
