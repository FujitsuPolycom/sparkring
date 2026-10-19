/* The large-reduction plan (src/link_plan.h): no link or chain op whose pack count does not fit the kernels'
 * int32_t arguments is chosen; such a body runs in pieces. */
#include "link_plan.h"

#include <stdio.h>
#include <stdlib.h>

static unsigned checks;
#define CHECK(expr) do { checks++; if (!(expr)) { \
  fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); exit(1); } } while (0)

int main(void) {
  const uint64_t limit = (uint64_t)INT32_MAX * SCCL_PLAN_PACK; /* the largest chain op: 32 GiB - 16 bytes */
  uint64_t bytes;
  CHECK(sccl_packs_fit(limit));
  CHECK(!sccl_packs_fit(limit + SCCL_PLAN_PACK));
  CHECK(!sccl_packs_fit(32ull << 30));

  /* The chain: up to the limit as one op; from 32 GiB in pieces, never a wrapped pack count. */
  CHECK(sccl_large_reduce_plan(8u << 20, 2, 0, 1, 1, &bytes) == SCCL_LARGE_CHAIN && bytes == 8u << 20);
  CHECK(sccl_large_reduce_plan(limit, 2, 0, 1, 1, &bytes) == SCCL_LARGE_CHAIN && bytes == limit);
  CHECK(sccl_large_reduce_plan(32ull << 30, 2, 0, 1, 1, &bytes) == SCCL_LARGE_PIECES && bytes == 0);
  CHECK(sccl_large_reduce_plan(64ull << 30, 4, 0, 1, 1, &bytes) == SCCL_LARGE_PIECES && bytes == 0);
  CHECK(sccl_large_reduce_plan(8u << 20, 2, 0, 1, 0, &bytes) == SCCL_LARGE_PIECES);

  /* The ring: the largest prefix of W equal chunks of whole packs, each below 2^31 bytes. */
  CHECK(sccl_large_reduce_plan((8u << 20) + 48, 4, 1, 1, 0, &bytes) == SCCL_LARGE_RING && bytes == 8u << 20);
  CHECK(sccl_large_reduce_plan(8ull << 30, 4, 1, 1, 0, &bytes) == SCCL_LARGE_PIECES);
  CHECK(sccl_large_reduce_plan((8ull << 30) - 64, 4, 1, 1, 0, &bytes) == SCCL_LARGE_RING &&
        bytes == (8ull << 30) - 64);
  CHECK(sccl_large_reduce_plan(48, 4, 1, 1, 0, &bytes) == SCCL_LARGE_PIECES);
  CHECK(sccl_large_reduce_plan(8u << 20, 4, 1, 0, 1, &bytes) == SCCL_LARGE_PIECES);

  /* One rank or an empty body: nothing to plan. */
  CHECK(sccl_large_reduce_plan(8u << 20, 1, 0, 1, 1, &bytes) == SCCL_LARGE_PIECES);
  CHECK(sccl_large_reduce_plan(0, 2, 0, 1, 1, &bytes) == SCCL_LARGE_PIECES);
  printf("link plan: %u checks passed\n", checks);
  return 0;
}
