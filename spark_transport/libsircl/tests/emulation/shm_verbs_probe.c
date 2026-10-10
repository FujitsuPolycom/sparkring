/* One rank of the CPU check of the shared-memory verbs stand-in across processes
 * (tests/test_shm_verbs.py). SIRCL's native proxy (emulation build) runs on an arena in
 * shared memory; the host plays the one-shot kernel's part of the command ring: it stages a
 * payload, rings the doorbell, waits for every peer's lane flags and checks every peer's bytes.
 *
 *   shm_verbs_probe <world> <rank> <lanes> <ops> <directory>
 *
 * Ranks exchange their connection records through files in <directory>. Exit 0 when every op's
 * bytes from every peer are exact. */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "shm_verbs.h"

int sccl_emu_roce_layout(int, uint64_t, uint64_t *);
unsigned sccl_emu_roce_local_features(void);
uint64_t sccl_emu_roce_blob_bytes(void);
void *sccl_emu_roce_create(int, int, const char *const *, int, const int *, int, const int *, int, void *, uint64_t,
                           uint64_t, char *, uint64_t);
int sccl_emu_roce_local_blob(void *, void *, uint64_t);
int sccl_emu_roce_connect(void *, const void *, uint64_t);
int sccl_emu_roce_lane_check(void *, int);
int sccl_emu_roce_start(void *);
int sccl_emu_roce_failed(void *);
const char *sccl_emu_roce_error(void *);
void sccl_emu_roce_destroy(void *);

static void fail(const char *what, const char *detail) {
  fprintf(stderr, "rank failure: %s %s\n", what, detail ? detail : "");
  exit(1);
}

static void barrier_file(const char *dir, const char *name, int world, int rank, const void *data, size_t bytes,
                         void *all) {
  char path[512], tmp[520];
  snprintf(path, sizeof path, "%s/%s.%d", dir, name, rank);
  snprintf(tmp, sizeof tmp, "%s.tmp", path);
  FILE *f = fopen(tmp, "wb");
  if (!f || fwrite(data, 1, bytes, f) != bytes) fail("writing", path);
  fclose(f);
  rename(tmp, path);
  for (int r = 0; r < world; ++r) {
    snprintf(path, sizeof path, "%s/%s.%d", dir, name, r);
    for (int tries = 0;; ++tries) {
      FILE *g = fopen(path, "rb");
      if (g) {
        size_t got = fread((char *)all + (size_t)r * bytes, 1, bytes, g);
        fclose(g);
        if (got == bytes) break;
      }
      if (tries > 200000) fail("waiting for", path);
      usleep(100);
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 6) return 2;
  int world = atoi(argv[1]), rank = atoi(argv[2]), lanes = atoi(argv[3]), ops = atoi(argv[4]);
  const char *dir = argv[5];
  char err[512] = {0};
  if (sccl_emu_attach(err, sizeof err)) fail("attach", err);
  /* SIRCL change LF: bit 0 (roce_destroy counts failed verbs calls) and bit 1 (flags-only own items). */
  if ((sccl_emu_roce_local_features() & 3u) != 3u) fail("local features (roce_local_features bits 0 and 1)", NULL);
  const uint64_t slot = 65536;
  uint64_t layout[7];
  if (sccl_emu_roce_layout(world, slot, layout)) fail("layout", NULL);
  uint8_t *arena = sccl_emu_segment_alloc(layout[4], err, sizeof err);
  if (!arena) fail("arena", err);
  char names[2][64];
  const char *list[2];
  int gids[2] = {0, 0};
  for (int d = 0; d < lanes; ++d) {
    snprintf(names[d], 64, "probe%d.%d", (int)getpid(), d);
    uint8_t gid[16] = {0xfe, 0x80};
    gid[8] = (uint8_t)(getpid() >> 8);
    gid[9] = (uint8_t)getpid();
    gid[15] = (uint8_t)(d + 1);
    if (sccl_emu_add_device(names[d], gid) < 0) fail("device", names[d]);
    list[d] = names[d];
  }
  int lane_devices[32];
  for (int p = 0; p < world; ++p)
    for (int l = 0; l < lanes; ++l) lane_devices[p * lanes + l] = p == rank ? -1 : l;
  void *proxy = sccl_emu_roce_create(world, rank, list, lanes, lane_devices, lanes, gids, 0, arena, layout[4], slot,
                                     err, sizeof err);
  if (!proxy) fail("create", err);
  uint64_t blob_bytes = sccl_emu_roce_blob_bytes();
  void *blob = calloc(1, blob_bytes), *blobs = calloc((size_t)world, blob_bytes);
  if (sccl_emu_roce_local_blob(proxy, blob, blob_bytes)) fail("blob", NULL);
  barrier_file(dir, "blob", world, rank, blob, blob_bytes, blobs);
  if (sccl_emu_roce_connect(proxy, blobs, blob_bytes * (uint64_t)world)) fail("connect", sccl_emu_roce_error(proxy));
  int ready = 1, readies[64];
  barrier_file(dir, "connected", world, rank, &ready, sizeof ready, readies);
  if (sccl_emu_roce_lane_check(proxy, 20000)) fail("lane check", sccl_emu_roce_error(proxy));
  if (sccl_emu_roce_start(proxy)) fail("start", sccl_emu_roce_error(proxy));
  volatile uint32_t *ctrl = (volatile uint32_t *)(arena + layout[3]);
  for (uint32_t seq = 1; seq <= (uint32_t)ops; ++seq) {
    uint32_t s = seq & 1u;
    uint32_t packs = 1 + (seq * 977u + 13u) % (uint32_t)(slot / 16 - 1);
    uint32_t *send = (uint32_t *)(arena + layout[2] + s * slot);
    for (uint32_t i = 0; i < packs * 4; ++i) send[i] = (uint32_t)rank * 0x01000000u + seq * 0x10000u + i;
    ctrl[1] = packs * 16;
    ctrl[4 + s] = packs * 16; /* op code 0: one-shot */
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __atomic_store_n((uint32_t *)&ctrl[0], seq, __ATOMIC_RELEASE);
    for (int p = 0; p < world; ++p) {
      if (p == rank) continue;
      for (int l = 0; l < lanes; ++l) {
        uint64_t line = ((uint64_t)p * 2 + s) * (uint64_t)lanes + (uint64_t)l;
        volatile uint32_t *flag = (volatile uint32_t *)(arena + layout[1] + line * 128);
        time_t start = time(NULL);
        while (__atomic_load_n((uint32_t *)flag, __ATOMIC_ACQUIRE) != seq) {
          if (sccl_emu_roce_failed(proxy)) fail("progress thread", sccl_emu_roce_error(proxy));
          if (time(NULL) - start > 30) {
            fprintf(stderr, "seq %u: flag of rank %d lane %d is %u\n", seq, p, l, *flag);
            fail("flag wait", NULL);
          }
        }
      }
      const uint32_t *recv = (const uint32_t *)(arena + layout[0] + ((uint64_t)p * 2 + s) * slot);
      for (uint32_t i = 0; i < packs * 4; ++i)
        if (recv[i] != (uint32_t)p * 0x01000000u + seq * 0x10000u + i) {
          fprintf(stderr, "seq %u: word %u from rank %d is %08x\n", seq, i, p, recv[i]);
          fail("payload", NULL);
        }
    }
    /* No barrier between ops: a peer writes this slot again (op seq + 2) only after it saw this
     * rank's flags of op seq + 1, which this rank posts after it checked op seq, as the kernels'
     * stream order guarantees on a session. */
  }
  printf("rank %d: %d ops exact from %d peers over %d lane(s), %" "llu writes executed\n", rank, ops, world - 1,
         lanes, (unsigned long long)sccl_emu_executed());
  int done = 1, dones[64];
  barrier_file(dir, "done", world, rank, &done, sizeof done, dones);
  sccl_emu_roce_destroy(proxy);
  sccl_emu_segment_free(arena);
  return 0;
}
