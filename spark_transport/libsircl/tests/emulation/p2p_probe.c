/* One rank of the CPU check of SIRCL's point-to-point native library (src/transport/sircl_p2p_proxy.c, the
 * library's emulation build) over the shared-memory verbs stand-in (tests/test_p2p_native.py). The rank's
 * arena is shared memory; host threads play the kernels' part of every channel (kernels/sircl_p2p.cu): a
 * sender thread per peer stages each item into its send slot after the sent word frees it and writes the
 * header and ready tag, a receiver thread per peer waits for each item's lane flags, checks its header, copies
 * it out and writes the consumed tag. The group sets up as a communicator does (connection records, queue-pair
 * connection, lane check, windows, progress thread).
 *
 *   p2p_probe <world> <rank> <lanes> <messages> <mode> <directory>
 *
 * Modes: `all` (a channel between every pair; every rank sends <messages> messages of 0 to 40,000 bytes to
 * every other rank while receiving theirs), `ring` (channels between ring neighbors only, and only ranks 0
 * and 1 exchange while the other ranks idle until the end), `windows` (as `all`, every lane of every channel
 * within a window of one 4096-byte chunk), `size` (rank 0 sends 100 bytes to rank 1, which expects 200: rank 1
 * records the size error and poisons its context, and every rank's progress thread must stop naming rank 1).
 * Slots are 4 of 8,192 bytes, so a message of up to 40,000 bytes is up to five items and waits for credits.
 * Ranks exchange their connection records through files in <directory>. Exit 0 when every byte from every
 * peer is exact (or, in `size`, every rank stopped naming rank 1), 1 otherwise. */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "shm_verbs.h"

int sccl_emu_p2p_abi_version(void);
unsigned sccl_emu_p2p_local_features(void);
int sccl_emu_p2p_layout(int, int, int, uint64_t, uint64_t *);
uint64_t sccl_emu_p2p_blob_bytes(void);
void *sccl_emu_p2p_create(int, int, const char *const *, int, const int *, int, const int *, int, const int *, void *,
                          uint64_t, int, uint64_t, char *, uint64_t);
int sccl_emu_p2p_local_blob(void *, void *, uint64_t);
int sccl_emu_p2p_connect(void *, const void *, uint64_t);
int sccl_emu_p2p_lane_check(void *, int);
int sccl_emu_p2p_set_windows(void *, const uint32_t *, uint32_t);
int sccl_emu_p2p_start(void *);
int sccl_emu_p2p_failed(void *);
const char *sccl_emu_p2p_error(void *);
uint64_t sccl_emu_p2p_stat(void *, int);
uint64_t sccl_emu_p2p_peer_stat(void *, int, int);
int sccl_emu_p2p_destroy(void *);

enum { L_CONTROL = 0, L_BLOCK = 1, L_RECV = 2, L_SEND = 3, L_FLAG = 4, L_DESC = 5, L_READY = 6, L_CONSUMED = 7,
       L_SENT = 8, L_CREDIT = 9, L_TOTAL = 10 };
enum { C_ERROR_TAG = 1, C_ERROR_PEER = 2, C_ERROR_LANE = 3, C_ERROR_KIND = 4, C_ERROR_EXPECTED = 5, C_ERROR_GOT = 6,
       C_POISON = 7 };
enum { SLOTS = 4, SLOT_BYTES = 8192, MAX_WORLD = 8, LINE = 128, CHUNK = 4096 };
#define LAST (1u << 31)

static int world, rank_, lanes, messages;
static const char *mode;
static uint8_t *arena;
static uint64_t layout[11];
static void *ctx;
static volatile int failures;

static void fail(const char *what, const char *detail) {
  fprintf(stderr, "rank %d failure: %s %s\n", rank_, what, detail ? detail : "");
  exit(1);
}

static void barrier_file(const char *dir, const char *name, const void *data, size_t bytes, void *all) {
  char path[512], tmp[520];
  snprintf(path, sizeof path, "%s/%s.%d", dir, name, rank_);
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
      if (tries > 600000) fail("waiting for", path);
      usleep(100);
    }
  }
}

static int has_channel(int a, int b) {
  if (a == b) return 0;
  if (!strcmp(mode, "ring")) return (a + 1) % world == b || (b + 1) % world == a;
  return 1;
}

/* Whether rank a sends to rank b in this mode. */
static int sends(int a, int b) {
  if (!has_channel(a, b)) return 0;
  if (!strcmp(mode, "size")) return a == 0 && b == 1;
  if (!strcmp(mode, "ring")) return (a == 0 && b == 1) || (a == 1 && b == 0);
  return 1;
}

static uint32_t message_bytes(int from, int to, int k) {
  if (!strcmp(mode, "size")) return 100;
  uint32_t seed = (uint32_t)(from * 131 + to * 17 + k * 7919);
  if (k % 7 == 3) return 0;
  return 1 + seed % 40000u;
}

static uint8_t pattern(int from, int to, int k, uint32_t i) {
  return (uint8_t)(from * 37 + to * 11 + k * 5 + i * 13 + (i >> 8));
}

static volatile uint32_t *word_at(uint64_t offset) { return (volatile uint32_t *)(arena + offset); }
static uint64_t block(int p) { return layout[L_CONTROL] + (uint64_t)p * layout[L_BLOCK]; }

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + t.tv_nsec / 1e9;
}

static int wait_word(volatile uint32_t *w, uint32_t value, int at_least, const char *what, int peer) {
  double start = now_s();
  for (;;) {
    uint32_t seen = __atomic_load_n((uint32_t *)w, __ATOMIC_ACQUIRE);
    if (at_least ? (int32_t)(seen - value) >= 0 : seen == value) return 0;
    if (__atomic_load_n((uint32_t *)word_at(4 * C_POISON), __ATOMIC_ACQUIRE)) return -1;
    if (now_s() - start > 60) {
      fprintf(stderr, "rank %d: %s of rank %d waited 60 s for %u, saw %u\n", rank_, what, peer, value, seen);
      failures = 1;
      return -1;
    }
  }
}

static void record_error(int peer, uint32_t lane, uint32_t kind, uint32_t tag, uint32_t expected, uint32_t got) {
  *word_at(4 * C_ERROR_PEER) = (uint32_t)peer;
  *word_at(4 * C_ERROR_LANE) = lane;
  *word_at(4 * C_ERROR_KIND) = kind;
  *word_at(4 * C_ERROR_EXPECTED) = expected;
  *word_at(4 * C_ERROR_GOT) = got;
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  __atomic_store_n((uint32_t *)word_at(4 * C_ERROR_TAG), tag, __ATOMIC_RELEASE);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  __atomic_store_n((uint32_t *)word_at(4 * C_POISON), 1u, __ATOMIC_RELEASE);
}

static void *sender(void *arg) {
  int peer = (int)(intptr_t)arg;
  uint32_t next = 0;
  uint8_t *data = malloc(40016);
  for (int k = 0; k < messages && !failures; ++k) {
    uint32_t n = message_bytes(rank_, peer, k), padded = (n + 15u) / 16u * 16u;
    for (uint32_t i = 0; i < n; ++i) data[i] = pattern(rank_, peer, k, i);
    uint32_t items = padded ? (padded + SLOT_BYTES - 1) / SLOT_BYTES : 1;
    for (uint32_t item = 0; item < items; ++item) {
      uint32_t g = next + item, m = g % SLOTS, tag = g + 1u;
      if (wait_word(word_at(block(peer) + layout[L_SENT]), tag - SLOTS, 1, "send slot", peer)) goto done;
      uint32_t start = item * SLOT_BYTES, count = padded - start < SLOT_BYTES ? padded - start : SLOT_BYTES;
      uint8_t *slot = arena + block(peer) + layout[L_SEND] + (uint64_t)m * SLOT_BYTES;
      for (uint32_t i = 0; i < count; ++i) slot[i] = start + i < n ? data[start + i] : 0;
      uint32_t header = count | (item == items - 1 ? LAST | (n % 16u) : 0u);
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
      *word_at(block(peer) + layout[L_DESC] + 4ull * m) = header;
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
      __atomic_store_n((uint32_t *)word_at(block(peer) + layout[L_READY] + 4ull * m), tag, __ATOMIC_RELEASE);
    }
    next += items;
  }
done:
  free(data);
  return NULL;
}

static void *receiver(void *arg) {
  int peer = (int)(intptr_t)arg;
  uint32_t next = 0;
  uint8_t *out = malloc(40016);
  for (int k = 0; k < messages && !failures; ++k) {
    uint32_t n = message_bytes(peer, rank_, k);
    if (!strcmp(mode, "size")) n = 200;
    uint32_t padded = (n + 15u) / 16u * 16u;
    uint32_t items = padded ? (padded + SLOT_BYTES - 1) / SLOT_BYTES : 1;
    for (uint32_t item = 0; item < items; ++item) {
      uint32_t g = next + item, m = g % SLOTS, tag = g + 1u;
      uint64_t line = block(peer) + layout[L_FLAG] + (uint64_t)m * (uint64_t)lanes * LINE;
      for (int l = 0; l < lanes; ++l)
        if (wait_word(word_at(line + (uint64_t)l * LINE), tag, 0, "lane flag", peer)) goto done;
      uint32_t start = item * SLOT_BYTES, count = padded - start < SLOT_BYTES ? padded - start : SLOT_BYTES;
      uint32_t expected = count | (item == items - 1 ? LAST | (n % 16u) : 0u);
      uint32_t got = *word_at(line + 4);
      if (got != expected) {
        if (!strcmp(mode, "size")) {
          record_error(peer, 255u, 3u, tag, expected, got);
          goto done;
        }
        fprintf(stderr, "rank %d: item %u from rank %d has header %08x, expected %08x\n", rank_, g, peer, got, expected);
        failures = 1;
        goto done;
      }
      const uint8_t *slot = arena + block(peer) + layout[L_RECV] + (uint64_t)m * SLOT_BYTES;
      memcpy(out + start, slot, count);
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
      __atomic_store_n((uint32_t *)word_at(block(peer) + layout[L_CONSUMED] + 4ull * m), tag, __ATOMIC_RELEASE);
    }
    next += items;
    for (uint32_t i = 0; i < n; ++i)
      if (out[i] != pattern(peer, rank_, k, i)) {
        fprintf(stderr, "rank %d: message %d from rank %d differs at byte %u of %u\n", rank_, k, peer, i, n);
        failures = 1;
        goto done;
      }
  }
done:
  free(out);
  return NULL;
}

int main(int argc, char **argv) {
  if (argc != 7) return 2;
  world = atoi(argv[1]);
  rank_ = atoi(argv[2]);
  lanes = atoi(argv[3]);
  messages = atoi(argv[4]);
  mode = argv[5];
  const char *dir = argv[6];
  if (world < 2 || world > MAX_WORLD || lanes < 1 || lanes > 2) return 2;
  char err[512] = {0};
  if (sccl_emu_p2p_abi_version() != 1) fail("ABI", NULL);
  /* SIRCL change LF: bit 0, p2p_destroy returns the number of verbs calls that failed. */
  if (!(sccl_emu_p2p_local_features() & 1u)) fail("local features (p2p_local_features bit 0)", NULL);
  if (sccl_emu_attach(err, sizeof err)) fail("attach", err);
  if (sccl_emu_p2p_layout(world, lanes, SLOTS, SLOT_BYTES, layout)) fail("layout", NULL);
  /* The layout of SIRCL's p2p/protocol.py. */
  uint64_t ring = (uint64_t)SLOTS * SLOT_BYTES, flag = 2 * ring, desc = flag + (uint64_t)SLOTS * lanes * LINE;
  uint64_t block_bytes = (desc + 5 * LINE + 4095) / 4096 * 4096;
  if (layout[L_CONTROL] != 4096 || layout[L_RECV] != 0 || layout[L_SEND] != ring || layout[L_FLAG] != flag ||
      layout[L_DESC] != desc || layout[L_READY] != desc + LINE || layout[L_CONSUMED] != desc + 2 * LINE ||
      layout[L_SENT] != desc + 3 * LINE || layout[L_CREDIT] != desc + 4 * LINE || layout[L_BLOCK] != block_bytes ||
      layout[L_TOTAL] != 4096 + (uint64_t)world * block_bytes)
    fail("layout", "differs from SIRCL's protocol");
  arena = sccl_emu_segment_alloc(layout[L_TOTAL], err, sizeof err);
  if (!arena) fail("arena", err);
  char names[2][64];
  const char *list[2];
  int gids[2] = {0, 0};
  for (int d = 0; d < lanes; ++d) {
    snprintf(names[d], 64, "p2pprobe%d.%d", (int)getpid(), d);
    uint8_t gid[16] = {0xfe, 0x80};
    gid[8] = (uint8_t)(getpid() >> 8);
    gid[9] = (uint8_t)getpid();
    gid[15] = (uint8_t)(d + 1);
    if (sccl_emu_add_device(names[d], gid) < 0) fail("device", names[d]);
    list[d] = names[d];
  }
  int lane_devices[MAX_WORLD * 2], channels[MAX_WORLD];
  for (int p = 0; p < world; ++p) {
    channels[p] = has_channel(rank_, p);
    for (int l = 0; l < lanes; ++l) lane_devices[p * lanes + l] = channels[p] ? l : -1;
  }
  ctx = sccl_emu_p2p_create(world, rank_, list, lanes, lane_devices, lanes, gids, 0, channels, arena, layout[L_TOTAL],
                            SLOTS, SLOT_BYTES, err, sizeof err);
  if (!ctx) fail("create", err);
  uint64_t blob_bytes = sccl_emu_p2p_blob_bytes();
  void *blob = calloc(1, blob_bytes), *blobs = calloc((size_t)world, blob_bytes);
  if (sccl_emu_p2p_local_blob(ctx, blob, blob_bytes)) fail("blob", NULL);
  barrier_file(dir, "blob", blob, blob_bytes, blobs);
  if (sccl_emu_p2p_connect(ctx, blobs, blob_bytes * (uint64_t)world)) fail("connect", sccl_emu_p2p_error(ctx));
  int one = 1, all[MAX_WORLD];
  barrier_file(dir, "connected", &one, sizeof one, all);
  if (sccl_emu_p2p_lane_check(ctx, 20000)) fail("lane check", sccl_emu_p2p_error(ctx));
  if (!strcmp(mode, "windows")) {
    uint32_t table[MAX_WORLD * 2] = {0};
    for (int p = 0; p < world; ++p)
      for (int l = 0; l < lanes; ++l) table[p * lanes + l] = channels[p] ? CHUNK : 0;
    if (sccl_emu_p2p_set_windows(ctx, table, CHUNK)) fail("windows", sccl_emu_p2p_error(ctx));
  }
  barrier_file(dir, "checked", &one, sizeof one, all);
  if (sccl_emu_p2p_start(ctx)) fail("start", sccl_emu_p2p_error(ctx));
  pthread_t threads[2 * MAX_WORLD];
  int started = 0;
  for (int p = 0; p < world; ++p) {
    if (sends(rank_, p)) pthread_create(&threads[started++], NULL, sender, (void *)(intptr_t)p);
    if (sends(p, rank_)) pthread_create(&threads[started++], NULL, receiver, (void *)(intptr_t)p);
  }
  for (int i = 0; i < started; ++i) pthread_join(threads[i], NULL);
  int code = 0;
  if (!strcmp(mode, "size")) {
    /* Every rank's progress thread stops within a second of rank 1's record, naming rank 1. */
    double start = now_s();
    while (!sccl_emu_p2p_failed(ctx) && now_s() - start < 30) usleep(1000);
    uint64_t from = sccl_emu_p2p_stat(ctx, 12);
    int ok = sccl_emu_p2p_failed(ctx) && (rank_ == 1 ? from == 0 : from == 2);
    printf("rank %d: channels stopped %d after %.3f s, abort from rank %d: %s\n", rank_, sccl_emu_p2p_failed(ctx),
           now_s() - start, (int)from - 1, sccl_emu_p2p_error(ctx));
    code = ok ? 0 : 1;
  } else {
    if (failures) code = 1;
    if (sccl_emu_p2p_failed(ctx)) {
      fprintf(stderr, "rank %d: progress thread failed: %s\n", rank_, sccl_emu_p2p_error(ctx));
      code = 1;
    }
    uint64_t posted = 0, items = 0;
    for (int p = 0; p < world; ++p) {
      if (!sends(rank_, p)) continue;
      for (int k = 0; k < messages; ++k) {
        uint32_t padded = (message_bytes(rank_, p, k) + 15u) / 16u * 16u;
        items += padded ? (padded + SLOT_BYTES - 1) / SLOT_BYTES : 1;
      }
    }
    /* Every item this rank staged was posted once its writes completed (the sent words). */
    double start = now_s();
    while (now_s() - start < 30) {
      posted = 0;
      for (int p = 0; p < world; ++p) posted += sends(rank_, p) ? sccl_emu_p2p_peer_stat(ctx, p, 5) : 0;
      if (posted == items) break;
      usleep(1000);
    }
    if (posted != items) {
      fprintf(stderr, "rank %d: %llu of %llu items completed\n", rank_, (unsigned long long)posted,
              (unsigned long long)items);
      code = 1;
    }
    printf("rank %d: %s, %d messages per channel exact, %llu items sent, window waits %llu, proven bytes %llu\n",
           rank_, mode, messages, (unsigned long long)items, (unsigned long long)sccl_emu_p2p_stat(ctx, 5),
           (unsigned long long)sccl_emu_p2p_stat(ctx, 9));
  }
  barrier_file(dir, "done", &one, sizeof one, all);
  int unreleased = sccl_emu_p2p_destroy(ctx);
  if (unreleased) {
    fprintf(stderr, "rank %d: %d verbs calls failed at destroy\n", rank_, unreleased);
    code = 1;
  }
  sccl_emu_segment_free(arena);
  return code;
}
