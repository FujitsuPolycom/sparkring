/* Kernel pack loading and launches (kernelpack.h). The fatbins are embedded by
 * the build (tools/embed_fatbin.py) as sccl_kernels_fatbin (transport pack),
 * sccl_fold_fatbin (fold pack), sccl_links_fatbin (link pack) and
 * sccl_p2p_fatbin (point-to-point pack). */
#define _GNU_SOURCE
#include "kernelpack.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

extern const unsigned char sccl_kernels_fatbin[];
extern const unsigned long sccl_kernels_fatbin_size;
extern const char sccl_kernels_fatbin_sha256[];
extern const unsigned char sccl_fold_fatbin[];
extern const char sccl_fold_fatbin_sha256[];
extern const unsigned char sccl_links_fatbin[];
extern const char sccl_links_fatbin_sha256[];
extern const unsigned char sccl_p2p_fatbin[];
extern const char sccl_p2p_fatbin_sha256[];

enum { MAX_CONTEXTS = 16 };
typedef struct {
  sccl_CUcontext ctx;
  sccl_CUmodule module, fold_module, links_module, p2p_module;
  sccl_CUfunction fn[SCCL_K_COUNT][SCCL_DT_COUNT][SCCL_KP_MAX_WORLD + 1];
  sccl_CUfunction fold[SCCL_FOLD_DTYPES][SCCL_FOLD_OPS];
  sccl_CUfunction chain[SCCL_DT_COUNT][SCCL_CHAIN_MAX_UNROLL + 1];
  sccl_CUfunction link[SCCL_LINK_KINDS][SCCL_DT_COUNT][SCCL_CHAIN_MAX_UNROLL + 1];
  sccl_CUfunction ring_reduce_two_pass[SCCL_DT_COUNT][SCCL_CHAIN_MAX_UNROLL + 1];
  sccl_CUfunction ring_exchange[SCCL_CHAIN_MAX_UNROLL + 1];
  sccl_CUfunction ring_exchange_reduce[SCCL_DT_COUNT][SCCL_CHAIN_MAX_UNROLL + 1];
  sccl_CUfunction p2p[2][SCCL_P2P_MAX_UNROLL + 1];
} loaded_pack;

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static loaded_pack packs[MAX_CONTEXTS];
static int npacks;
static _Thread_local char error_text[256];

static const char *const kind_names[SCCL_K_COUNT] = {"oneshot", "twoshot", "scatter", "allgather", "alltoall"};
static const char *const dt_names[SCCL_DT_COUNT] = {"f32", "f16", "bf16"};
/* Entry name prefixes of the link kinds; the all-gathers have no dtype in their names. */
static const char *const link_names[SCCL_LINK_KINDS] = {"sircl_link_gather", "sircl_link_scatter",
                                                        "sircl_ring_gather", "sircl_ring_scatter",
                                                        "sircl_ring_reduce"};
static int link_moves_bytes(int kind) { return kind == SCCL_LINK_GATHER || kind == SCCL_RING_GATHER; }

const char *sccl_kp_error(void) { return error_text; }
const char *sccl_kp_hash(void) { return sccl_kernels_fatbin_sha256; }
const char *sccl_kp_fold_hash(void) { return sccl_fold_fatbin_sha256; }
const char *sccl_kp_links_hash(void) { return sccl_links_fatbin_sha256; }
const char *sccl_kp_p2p_hash(void) { return sccl_p2p_fatbin_sha256; }

static loaded_pack *find(sccl_CUcontext ctx) {
  for (int i = 0; i < npacks; ++i)
    if (packs[i].ctx == ctx) return &packs[i];
  return NULL;
}

int sccl_kp_load(sccl_CUcontext ctx) {
  const sccl_cuda *cu = sccl_cuda_get();
  if (!cu) {
    snprintf(error_text, sizeof error_text, "%s", sccl_cuda_error());
    return -1;
  }
  pthread_mutex_lock(&lock);
  if (find(ctx)) {
    pthread_mutex_unlock(&lock);
    return 0;
  }
  if (npacks == MAX_CONTEXTS) {
    pthread_mutex_unlock(&lock);
    snprintf(error_text, sizeof error_text, "kernel pack loaded in %d contexts already", MAX_CONTEXTS);
    return -1;
  }
  loaded_pack *p = &packs[npacks];
  memset(p, 0, sizeof *p);
  sccl_CUresult r = cu->CtxPushCurrent(ctx);
  if (r != SCCL_CUDA_SUCCESS) {
    pthread_mutex_unlock(&lock);
    snprintf(error_text, sizeof error_text, "cuCtxPushCurrent: %s", sccl_cuda_result_text(r));
    return -1;
  }
  r = cu->ModuleLoadData(&p->module, sccl_kernels_fatbin);
  int failed = r != SCCL_CUDA_SUCCESS;
  if (failed) snprintf(error_text, sizeof error_text, "loading the kernel pack: %s", sccl_cuda_result_text(r));
  for (int k = 0; !failed && k < SCCL_K_COUNT; ++k)
    for (int d = 0; !failed && d < SCCL_DT_COUNT; ++d)
      for (int w = SCCL_KP_MIN_WORLD; !failed && w <= SCCL_KP_MAX_WORLD; ++w) {
        int bytes_only = k == SCCL_K_ALLGATHER || k == SCCL_K_ALLTOALL;
        if (bytes_only && d) continue;
        char name[64];
        if (bytes_only)
          snprintf(name, sizeof name, "sircl_%s_w%d", kind_names[k], w);
        else
          snprintf(name, sizeof name, "sircl_%s_%s_w%d", kind_names[k], dt_names[d], w);
        r = cu->ModuleGetFunction(&p->fn[k][d][w], p->module, name);
        if (r != SCCL_CUDA_SUCCESS) {
          failed = 1;
          snprintf(error_text, sizeof error_text, "kernel pack has no entry %s: %s", name, sccl_cuda_result_text(r));
        }
      }
  if (!failed) {
    r = cu->ModuleLoadData(&p->fold_module, sccl_fold_fatbin);
    failed = r != SCCL_CUDA_SUCCESS;
    if (failed) snprintf(error_text, sizeof error_text, "loading the fold pack: %s", sccl_cuda_result_text(r));
  }
  for (int d = 0; !failed && d < SCCL_FOLD_DTYPES; ++d)
    for (int o = 0; !failed && o < SCCL_FOLD_OPS; ++o) {
      char name[64];
      snprintf(name, sizeof name, "sircl_fold_d%d_o%d", d, o);
      r = cu->ModuleGetFunction(&p->fold[d][o], p->fold_module, name);
      if (r != SCCL_CUDA_SUCCESS) {
        failed = 1;
        snprintf(error_text, sizeof error_text, "fold pack has no entry %s: %s", name, sccl_cuda_result_text(r));
      }
    }
  if (!failed) {
    r = cu->ModuleLoadData(&p->links_module, sccl_links_fatbin);
    failed = r != SCCL_CUDA_SUCCESS;
    if (failed) snprintf(error_text, sizeof error_text, "loading the link pack: %s", sccl_cuda_result_text(r));
  }
  for (int d = 0; !failed && d < SCCL_DT_COUNT; ++d)
    for (int u = 1; !failed && u <= SCCL_CHAIN_MAX_UNROLL; ++u) {
      char name[64];
      snprintf(name, sizeof name, "sircl_chain_%s_u%d", dt_names[d], u);
      r = cu->ModuleGetFunction(&p->chain[d][u], p->links_module, name);
      if (r != SCCL_CUDA_SUCCESS) {
        failed = 1;
        snprintf(error_text, sizeof error_text, "link pack has no entry %s: %s", name, sccl_cuda_result_text(r));
      }
    }
  for (int k = 0; !failed && k < SCCL_LINK_KINDS; ++k)
    for (int d = 0; !failed && d < SCCL_DT_COUNT; ++d)
      for (int u = 1; !failed && u <= SCCL_CHAIN_MAX_UNROLL; ++u) {
        char name[64];
        if (link_moves_bytes(k))
          snprintf(name, sizeof name, "%s_u%d", link_names[k], u);
        else
          snprintf(name, sizeof name, "%s_%s_u%d", link_names[k], dt_names[d], u);
        r = cu->ModuleGetFunction(&p->link[k][d][u], p->links_module, name);
        if (r != SCCL_CUDA_SUCCESS) {
          failed = 1;
          snprintf(error_text, sizeof error_text, "link pack has no entry %s: %s", name, sccl_cuda_result_text(r));
        }
      }
  for (int d = 0; !failed && d < SCCL_DT_COUNT; ++d)
    for (int u = 1; !failed && u <= SCCL_CHAIN_MAX_UNROLL; ++u) {
      char name[64];
      snprintf(name, sizeof name, "sircl_ring_reduce_two_pass_%s_u%d", dt_names[d], u);
      r = cu->ModuleGetFunction(&p->ring_reduce_two_pass[d][u], p->links_module, name);
      if (r != SCCL_CUDA_SUCCESS) {
        failed = 1;
        snprintf(error_text, sizeof error_text, "link pack has no entry %s: %s", name, sccl_cuda_result_text(r));
      }
    }
  for (int u = 1; !failed && u <= SCCL_CHAIN_MAX_UNROLL; ++u) {
    char name[64];
    snprintf(name, sizeof name, "sircl_ring_exchange_u%d", u);
    r = cu->ModuleGetFunction(&p->ring_exchange[u], p->links_module, name);
    if (r != SCCL_CUDA_SUCCESS) {
      failed = 1;
      snprintf(error_text, sizeof error_text, "link pack has no entry %s: %s", name, sccl_cuda_result_text(r));
    }
  }
  for (int d = 0; !failed && d < SCCL_DT_COUNT; ++d)
    for (int u = 1; !failed && u <= SCCL_CHAIN_MAX_UNROLL; ++u) {
      char name[64];
      snprintf(name, sizeof name, "sircl_ring_exchange_reduce_%s_u%d", dt_names[d], u);
      r = cu->ModuleGetFunction(&p->ring_exchange_reduce[d][u], p->links_module, name);
      if (r != SCCL_CUDA_SUCCESS) {
        failed = 1;
        snprintf(error_text, sizeof error_text, "link pack has no entry %s: %s", name, sccl_cuda_result_text(r));
      }
    }
  if (!failed) {
    r = cu->ModuleLoadData(&p->p2p_module, sccl_p2p_fatbin);
    failed = r != SCCL_CUDA_SUCCESS;
    if (failed) snprintf(error_text, sizeof error_text, "loading the point-to-point pack: %s", sccl_cuda_result_text(r));
  }
  for (int s = 0; !failed && s < 2; ++s)
    for (int u = 1; !failed && u <= SCCL_P2P_MAX_UNROLL; ++u) {
      char name[64];
      snprintf(name, sizeof name, "sircl_p2p_%s_u%d", s ? "send" : "recv", u);
      r = cu->ModuleGetFunction(&p->p2p[s][u], p->p2p_module, name);
      if (r != SCCL_CUDA_SUCCESS) {
        failed = 1;
        snprintf(error_text, sizeof error_text, "point-to-point pack has no entry %s: %s", name,
                 sccl_cuda_result_text(r));
      }
    }
  if (failed && p->p2p_module) cu->ModuleUnload(p->p2p_module);
  if (failed && p->links_module) cu->ModuleUnload(p->links_module);
  if (failed && p->fold_module) cu->ModuleUnload(p->fold_module);
  if (failed && p->module) cu->ModuleUnload(p->module);
  sccl_CUcontext popped;
  cu->CtxPopCurrent(&popped);
  if (!failed) {
    p->ctx = ctx;
    ++npacks;
  }
  pthread_mutex_unlock(&lock);
  return failed ? -1 : 0;
}

int sccl_kp_function(sccl_CUcontext ctx, int kind, int dtype, int world, sccl_CUfunction *out) {
  if (kind < 0 || kind >= SCCL_K_COUNT || dtype < 0 || dtype >= SCCL_DT_COUNT || world < SCCL_KP_MIN_WORLD ||
      world > SCCL_KP_MAX_WORLD) {
    snprintf(error_text, sizeof error_text, "no kernel for kind %d, dtype %d, world %d", kind, dtype, world);
    return -1;
  }
  if (kind == SCCL_K_ALLGATHER || kind == SCCL_K_ALLTOALL) dtype = 0;
  pthread_mutex_lock(&lock);
  loaded_pack *p = find(ctx);
  if (p) *out = p->fn[kind][dtype][world];
  pthread_mutex_unlock(&lock);
  if (!p) {
    snprintf(error_text, sizeof error_text, "the kernel pack is not loaded in this context");
    return -1;
  }
  return 0;
}

int sccl_kp_fold_function(sccl_CUcontext ctx, int datatype, int op, sccl_CUfunction *out) {
  if (datatype < 0 || datatype >= SCCL_FOLD_DTYPES || op < 0 || op >= SCCL_FOLD_OPS) {
    snprintf(error_text, sizeof error_text, "no fold for datatype %d, op %d", datatype, op);
    return -1;
  }
  pthread_mutex_lock(&lock);
  loaded_pack *p = find(ctx);
  if (p) *out = p->fold[datatype][op];
  pthread_mutex_unlock(&lock);
  if (!p) {
    snprintf(error_text, sizeof error_text, "the kernel packs are not loaded in this context");
    return -1;
  }
  return 0;
}

int sccl_kp_chain_function(sccl_CUcontext ctx, int dtype, int unroll, sccl_CUfunction *out) {
  if (dtype < 0 || dtype >= SCCL_DT_COUNT || unroll < 1 || unroll > SCCL_CHAIN_MAX_UNROLL) {
    snprintf(error_text, sizeof error_text, "no chain all-reduce for dtype %d, unroll %d", dtype, unroll);
    return -1;
  }
  pthread_mutex_lock(&lock);
  loaded_pack *p = find(ctx);
  if (p) *out = p->chain[dtype][unroll];
  pthread_mutex_unlock(&lock);
  if (!p) {
    snprintf(error_text, sizeof error_text, "the kernel packs are not loaded in this context");
    return -1;
  }
  return 0;
}

sccl_CUresult sccl_kp_launch_chain(sccl_CUfunction function, const sccl_chain_args *args, unsigned grid,
                                   unsigned threads, sccl_CUstream stream) {
  const sccl_cuda *cu = sccl_cuda_get();
  sccl_chain_args a = *args;
  void *params[] = {&a.input, &a.output, &a.a_packs, &a.b_packs, &a.chunk_packs, &a.chain_base, &a.counters,
                    &a.ctrl_base, &a.poison, &a.spin_limit, &a.trace_base, &a.trace_capacity, &a.world, &a.index,
                    &a.prev, &a.next, &a.rank, &a.lanes, &a.slots, &a.slot_bytes, &a.blocks_per_role};
  return cu->LaunchKernel(function, grid, 1, 1, threads, 1, 1, 0, stream, params, NULL);
}

int sccl_kp_link_function(sccl_CUcontext ctx, int kind, int dtype, int unroll, sccl_CUfunction *out) {
  if (kind < 0 || kind >= SCCL_LINK_KINDS || dtype < 0 || dtype >= SCCL_DT_COUNT || unroll < 1 ||
      unroll > SCCL_CHAIN_MAX_UNROLL) {
    snprintf(error_text, sizeof error_text, "no link collective for kind %d, dtype %d, unroll %d", kind, dtype,
             unroll);
    return -1;
  }
  pthread_mutex_lock(&lock);
  loaded_pack *p = find(ctx);
  if (p) *out = p->link[kind][dtype][unroll];
  pthread_mutex_unlock(&lock);
  if (!p) {
    snprintf(error_text, sizeof error_text, "the kernel packs are not loaded in this context");
    return -1;
  }
  return 0;
}

int sccl_kp_ring_reduce_two_pass_function(sccl_CUcontext ctx, int dtype, int unroll, sccl_CUfunction *out) {
  if (dtype < 0 || dtype >= SCCL_DT_COUNT || unroll < 1 || unroll > SCCL_CHAIN_MAX_UNROLL) {
    snprintf(error_text, sizeof error_text, "no two-pass ring all-reduce for dtype %d, unroll %d", dtype, unroll);
    return -1;
  }
  pthread_mutex_lock(&lock);
  loaded_pack *p = find(ctx);
  if (p) *out = p->ring_reduce_two_pass[dtype][unroll];
  pthread_mutex_unlock(&lock);
  if (!p) {
    snprintf(error_text, sizeof error_text, "the kernel packs are not loaded in this context");
    return -1;
  }
  return 0;
}

int sccl_kp_ring_exchange_function(sccl_CUcontext ctx, int unroll, sccl_CUfunction *out) {
  if (unroll < 1 || unroll > SCCL_CHAIN_MAX_UNROLL) {
    snprintf(error_text, sizeof error_text, "no pair exchange for unroll %d", unroll);
    return -1;
  }
  pthread_mutex_lock(&lock);
  loaded_pack *p = find(ctx);
  if (p) *out = p->ring_exchange[unroll];
  pthread_mutex_unlock(&lock);
  if (!p) {
    snprintf(error_text, sizeof error_text, "the kernel packs are not loaded in this context");
    return -1;
  }
  return 0;
}

int sccl_kp_ring_exchange_reduce_function(sccl_CUcontext ctx, int dtype, int unroll, sccl_CUfunction *out) {
  if (dtype < 0 || dtype >= SCCL_DT_COUNT || unroll < 1 || unroll > SCCL_CHAIN_MAX_UNROLL) {
    snprintf(error_text, sizeof error_text, "no pair reduce exchange for dtype %d, unroll %d", dtype, unroll);
    return -1;
  }
  pthread_mutex_lock(&lock);
  loaded_pack *p = find(ctx);
  if (p) *out = p->ring_exchange_reduce[dtype][unroll];
  pthread_mutex_unlock(&lock);
  if (!p) {
    snprintf(error_text, sizeof error_text, "the kernel packs are not loaded in this context");
    return -1;
  }
  return 0;
}

sccl_CUresult sccl_kp_launch_link(sccl_CUfunction function, const sccl_link_args *args, unsigned grid,
                                  unsigned threads, sccl_CUstream stream) {
  const sccl_cuda *cu = sccl_cuda_get();
  sccl_link_args a = *args;
  void *params[] = {&a};
  return cu->LaunchKernel(function, grid, 1, 1, threads, 1, 1, 0, stream, params, NULL);
}

int sccl_kp_p2p_function(sccl_CUcontext ctx, int send, int unroll, sccl_CUfunction *out) {
  if (unroll < 1 || unroll > SCCL_P2P_MAX_UNROLL) {
    snprintf(error_text, sizeof error_text, "no point-to-point kernel for unroll %d", unroll);
    return -1;
  }
  pthread_mutex_lock(&lock);
  loaded_pack *p = find(ctx);
  if (p) *out = p->p2p[send ? 1 : 0][unroll];
  pthread_mutex_unlock(&lock);
  if (!p) {
    snprintf(error_text, sizeof error_text, "the kernel packs are not loaded in this context");
    return -1;
  }
  return 0;
}

sccl_CUresult sccl_kp_launch_p2p(sccl_CUfunction function, const sccl_p2p_args *args, unsigned grid,
                                 unsigned threads, sccl_CUstream stream) {
  const sccl_cuda *cu = sccl_cuda_get();
  sccl_p2p_args a = *args;
  void *params[] = {&a};
  return cu->LaunchKernel(function, grid, 1, 1, threads, 1, 1, 0, stream, params, NULL);
}

sccl_CUresult sccl_kp_launch_fold(sccl_CUfunction function, uint64_t rows, int64_t pitch, uint64_t out, int64_t count,
                                  int32_t world, unsigned grid, unsigned threads, sccl_CUstream stream) {
  const sccl_cuda *cu = sccl_cuda_get();
  void *params[] = {&rows, &pitch, &out, &count, &world};
  return cu->LaunchKernel(function, grid, 1, 1, threads, 1, 1, 0, stream, params, NULL);
}

sccl_CUresult sccl_kp_launch_allreduce(sccl_CUfunction function, int algorithm, const sccl_allreduce_args *args,
                                       unsigned grid, unsigned threads, sccl_CUstream stream) {
  const sccl_cuda *cu = sccl_cuda_get();
  sccl_allreduce_args a = *args;
  void *oneshot_params[] = {&a.input, &a.output, &a.size_packs, &a.nbytes, &a.recv_base, &a.flag_base,
                            &a.send_base, &a.ctrl_base, &a.slot_bytes, &a.epoch, &a.stage_counter,
                            &a.tail_counter, &a.poison, &a.arrival, &a.spin_limit, &a.rank, &a.lanes,
                            &a.one_block};
  void *twoshot_params[] = {&a.input, &a.output, &a.size_packs, &a.nbytes, &a.recv_base, &a.flag_base,
                            &a.send_base, &a.ctrl_base, &a.slot_bytes, &a.epoch, &a.stage_counter,
                            &a.phase_counter, &a.tail_counter, &a.poison, &a.arrival, &a.spin_limit, &a.rank,
                            &a.lanes, &a.one_block};
  return cu->LaunchKernel(function, grid, 1, 1, threads, 1, 1, 0, stream,
                          algorithm == SCCL_ALG_TWOSHOT ? twoshot_params : oneshot_params, NULL);
}

sccl_CUresult sccl_kp_launch_allgather(sccl_CUfunction function, const sccl_allgather_args *args, unsigned grid,
                                       unsigned threads, sccl_CUstream stream) {
  const sccl_cuda *cu = sccl_cuda_get();
  sccl_allgather_args a = *args;
  void *params[] = {&a.input, &a.output, &a.shard_packs, &a.nbytes, &a.tile_cols, &a.in_row_stride,
                    &a.out_row_stride, &a.out_src_stride, &a.recv_base, &a.flag_base, &a.send_base, &a.ctrl_base,
                    &a.slot_bytes, &a.epoch, &a.stage_counter, &a.tail_counter, &a.poison, &a.arrival,
                    &a.spin_limit, &a.rank, &a.lanes, &a.one_block};
  return cu->LaunchKernel(function, grid, 1, 1, threads, 1, 1, 0, stream, params, NULL);
}

sccl_CUresult sccl_kp_launch_scatter(sccl_CUfunction function, const sccl_scatter_args *args, unsigned grid,
                                     unsigned threads, sccl_CUstream stream) {
  const sccl_cuda *cu = sccl_cuda_get();
  sccl_scatter_args a = *args;
  void *params[] = {&a.input, &a.output, &a.size_packs, &a.nbytes, &a.chunk_packs, &a.src_stride, &a.dst_stride,
                    &a.recv_base, &a.flag_base, &a.send_base, &a.ctrl_base, &a.slot_bytes, &a.epoch,
                    &a.stage_counter, &a.tail_counter, &a.poison, &a.spin_limit, &a.rank, &a.lanes};
  return cu->LaunchKernel(function, grid, 1, 1, threads, 1, 1, 0, stream, params, NULL);
}
