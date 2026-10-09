/* The libibverbs entry points that SIRCL's native proxy and point-to-point
 * library call (proxy_hw.c, p2p_hw.c), resolved from libibverbs.so.1 at first use, so the library loads on hosts
 * without rdma-core and binds the process's own libibverbs where one is
 * loaded. Data-path calls (ibv_post_send, ibv_poll_cq) are inline in the
 * rdma-core header and dispatch through the device context; they need no
 * symbol. Every definition here is hidden: the library exports only the NCCL
 * API and its libsircl extension. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <stddef.h>

#undef ibv_get_device_list
#undef ibv_query_port
#undef ibv_reg_mr

#define HIDDEN __attribute__((visibility("hidden")))

static struct {
  struct ibv_device **(*get_device_list)(int *);
  void (*free_device_list)(struct ibv_device **);
  const char *(*get_device_name)(struct ibv_device *);
  struct ibv_context *(*open_device)(struct ibv_device *);
  int (*close_device)(struct ibv_context *);
  int (*query_port)(struct ibv_context *, uint8_t, struct _compat_ibv_port_attr *);
  int (*query_gid)(struct ibv_context *, uint8_t, int, union ibv_gid *);
  struct ibv_pd *(*alloc_pd)(struct ibv_context *);
  int (*dealloc_pd)(struct ibv_pd *);
  struct ibv_mr *(*reg_mr)(struct ibv_pd *, void *, size_t, int);
  int (*dereg_mr)(struct ibv_mr *);
  struct ibv_cq *(*create_cq)(struct ibv_context *, int, void *, struct ibv_comp_channel *, int);
  int (*destroy_cq)(struct ibv_cq *);
  struct ibv_qp *(*create_qp)(struct ibv_pd *, struct ibv_qp_init_attr *);
  int (*modify_qp)(struct ibv_qp *, struct ibv_qp_attr *, int);
  int (*destroy_qp)(struct ibv_qp *);
  const char *(*wc_status_str)(enum ibv_wc_status);
  struct ibv_mr *(*reg_mr_iova2)(struct ibv_pd *, void *, size_t, uint64_t, unsigned int);
} v;
static int available;
static pthread_once_t once = PTHREAD_ONCE_INIT;

static void load(void) {
  void *h = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!h) return;
#define SYM(field, name)                              \
  do {                                                \
    *(void **)(&v.field) = dlsym(h, name);            \
    if (!v.field) return;                             \
  } while (0)
  SYM(get_device_list, "ibv_get_device_list");
  SYM(free_device_list, "ibv_free_device_list");
  SYM(get_device_name, "ibv_get_device_name");
  SYM(open_device, "ibv_open_device");
  SYM(close_device, "ibv_close_device");
  SYM(query_port, "ibv_query_port");
  SYM(query_gid, "ibv_query_gid");
  SYM(alloc_pd, "ibv_alloc_pd");
  SYM(dealloc_pd, "ibv_dealloc_pd");
  SYM(reg_mr, "ibv_reg_mr");
  SYM(dereg_mr, "ibv_dereg_mr");
  SYM(create_cq, "ibv_create_cq");
  SYM(destroy_cq, "ibv_destroy_cq");
  SYM(create_qp, "ibv_create_qp");
  SYM(modify_qp, "ibv_modify_qp");
  SYM(destroy_qp, "ibv_destroy_qp");
  SYM(wc_status_str, "ibv_wc_status_str");
#undef SYM
  /* Optional: libibverbs before rdma-core 29 lacks it (see ibv_reg_mr_iova2 below). */
  *(void **)(&v.reg_mr_iova2) = dlsym(h, "ibv_reg_mr_iova2");
  available = 1;
}

/* 1 when libibverbs.so.1 and every entry point above resolved. */
HIDDEN int sccl_verbs_available(void) {
  pthread_once(&once, load);
  return available;
}

#define READY_OR(value)                  \
  do {                                   \
    if (!sccl_verbs_available()) {       \
      errno = ENOSYS;                    \
      return value;                      \
    }                                    \
  } while (0)

HIDDEN struct ibv_device **ibv_get_device_list(int *n) { READY_OR(NULL); return v.get_device_list(n); }
HIDDEN void ibv_free_device_list(struct ibv_device **l) { if (sccl_verbs_available()) v.free_device_list(l); }
HIDDEN const char *ibv_get_device_name(struct ibv_device *d) { READY_OR(NULL); return v.get_device_name(d); }
HIDDEN struct ibv_context *ibv_open_device(struct ibv_device *d) { READY_OR(NULL); return v.open_device(d); }
HIDDEN int ibv_close_device(struct ibv_context *c) { READY_OR(-1); return v.close_device(c); }
HIDDEN int ibv_query_port(struct ibv_context *c, uint8_t p, struct _compat_ibv_port_attr *a) {
  READY_OR(ENOSYS);
  return v.query_port(c, p, a);
}
HIDDEN int ibv_query_gid(struct ibv_context *c, uint8_t p, int i, union ibv_gid *g) {
  READY_OR(-1);
  return v.query_gid(c, p, i, g);
}
HIDDEN struct ibv_pd *ibv_alloc_pd(struct ibv_context *c) { READY_OR(NULL); return v.alloc_pd(c); }
HIDDEN int ibv_dealloc_pd(struct ibv_pd *pd) { READY_OR(-1); return v.dealloc_pd(pd); }
HIDDEN struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *a, size_t n, int access) {
  READY_OR(NULL);
  return v.reg_mr(pd, a, n, access);
}
/* rdma-core's inline ibv_reg_mr calls ibv_reg_mr_iova2 for access flags that are not a compile-time constant
 * or that ask for an optional access flag. The native layers pass constant flags without those, so an
 * optimized build drops that branch; an unoptimized one (a CMake build without a build type) keeps the call,
 * so the symbol must resolve. Without it in the process's libibverbs the call fails with EOPNOTSUPP. */
HIDDEN struct ibv_mr *ibv_reg_mr_iova2(struct ibv_pd *pd, void *a, size_t n, uint64_t iova, unsigned int access) {
  READY_OR(NULL);
  if (!v.reg_mr_iova2) {
    errno = EOPNOTSUPP;
    return NULL;
  }
  return v.reg_mr_iova2(pd, a, n, iova, access);
}
HIDDEN int ibv_dereg_mr(struct ibv_mr *mr) { READY_OR(-1); return v.dereg_mr(mr); }
HIDDEN struct ibv_cq *ibv_create_cq(struct ibv_context *c, int n, void *ctx, struct ibv_comp_channel *ch, int vec) {
  READY_OR(NULL);
  return v.create_cq(c, n, ctx, ch, vec);
}
HIDDEN int ibv_destroy_cq(struct ibv_cq *cq) { READY_OR(-1); return v.destroy_cq(cq); }
HIDDEN struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *a) {
  READY_OR(NULL);
  return v.create_qp(pd, a);
}
HIDDEN int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *a, int mask) {
  READY_OR(ENOSYS);
  return v.modify_qp(qp, a, mask);
}
HIDDEN int ibv_destroy_qp(struct ibv_qp *qp) { READY_OR(-1); return v.destroy_qp(qp); }
HIDDEN const char *ibv_wc_status_str(enum ibv_wc_status s) {
  if (!sccl_verbs_available()) return "libibverbs unavailable";
  return v.wc_status_str(s);
}
