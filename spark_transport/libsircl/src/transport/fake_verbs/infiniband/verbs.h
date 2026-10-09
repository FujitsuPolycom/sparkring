/*
 * Fake libibverbs for CPU tests of the SIRCL native proxy.
 *
 * Declares the subset of <infiniband/verbs.h> the proxy uses, with the same
 * names, field names and enumerator values, so sircl_proxy.c compiles
 * unchanged against either header. The implementation (fake_verbs.c) moves
 * bytes between registered regions of one process and models cables, NIC
 * functions and tagged relays; see fake_verbs.h for its control interface.
 * Nothing here talks to hardware.
 */
#ifndef SIRCL_FAKE_INFINIBAND_VERBS_H
#define SIRCL_FAKE_INFINIBAND_VERBS_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define FV_API __declspec(dllexport)
#else
#define FV_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct ibv_device {
    char name[64];
    int fv_index;
};

struct ibv_context {
    struct ibv_device *device;
    int fv_index;
};

struct ibv_pd {
    struct ibv_context *context;
};

struct ibv_mr {
    struct ibv_context *context;
    struct ibv_pd *pd;
    void *addr;
    size_t length;
    uint32_t handle;
    uint32_t lkey;
    uint32_t rkey;
};

struct ibv_comp_channel;

struct ibv_cq {
    struct ibv_context *context;
    int cqe;
    int fv_index;
};

union ibv_gid {
    uint8_t raw[16];
    struct {
        uint64_t subnet_prefix;
        uint64_t interface_id;
    } global;
};

enum ibv_port_state {
    IBV_PORT_NOP = 0,
    IBV_PORT_DOWN = 1,
    IBV_PORT_INIT = 2,
    IBV_PORT_ARMED = 3,
    IBV_PORT_ACTIVE = 4,
    IBV_PORT_ACTIVE_DEFER = 5,
};

enum ibv_mtu {
    IBV_MTU_256 = 1,
    IBV_MTU_512 = 2,
    IBV_MTU_1024 = 3,
    IBV_MTU_2048 = 4,
    IBV_MTU_4096 = 5,
};

struct ibv_port_attr {
    enum ibv_port_state state;
    enum ibv_mtu max_mtu;
    enum ibv_mtu active_mtu;
    int gid_tbl_len;
    uint16_t lid;
};

enum ibv_access_flags {
    IBV_ACCESS_LOCAL_WRITE = 1,
    IBV_ACCESS_REMOTE_WRITE = 1 << 1,
    IBV_ACCESS_REMOTE_READ = 1 << 2,
    IBV_ACCESS_REMOTE_ATOMIC = 1 << 3,
};

enum ibv_qp_type {
    IBV_QPT_RC = 2,
    IBV_QPT_UC = 3,
    IBV_QPT_UD = 4,
};

struct ibv_qp_cap {
    uint32_t max_send_wr;
    uint32_t max_recv_wr;
    uint32_t max_send_sge;
    uint32_t max_recv_sge;
    uint32_t max_inline_data;
};

struct ibv_qp_init_attr {
    void *qp_context;
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    void *srq;
    struct ibv_qp_cap cap;
    enum ibv_qp_type qp_type;
    int sq_sig_all;
};

enum ibv_qp_state {
    IBV_QPS_RESET,
    IBV_QPS_INIT,
    IBV_QPS_RTR,
    IBV_QPS_RTS,
    IBV_QPS_SQD,
    IBV_QPS_SQE,
    IBV_QPS_ERR,
};

enum ibv_qp_attr_mask {
    IBV_QP_STATE = 1 << 0,
    IBV_QP_CUR_STATE = 1 << 1,
    IBV_QP_EN_SQD_ASYNC_NOTIFY = 1 << 2,
    IBV_QP_ACCESS_FLAGS = 1 << 3,
    IBV_QP_PKEY_INDEX = 1 << 4,
    IBV_QP_PORT = 1 << 5,
    IBV_QP_QKEY = 1 << 6,
    IBV_QP_AV = 1 << 7,
    IBV_QP_PATH_MTU = 1 << 8,
    IBV_QP_TIMEOUT = 1 << 9,
    IBV_QP_RETRY_CNT = 1 << 10,
    IBV_QP_RNR_RETRY = 1 << 11,
    IBV_QP_RQ_PSN = 1 << 12,
    IBV_QP_MAX_QP_RD_ATOMIC = 1 << 13,
    IBV_QP_ALT_PATH = 1 << 14,
    IBV_QP_MIN_RNR_TIMER = 1 << 15,
    IBV_QP_SQ_PSN = 1 << 16,
    IBV_QP_MAX_DEST_RD_ATOMIC = 1 << 17,
    IBV_QP_PATH_MIG_STATE = 1 << 18,
    IBV_QP_CAP = 1 << 19,
    IBV_QP_DEST_QPN = 1 << 20,
};

struct ibv_global_route {
    union ibv_gid dgid;
    uint32_t flow_label;
    uint8_t sgid_index;
    uint8_t hop_limit;
    uint8_t traffic_class;
};

struct ibv_ah_attr {
    struct ibv_global_route grh;
    uint16_t dlid;
    uint8_t sl;
    uint8_t src_path_bits;
    uint8_t static_rate;
    uint8_t is_global;
    uint8_t port_num;
};

struct ibv_qp_attr {
    enum ibv_qp_state qp_state;
    enum ibv_qp_state cur_qp_state;
    enum ibv_mtu path_mtu;
    int path_mig_state;
    uint32_t qkey;
    uint32_t rq_psn;
    uint32_t sq_psn;
    uint32_t dest_qp_num;
    unsigned int qp_access_flags;
    struct ibv_qp_cap cap;
    struct ibv_ah_attr ah_attr;
    struct ibv_ah_attr alt_ah_attr;
    uint16_t pkey_index;
    uint16_t alt_pkey_index;
    uint8_t en_sqd_async_notify;
    uint8_t sq_draining;
    uint8_t max_rd_atomic;
    uint8_t max_dest_rd_atomic;
    uint8_t min_rnr_timer;
    uint8_t port_num;
    uint8_t timeout;
    uint8_t retry_cnt;
    uint8_t rnr_retry;
    uint8_t alt_port_num;
    uint8_t alt_timeout;
    uint32_t rate_limit;
};

struct ibv_qp {
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    uint32_t qp_num;
    enum ibv_qp_state state;
    int fv_index;
};

enum ibv_wr_opcode {
    IBV_WR_RDMA_WRITE = 0,
    IBV_WR_RDMA_WRITE_WITH_IMM = 1,
    IBV_WR_SEND = 2,
};

enum ibv_send_flags {
    IBV_SEND_FENCE = 1,
    IBV_SEND_SIGNALED = 1 << 1,
    IBV_SEND_SOLICITED = 1 << 2,
    IBV_SEND_INLINE = 1 << 3,
};

struct ibv_sge {
    uint64_t addr;
    uint32_t length;
    uint32_t lkey;
};

struct ibv_send_wr {
    uint64_t wr_id;
    struct ibv_send_wr *next;
    struct ibv_sge *sg_list;
    int num_sge;
    enum ibv_wr_opcode opcode;
    unsigned int send_flags;
    uint32_t imm_data;
    union {
        struct {
            uint64_t remote_addr;
            uint32_t rkey;
        } rdma;
    } wr;
};

enum ibv_wc_status {
    IBV_WC_SUCCESS = 0,
    IBV_WC_LOC_LEN_ERR = 1,
    IBV_WC_LOC_QP_OP_ERR = 2,
    IBV_WC_LOC_EEC_OP_ERR = 3,
    IBV_WC_LOC_PROT_ERR = 4,
    IBV_WC_WR_FLUSH_ERR = 5,
    IBV_WC_MW_BIND_ERR = 6,
    IBV_WC_BAD_RESP_ERR = 7,
    IBV_WC_LOC_ACCESS_ERR = 8,
    IBV_WC_REM_INV_REQ_ERR = 9,
    IBV_WC_REM_ACCESS_ERR = 10,
    IBV_WC_REM_OP_ERR = 11,
    IBV_WC_RETRY_EXC_ERR = 12,
};

enum ibv_wc_opcode {
    IBV_WC_SEND = 0,
    IBV_WC_RDMA_WRITE = 1,
};

struct ibv_wc {
    uint64_t wr_id;
    enum ibv_wc_status status;
    enum ibv_wc_opcode opcode;
    uint32_t vendor_err;
    uint32_t byte_len;
    uint32_t imm_data;
    uint32_t qp_num;
    uint32_t src_qp;
    unsigned int wc_flags;
};

FV_API struct ibv_device **ibv_get_device_list(int *num_devices);
FV_API void ibv_free_device_list(struct ibv_device **list);
FV_API const char *ibv_get_device_name(struct ibv_device *device);
FV_API struct ibv_context *ibv_open_device(struct ibv_device *device);
FV_API int ibv_close_device(struct ibv_context *context);
FV_API int ibv_query_port(struct ibv_context *context, uint8_t port_num,
                          struct ibv_port_attr *port_attr);
FV_API int ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index,
                         union ibv_gid *gid);
FV_API struct ibv_pd *ibv_alloc_pd(struct ibv_context *context);
FV_API int ibv_dealloc_pd(struct ibv_pd *pd);
FV_API struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access);
FV_API int ibv_dereg_mr(struct ibv_mr *mr);
FV_API struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe, void *cq_context,
                                    struct ibv_comp_channel *channel, int comp_vector);
FV_API int ibv_destroy_cq(struct ibv_cq *cq);
FV_API struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr);
FV_API int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
FV_API int ibv_destroy_qp(struct ibv_qp *qp);
FV_API int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr);
FV_API int ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc);
FV_API const char *ibv_wc_status_str(enum ibv_wc_status status);

#ifdef __cplusplus
}
#endif

#endif /* SIRCL_FAKE_INFINIBAND_VERBS_H */
