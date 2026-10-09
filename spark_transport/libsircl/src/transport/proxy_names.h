/* Renames SIRCL's native proxy entry points (sircl_roce_proxy.c) with the
 * transport tag SCCL_PROXY_TAG, so the proxy compiles once against the real
 * libibverbs (tag hw) and once against the shared-memory verbs stand-in (tag
 * emu) into one library. With SCCL_RENAME_VERBS the verbs entry points are
 * renamed too (the stand-in's definitions and the emu proxy's calls). */
#ifndef SCCL_PROXY_NAMES_H
#define SCCL_PROXY_NAMES_H
#define SCCL_CAT2(a, b) a##b
#define SCCL_CAT(a, b) SCCL_CAT2(a, b)
#define SCCL_PN(name) SCCL_CAT(SCCL_CAT(sccl_, SCCL_PROXY_TAG), SCCL_CAT(_, name))
#define roce_abi_version SCCL_PN(roce_abi_version)
#define roce_layout SCCL_PN(roce_layout)
#define roce_blob_bytes SCCL_PN(roce_blob_bytes)
#define roce_flag_lines SCCL_PN(roce_flag_lines)
#define roce_store_release_u32 SCCL_PN(roce_store_release_u32)
#define roce_load_acquire_u32 SCCL_PN(roce_load_acquire_u32)
#define roce_create SCCL_PN(roce_create)
#define roce_local_blob SCCL_PN(roce_local_blob)
#define roce_connect SCCL_PN(roce_connect)
#define roce_set_forward SCCL_PN(roce_set_forward)
#define roce_chain_layout SCCL_PN(roce_chain_layout)
#define roce_set_chain SCCL_PN(roce_set_chain)
#define roce_link_layout SCCL_PN(roce_link_layout)
#define roce_set_links SCCL_PN(roce_set_links)
#define roce_lane_check SCCL_PN(roce_lane_check)
#define roce_start SCCL_PN(roce_start)
#define roce_stop SCCL_PN(roce_stop)
#define roce_failed SCCL_PN(roce_failed)
#define roce_error SCCL_PN(roce_error)
#define roce_set_trace SCCL_PN(roce_set_trace)
#define roce_trace_take SCCL_PN(roce_trace_take)
#define roce_tracing SCCL_PN(roce_tracing)
#define roce_trace_read SCCL_PN(roce_trace_read)
#define roce_stat SCCL_PN(roce_stat)
#define roce_hca_stat SCCL_PN(roce_hca_stat)
#define roce_destroy SCCL_PN(roce_destroy)
#define roce_local_features SCCL_PN(roce_local_features)
#ifdef SCCL_RENAME_VERBS
#define ibv_get_device_list sccl_emu_ibv_get_device_list
#define ibv_free_device_list sccl_emu_ibv_free_device_list
#define ibv_get_device_name sccl_emu_ibv_get_device_name
#define ibv_open_device sccl_emu_ibv_open_device
#define ibv_close_device sccl_emu_ibv_close_device
#define ibv_query_port sccl_emu_ibv_query_port
#define ibv_query_gid sccl_emu_ibv_query_gid
#define ibv_alloc_pd sccl_emu_ibv_alloc_pd
#define ibv_dealloc_pd sccl_emu_ibv_dealloc_pd
#define ibv_reg_mr sccl_emu_ibv_reg_mr
#define ibv_dereg_mr sccl_emu_ibv_dereg_mr
#define ibv_create_cq sccl_emu_ibv_create_cq
#define ibv_destroy_cq sccl_emu_ibv_destroy_cq
#define ibv_create_qp sccl_emu_ibv_create_qp
#define ibv_modify_qp sccl_emu_ibv_modify_qp
#define ibv_destroy_qp sccl_emu_ibv_destroy_qp
#define ibv_post_send sccl_emu_ibv_post_send
#define ibv_poll_cq sccl_emu_ibv_poll_cq
#define ibv_wc_status_str sccl_emu_ibv_wc_status_str
#endif
#endif
