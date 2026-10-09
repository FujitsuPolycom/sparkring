/* Renames the entry points of SIRCL's point-to-point native library (sircl_p2p_proxy.c, a byte-identical
 * copy of SIRCL's p2p/_p2p_proxy.c) with the transport tag SCCL_PROXY_TAG, as proxy_names.h does for the
 * collective proxy: the library compiles once against the real libibverbs (tag hw, sccl_hw_p2p_*) and once
 * against the shared-memory verbs stand-in (tag emu, sccl_emu_p2p_*, with SCCL_RENAME_VERBS), and neither
 * copy exports an unprefixed p2p_* symbol that could bind to SIRCL's own build of the library in the same
 * process. p2p_local_features is named for the source that defines it (SIRCL change LF); the copy vendored
 * here predates it, so the name is unused until that source is vendored. */
#ifndef SCCL_P2P_NAMES_H
#define SCCL_P2P_NAMES_H
#include "proxy_names.h"
#define p2p_abi_version SCCL_PN(p2p_abi_version)
#define p2p_layout SCCL_PN(p2p_layout)
#define p2p_blob_bytes SCCL_PN(p2p_blob_bytes)
#define p2p_store_release_u32 SCCL_PN(p2p_store_release_u32)
#define p2p_load_acquire_u32 SCCL_PN(p2p_load_acquire_u32)
#define p2p_create SCCL_PN(p2p_create)
#define p2p_local_blob SCCL_PN(p2p_local_blob)
#define p2p_connect SCCL_PN(p2p_connect)
#define p2p_lane_check SCCL_PN(p2p_lane_check)
#define p2p_set_windows SCCL_PN(p2p_set_windows)
#define p2p_start SCCL_PN(p2p_start)
#define p2p_stop SCCL_PN(p2p_stop)
#define p2p_failed SCCL_PN(p2p_failed)
#define p2p_error SCCL_PN(p2p_error)
#define p2p_stat SCCL_PN(p2p_stat)
#define p2p_peer_stat SCCL_PN(p2p_peer_stat)
#define p2p_hca_stat SCCL_PN(p2p_hca_stat)
#define p2p_destroy SCCL_PN(p2p_destroy)
#define p2p_test_set_base SCCL_PN(p2p_test_set_base)
#define p2p_test_qp_num SCCL_PN(p2p_test_qp_num)
#define p2p_local_features SCCL_PN(p2p_local_features)
#endif
