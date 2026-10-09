#ifndef SIRCL_FABRIC_H
#define SIRCL_FABRIC_H

/* Implemented offline planning. Hardware qualification is pending. */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SIRCL_FABRIC_MAX_WORLD 16
#define SIRCL_FABRIC_MAX_LANES 2
#define SIRCL_FABRIC_MAX_CABLES 16
#define SIRCL_FABRIC_DEFAULT_MAX_RELAYS 3
#define SIRCL_PROTOCOL_MAX_PHASES 8
#define SIRCL_PROTOCOL_OP_BYTES_MASK UINT32_C(0x3fffffff)

typedef struct { int code; char message[192]; } sircl_fabric_error;
enum { SIRCL_FABRIC_INVALID = 1, SIRCL_FABRIC_UNREACHABLE = 2,
       SIRCL_FABRIC_MISMATCH = 3, SIRCL_FABRIC_LIMIT = 4 };
/* The role value is port * 2 + secondary. Device names are case sensitive. */
enum { SIRCL_CW_PRIMARY = 0, SIRCL_CW_SECONDARY = 1,
       SIRCL_CCW_PRIMARY = 2, SIRCL_CCW_SECONDARY = 3 };
typedef struct { int a, a_port, b, b_port; } sircl_cable;
typedef struct { int src, src_port, dst, dst_port; } sircl_fabric_step;
typedef struct {
  int cable_count, node_count, world, cycle;
  sircl_cable cables[SIRCL_FABRIC_MAX_CABLES];
  int nodes[SIRCL_FABRIC_MAX_WORLD], positions[SIRCL_FABRIC_MAX_WORLD];
} sircl_fabric_layout;
typedef struct {
  uint8_t count[SIRCL_FABRIC_MAX_WORLD];
  uint8_t role[SIRCL_FABRIC_MAX_WORLD][SIRCL_FABRIC_MAX_LANES];
} sircl_route_map;
typedef struct {
  int local_role, remote_role, hops;
  sircl_fabric_step steps[SIRCL_FABRIC_MAX_WORLD];
} sircl_lane_route;
typedef struct {
  sircl_fabric_layout layout;
  int lanes;
  sircl_lane_route route[SIRCL_FABRIC_MAX_WORLD][SIRCL_FABRIC_MAX_WORLD][SIRCL_FABRIC_MAX_LANES];
} sircl_fabric_routes;
typedef struct {
  const char *local_error;
  /* Canonical serialized shared settings only: no pointers, padding, or rank-local fields. */
  const void *shared_settings;
  size_t shared_size;
  sircl_route_map map;
} sircl_fabric_rank_record;
typedef struct { uint64_t first, count; } sircl_pack_range;
typedef struct { int first, end, peer, namespace_id; } sircl_phase;
enum { SIRCL_ALGORITHM_AUTO = -1, SIRCL_ALGORITHM_ONESHOT = 0,
       SIRCL_ALGORITHM_TWOSHOT = 1, SIRCL_ALGORITHM_SWING = 2 };

/* Planning/encoding functions return 0 on success and -1 on error. Error is optional.
 * Role getters return the role or name, or -1/NULL if unknown. Input bounds and
 * wire encodings are checked. Output contents after an error are unspecified. */
const char *sircl_fabric_role_device(int role);
int sircl_fabric_role_of(const char *device);
int sircl_fabric_layout_init(sircl_fabric_layout *out, const sircl_cable *cables,
                            int cable_count, const int *positions, int world,
                            sircl_fabric_error *error);
int sircl_fabric_layout_parse(const char *text, sircl_fabric_layout *out, sircl_fabric_error *error);
int sircl_fabric_derive(const sircl_fabric_layout *layout, int lanes,
                        sircl_fabric_routes *out, sircl_fabric_error *error);
int sircl_fabric_map(const sircl_fabric_routes *routes, int rank,
                     sircl_route_map *out, sircl_fabric_error *error);
int sircl_fabric_map_parse(const char *text, sircl_route_map *out, sircl_fabric_error *error);
int sircl_fabric_map_format(const sircl_route_map *map, int world, char *out,
                            size_t capacity, sircl_fabric_error *error);
int sircl_fabric_map_validate(const sircl_fabric_layout *layout, const sircl_route_map *map,
                              int rank, int world, int max_relays, sircl_fabric_error *error);
int sircl_fabric_complementary(const sircl_fabric_layout *layout, const sircl_route_map *maps,
                               sircl_fabric_error *error);
int sircl_fabric_agree(const sircl_fabric_layout *layout, int world, int lanes,
                       const sircl_fabric_rank_record *records, int max_relays,
                       sircl_fabric_error *error);
int sircl_fabric_post_order(const sircl_fabric_routes *routes, int rank,
                            const char *order, int *peers, sircl_fabric_error *error);
int sircl_fabric_relay_load(const sircl_fabric_routes *routes, int *busiest,
                            double *factor, sircl_fabric_error *error);
int sircl_fabric_forward_windows(const sircl_fabric_routes *routes, int rank,
                                 uint64_t max_window, uint64_t chunk, uint64_t queue_bytes,
                                 uint64_t out[SIRCL_FABRIC_MAX_WORLD][SIRCL_FABRIC_MAX_LANES],
                                 sircl_fabric_error *error);
int sircl_fabric_chain_order(const sircl_fabric_routes *routes, int *order,
                             sircl_fabric_error *error);
int sircl_fabric_ring_window(const sircl_fabric_routes *routes, const int *order,
                             uint64_t chunk, uint64_t queue_bytes, uint64_t *window,
                             sircl_fabric_error *error);

int sircl_protocol_stripe(uint64_t packs, int lanes, int lane, sircl_pack_range *out);
int sircl_protocol_chunk(uint64_t packs, int world, int index, sircl_pack_range *out);
int sircl_protocol_flag_index(int namespace_id, int source, int slot, int lane,
                              int world, int lanes, uint32_t *out);
int sircl_protocol_op_word(int op, uint32_t bytes, uint32_t *out);
int sircl_protocol_descriptor(const sircl_phase *phase, int world, uint32_t *out);
int sircl_protocol_swing(int world, int rank, int *owners,
                         sircl_phase phases[SIRCL_PROTOCOL_MAX_PHASES], int *phase_count);
int sircl_protocol_select(uint64_t bytes, int algorithm, int large_algorithm,
                          uint64_t oneshot_max, uint64_t swing_above,
                          unsigned available_mask, int *out);

#ifdef __cplusplus
}
#endif
#endif
