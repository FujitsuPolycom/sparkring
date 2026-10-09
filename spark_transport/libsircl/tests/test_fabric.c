#include "sircl_fabric.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "data/reference_vectors.h"

#define LENGTH(a) (sizeof(a) / sizeof((a)[0]))
static unsigned checks;
static const char *case_name;
#define CHECK(expr) do { checks++; if (!(expr)) { \
  fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, case_name ? case_name : "test", #expr); \
  exit(1); } } while (0)
static sircl_fabric_routes group;
static sircl_fabric_error error;

static void load(const char *text, int lanes) {
  sircl_fabric_layout layout;
  case_name = text;
  CHECK(sircl_fabric_layout_parse(text, &layout, &error) == 0);
  CHECK(sircl_fabric_derive(&layout, lanes, &group, &error) == 0);
}

static void reference_routes(void) {
  size_t i;
  for (i = 0; i < LENGTH(route_layouts); i++) {
    const layout_vector *v = &route_layouts[i];
    sircl_fabric_layout layout;
    sircl_route_map maps[16];
    int rank, j;
    case_name = v->name;
    CHECK(sircl_fabric_layout_init(&layout, v->cables, v->cable_count, v->positions, v->world, &error) == 0);
    CHECK(sircl_fabric_derive(&layout, v->lanes, &group, &error) == 0);
    for (rank = 0; rank < v->world; rank++) {
      sircl_route_map parsed;
      char text[512];
      CHECK(sircl_fabric_map(&group, rank, &maps[rank], &error) == 0);
      CHECK(sircl_fabric_map_format(&maps[rank], v->world, text, sizeof text, &error) == 0);
      CHECK(strcmp(text, v->texts[rank]) == 0);
      CHECK(sircl_fabric_map_parse(v->texts[rank], &parsed, &error) == 0);
      CHECK(memcmp(&parsed, &maps[rank], sizeof parsed) == 0);
      CHECK(sircl_fabric_map_validate(&layout, &parsed, rank, v->world, 16, &error) == 0);
    }
    CHECK(sircl_fabric_complementary(&layout, maps, &error) == 0);
    for (j = 0; j < v->count; j++) {
      const lane_vector *expected = &route_lanes[v->start + j];
      const sircl_lane_route *actual = &group.route[expected->rank][expected->peer][expected->lane];
      int k;
      CHECK(actual->hops == expected->hops);
      CHECK(actual->local_role == expected->local_role);
      CHECK(actual->remote_role == expected->remote_role);
      for (k = 0; k < expected->hops; k++) {
        CHECK(actual->steps[k].src == expected->steps[k].src);
        CHECK(actual->steps[k].src_port == expected->steps[k].src_port);
        CHECK(actual->steps[k].dst == expected->steps[k].dst);
        CHECK(actual->steps[k].dst_port == expected->steps[k].dst_port);
      }
    }
    if (!strcmp(v->name, "ring-of-8-tp6-0-5")) {
      CHECK(sircl_fabric_map_validate(&layout, &maps[0], 0, v->world, 3, &error) == -1);
      CHECK(error.code == SIRCL_FABRIC_LIMIT);
    }
  }
}

static void reference_numeric(void) {
  size_t i;
  case_name = "numeric reference vectors";
  for (i = 0; i < LENGTH(numeric_ranges); i++) {
    const range_vector *v = &numeric_ranges[i]; sircl_pack_range actual;
    CHECK((v->kind == 0 ? sircl_protocol_stripe(v->packs, v->parts, v->index, &actual) :
                        sircl_protocol_chunk(v->packs, v->parts, v->index, &actual)) == 0);
    CHECK(actual.first == v->first && actual.count == v->count);
  }
  for (i = 0; i < LENGTH(numeric_flags); i++) {
    const flag_vector *v = &numeric_flags[i]; uint32_t word;
    CHECK(sircl_protocol_flag_index(v->ns, v->source, v->slot, v->lane, v->world, v->lanes, &word) == 0);
    CHECK(word == v->expected);
  }
  for (i = 0; i < LENGTH(numeric_ops); i++) {
    uint32_t word; CHECK(sircl_protocol_op_word(numeric_ops[i].op, numeric_ops[i].bytes, &word) == 0);
    CHECK(word == numeric_ops[i].expected);
  }
  for (i = 0; i < LENGTH(numeric_descriptors); i++) {
    uint32_t word; CHECK(sircl_protocol_descriptor(&numeric_descriptors[i].phase, 0, &word) == 0);
    CHECK(word == numeric_descriptors[i].expected);
  }
  for (i = 0; i < LENGTH(numeric_posting); i++) {
    const post_vector *v = &numeric_posting[i]; int peers[16], j; char layout[32];
    snprintf(layout, sizeof layout, "ring:%d", v->world); load(layout, 2);
    CHECK(sircl_fabric_post_order(&group, v->rank, "ring-farthest", peers, &error) == 0);
    for (j = 0; j < v->world - 1; j++) CHECK(peers[j] == v->peers[j]);
    CHECK(sircl_fabric_post_order(&group, v->rank, "farthest", peers, &error) == 0);
    for (j = 0; j < v->world - 1; j++) CHECK(peers[j] == v->peers[j]);
  }
  for (i = 0; i < LENGTH(numeric_owners); i++) {
    const owner_vector *v = &numeric_owners[i]; int owners[16], count, j; sircl_phase phases[8];
    CHECK(sircl_protocol_swing(v->world, 0, owners, phases, &count) == 0);
    CHECK(count == v->phase_count);
    for (j = 0; j < v->world; j++) CHECK(owners[j] == v->owners[j]);
  }
  for (i = 0; i < LENGTH(numeric_phases); i++) {
    const phase_vector *v = &numeric_phases[i]; int owners[16], count; sircl_phase phases[8], *p;
    CHECK(sircl_protocol_swing(v->world, v->rank, owners, phases, &count) == 0);
    CHECK(count > v->phase_index); p = &phases[v->phase_index];
    CHECK(p->first == v->phase.first && p->end == v->phase.end && p->peer == v->phase.peer && p->namespace_id == v->phase.namespace_id);
  }
  for (i = 0; i < LENGTH(numeric_selection); i++) {
    const selection_vector *v = &numeric_selection[i]; int algorithm;
    unsigned available = (v->world & (v->world - 1)) ? 3u : 7u;
    CHECK(sircl_protocol_select(v->bytes, -1, 1, 28672, v->swing_above, available, &algorithm) == 0);
    CHECK(algorithm == v->algorithm);
  }
}

static void posting_and_windows(void) {
  uint64_t windows[16][2], ring_window;
  int peers[16], order[16], busiest, rank, peer, lane;
  double factor;
  const int path_orders[4][3] = {{3,2,1}, {3,2,0}, {0,3,1}, {0,1,2}};
  load("path:0-3", 2);
  for (rank = 0; rank < 4; rank++) {
    CHECK(sircl_fabric_post_order(&group, rank, "farthest", peers, &error) == 0);
    CHECK(memcmp(peers, path_orders[rank], sizeof path_orders[rank]) == 0);
  }
  CHECK(sircl_fabric_relay_load(&group, &busiest, &factor, &error) == 0);
  CHECK(busiest == 2 && factor == 1);
  CHECK(sircl_fabric_forward_windows(&group, 0, 131072, 32768, 524288, windows, &error) == 0);
  CHECK(windows[0][0] == 0 && windows[1][0] == 0 && windows[2][0] == 131072 && windows[3][1] == 131072);
  CHECK(sircl_fabric_chain_order(&group, order, &error) == 0);
  for (rank = 0; rank < 4; rank++) CHECK(order[rank] == rank);
  CHECK(sircl_fabric_ring_window(&group, order, 32768, 524288, &ring_window, &error) == 0);
  CHECK(ring_window == 393216);
  CHECK(sircl_fabric_ring_window(&group, order, 32768, 1, &ring_window, &error) == -1);
  /* Deliberately model duplicate ring lanes on one function; its queue is overloaded. */
  group.route[3][0][1] = group.route[3][0][0];
  CHECK(sircl_fabric_ring_window(&group, order, 32768, 524288, &ring_window, &error) == -1);
  CHECK(error.code == SIRCL_FABRIC_LIMIT);
  load("ring:8", 2);
  CHECK(sircl_fabric_relay_load(&group, &busiest, &factor, &error) == 0);
  CHECK(busiest == 6 && factor == 3);
  CHECK(sircl_fabric_forward_windows(&group, 0, 131072, 32768, 524288, windows, &error) == 0);
  CHECK(windows[1][0] == 0 && windows[7][1] == 0 && windows[4][0] == 65536 && windows[4][1] == 65536);
  CHECK(sircl_fabric_forward_windows(&group, 0, 0, 0, 0, windows, &error) == 0);
  for (peer = 0; peer < 8; peer++) for (lane = 0; lane < 2; lane++) CHECK(windows[peer][lane] == 0);
  CHECK(sircl_fabric_forward_windows(&group, 0, 131072, 1000, 524288, windows, &error) == -1);
  CHECK(sircl_fabric_forward_windows(&group, 0, 131072, 32768, 16, windows, &error) == -1);
  CHECK(sircl_fabric_chain_order(&group, order, &error) == 0);
  CHECK(sircl_fabric_ring_window(&group, order, 32768, 524288, &ring_window, &error) == 0 && ring_window == 0);
  load("ring:8:0,2,4,6", 2);
  CHECK(sircl_fabric_chain_order(&group, order, &error) == -1);
  CHECK(sircl_fabric_post_order(&group, 0, "farthest", peers, &error) == 0);
  CHECK(peers[0] == 2 && peers[1] == 1 && peers[2] == 3);
  load("cables=0.port0-1.port1,1.port0-2.port1;positions=2,1,0", 2);
  CHECK(sircl_fabric_chain_order(&group, order, &error) == 0);
  CHECK(order[0] == 2 && order[1] == 1 && order[2] == 0);
}

static void agreement_and_invalid(void) {
  sircl_fabric_rank_record records[16];
  sircl_route_map maps[16], map;
  sircl_fabric_layout l;
  uint32_t shared[3] = {1, 131072, 512}, different[3] = {1, 131072, 256};
  int i, peers[16]; char text[8];
  static const char *bad_layouts[] = {"", "star:4", "ring:x", "ring:1", "ring:17", "path:3-0", "path:0-3:0,0,1,2", "cables=0.port0-1.port0", "cables=0.port01-1.port0;positions=0,1", "ring:4:0,9", "ring:4:", "ring:4:0,1,", "ring:2147483648"};
  static const char *bad_maps[] = {"", "1", "x=rocep1s0f0", "1=rocep1s0f0,,2=rocep1s0f1", "1=rocep1s0f0,", "1=", "16=rocep1s0f0", "1=rocep1s0f0/rocep1s0f1/roceP2p1s0f0", "1=mlx5_9", "1=rocep1s0f0,1=rocep1s0f1"};
  case_name = "invalid inputs";
  for (i = 0; i < (int)LENGTH(bad_layouts); i++) CHECK(sircl_fabric_layout_parse(bad_layouts[i], &l, &error) == -1);
  for (i = 0; i < (int)LENGTH(bad_maps); i++) CHECK(sircl_fabric_map_parse(bad_maps[i], &map, &error) == -1);
  { sircl_cable disconnected[] = {{0,0,1,1}, {2,0,3,1}}, reused[] = {{0,0,1,1}, {0,0,2,1}};
    int positions[] = {0,1};
    CHECK(sircl_fabric_layout_init(&l, disconnected, 2, positions, 2, &error) == -1);
    CHECK(sircl_fabric_layout_init(&l, reused, 2, positions, 2, &error) == -1);
  }
  load("ring:4", 2); memset(records, 0, sizeof records);
  for (i = 0; i < 4; i++) {
    CHECK(sircl_fabric_map(&group, i, &maps[i], &error) == 0);
    records[i].map = maps[i]; records[i].shared_settings = shared; records[i].shared_size = sizeof shared;
  }
  CHECK(sircl_fabric_agree(&group.layout, 4, 2, records, 3, &error) == 0);
  CHECK(sircl_fabric_agree(NULL, 4, 2, records, 3, &error) == 0);
  records[2].shared_settings = different;
  CHECK(sircl_fabric_agree(&group.layout, 4, 2, records, 3, &error) == -1);
  CHECK(strstr(error.message, "rank 2 shared settings"));
  records[2].shared_settings = shared; records[1].local_error = "inactive RDMA device";
  CHECK(sircl_fabric_agree(&group.layout, 4, 2, records, 3, &error) == -1);
  CHECK(strstr(error.message, "rank 1: inactive")); records[1].local_error = NULL;
  records[0].map.count[2] = 1;
  CHECK(sircl_fabric_agree(&group.layout, 4, 2, records, 3, &error) == -1);
  records[0].map = maps[0]; records[2].map.role[0][0] = SIRCL_CW_PRIMARY;
  CHECK(sircl_fabric_agree(&group.layout, 4, 2, records, 3, &error) == -1);
  CHECK(strstr(error.message, "arrives on rank"));
  map = maps[0]; map.count[2] = 0;
  CHECK(sircl_fabric_map_validate(&group.layout, &map, 0, 4, 3, &error) == -1);
  map = maps[0]; map.count[0] = 1;
  CHECK(sircl_fabric_map_validate(&group.layout, &map, 0, 4, 3, &error) == -1);
  map = maps[0]; map.role[1][1] = map.role[1][0];
  CHECK(sircl_fabric_map_validate(&group.layout, &map, 0, 4, 3, &error) == -1);
  CHECK(sircl_fabric_map_format(&maps[0], 4, text, sizeof text, &error) == -1);
  CHECK(sircl_fabric_map_parse(" 1 = rocep1s0f0 / roceP2p1s0f0 , 2=rocep1s0f1 ", &map, &error) == 0);
  CHECK(map.count[1] == 2 && map.count[2] == 1 && map.role[2][0] == SIRCL_CCW_PRIMARY);
  CHECK(sircl_fabric_post_order(&group, 1, " 3,0,2 ", peers, &error) == 0 && peers[0] == 3 && peers[1] == 0 && peers[2] == 2);
  CHECK(sircl_fabric_post_order(&group, 1, NULL, peers, &error) == 0 && peers[0] == 0 && peers[1] == 2 && peers[2] == 3);
  CHECK(sircl_fabric_post_order(&group, 1, "0,2", peers, &error) == -1);
  CHECK(sircl_fabric_post_order(&group, 1, "0,2,2", peers, &error) == -1);
  CHECK(sircl_fabric_post_order(&group, 1, "0,2,3,", peers, &error) == -1);
  CHECK(sircl_fabric_post_order(&group, 1, "nearest", peers, &error) == -1);
  load("path:0-3", 2); CHECK(sircl_fabric_map(&group, 0, &map, &error) == 0);
  map.role[3][0] = SIRCL_CCW_PRIMARY;
  CHECK(sircl_fabric_map_validate(&group.layout, &map, 0, 4, 16, &error) == -1);
  group.route[0][1][0].local_role = 255;
  CHECK(sircl_fabric_post_order(&group, 0, "farthest", peers, &error) == -1);
  load("path:0-3", 2); group.layout.nodes[0] = -1;
  CHECK(sircl_fabric_chain_order(&group, peers, &error) == -1);
  load("path:0-3", 2); group.route[0][1][0].steps[0].dst_port = 9;
  CHECK(sircl_fabric_post_order(&group, 0, "rank", peers, &error) == -1);
}

static void numeric_limits(void) {
  uint32_t word;
  sircl_pack_range range;
  sircl_phase phases[8], p = {0, 4, 1, 0};
  int owners[16], count, world, rank, i, algorithm;
  case_name = "numeric bounds and conservation";
  CHECK(sircl_protocol_stripe(1, 0, 0, &range) == -1);
  CHECK(sircl_protocol_stripe(1, 2, 2, &range) == -1);
  CHECK(sircl_protocol_chunk(1, 1, 0, &range) == -1);
  CHECK(sircl_protocol_chunk(1, 4, 4, &range) == -1);
  CHECK(sircl_protocol_flag_index(2, 0, 0, 0, 4, 2, &word) == -1);
  CHECK(sircl_protocol_flag_index(0, 4, 0, 0, 4, 2, &word) == -1);
  CHECK(sircl_protocol_op_word(4, 16, &word) == -1);
  CHECK(sircl_protocol_op_word(0, 0, &word) == -1);
  CHECK(sircl_protocol_op_word(0, 15, &word) == -1);
  CHECK(sircl_protocol_op_word(0, UINT32_C(0x40000000), &word) == -1);
  CHECK(sircl_protocol_descriptor(&p, 2, &word) == -1);
  p.end = 32; CHECK(sircl_protocol_descriptor(&p, 0, &word) == -1);
  CHECK(sircl_protocol_swing(3, 0, owners, phases, &count) == -1);
  for (world = 2; world <= 16; world++) {
    uint64_t next = 0;
    for (i = 0; i < world; i++) {
      CHECK(sircl_protocol_chunk(UINT64_MAX, world, i, &range) == 0);
      CHECK(range.first == next); next += range.count;
    }
    CHECK(next == UINT64_MAX);
    if (world & (world - 1)) continue;
    for (rank = 0; rank < world; rank++) {
      unsigned mask = 0;
      CHECK(sircl_protocol_swing(world, rank, owners, phases, &count) == 0);
      for (i = 0; i < world; i++) { CHECK(owners[i] >= 0 && owners[i] < world); CHECK(!(mask & (1u << owners[i]))); mask |= 1u << owners[i]; }
      CHECK(mask == (1u << world) - 1);
      for (i = 0; i < count; i++) {
        sircl_phase other[8]; int other_owners[16], other_count;
        CHECK(sircl_protocol_descriptor(&phases[i], world, &word) == 0);
        CHECK(sircl_protocol_swing(world, phases[i].peer, other_owners, other, &other_count) == 0);
        CHECK(other_count == count && other[i].peer == rank && other[i].namespace_id == phases[i].namespace_id);
      }
    }
  }
  CHECK(sircl_protocol_select(1 << 20, -1, 1, 28672, 0, 1u, &algorithm) == 0 && algorithm == 0);
  CHECK(sircl_protocol_select(16, 2, 1, 28672, 0, 1u, &algorithm) == 0 && algorithm == 2);
  CHECK(sircl_protocol_select(16, 9, 1, 28672, 0, 7u, &algorithm) == -1);
}

int main(void) {
  reference_routes(); reference_numeric(); posting_and_windows(); agreement_and_invalid(); numeric_limits();
  printf("fabric: %zu supplied layouts, %zu lane routes, all numeric vectors; %u checks passed\n", LENGTH(route_layouts), LENGTH(route_lanes), checks);
  return 0;
}
