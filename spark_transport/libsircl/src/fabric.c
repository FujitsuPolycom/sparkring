#include "sircl_fabric.h"
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FW SIRCL_FABRIC_MAX_WORLD
#define FL SIRCL_FABRIC_MAX_LANES

static int fail(sircl_fabric_error *e, int code, const char *fmt, ...) {
  if (e) { va_list ap; e->code = code; va_start(ap, fmt);
    vsnprintf(e->message, sizeof e->message, fmt, ap); va_end(ap); }
  return -1;
}
static void clear_error(sircl_fabric_error *e) { if (e) memset(e, 0, sizeof *e); }
static const char *devices[4] = {"rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1", "roceP2p1s0f1"};
const char *sircl_fabric_role_device(int role) { return role >= 0 && role < 4 ? devices[role] : NULL; }
int sircl_fabric_role_of(const char *device) {
  int i; if (!device) return -1;
  for (i = 0; i < 4; i++) if (!strcmp(device, devices[i])) return i;
  return -1;
}
static int node_at(const sircl_fabric_layout *l, int position) {
  int i; for (i = 0; i < l->node_count; i++) if (l->nodes[i] == position) return i;
  return -1;
}
static int cross(const sircl_fabric_layout *l, int position, int port, sircl_fabric_step *s) {
  int i; for (i = 0; i < l->cable_count; i++) {
    const sircl_cable *c = &l->cables[i];
    if (c->a == position && c->a_port == port) {
      *s = (sircl_fabric_step){position, port, c->b, c->b_port}; return 1;
    }
    if (c->b == position && c->b_port == port) {
      *s = (sircl_fabric_step){position, port, c->a, c->a_port}; return 1;
    }
  } return 0;
}
static int walk(const sircl_fabric_layout *l, int a, int port, int b,
                sircl_fabric_step steps[FW]) {
  int n, here = a, out = port;
  for (n = 0; n < l->node_count; n++) {
    if (!cross(l, here, out, &steps[n])) return 0;
    if (steps[n].dst == b) return n + 1;
    here = steps[n].dst; out = 1 - steps[n].dst_port;
  } return 0;
}
static int layout_valid(const sircl_fabric_layout *l) {
  sircl_fabric_layout checked;
  if (!l || l->world < 2 || l->world > FW || l->node_count < l->world ||
      l->node_count > FW || l->cable_count < 1 || l->cable_count > FW) return 0;
  if (sircl_fabric_layout_init(&checked, l->cables, l->cable_count, l->positions, l->world, NULL)) return 0;
  return l->node_count == checked.node_count && l->cycle == checked.cycle &&
    !memcmp(l->nodes, checked.nodes, (size_t)l->node_count * sizeof l->nodes[0]);
}
static int routes_valid(const sircl_fabric_routes *r) {
  int rank, peer, lane;
  if (!r || !layout_valid(&r->layout) || r->lanes < 1 || r->lanes > FL) return 0;
  for (rank = 0; rank < r->layout.world; rank++) for (peer = 0; peer < r->layout.world; peer++) if (rank != peer)
    for (lane = 0; lane < r->lanes; lane++) {
      const sircl_lane_route *route = &r->route[rank][peer][lane];
      int k, here = r->layout.positions[rank], port;
      if (route->hops < 1 || route->hops > r->layout.node_count || route->local_role < 0 ||
          route->local_role > 3 || route->remote_role < 0 || route->remote_role > 3) return 0;
      port = route->local_role / 2;
      for (k = 0; k < route->hops; k++) {
        sircl_fabric_step expected;
        const sircl_fabric_step *step = &route->steps[k];
        if (!cross(&r->layout, here, port, &expected) || step->src != expected.src ||
            step->src_port != expected.src_port || step->dst != expected.dst || step->dst_port != expected.dst_port) return 0;
        here = step->dst; port = 1 - step->dst_port;
      }
      if (here != r->layout.positions[peer] || route->remote_role !=
          route->steps[route->hops - 1].dst_port * 2 + route->local_role % 2) return 0;
    }
  return 1;
}
int sircl_fabric_layout_init(sircl_fabric_layout *out, const sircl_cable *cables,
                            int cable_count, const int *positions, int world, sircl_fabric_error *e) {
  sircl_fabric_layout t; int i, j, degree[FW] = {0}, reached[FW] = {0};
  clear_error(e);
  if (!out || !cables || !positions || world < 2 || world > FW || cable_count < 1 || cable_count > FW)
    return fail(e, SIRCL_FABRIC_INVALID, "a fabric needs 2-16 ranks and 1-16 cables");
  memset(&t, 0, sizeof t); t.world = world; t.cable_count = cable_count;
  for (i = 0; i < cable_count; i++) {
    const sircl_cable *c = &cables[i]; int ends[2] = {c->a, c->b};
    if (c->a < 0 || c->b < 0 || c->a == c->b || c->a_port < 0 || c->a_port > 1 || c->b_port < 0 || c->b_port > 1)
      return fail(e, SIRCL_FABRIC_INVALID, "cable %d has invalid endpoints or ports", i);
    for (j = 0; j < i; j++) {
      const sircl_cable *d = &cables[j];
      if ((c->a == d->a && c->a_port == d->a_port) || (c->a == d->b && c->a_port == d->b_port) ||
          (c->b == d->a && c->b_port == d->a_port) || (c->b == d->b && c->b_port == d->b_port))
        return fail(e, SIRCL_FABRIC_INVALID, "cable %d reuses a fabric port", i);
    }
    t.cables[i] = *c;
    for (j = 0; j < 2; j++) if (node_at(&t, ends[j]) < 0) {
      if (t.node_count == FW) return fail(e, SIRCL_FABRIC_LIMIT, "fabric exceeds 16 positions");
      t.nodes[t.node_count++] = ends[j];
    }
  }
  for (i = 0; i < t.node_count; i++) for (j = i + 1; j < t.node_count; j++)
    if (t.nodes[j] < t.nodes[i]) { int p = t.nodes[i]; t.nodes[i] = t.nodes[j]; t.nodes[j] = p; }
  for (i = 0; i < cable_count; i++) { degree[node_at(&t, cables[i].a)]++; degree[node_at(&t, cables[i].b)]++; }
  t.cycle = 1; for (i = 0; i < t.node_count; i++) if (degree[i] != 2) t.cycle = 0;
  reached[0] = 1;
  for (i = 0; i < t.node_count; i++) for (j = 0; j < cable_count; j++) {
    int a = node_at(&t, cables[j].a), b = node_at(&t, cables[j].b);
    if (reached[a] || reached[b]) reached[a] = reached[b] = 1;
  }
  for (i = 0; i < t.node_count; i++) if (!reached[i])
    return fail(e, SIRCL_FABRIC_UNREACHABLE, "fabric contains disconnected position %d", t.nodes[i]);
  for (i = 0; i < world; i++) {
    if (node_at(&t, positions[i]) < 0) return fail(e, SIRCL_FABRIC_INVALID, "rank %d position %d is not on fabric", i, positions[i]);
    for (j = 0; j < i; j++) if (positions[j] == positions[i]) return fail(e, SIRCL_FABRIC_INVALID, "ranks need distinct fabric positions");
    t.positions[i] = positions[i];
  }
  *out = t; return 0;
}
static void skip_space(const char **p) { while (isspace((unsigned char)**p)) (*p)++; }
static int number(const char **p, int *out) {
  char *end; long value; skip_space(p); if (!isdigit((unsigned char)**p)) return -1;
  errno = 0; value = strtol(*p, &end, 10); if (errno || value > INT_MAX) return -1;
  *p = end; *out = (int)value; skip_space(p); return 0;
}
static int token(const char **p, const char *text) {
  size_t n = strlen(text); skip_space(p); if (strncmp(*p, text, n)) return -1;
  *p += n; skip_space(p); return 0;
}
static int parse_cable(const char **p, sircl_cable *c) {
  if (number(p, &c->a) || token(p, ".port") || (**p != '0' && **p != '1')) return -1;
  c->a_port = *(*p)++ - '0';
  if (token(p, "-") || number(p, &c->b) || token(p, ".port") || (**p != '0' && **p != '1')) return -1;
  c->b_port = *(*p)++ - '0'; skip_space(p);
  return 0;
}
static int parse_positions(const char **p, int positions[FW], int *count) {
  *count = 0;
  do {
    if (*count == FW || number(p, &positions[(*count)++])) return -1;
    if (**p != ',') break;
    (*p)++;
  } while (1);
  return **p ? -1 : 0;
}
int sircl_fabric_layout_parse(const char *text, sircl_fabric_layout *out, sircl_fabric_error *e) {
  sircl_cable cables[FW]; int positions[FW], count = 0, world = 0, i, first, last; const char *p = text;
  clear_error(e); if (!text || !out) return fail(e, SIRCL_FABRIC_INVALID, "layout is required");
  skip_space(&p);
  if (!strncmp(p, "cables=", 7)) {
    p += 7;
    do { if (count == FW || parse_cable(&p, &cables[count++])) goto malformed;
      if (*p != ',') break;
      p++;
    } while (1);
    if (token(&p, ";positions=") || parse_positions(&p, positions, &world)) goto malformed;
  } else if (!strncmp(p, "ring:", 5)) {
    p += 5; if (number(&p, &count) || count < 2 || count > FW) goto malformed;
    for (i = 0; i < count; i++) { cables[i] = (sircl_cable){i, 0, (i + 1) % count, 1}; positions[i] = i; }
    world = count;
    if (*p == ':') { p++; if (parse_positions(&p, positions, &world)) goto malformed; }
    else if (*p) goto malformed;
  } else if (!strncmp(p, "path:", 5)) {
    p += 5; if (number(&p, &first) || token(&p, "-") || number(&p, &last) || last <= first || last - first >= FW) goto malformed;
    count = last - first; world = count + 1;
    for (i = 0; i < world; i++) positions[i] = first + i;
    for (i = 0; i < count; i++) cables[i] = (sircl_cable){first + i, 0, first + i + 1, 1};
    if (*p == ':') { p++; if (parse_positions(&p, positions, &world)) goto malformed; }
    else if (*p) goto malformed;
  } else goto malformed;
  return sircl_fabric_layout_init(out, cables, count, positions, world, e);
malformed: return fail(e, SIRCL_FABRIC_INVALID, "malformed layout; use ring:, path:, or cables=...;positions=...");
}
int sircl_fabric_derive(const sircl_fabric_layout *l, int lanes, sircl_fabric_routes *out, sircl_fabric_error *e) {
  int rank, peer, lane; clear_error(e);
  if (!layout_valid(l) || !out || lanes < 1 || lanes > FL) return fail(e, SIRCL_FABRIC_INVALID, "invalid layout or lane count");
  /* Revalidate public structures, including port uniqueness and connectivity. */
  { sircl_fabric_layout checked;
    if (sircl_fabric_layout_init(&checked, l->cables, l->cable_count, l->positions, l->world, e)) return -1;
    out->layout = checked;
  }
  out->lanes = lanes; memset(out->route, 0, sizeof out->route); l = &out->layout;
  for (rank = 0; rank < l->world; rank++) for (peer = 0; peer < l->world; peer++) if (rank != peer) {
    sircl_fabric_step paths[2][FW]; int n[2], chosen[2];
    n[0] = walk(l, l->positions[rank], 0, l->positions[peer], paths[0]);
    n[1] = walk(l, l->positions[rank], 1, l->positions[peer], paths[1]);
    if (!n[0] && !n[1]) return fail(e, SIRCL_FABRIC_UNREACHABLE, "rank %d cannot reach rank %d", rank, peer);
    if (!n[1] || (n[0] && n[0] < n[1])) chosen[0] = chosen[1] = 0;
    else if (!n[0] || n[1] < n[0]) chosen[0] = chosen[1] = 1;
    else {
      /* Path A leaves the numerically smaller endpoint through port 0. */
      int a = l->positions[rank] < l->positions[peer] ? 0 : (paths[0][n[0]-1].dst_port == 0 ? 0 : 1);
      chosen[0] = a; chosen[1] = 1 - a;
    }
    for (lane = 0; lane < lanes; lane++) {
      int which = chosen[lane]; sircl_lane_route *r = &out->route[rank][peer][lane];
      r->hops = n[which]; memcpy(r->steps, paths[which], (size_t)r->hops * sizeof r->steps[0]);
      r->local_role = r->steps[0].src_port * 2 + lane;
      r->remote_role = r->steps[r->hops - 1].dst_port * 2 + lane;
    }
  } return 0;
}
int sircl_fabric_map(const sircl_fabric_routes *r, int rank, sircl_route_map *out, sircl_fabric_error *e) {
  int peer, lane; clear_error(e);
  if (!routes_valid(r) || !out || rank < 0 || rank >= r->layout.world) return fail(e, SIRCL_FABRIC_INVALID, "invalid routes or rank");
  memset(out, 0, sizeof *out);
  for (peer = 0; peer < r->layout.world; peer++) if (peer != rank) {
    out->count[peer] = (uint8_t)r->lanes;
    for (lane = 0; lane < r->lanes; lane++) out->role[peer][lane] = (uint8_t)r->route[rank][peer][lane].local_role;
  } return 0;
}
int sircl_fabric_map_parse(const char *text, sircl_route_map *out, sircl_fabric_error *e) {
  sircl_route_map t; const char *p = text; int seen[FW] = {0}; clear_error(e);
  if (!text || !out) return fail(e, SIRCL_FABRIC_INVALID, "route map is required");
  memset(&t, 0, sizeof t); skip_space(&p); if (!*p) return fail(e, SIRCL_FABRIC_INVALID, "route map is empty");
  while (*p) {
    int peer, lane = 0;
    if (number(&p, &peer) || peer >= FW || token(&p, "=")) return fail(e, SIRCL_FABRIC_INVALID, "malformed route map entry");
    if (seen[peer]++) return fail(e, SIRCL_FABRIC_INVALID, "route map names rank %d twice", peer);
    do {
      const char *start; size_t len; char device[64]; int role;
      skip_space(&p); start = p; while (*p && *p != '/' && *p != ',' && !isspace((unsigned char)*p)) p++;
      len = (size_t)(p - start); skip_space(&p);
      if (!len || len >= sizeof device || lane == FL) return fail(e, SIRCL_FABRIC_INVALID, "invalid lane list of rank %d", peer);
      memcpy(device, start, len); device[len] = 0; role = sircl_fabric_role_of(device);
      if (role < 0) return fail(e, SIRCL_FABRIC_INVALID, "device %s has no known role", device);
      t.role[peer][lane++] = (uint8_t)role;
      if (*p != '/') break;
      p++;
    } while (1);
    t.count[peer] = (uint8_t)lane;
    if (!*p) break;
    if (*p != ',') return fail(e, SIRCL_FABRIC_INVALID, "malformed route map separator");
    p++; skip_space(&p); if (!*p) return fail(e, SIRCL_FABRIC_INVALID, "route map has an empty entry");
  } *out = t; return 0;
}
int sircl_fabric_map_format(const sircl_route_map *map, int world, char *out, size_t cap, sircl_fabric_error *e) {
  int peer, lane, entries = 0; size_t used = 0; clear_error(e);
  if (!map || !out || !cap || world < 2 || world > FW) return fail(e, SIRCL_FABRIC_INVALID, "invalid route formatter arguments");
  out[0] = 0;
  for (peer = 0; peer < world; peer++) if (map->count[peer]) {
    int n; if (map->count[peer] > FL) return fail(e, SIRCL_FABRIC_INVALID, "unsupported lane count");
    n = snprintf(out + used, cap - used, "%s%d=", entries++ ? "," : "", peer);
    if (n < 0 || (size_t)n >= cap - used) goto too_small;
    used += (size_t)n;
    for (lane = 0; lane < map->count[peer]; lane++) {
      const char *device = sircl_fabric_role_device(map->role[peer][lane]);
      if (!device) return fail(e, SIRCL_FABRIC_INVALID, "invalid device role");
      n = snprintf(out + used, cap - used, "%s%s", lane ? "/" : "", device);
      if (n < 0 || (size_t)n >= cap - used) goto too_small;
      used += (size_t)n;
    }
  } return 0;
too_small: return fail(e, SIRCL_FABRIC_LIMIT, "route text buffer is too small");
}
int sircl_fabric_map_validate(const sircl_fabric_layout *l, const sircl_route_map *map,
                              int rank, int world, int max_relays, sircl_fabric_error *e) {
  int peer, lane, lanes = 0; clear_error(e);
  if (!map || world < 2 || world > FW || rank < 0 || rank >= world || max_relays < 0 ||
      (l && (!layout_valid(l) || l->world != world))) return fail(e, SIRCL_FABRIC_INVALID, "invalid route validation arguments");
  if (map->count[rank]) return fail(e, SIRCL_FABRIC_INVALID, "rank %d's map names its own rank", rank);
  for (peer = 0; peer < FW; peer++) if (peer != rank) {
    if (peer >= world) { if (map->count[peer]) return fail(e, SIRCL_FABRIC_INVALID, "rank %d is outside session", peer); continue; }
    if (!map->count[peer]) return fail(e, SIRCL_FABRIC_INVALID, "rank %d's map has no entry for rank %d", rank, peer);
    if (map->count[peer] > FL || (lanes && map->count[peer] != lanes)) return fail(e, SIRCL_FABRIC_INVALID, "rank %d has different or unsupported lane counts", rank);
    lanes = map->count[peer];
    if (lanes == 2 && map->role[peer][0] == map->role[peer][1]) return fail(e, SIRCL_FABRIC_INVALID, "rank %d's entry for rank %d names device twice", rank, peer);
    for (lane = 0; lane < lanes; lane++) {
      int role = map->role[peer][lane]; sircl_fabric_step steps[FW]; int hops;
      if (role > 3) return fail(e, SIRCL_FABRIC_INVALID, "rank %d has unknown role", rank);
      if (!l) continue;
      hops = walk(l, l->positions[rank], role / 2, l->positions[peer], steps);
      if (!hops) return fail(e, SIRCL_FABRIC_UNREACHABLE, "rank %d lane %d to rank %d leaves the group's fabric", rank, lane, peer);
      if (hops - 1 > max_relays) return fail(e, SIRCL_FABRIC_LIMIT, "rank %d lane %d to rank %d crosses %d relays; limit is %d", rank, lane, peer, hops - 1, max_relays);
    }
  } return 0;
}
int sircl_fabric_complementary(const sircl_fabric_layout *l, const sircl_route_map *maps, sircl_fabric_error *e) {
  int rank, peer, lane; clear_error(e);
  if (!layout_valid(l) || !maps) return fail(e, SIRCL_FABRIC_INVALID, "layout and maps are required");
  for (rank = 0; rank < l->world; rank++) if (sircl_fabric_map_validate(l, &maps[rank], rank, l->world, FW, e)) return -1;
  for (rank = 0; rank < l->world; rank++) for (peer = 0; peer < l->world; peer++) if (peer != rank) {
    if (maps[rank].count[peer] != maps[peer].count[rank]) return fail(e, SIRCL_FABRIC_MISMATCH, "ranks %d and %d disagree on lanes", rank, peer);
    for (lane = 0; lane < maps[rank].count[peer]; lane++) {
      int mine = maps[rank].role[peer][lane]; sircl_fabric_step steps[FW];
      int hops = walk(l, l->positions[rank], mine / 2, l->positions[peer], steps);
      int expected = steps[hops - 1].dst_port * 2 + mine % 2;
      if (maps[peer].role[rank][lane] != expected) return fail(e, SIRCL_FABRIC_MISMATCH,
        "rank %d lane %d arrives on rank %d port %d (%s); peer lists %s", rank, lane, peer,
        steps[hops - 1].dst_port, devices[expected], devices[maps[peer].role[rank][lane]]);
    }
  } return 0;
}
int sircl_fabric_agree(const sircl_fabric_layout *l, int world, int lanes,
                       const sircl_fabric_rank_record *records, int max_relays, sircl_fabric_error *e) {
  sircl_route_map maps[FW]; int rank, peer, lane; clear_error(e);
  if (!records || world < 2 || world > FW || lanes < 1 || lanes > FL || (l && (!layout_valid(l) || l->world != world)))
    return fail(e, SIRCL_FABRIC_INVALID, "invalid setup agreement arguments");
  for (rank = 0; rank < world; rank++) if (records[rank].local_error)
    return fail(e, SIRCL_FABRIC_MISMATCH, "rank %d: %s", rank, records[rank].local_error);
  for (rank = 0; rank < world; rank++) {
    if (records[rank].shared_size && !records[rank].shared_settings) return fail(e, SIRCL_FABRIC_INVALID, "rank %d lacks shared settings", rank);
    if (records[rank].shared_size != records[0].shared_size || (records[0].shared_size &&
        memcmp(records[rank].shared_settings, records[0].shared_settings, records[0].shared_size)))
      return fail(e, SIRCL_FABRIC_MISMATCH, "rank %d shared settings differ from rank 0", rank);
    maps[rank] = records[rank].map;
    for (peer = 0; peer < world; peer++) if (maps[rank].count[peer] != (peer == rank ? 0 : lanes))
      return fail(e, SIRCL_FABRIC_MISMATCH, "rank %d has incorrect lane count toward rank %d", rank, peer);
    if (sircl_fabric_map_validate(l, &maps[rank], rank, world, max_relays, e)) return -1;
  }
  if (l) return sircl_fabric_complementary(l, maps, e);
  for (rank = 0; rank < world; rank++) for (peer = 0; peer < world; peer++) if (peer != rank)
    for (lane = 0; lane < lanes; lane++) if (maps[rank].role[peer][lane] % 2 != maps[peer].role[rank][lane] % 2)
      return fail(e, SIRCL_FABRIC_MISMATCH, "rank %d and rank %d lane %d use different function classes", rank, peer, lane);
  return 0;
}
int sircl_fabric_post_order(const sircl_fabric_routes *r, int rank, const char *order, int *peers, sircl_fabric_error *e) {
  int world, i, n = 0, seen[FW] = {0}; const char *p = order; char text[256]; size_t len; clear_error(e);
  if (!routes_valid(r) || !peers || rank < 0 || rank >= r->layout.world) return fail(e, SIRCL_FABRIC_INVALID, "invalid posting order arguments");
  world = r->layout.world;
  if (!p) p = "rank";
  skip_space(&p); len = strlen(p); while (len && isspace((unsigned char)p[len-1])) len--;
  if (len >= sizeof text) return fail(e, SIRCL_FABRIC_INVALID, "posting order is too long");
  memcpy(text, p, len); text[len] = 0; p = text;
  if (!len || !strcmp(p, "rank")) { for (i = 0; i < world; i++) if (i != rank) peers[n++] = i; return 0; }
  if (!strcmp(p, "ring-farthest")) {
    for (i = world / 2; i >= 1; i--) {
      int a = (rank + i) % world, b = (rank + world - i) % world; peers[n++] = a; if (a != b) peers[n++] = b;
    } return 0;
  }
  if (!strcmp(p, "farthest")) {
    int distances[FW] = {0}, max = 0, d, port;
    for (i = 0; i < world; i++) if (i != rank) { int lane;
      for (lane = 0; lane < r->lanes; lane++) if (r->route[rank][i][lane].hops - 1 > distances[i]) distances[i] = r->route[rank][i][lane].hops - 1;
      if (distances[i] > max) max = distances[i];
    }
    for (d = max; d >= 0; d--) {
      int next[2] = {0, 0}, queues[2][FW], count[2] = {0, 0};
      for (i = 0; i < world; i++) if (i != rank && distances[i] == d) { port = r->route[rank][i][0].local_role / 2; queues[port][count[port]++] = i; }
      while (next[0] < count[0] || next[1] < count[1]) for (port = 0; port < 2; port++)
        if (next[port] < count[port]) peers[n++] = queues[port][next[port]++];
    } return 0;
  }
  while (*p) { int peer;
    if (number(&p, &peer) || peer >= world || peer == rank || seen[peer]++ || n == world - 1)
      return fail(e, SIRCL_FABRIC_INVALID, "posting order must name every peer exactly once");
    peers[n++] = peer; if (!*p) break; if (*p != ',') return fail(e, SIRCL_FABRIC_INVALID, "malformed posting order"); p++;
    skip_space(&p); if (!*p) return fail(e, SIRCL_FABRIC_INVALID, "posting order ends in an empty peer");
  }
  return n == world - 1 ? 0 : fail(e, SIRCL_FABRIC_INVALID, "posting order must name every peer exactly once");
}
static int queue_loads(const sircl_fabric_routes *r, const int *order, int load[FW][2][2]) {
  int rank, peer, lane, k; memset(load, 0, sizeof(int) * FW * 2 * 2);
  for (rank = 0; rank < r->layout.world; rank++) for (peer = 0; peer < r->layout.world; peer++) if (peer != rank) {
    if (order) { int i, enabled = 0;
      for (i = 0; i < r->layout.world; i++) {
        if (order[i] == rank && order[(i + 1) % r->layout.world] == peer) enabled = 1;
      }
      if (!enabled) continue;
    }
    for (lane = 0; lane < r->lanes; lane++) {
      const sircl_lane_route *route = &r->route[rank][peer][lane];
      if (route->hops < 1 || route->hops > r->layout.node_count || route->local_role < 0 || route->local_role > 3) return -1;
      for (k = 0; k < route->hops - 1; k++) {
        int node = node_at(&r->layout, route->steps[k].dst), port = route->steps[k].dst_port;
        if (node < 0 || port < 0 || port > 1) return -1;
        load[node][route->local_role % 2][1 - port]++;
      }
    }
  } return 0;
}
int sircl_fabric_relay_load(const sircl_fabric_routes *r, int *busiest, double *factor, sircl_fabric_error *e) {
  int load[FW][2][2], i, p, f, max = 0; clear_error(e);
  if (!routes_valid(r) || !busiest || !factor || queue_loads(r, NULL, load)) return fail(e, SIRCL_FABRIC_INVALID, "invalid relay routes");
  for (i = 0; i < r->layout.node_count; i++) for (p = 0; p < 2; p++) for (f = 0; f < 2; f++) if (load[i][f][p] > max) max = load[i][f][p];
  *busiest = max; *factor = (double)max / r->lanes; return 0;
}
static uint64_t queue_share(uint64_t bytes) { return (bytes / 4) * 3 + (bytes % 4) * 3 / 4; }
int sircl_fabric_forward_windows(const sircl_fabric_routes *r, int rank, uint64_t max_window,
                                 uint64_t chunk, uint64_t queue_bytes, uint64_t out[FW][FL], sircl_fabric_error *e) {
  int load[FW][2][2], peer, lane; clear_error(e);
  if (!routes_valid(r) || !out || rank < 0 || rank >= r->layout.world) return fail(e, SIRCL_FABRIC_INVALID, "invalid forward window arguments");
  memset(out, 0, sizeof(uint64_t) * FW * FL); if (!max_window) return 0;
  if (!chunk || chunk % 16 || max_window < chunk) return fail(e, SIRCL_FABRIC_INVALID, "forward chunk must be a positive multiple of 16 within window");
  if (queue_loads(r, NULL, load)) return fail(e, SIRCL_FABRIC_INVALID, "invalid relay routes");
  for (peer = 0; peer < r->layout.world; peer++) if (peer != rank) for (lane = 0; lane < r->lanes; lane++) {
    const sircl_lane_route *route = &r->route[rank][peer][lane]; int k, sharing = 0;
    for (k = 0; k < route->hops - 1; k++) { int q = load[node_at(&r->layout, route->steps[k].dst)][route->local_role % 2][1 - route->steps[k].dst_port]; if (q > sharing) sharing = q; }
    if (sharing) { uint64_t limit = queue_share(queue_bytes) / (uint64_t)sharing;
      /* Fail closed if one chunk already exceeds the queue budget. */
      if (limit < chunk) return fail(e, SIRCL_FABRIC_LIMIT, "relay queue budget cannot hold one chunk per lane");
      if (max_window < limit) limit = max_window;
      out[peer][lane] = limit / chunk * chunk;
    }
  } return 0;
}
int sircl_fabric_chain_order(const sircl_fabric_routes *r, int *order, sircl_fabric_error *e) {
  const sircl_fabric_layout *l; int nodes[FW], visited[FW] = {0}, i, start, previous = -1; clear_error(e);
  if (!routes_valid(r) || !order) return fail(e, SIRCL_FABRIC_INVALID, "invalid chain arguments");
  l = &r->layout;
  if (l->world != l->node_count) return fail(e, SIRCL_FABRIC_UNREACHABLE, "chain requires a rank at every fabric position");
  start = l->positions[0];
  if (!l->cycle) for (i = 0; i < l->node_count; i++) {
    sircl_fabric_step s; if (!cross(l, l->nodes[i], 0, &s) || !cross(l, l->nodes[i], 1, &s)) { start = l->nodes[i]; break; }
  }
  nodes[0] = start; visited[node_at(l, start)] = 1;
  for (i = 1; i < l->world; i++) {
    int port, found = 0; sircl_fabric_step step;
    for (port = 0; port < 2; port++) if (cross(l, nodes[i-1], port, &step) &&
        step.dst != previous && !visited[node_at(l, step.dst)]) {
      nodes[i] = step.dst; visited[node_at(l, step.dst)] = 1; found = 1; break;
    }
    if (!found) return fail(e, SIRCL_FABRIC_UNREACHABLE, "fabric has no complete chain");
    previous = nodes[i-1];
  }
  for (i = 0; i < l->world; i++) { int rank; for (rank = 0; rank < l->world && l->positions[rank] != nodes[i]; rank++) { }
    order[i] = rank;
  }
  for (i = 0; i < l->world - 1; i++) { int lane; for (lane = 0; lane < r->lanes; lane++)
    if (r->route[order[i]][order[i+1]][lane].hops != 1 || r->route[order[i+1]][order[i]][lane].hops != 1)
      return fail(e, SIRCL_FABRIC_UNREACHABLE, "chain edge has a relayed lane");
  } return 0;
}
int sircl_fabric_ring_window(const sircl_fabric_routes *r, const int *order, uint64_t chunk,
                             uint64_t queue_bytes, uint64_t *window, sircl_fabric_error *e) {
  int load[FW][2][2], seen[FW] = {0}, i, f, p, any = 0; clear_error(e);
  if (!routes_valid(r) || !order || !window || !chunk || chunk % 16) return fail(e, SIRCL_FABRIC_INVALID, "invalid ring window arguments");
  for (i = 0; i < r->layout.world; i++) if (order[i] < 0 || order[i] >= r->layout.world || seen[order[i]]++)
    return fail(e, SIRCL_FABRIC_INVALID, "ring order must name every rank exactly once");
  if (queue_loads(r, order, load)) return fail(e, SIRCL_FABRIC_INVALID, "invalid relay routes");
  for (i = 0; i < r->layout.node_count; i++) for (f = 0; f < 2; f++) for (p = 0; p < 2; p++) {
    if (load[i][f][p] > 1) return fail(e, SIRCL_FABRIC_LIMIT, "relay %d function %d port %d carries multiple ring lanes", r->layout.nodes[i], f, p);
    if (load[i][f][p]) any = 1;
  }
  *window = any ? queue_share(queue_bytes) / chunk * chunk : 0;
  return any && *window < chunk ? fail(e, SIRCL_FABRIC_LIMIT, "relay queue holds less than one chunk") : 0;
}
int sircl_protocol_stripe(uint64_t packs, int lanes, int lane, sircl_pack_range *out) {
  uint64_t base, rest; if (!out || lanes < 1 || lanes > FL || lane < 0 || lane >= lanes) return -1;
  base = packs / (uint64_t)lanes; rest = packs % (uint64_t)lanes;
  out->first = (uint64_t)lane * base + ((uint64_t)lane < rest ? (uint64_t)lane : rest);
  out->count = base + ((uint64_t)lane < rest); return 0;
}
int sircl_protocol_chunk(uint64_t packs, int world, int index, sircl_pack_range *out) {
  uint64_t base, rest, lo, hi; if (!out || world < 2 || world > FW || index < 0 || index >= world) return -1;
  /* Division before multiplication preserves floor(i*P/W) without overflow. */
  base = packs / (uint64_t)world; rest = packs % (uint64_t)world;
  lo = (uint64_t)index * base + (uint64_t)index * rest / (uint64_t)world;
  hi = (uint64_t)(index + 1) * base + (uint64_t)(index + 1) * rest / (uint64_t)world;
  out->first = lo; out->count = hi - lo; return 0;
}
int sircl_protocol_flag_index(int ns, int source, int slot, int lane, int world, int lanes, uint32_t *out) {
  /* The reference permits lane 1 with one active lane for reserved flag indexing. */
  if (!out || ns < 0 || ns > 1 || world < 2 || world > FW || source < 0 || source >= world ||
      slot < 0 || slot > 1 || lanes < 1 || lanes > FL || lane < 0 || lane >= FL) return -1;
  *out = (uint32_t)(ns * world * 2 * lanes + (source * 2 + slot) * lanes + lane); return 0;
}
int sircl_protocol_op_word(int op, uint32_t bytes, uint32_t *out) {
  if (!out || op < 0 || op > 3 || !bytes || bytes % 16 || bytes > SIRCL_PROTOCOL_OP_BYTES_MASK) return -1;
  *out = (uint32_t)op << 30 | bytes; return 0;
}
int sircl_protocol_descriptor(const sircl_phase *p, int world, uint32_t *out) {
  if (!p || !out || p->first < 0 || p->first > p->end || p->end > 31 || p->peer < 0 || p->peer > 15 ||
      p->namespace_id < 0 || p->namespace_id > 1 || world < 0 || world > FW ||
      (world && (p->end > world || p->peer >= world))) return -1;
  *out = UINT32_C(0x80000000) | (uint32_t)p->namespace_id << 14 | (uint32_t)p->peer << 10 |
    (uint32_t)p->end << 5 | (uint32_t)p->first; return 0;
}
static int swing_peer(int world, int rank, int step) {
  static const int offsets[4] = {1, -1, 3, -5}; int p;
  if (step < 0 || step >= 4) return rank;
  p = rank + (rank % 2 ? -offsets[step] : offsets[step]);
  return (p + world) % world;
}
static uint32_t reach(int world, int steps, int rank, int step) {
  if (step >= steps) return UINT32_C(1) << rank;
  return reach(world, steps, rank, step + 1) | reach(world, steps, swing_peer(world, rank, step), step + 1);
}
static int first_bit(uint32_t bits) { int i; for (i = 0; i < FW; i++) if (bits & (UINT32_C(1) << i)) return i; return FW; }
static void owners_walk(int world, int steps, int rank, int step, int *owners, int *count) {
  uint32_t a, b; int x, y;
  if (step >= steps) { owners[(*count)++] = rank; return; }
  a = reach(world, steps, rank, step + 1); b = reach(world, steps, swing_peer(world, rank, step), step + 1);
  x = first_bit(a); y = first_bit(b);
  if (y < x) { int t = x; x = y; y = t; }
  owners_walk(world, steps, x, step + 1, owners, count); owners_walk(world, steps, y, step + 1, owners, count);
}
static void phase_span(uint32_t bits, const int *owners, int world, sircl_phase *phase) {
  int i; phase->first = world; phase->end = 0;
  for (i = 0; i < world; i++) if (bits & (UINT32_C(1) << owners[i])) { if (i < phase->first) phase->first = i; phase->end = i + 1; }
}
int sircl_protocol_swing(int world, int rank, int *owners, sircl_phase phases[SIRCL_PROTOCOL_MAX_PHASES], int *phase_count) {
  int steps = 0, i, count = 0, n = 0;
  if (!owners || !phases || !phase_count || world < 2 || world > FW || (world & (world - 1)) || rank < 0 || rank >= world) return -1;
  for (i = world; i > 1; i >>= 1) steps++;
  owners_walk(world, steps, 0, 0, owners, &count);
  for (i = 0; i < steps; i++) { sircl_phase *p = &phases[n++]; p->peer = swing_peer(world, rank, i); p->namespace_id = 0; phase_span(reach(world, steps, p->peer, i + 1), owners, world, p); }
  for (i = steps - 1; i >= 0; i--) { sircl_phase *p = &phases[n++]; p->peer = swing_peer(world, rank, i); p->namespace_id = 1; phase_span(reach(world, steps, rank, i + 1), owners, world, p); }
  *phase_count = n; return 0;
}
int sircl_protocol_select(uint64_t bytes, int algorithm, int large, uint64_t oneshot_max,
                          uint64_t swing_above, unsigned available, int *out) {
  if (!out || algorithm < -1 || algorithm > 2 || large < 1 || large > 2) return -1;
  if (algorithm >= 0) { *out = algorithm; return 0; }
  if (bytes <= oneshot_max || !(available & (1u << large))) *out = SIRCL_ALGORITHM_ONESHOT;
  else if (swing_above && bytes > swing_above && (available & (1u << SIRCL_ALGORITHM_SWING))) *out = SIRCL_ALGORITHM_SWING;
  else *out = large;
  return 0;
}
