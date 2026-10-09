#define _GNU_SOURCE
#include "env_names.h"
#include "bootstrap.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <limits.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

enum { OK = 0, SYSTEM = 2, ARGUMENT = 4, USAGE = 5, REMOTE = 6, TIMEOUT = 8 };
enum { ID_SIZE = 128, HELLO_SIZE = 136, MAX_RANKS = 8, MAX_PENDING = 32,
       MAX_ROOTS = 64, FRAME_HEADER = 16, MAX_FRAME = 1 << 20 };
enum { VERSION_OFF = 8, IP_OFF = 12, PORT_OFF = 16,
       NONCE_OFF = 20, HASH_OFF = 36, RESERVED_OFF = 68 };
static const unsigned char id_magic[8] = {'S','C','C','L','B','O','0','1'};
static const unsigned char reply_magic[4] = {'S','C','C','1'};
/* All-gather frames after the group is complete: a rank sends a header (magic,
 * round, payload bytes, reserved) and its payload; the broker answers every
 * rank, once all ranks sent the round, with a header (magic, round, ranks,
 * total bytes) and each rank's byte count and payload in rank order. */
static const unsigned char frame_magic[4] = {'S','C','F','1'};
static const unsigned char gather_magic[4] = {'S','C','R','1'};
static _Thread_local char last_error[192];

struct sccl_bootstrap { int fd; int nranks; int rank; uint32_t round; };
struct peer {
  int fd;
  int rank;
  size_t used;
  unsigned char hello[HELLO_SIZE];
  unsigned char frame[FRAME_HEADER];
  size_t frame_used;
  unsigned char *payload;
  uint32_t payload_bytes;
  size_t payload_used;
  int frame_ready;
};
struct root {
  unsigned char id[ID_SIZE];
  int listen_fd;
  int wake[2];
  int nranks;
  int joined;
  uint32_t round;
  int64_t deadline;
  struct peer peers[MAX_PENDING];
  pthread_t thread;
  pthread_mutex_t wake_lock;
  atomic_int stopped;
  atomic_int finished;
  struct root *next;
};
static pthread_mutex_t roots_lock = PTHREAD_MUTEX_INITIALIZER;
static struct root *roots;
static atomic_long process_owner = ATOMIC_VAR_INIT(0);

int sccl_bootstrap_process_valid(void) {
  long pid = (long)getpid();
  long owner = atomic_load_explicit(&process_owner, memory_order_relaxed);
  if (!owner) {
    long expected = 0;
    if (atomic_compare_exchange_strong_explicit(&process_owner, &expected, pid,
          memory_order_relaxed, memory_order_relaxed)) return 1;
    owner = expected;
  }
  return owner == pid;
}

static int error_result(int code, const char *message) {
  (void)snprintf(last_error, sizeof last_error, "%s", message);
  return code;
}

const char *sccl_bootstrap_error(void) { return last_error; }

static int64_t now_ms(void) {
  struct timespec t;
  if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) return -1;
  return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}

static int timeout_ms(int *out) {
  const char *value = sccl_env("SIRCL_BOOTSTRAP_TIMEOUT_MS");
  char *end;
  long parsed;
  if (!value || !*value) { *out = 10000; return OK; }
  errno = 0;
  parsed = strtol(value, &end, 10);
  if (errno || *end || parsed < 1 || parsed > 600000)
    return error_result(ARGUMENT,
      "SIRCL_BOOTSTRAP_TIMEOUT_MS must be an integer from 1 to 600000");
  *out = (int)parsed;
  return OK;
}

/* Deadline of one all-gather round (LIBSIRCL_SETUP_TIMEOUT_MS, 1 ms to 1 h;
 * default 600 s, the startup wait regime: ranks may lag while they allocate,
 * register and connect). */
static int frame_deadline_ms(void) {
  const char *value = sccl_env("LIBSIRCL_SETUP_TIMEOUT_MS");
  char *end;
  if (value && *value) {
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (!errno && !*end && parsed >= 1 && parsed <= 3600000) return (int)parsed;
  }
  return 600000;
}

static int hex_digit(unsigned char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static int site_hash(unsigned char out[32]) {
  const char *value = sccl_env("SIRCL_SITE_HASH");
  memset(out, 0, 32);
  if (!value || !*value) return OK;
  if (strlen(value) != 64)
    return error_result(ARGUMENT, "SIRCL_SITE_HASH must contain 64 hexadecimal digits");
  for (int i = 0; i < 32; ++i) {
    int a = hex_digit((unsigned char)value[2*i]);
    int b = hex_digit((unsigned char)value[2*i+1]);
    if (a < 0 || b < 0)
      return error_result(ARGUMENT, "SIRCL_SITE_HASH must contain 64 hexadecimal digits");
    out[i] = (unsigned char)((a << 4) | b);
  }
  return OK;
}

static void put32(unsigned char *p, uint32_t value) {
  p[0] = (unsigned char)(value >> 24); p[1] = (unsigned char)(value >> 16);
  p[2] = (unsigned char)(value >> 8); p[3] = (unsigned char)value;
}

static uint32_t get32(const unsigned char *p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
         ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static int configure_fd(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0 ||
      fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) return -1;
  return 0;
}

static int remaining_ms(int64_t deadline) {
  int64_t now = now_ms();
  if (now < 0 || now >= deadline) return 0;
  int64_t remaining = deadline - now;
  return remaining > INT_MAX ? INT_MAX : (int)remaining;
}

static int is_cancelled(const atomic_int *cancelled) {
  return cancelled && atomic_load_explicit(cancelled, memory_order_relaxed);
}

static int wait_fd(int fd, short events, int64_t deadline,
                   const atomic_int *cancelled) {
  struct pollfd p = {fd, events, 0};
  for (;;) {
    if (is_cancelled(cancelled)) return USAGE;
    int remaining = remaining_ms(deadline);
    if (!remaining) return TIMEOUT;
    if (cancelled && remaining > 50) remaining = 50;
    int rc = poll(&p, 1, remaining);
    if (rc < 0) { if (errno == EINTR) continue; return SYSTEM; }
    if (!rc) continue;
    /* Let recv consume buffered replies even when POLLHUP is also present. */
    if (p.revents & events) return OK;
    if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) return REMOTE;
  }
}

static int send_all(int fd, const unsigned char *data, size_t count,
                    int64_t deadline, const atomic_int *cancelled) {
  while (count) {
    if (is_cancelled(cancelled)) return USAGE;
    ssize_t n = send(fd, data, count, MSG_NOSIGNAL);
    if (n > 0) { data += n; count -= (size_t)n; continue; }
    if (n < 0 && errno == EINTR) continue;
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      int rc = wait_fd(fd, POLLOUT, deadline, cancelled);
      if (rc != OK) return rc;
      continue;
    }
    return REMOTE;
  }
  return OK;
}

static int recv_all(int fd, unsigned char *data, size_t count,
                    int64_t deadline, const atomic_int *cancelled) {
  while (count) {
    if (is_cancelled(cancelled)) return USAGE;
    ssize_t n = recv(fd, data, count, 0);
    if (n > 0) { data += n; count -= (size_t)n; continue; }
    if (n < 0 && errno == EINTR) continue;
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      int rc = wait_fd(fd, POLLIN, deadline, cancelled);
      if (rc != OK) return rc;
      continue;
    }
    return REMOTE;
  }
  return OK;
}

static void reply(int fd, int result) {
  unsigned char data[8];
  memcpy(data, reply_magic, 4);
  put32(data + 4, (uint32_t)result);
  /* Eight bytes fit in the empty TCP send queue. This never waits on an
   * untrusted connection; the client reports a lost reply as a remote error. */
  (void)send(fd, data, sizeof data, MSG_NOSIGNAL);
}

static void drop_peer(struct root *r, int index, int result) {
  struct peer *p = &r->peers[index];
  if (p->fd < 0) return;
  if (result >= 0) reply(p->fd, result);
  if (p->rank >= 0) --r->joined;
  close(p->fd);
  p->fd = -1; p->rank = -1; p->used = 0;
  free(p->payload);
  p->payload = NULL; p->payload_bytes = 0; p->payload_used = 0;
  p->frame_used = 0; p->frame_ready = 0;
}

/* Read the next part of a joined peer's all-gather frame. Returns 0 while the
 * frame is incomplete or complete, -1 when the peer must be dropped. */
static int read_frame(struct root *r, struct peer *p) {
  if (p->frame_ready) {
    unsigned char byte;
    ssize_t got = recv(p->fd, &byte, 1, MSG_PEEK);
    if (got < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) return 0;
    return -1; /* closed, or bytes ahead of the broker's answer */
  }
  if (p->frame_used < FRAME_HEADER) {
    ssize_t got = recv(p->fd, p->frame + p->frame_used, FRAME_HEADER - p->frame_used, 0);
    if (got == 0) return -1;
    if (got < 0) return errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK ? 0 : -1;
    p->frame_used += (size_t)got;
    if (p->frame_used < FRAME_HEADER) return 0;
    if (memcmp(p->frame, frame_magic, 4) || get32(p->frame + 4) != r->round ||
        get32(p->frame + 8) > MAX_FRAME) return -1;
    p->payload_bytes = get32(p->frame + 8);
    p->payload_used = 0;
    p->payload = malloc(p->payload_bytes ? p->payload_bytes : 1);
    if (!p->payload) return -1;
  }
  if (p->payload_used < p->payload_bytes) {
    ssize_t got = recv(p->fd, p->payload + p->payload_used, p->payload_bytes - p->payload_used, 0);
    if (got == 0) return -1;
    if (got < 0) return errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK ? 0 : -1;
    p->payload_used += (size_t)got;
  }
  if (p->payload_used == p->payload_bytes) p->frame_ready = 1;
  return 0;
}

/* When every rank sent this round's frame, answer every rank and start the next
 * round. Returns -1 when an answer could not be delivered. */
static int finish_round(struct root *r) {
  struct peer *by_rank[MAX_RANKS] = {0};
  size_t total = 0;
  for (int i = 0; i < MAX_PENDING; ++i) {
    struct peer *p = &r->peers[i];
    if (p->fd < 0 || p->rank < 0) continue;
    if (!p->frame_ready) return 0;
    by_rank[p->rank] = p;
    total += 4 + p->payload_bytes;
  }
  for (int k = 0; k < r->nranks; ++k) if (!by_rank[k]) return 0;
  unsigned char *answer = malloc(FRAME_HEADER + total);
  if (!answer) return -1;
  memcpy(answer, gather_magic, 4);
  put32(answer + 4, r->round);
  put32(answer + 8, (uint32_t)r->nranks);
  put32(answer + 12, (uint32_t)total);
  size_t at = FRAME_HEADER;
  for (int k = 0; k < r->nranks; ++k) {
    put32(answer + at, by_rank[k]->payload_bytes);
    memcpy(answer + at + 4, by_rank[k]->payload, by_rank[k]->payload_bytes);
    at += 4 + by_rank[k]->payload_bytes;
  }
  int result = 0;
  int64_t deadline = now_ms() + frame_deadline_ms();
  for (int k = 0; k < r->nranks; ++k) {
    if (send_all(by_rank[k]->fd, answer, at, deadline, NULL) != OK) result = -1;
    free(by_rank[k]->payload);
    by_rank[k]->payload = NULL;
    by_rank[k]->payload_bytes = 0; by_rank[k]->payload_used = 0;
    by_rank[k]->frame_used = 0; by_rank[k]->frame_ready = 0;
  }
  free(answer);
  ++r->round;
  return result;
}

static int accept_hello(struct root *r, int index) {
  struct peer *p = &r->peers[index];
  if (memcmp(p->hello, r->id, ID_SIZE) != 0) return REMOTE;
  uint32_t nranks = get32(p->hello + ID_SIZE);
  uint32_t rank = get32(p->hello + ID_SIZE + 4);
  if (nranks < 1 || nranks > MAX_RANKS || rank >= nranks) return ARGUMENT;
  if (r->nranks && r->nranks != (int)nranks) return USAGE;
  for (int i = 0; i < MAX_PENDING; ++i)
    if (r->peers[i].rank == (int)rank) return USAGE;
  r->nranks = (int)nranks;
  p->rank = (int)rank;
  ++r->joined;
  return OK;
}

static void accept_connections(struct root *r) {
  /* Cap each batch so connections cannot starve the absolute deadline. */
  for (int attempt = 0; attempt < MAX_PENDING; ++attempt) {
    int fd = accept(r->listen_fd, NULL, NULL);
    if (fd < 0) { if (errno == EINTR) continue; return; }
    if (configure_fd(fd) != 0) { close(fd); continue; }
    int slot;
    for (slot = 0; slot < MAX_PENDING; ++slot)
      if (r->peers[slot].fd < 0) break;
    if (slot == MAX_PENDING) { reply(fd, USAGE); close(fd); continue; }
    r->peers[slot].fd = fd;
    r->peers[slot].rank = -1;
    r->peers[slot].used = 0;
  }
}

static void *root_thread(void *arg) {
  struct root *r = arg;
  int ready = 0, broken = 0;
  while (!atomic_load_explicit(&r->stopped, memory_order_relaxed)) {
    struct pollfd pfds[MAX_PENDING + 2];
    int mapping[MAX_PENDING + 2];
    int n = 0;
    pfds[n] = (struct pollfd){r->wake[0], POLLIN, 0}; mapping[n++] = -2;
    if (!ready) {
      int remaining = remaining_ms(r->deadline);
      if (!remaining) break;
      pfds[n] = (struct pollfd){r->listen_fd, POLLIN, 0}; mapping[n++] = -1;
    }
    for (int i = 0; i < MAX_PENDING; ++i) {
      if (r->peers[i].fd < 0) continue;
      pfds[n] = (struct pollfd){r->peers[i].fd, POLLIN, 0}; mapping[n++] = i;
    }
    if (ready && !r->joined) break;
    int delay = ready ? 100 : remaining_ms(r->deadline);
    if (!ready && !delay) break;
    int rc = poll(pfds, (nfds_t)n, delay);
    if (rc < 0) { if (errno == EINTR) continue; break; }
    for (int j = 0; j < n; ++j) {
      if (!pfds[j].revents) continue;
      int i = mapping[j];
      if (i == -2) goto done;
      if (i == -1) { accept_connections(r); continue; }
      struct peer *p = &r->peers[i];
      if (p->fd < 0) continue;
      if (pfds[j].revents & (POLLERR | POLLNVAL)) {
        drop_peer(r, i, -1); continue;
      }
      if (ready && p->rank >= 0) {
        /* A rank that leaves a complete group breaks every later round:
         * every rank is disconnected, so none waits for its deadline. */
        if (read_frame(r, p) != 0) broken = 1;
        continue;
      }
      if (ready || p->rank >= 0) {
        unsigned char byte;
        ssize_t got = recv(p->fd, &byte, 1, MSG_PEEK);
        if (got == 0 || (got < 0 && errno != EINTR && errno != EAGAIN &&
                         errno != EWOULDBLOCK)) drop_peer(r, i, -1);
        /* Control frames have no implementation in the offline scaffold.
         * Refuse unexpected bytes rather than spinning on unread data. */
        else if (got > 0) drop_peer(r, i, USAGE);
        continue;
      }
      ssize_t got = recv(p->fd, p->hello + p->used, HELLO_SIZE - p->used, 0);
      if (got == 0 || (got < 0 && errno != EINTR && errno != EAGAIN &&
                       errno != EWOULDBLOCK)) { drop_peer(r, i, -1); continue; }
      if (got <= 0) continue;
      p->used += (size_t)got;
      if (p->used == HELLO_SIZE) {
        int result = accept_hello(r, i);
        if (result != OK) drop_peer(r, i, result);
      }
    }
    if (ready && !broken && finish_round(r) != 0) broken = 1;
    if (broken) {
      for (int i = 0; i < MAX_PENDING; ++i) drop_peer(r, i, -1);
      break;
    }
    if (!ready && r->nranks && r->joined == r->nranks) {
      ready = 1;
      close(r->listen_fd); r->listen_fd = -1;
      for (int i = 0; i < MAX_PENDING; ++i) {
        if (r->peers[i].fd < 0) continue;
        if (r->peers[i].rank >= 0) reply(r->peers[i].fd, OK);
        else drop_peer(r, i, REMOTE);
      }
    }
  }
done:
  if (r->listen_fd >= 0) { close(r->listen_fd); r->listen_fd = -1; }
  for (int i = 0; i < MAX_PENDING; ++i)
    drop_peer(r, i, ready ? -1 : TIMEOUT);
  pthread_mutex_lock(&r->wake_lock);
  close(r->wake[0]); close(r->wake[1]);
  r->wake[0] = -1; r->wake[1] = -1;
  pthread_mutex_unlock(&r->wake_lock);
  /* Finished roots retain only join bookkeeping until the next API call or
   * library unload. Listeners, peer sockets and worker execution end here. */
  atomic_store_explicit(&r->finished, 1, memory_order_release);
  return NULL;
}

static void free_root(struct root *r) {
  (void)pthread_join(r->thread, NULL);
  pthread_mutex_destroy(&r->wake_lock);
  free(r);
}

static void reap_finished(void) {
  if (!sccl_bootstrap_process_valid()) return;
  pthread_mutex_lock(&roots_lock);
  struct root **link = &roots;
  while (*link) {
    struct root *r = *link;
    if (!atomic_load_explicit(&r->finished, memory_order_acquire)) {
      link = &r->next; continue;
    }
    *link = r->next;
    free_root(r);
  }
  pthread_mutex_unlock(&roots_lock);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((destructor))
#endif
static void shutdown_roots(void) {
  /* Fork inherits pipe descriptors but not their servicing worker threads.
   * Writing an inherited wake pipe would terminate the parent's broker. */
  if (!sccl_bootstrap_process_valid()) return;
  pthread_mutex_lock(&roots_lock);
  for (struct root *r = roots; r; r = r->next) {
    atomic_store_explicit(&r->stopped, 1, memory_order_relaxed);
    pthread_mutex_lock(&r->wake_lock);
    if (r->wake[1] >= 0) {
      ssize_t ignored = write(r->wake[1], "x", 1);
      (void)ignored;
    }
    pthread_mutex_unlock(&r->wake_lock);
  }
  while (roots) {
    struct root *r = roots;
    roots = r->next;
    free_root(r);
  }
  pthread_mutex_unlock(&roots_lock);
}

/* LAN bootstrap is configured when any of SIRCL_BOOTSTRAP_ADDR,
 * SIRCL_BOOTSTRAP_IFNAME or NCCL_SOCKET_IFNAME is set. */
static int lan_configured(void) {
  const char *names[] = {"SIRCL_BOOTSTRAP_ADDR", "SIRCL_BOOTSTRAP_IFNAME", "NCCL_SOCKET_IFNAME"};
  for (int i = 0; i < 3; ++i) {
    const char *value = sccl_env(names[i]);
    if (value && *value) return 1;
  }
  return 0;
}

/* Interface list match: comma-separated prefixes, "=name" for an exact name,
 * "^prefix" entries exclude. */
static int interface_matches(const char *list, const char *name) {
  char copy[256];
  snprintf(copy, sizeof copy, "%s", list);
  int included = 0, any_include = 0;
  for (char *save = NULL, *item = strtok_r(copy, ",", &save); item; item = strtok_r(NULL, ",", &save)) {
    while (*item == ' ') ++item;
    if (*item == '^') {
      if (!strncmp(name, item + 1, strlen(item + 1))) return 0;
      continue;
    }
    any_include = 1;
    if (*item == '=' ? !strcmp(name, item + 1) : !strncmp(name, item, strlen(item))) included = 1;
  }
  return any_include ? included : 1;
}

/* The IPv4 address a new root listens on and advertises: SIRCL_BOOTSTRAP_ADDR,
 * else the first address of an interface named by SIRCL_BOOTSTRAP_IFNAME or
 * NCCL_SOCKET_IFNAME, else 127.0.0.1. */
static int root_address(struct in_addr *out) {
  out->s_addr = htonl(INADDR_LOOPBACK);
  const char *address = sccl_env("SIRCL_BOOTSTRAP_ADDR");
  if (address && *address) {
    if (inet_pton(AF_INET, address, out) != 1)
      return error_result(ARGUMENT, "SIRCL_BOOTSTRAP_ADDR must be an IPv4 address");
    return OK;
  }
  const char *list = sccl_env("SIRCL_BOOTSTRAP_IFNAME");
  if (!list || !*list) list = sccl_env("NCCL_SOCKET_IFNAME");
  if (!list || !*list) return OK;
  struct ifaddrs *all = NULL;
  if (getifaddrs(&all) != 0) return error_result(SYSTEM, "cannot list network interfaces");
  int found = 0;
  for (struct ifaddrs *a = all; a && !found; a = a->ifa_next) {
    if (!a->ifa_addr || a->ifa_addr->sa_family != AF_INET || !(a->ifa_flags & IFF_UP)) continue;
    if (!interface_matches(list, a->ifa_name)) continue;
    *out = ((struct sockaddr_in *)a->ifa_addr)->sin_addr;
    found = 1;
  }
  freeifaddrs(all);
  if (!found) return error_result(ARGUMENT, "no IPv4 address on the bootstrap interfaces named by "
                                            "SIRCL_BOOTSTRAP_IFNAME or NCCL_SOCKET_IFNAME");
  return OK;
}

static int random_nonce(unsigned char *out) {
  int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
  if (fd < 0) return -1;
  size_t used = 0;
  while (used < 16) {
    ssize_t n = read(fd, out + used, 16 - used);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) { close(fd); return -1; }
    used += (size_t)n;
  }
  close(fd);
  return 0;
}

int sccl_bootstrap_id(unsigned char id[ID_SIZE]) {
  int timeout, result;
  unsigned char hash[32];
  if (!sccl_bootstrap_process_valid())
    return error_result(USAGE, "fork after library use is unsupported; exec is required in the child");
  last_error[0] = 0;
  if (!id) return error_result(ARGUMENT, "bootstrap ID output is null");
  memset(id, 0, ID_SIZE);
  if ((result = timeout_ms(&timeout)) != OK ||
      (result = site_hash(hash)) != OK) return result;
  reap_finished();
  struct root *r = calloc(1, sizeof *r);
  if (!r) return error_result(SYSTEM, "cannot allocate bootstrap root");
  if (pthread_mutex_init(&r->wake_lock, NULL) != 0) {
    free(r);
    return error_result(SYSTEM, "cannot allocate bootstrap wake lock");
  }
  r->listen_fd = -1; r->wake[0] = -1; r->wake[1] = -1;
  for (int i = 0; i < MAX_PENDING; ++i) {
    r->peers[i].fd = -1; r->peers[i].rank = -1;
  }
  atomic_init(&r->stopped, 0); atomic_init(&r->finished, 0);
  memcpy(r->id, id_magic, 8); put32(r->id + VERSION_OFF, 1);
  struct in_addr bind_address;
  if ((result = root_address(&bind_address)) != OK) {
    pthread_mutex_destroy(&r->wake_lock);
    free(r);
    return result;
  }
  memcpy(r->id + IP_OFF, &bind_address.s_addr, 4);
  memcpy(r->id + HASH_OFF, hash, 32);
  if (random_nonce(r->id + NONCE_OFF) != 0) goto failed;
  r->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (r->listen_fd < 0 || configure_fd(r->listen_fd) != 0) goto failed;
  struct sockaddr_in address;
  memset(&address, 0, sizeof address);
  address.sin_family = AF_INET;
  address.sin_addr = bind_address;
  if (bind(r->listen_fd, (struct sockaddr *)&address, sizeof address) < 0 ||
      listen(r->listen_fd, MAX_PENDING) < 0) goto failed;
  socklen_t length = sizeof address;
  if (getsockname(r->listen_fd, (struct sockaddr *)&address, &length) < 0)
    goto failed;
  memcpy(r->id + PORT_OFF, &address.sin_port, 2);
  if (pipe(r->wake) != 0 || configure_fd(r->wake[0]) != 0 ||
      configure_fd(r->wake[1]) != 0) goto failed;
  int64_t now = now_ms();
  if (now < 0) goto failed;
  r->deadline = now + timeout;
  pthread_mutex_lock(&roots_lock);
  int count = 0;
  for (struct root *entry = roots; entry; entry = entry->next) ++count;
  if (count >= MAX_ROOTS) {
    pthread_mutex_unlock(&roots_lock);
    result = error_result(USAGE, "at most 64 active bootstrap IDs are supported");
    goto cleanup;
  }
  if (pthread_create(&r->thread, NULL, root_thread, r) != 0) {
    pthread_mutex_unlock(&roots_lock); goto failed;
  }
  r->next = roots; roots = r;
  memcpy(id, r->id, ID_SIZE);
  pthread_mutex_unlock(&roots_lock);
  return OK;
failed:
  result = error_result(SYSTEM, "cannot create the bootstrap listener or nonce");
cleanup:
  if (r->listen_fd >= 0) close(r->listen_fd);
  if (r->wake[0] >= 0) close(r->wake[0]);
  if (r->wake[1] >= 0) close(r->wake[1]);
  pthread_mutex_destroy(&r->wake_lock);
  free(r);
  return result;
}

static int validate_id(const unsigned char id[ID_SIZE]) {
  if (memcmp(id, id_magic, 8) || get32(id + VERSION_OFF) != 1)
    return error_result(ARGUMENT, "invalid or unsupported bootstrap ID format");
  int loopback = id[IP_OFF] == 127 && !id[IP_OFF+1] && !id[IP_OFF+2] && id[IP_OFF+3] == 1;
  if (!loopback && !lan_configured())
    return error_result(USAGE, "a bootstrap ID with a non-loopback address needs SIRCL_BOOTSTRAP_ADDR, "
                               "SIRCL_BOOTSTRAP_IFNAME or NCCL_SOCKET_IFNAME on the joining rank");
  if (!id[IP_OFF] && !id[IP_OFF+1] && !id[IP_OFF+2] && !id[IP_OFF+3])
    return error_result(ARGUMENT, "invalid bootstrap ID address");
  if ((!id[PORT_OFF] && !id[PORT_OFF+1]) || id[18] || id[19])
    return error_result(ARGUMENT, "invalid bootstrap ID port or reserved field");
  for (int i = RESERVED_OFF; i < ID_SIZE; ++i)
    if (id[i]) return error_result(ARGUMENT, "invalid bootstrap ID reserved field");
  return OK;
}

int sccl_bootstrap_join_cancel(const unsigned char id[ID_SIZE], int nranks, int rank,
                               sccl_bootstrap **out, const atomic_int *cancelled) {
  unsigned char hash[32], hello[HELLO_SIZE], response[8];
  int timeout, result, fd = -1;
  if (out) *out = NULL;
  if (!sccl_bootstrap_process_valid())
    return error_result(USAGE, "fork after library use is unsupported; exec is required in the child");
  last_error[0] = 0;
  if (!id || !out || nranks < 1 || nranks > MAX_RANKS || rank < 0 || rank >= nranks)
    return error_result(ARGUMENT, "bootstrap requires a non-null ID/output and ranks within 1..8");
  if (is_cancelled(cancelled))
    return error_result(USAGE, "bootstrap initialization cancelled");
  if ((result = timeout_ms(&timeout)) != OK ||
      (result = validate_id(id)) != OK ||
      (result = site_hash(hash)) != OK) return result;
  if (memcmp(hash, id + HASH_OFF, 32))
    return error_result(USAGE, "bootstrap site hash does not match SIRCL_SITE_HASH");
  reap_finished();
  int64_t now = now_ms();
  if (now < 0) return error_result(SYSTEM, "cannot read monotonic clock");
  int64_t deadline = now + timeout;
  fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0 || configure_fd(fd) != 0) { result = SYSTEM; goto failed; }
  struct sockaddr_in address;
  memset(&address, 0, sizeof address);
  address.sin_family = AF_INET;
  memcpy(&address.sin_addr.s_addr, id + IP_OFF, 4);
  memcpy(&address.sin_port, id + PORT_OFF, 2);
  if (connect(fd, (struct sockaddr *)&address, sizeof address) < 0) {
    if (errno != EINPROGRESS) { result = REMOTE; goto failed; }
    if ((result = wait_fd(fd, POLLOUT, deadline, cancelled)) != OK) goto failed;
    int socket_error = 0;
    socklen_t length = sizeof socket_error;
    if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &length) < 0) {
      result = SYSTEM; goto failed;
    }
    if (socket_error) { result = REMOTE; goto failed; }
  }
  memcpy(hello, id, ID_SIZE);
  put32(hello + ID_SIZE, (uint32_t)nranks);
  put32(hello + ID_SIZE + 4, (uint32_t)rank);
  if ((result = send_all(fd, hello, sizeof hello, deadline, cancelled)) != OK ||
      (result = recv_all(fd, response, sizeof response, deadline, cancelled)) != OK) goto failed;
  if (memcmp(response, reply_magic, 4)) { result = REMOTE; goto failed; }
  result = (int)get32(response + 4);
  if (result != OK) {
    if (result != ARGUMENT && result != USAGE && result != REMOTE && result != TIMEOUT)
      result = REMOTE;
    goto failed;
  }
  if (is_cancelled(cancelled)) { result = USAGE; goto failed; }
  sccl_bootstrap *handle = malloc(sizeof *handle);
  if (!handle) { result = SYSTEM; goto failed; }
  handle->fd = fd;
  handle->nranks = nranks;
  handle->rank = rank;
  handle->round = 0;
  *out = handle;
  return OK;
failed:
  if (fd >= 0) close(fd);
  if (is_cancelled(cancelled))
    return error_result(USAGE, "bootstrap initialization cancelled");
  switch (result) {
    case TIMEOUT: return error_result(result, "bootstrap timed out waiting for every rank");
    case USAGE: return error_result(result, "bootstrap rejected a duplicate rank or mismatched world size");
    case ARGUMENT: return error_result(result, "bootstrap rejected invalid rank arguments");
    case REMOTE: return error_result(result, "bootstrap endpoint rejected the ID or disconnected");
    default: return error_result(SYSTEM, "bootstrap socket or allocation failure");
  }
}

int sccl_bootstrap_join(const unsigned char id[ID_SIZE], int nranks, int rank,
                        sccl_bootstrap **out) {
  return sccl_bootstrap_join_cancel(id, nranks, rank, out, NULL);
}

void sccl_bootstrap_close(sccl_bootstrap *handle) {
  if (!sccl_bootstrap_process_valid()) {
    (void)error_result(USAGE, "fork after library use is unsupported; exec is required in the child");
    return;
  }
  if (handle) { close(handle->fd); free(handle); }
  reap_finished();
}

int sccl_bootstrap_fd(const sccl_bootstrap *handle) {
  if (!sccl_bootstrap_process_valid()) {
    (void)error_result(USAGE, "fork after library use is unsupported; exec is required in the child");
    return -1;
  }
  return handle ? handle->fd : -1;
}

int sccl_bootstrap_allgather(sccl_bootstrap *handle, const void *data, uint32_t bytes,
                             void **out, uint32_t *lengths, const atomic_int *cancelled) {
  return sccl_bootstrap_allgather_within(handle, data, bytes, out, lengths, frame_deadline_ms(), cancelled);
}

int sccl_bootstrap_allgather_within(sccl_bootstrap *handle, const void *data, uint32_t bytes, void **out,
                                    uint32_t *lengths, int timeout_ms, const atomic_int *cancelled) {
  if (out) *out = NULL;
  if (!sccl_bootstrap_process_valid())
    return error_result(USAGE, "fork after library use is unsupported; exec is required in the child");
  if (!handle || !out || !lengths || (bytes && !data) || bytes > MAX_FRAME || timeout_ms < 1)
    return error_result(ARGUMENT, "all-gather needs a handle, outputs, at most 1 MiB per rank and a timeout");
  int64_t deadline = now_ms() + timeout_ms;
  unsigned char header[FRAME_HEADER];
  memcpy(header, frame_magic, 4);
  put32(header + 4, handle->round);
  put32(header + 8, bytes);
  put32(header + 12, 0);
  int result = send_all(handle->fd, header, sizeof header, deadline, cancelled);
  if (result == OK && bytes) result = send_all(handle->fd, data, bytes, deadline, cancelled);
  if (result == OK) result = recv_all(handle->fd, header, sizeof header, deadline, cancelled);
  if (result != OK) goto failed;
  uint32_t total = get32(header + 12);
  if (memcmp(header, gather_magic, 4) || get32(header + 4) != handle->round ||
      get32(header + 8) != (uint32_t)handle->nranks ||
      total > (uint32_t)handle->nranks * (4u + MAX_FRAME)) { result = REMOTE; goto failed; }
  unsigned char *body = malloc(total ? total : 1);
  if (!body) return error_result(SYSTEM, "cannot allocate the all-gather answer");
  result = recv_all(handle->fd, body, total, deadline, cancelled);
  if (result != OK) { free(body); goto failed; }
  /* Repack as the contributions back to back, without their length words. */
  size_t at = 0, packed = 0;
  for (int k = 0; k < handle->nranks; ++k) {
    if (at + 4 > total) { free(body); result = REMOTE; goto failed; }
    uint32_t length = get32(body + at);
    if (at + 4 + length > total) { free(body); result = REMOTE; goto failed; }
    memmove(body + packed, body + at + 4, length);
    lengths[k] = length;
    packed += length;
    at += 4 + length;
  }
  ++handle->round;
  *out = body;
  return OK;
failed:
  switch (result) {
    case TIMEOUT: return error_result(TIMEOUT, "bootstrap all-gather timed out waiting for every rank");
    case USAGE: return error_result(USAGE, "bootstrap all-gather cancelled");
    default: return error_result(REMOTE, "bootstrap all-gather lost the broker or a rank");
  }
}
