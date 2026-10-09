/* The MPI subset of include/mpi.h over TCP between the processes of one job.
 *
 * Connections: every rank connects to rank 0 (SIRCL_MPI_ROOT) and announces the
 * address of a listener of its own; rank 0 answers with every rank's address,
 * and each rank then connects to the listeners of the ranks below it, so every
 * pair of processes shares one socket.
 *
 * Messages: a header (communicator id, the communicator's collective number,
 * payload bytes) and the payload. A process waiting for a message from a peer
 * reads that peer's socket; messages of other collectives that arrive first are
 * queued and matched later, so collectives on different communicators may
 * interleave.
 *
 * Collectives among a communicator's members: an all-to-all exchange goes
 * through the communicator's rank 0 (every member sends its contribution, rank
 * 0 answers every member with all contributions in communicator rank order);
 * rooted collectives send to or from the root only. Reductions combine the
 * contributions in communicator rank order. A peer that does not answer within
 * SIRCL_MPI_TIMEOUT_S fails the process.
 *
 * Jobs: every hello carries a hash of SIRCL_MPI_JOB, so processes of different jobs that meet at one
 * address (consecutive test programs whose ranks fell out of step) refuse each other; a rank whose hello
 * rank 0 refuses tries again until the deadline. Watchdog: after MPI_Init a thread watches every peer
 * connection; when a peer's process ends outside MPI_Finalize (a crash, MPI_Abort, a kill), this process
 * ends too (exit status 1), so no rank waits for a peer that is gone, inside MPI or inside other work. */
#define _GNU_SOURCE
#include "mpi.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#define MAGIC 0x4d504953u /* "SIPM" */
#define MAX_PROCS 64
#define MAX_COMMS 256

typedef struct {
  int used;
  uint64_t id;          /* the same on every member: MPI_COMM_WORLD 1, a split's derived from its parent */
  int size, rank;       /* members and this process's rank among them */
  int members[MAX_PROCS]; /* world rank of each communicator rank */
  uint32_t round;       /* collectives so far on this communicator */
} comm_t;

typedef struct message {
  int64_t comm;
  uint32_t round, bytes;
  char *data;
  struct message *next;
} message;

static int world_rank, world_size = 1, initialized;
static int sockets[MAX_PROCS];
static message *pending[MAX_PROCS];
static comm_t comms[MAX_COMMS];
static int timeout_s = 300;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static uint32_t job_hash;
static volatile int finalizing;

static void die(const char *what) {
  fprintf(stderr, "sircl mpi shim (rank %d of %d): %s%s%s\n", world_rank, world_size, what, errno ? ": " : "",
          errno ? strerror(errno) : "");
  exit(1);
}

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static void wait_fd(int fd, short events, double deadline) {
  for (;;) {
    int left = (int)((deadline - now_s()) * 1000);
    if (left <= 0) {
      errno = ETIMEDOUT;
      die("timed out waiting for a peer");
    }
    struct pollfd p = {fd, events, 0};
    int rc = poll(&p, 1, left);
    if (rc > 0) return;
    if (rc < 0 && errno != EINTR) die("poll");
  }
}

static void write_full(int fd, const void *data, size_t n) {
  const char *p = data;
  double deadline = now_s() + timeout_s;
  while (n) {
    ssize_t w = send(fd, p, n, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (w > 0) {
      p += w;
      n -= (size_t)w;
    } else if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
      wait_fd(fd, POLLOUT, deadline);
    } else {
      die("lost a peer while sending");
    }
  }
}

static void read_full(int fd, void *data, size_t n) {
  char *p = data;
  double deadline = now_s() + timeout_s;
  while (n) {
    ssize_t r = recv(fd, p, n, MSG_DONTWAIT);
    if (r > 0) {
      p += r;
      n -= (size_t)r;
    } else if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
      wait_fd(fd, POLLIN, deadline);
    } else {
      if (r == 0) errno = 0;
      die("lost a peer while receiving");
    }
  }
}

static int env_int(const char *name, int fallback) {
  const char *v = getenv(name);
  return v && *v ? atoi(v) : fallback;
}

/* SIRCL_MPI_DISTINCT_HOSTS=1, for ranks that share one host in GPU emulation: gethostname reports the
 * host's name with "-rank<r>" (r = SIRCL_MPI_RANK) inserted before its first dot, so a program that counts
 * the ranks of its host by name (nccl-tests uses device <local rank>) sees one rank per host and every rank
 * uses device 0. Without the setting it reports the name unchanged. As a dependency of the program, this
 * definition takes the place of the C library's for the whole process. */
int gethostname(char *name, size_t len) {
  struct utsname u;
  if (uname(&u) != 0) return -1;
  char out[sizeof u.nodename + 32];
  const char *distinct = getenv("SIRCL_MPI_DISTINCT_HOSTS");
  if (distinct && !strcmp(distinct, "1")) {
    const char *dot = strchr(u.nodename, '.');
    int head = dot ? (int)(dot - u.nodename) : (int)strlen(u.nodename);
    snprintf(out, sizeof out, "%.*s-rank%d%s", head, u.nodename, env_int("SIRCL_MPI_RANK", 0), dot ? dot : "");
  } else {
    snprintf(out, sizeof out, "%s", u.nodename);
  }
  size_t n = strlen(out);
  if (n + 1 > len) {
    errno = ENAMETOOLONG;
    return -1;
  }
  memcpy(name, out, n + 1);
  return 0;
}

static void no_delay(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
}

/* A listener on `port` (0: any) of every interface; its port in *bound. */
static int listen_on(uint16_t port, uint16_t *bound) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in address = {0};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(port);
  if (fd < 0 || bind(fd, (struct sockaddr *)&address, sizeof address) || listen(fd, MAX_PROCS)) die("listen");
  socklen_t length = sizeof address;
  getsockname(fd, (struct sockaddr *)&address, &length);
  *bound = ntohs(address.sin_port);
  return fd;
}

static int connect_to(const struct sockaddr_in *address, double deadline) {
  for (;;) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (connect(fd, (const struct sockaddr *)address, sizeof *address) == 0) {
      no_delay(fd);
      return fd;
    }
    close(fd);
    if (now_s() > deadline) die("connecting to a peer");
    usleep(50000);
  }
}

/* Read `n` bytes by `deadline`; -1 when the peer closes the connection, fails or the deadline passes. */
static int try_read_full(int fd, void *data, size_t n, double deadline) {
  char *p = data;
  while (n) {
    ssize_t r = recv(fd, p, n, MSG_DONTWAIT);
    if (r > 0) {
      p += r;
      n -= (size_t)r;
      continue;
    }
    if (r == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) return -1;
    int left = (int)((deadline - now_s()) * 1000);
    if (left <= 0) return -1;
    struct pollfd q = {fd, POLLIN, 0};
    if (poll(&q, 1, left) < 0 && errno != EINTR) return -1;
  }
  return 0;
}

/* The FNV-1a hash of SIRCL_MPI_JOB (unset: the empty name). */
static uint32_t job_of_environment(void) {
  const char *name = getenv("SIRCL_MPI_JOB");
  uint32_t hash = 2166136261u;
  for (const char *c = name ? name : ""; *c; ++c) hash = (hash ^ (uint8_t)*c) * 16777619u;
  return hash;
}

/* Accept one connection and read its hello {MAGIC, rank, size, address, port, job}; connections of
 * another job, group size or protocol are closed. */
static int accept_hello(int listener, uint32_t *hello, size_t words, double deadline) {
  for (;;) {
    wait_fd(listener, POLLIN, deadline);
    int fd = accept(listener, NULL, NULL);
    if (fd < 0) continue;
    no_delay(fd);
    if (try_read_full(fd, hello, words * sizeof(uint32_t), deadline) == 0 && hello[0] == MAGIC &&
        (int)hello[2] == world_size && hello[1] < (uint32_t)world_size && hello[5] == job_hash)
      return fd;
    close(fd);
  }
}

/* The watchdog: a peer connection that hangs up outside MPI_Finalize ends this process. Data never wakes
 * it (it asks for hang-ups only), so it reads nothing the collectives expect. */
static void *watchdog(void *unused) {
  (void)unused;
  for (;;) {
    struct pollfd q[MAX_PROCS];
    int peer[MAX_PROCS], n = 0;
    for (int r = 0; r < world_size; ++r)
      if (sockets[r] >= 0) {
        q[n].fd = sockets[r];
        q[n].events = POLLRDHUP;
        q[n].revents = 0;
        peer[n++] = r;
      }
    if (poll(q, (nfds_t)n, 1000) < 0 && errno != EINTR) return NULL;
    if (finalizing) return NULL;
    for (int i = 0; i < n; ++i)
      if (q[i].revents & (POLLRDHUP | POLLHUP | POLLERR)) {
        fprintf(stderr, "sircl mpi shim (rank %d of %d): rank %d's process ended outside MPI_Finalize; ending "
                        "this process\n", world_rank, world_size, peer[i]);
        _exit(1);
      }
  }
}

/* Handles are communicator table slots: 1 is MPI_COMM_WORLD, splits take 2 and above. */
static intptr_t slot_of(MPI_Comm handle) { return (intptr_t)handle; }

static comm_t *comm_of(MPI_Comm handle) {
  intptr_t slot = slot_of(handle);
  if (slot <= 0 || slot >= MAX_COMMS || !comms[slot].used) return NULL;
  return &comms[slot];
}

int MPI_Init(int *argc, char ***argv) {
  (void)argc;
  (void)argv;
  pthread_mutex_lock(&lock);
  world_size = env_int("SIRCL_MPI_SIZE", 1);
  world_rank = env_int("SIRCL_MPI_RANK", 0);
  timeout_s = env_int("SIRCL_MPI_TIMEOUT_S", 300);
  if (world_size < 1 || world_size > MAX_PROCS || world_rank < 0 || world_rank >= world_size) {
    errno = 0;
    die("SIRCL_MPI_SIZE must be 1-64 and SIRCL_MPI_RANK below it");
  }
  for (int i = 0; i < MAX_PROCS; ++i) sockets[i] = -1;
  job_hash = job_of_environment();
  comm_t *world = &comms[slot_of(MPI_COMM_WORLD)];
  world->used = 1;
  world->id = 1;
  world->size = world_size;
  world->rank = world_rank;
  for (int r = 0; r < world_size; ++r) world->members[r] = r;
  initialized = 1;
  if (world_size == 1) {
    pthread_mutex_unlock(&lock);
    return MPI_SUCCESS;
  }
  const char *root = getenv("SIRCL_MPI_ROOT");
  if (!root || !strchr(root, ':')) {
    errno = 0;
    die("SIRCL_MPI_ROOT must be host:port");
  }
  char host[256];
  snprintf(host, sizeof host, "%.*s", (int)(strrchr(root, ':') - root), root);
  const char *port = strrchr(root, ':') + 1;
  double deadline = now_s() + timeout_s;
  /* addresses[r]: rank r's listener (IPv4 address, port), as rank 0 saw it. */
  uint32_t addresses[MAX_PROCS][2];
  memset(addresses, 0, sizeof addresses);
  if (world_rank == 0) {
    uint16_t bound;
    int listener = listen_on((uint16_t)atoi(port), &bound);
    for (int joined = 1; joined < world_size;) {
      uint32_t hello[6];
      int fd = accept_hello(listener, hello, 6, deadline);
      if (hello[1] == 0 || sockets[hello[1]] >= 0) {
        close(fd);
        continue;
      }
      sockets[hello[1]] = fd;
      addresses[hello[1]][0] = hello[3];
      addresses[hello[1]][1] = hello[4];
      ++joined;
    }
    close(listener);
    for (int r = 1; r < world_size; ++r) write_full(sockets[r], addresses, sizeof addresses);
  } else {
    uint16_t bound;
    int listener = listen_on(0, &bound);
    struct addrinfo hints = {0}, *found = NULL;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &found) || !found) die("resolving SIRCL_MPI_ROOT");
    /* Rank 0 answers an accepted hello with every rank's address and closes a refused one (another job's
     * rank 0, or a rank 0 that already has this rank): then try again until the deadline. */
    for (;;) {
      sockets[0] = connect_to((const struct sockaddr_in *)found->ai_addr, deadline);
      /* This rank's address on the interface that reaches rank 0. */
      struct sockaddr_in local;
      socklen_t length = sizeof local;
      getsockname(sockets[0], (struct sockaddr *)&local, &length);
      uint32_t hello[6] = {MAGIC, (uint32_t)world_rank, (uint32_t)world_size, local.sin_addr.s_addr, bound, job_hash};
      write_full(sockets[0], hello, sizeof hello);
      if (try_read_full(sockets[0], addresses, sizeof addresses, deadline) == 0) break;
      close(sockets[0]);
      sockets[0] = -1;
      if (now_s() > deadline) {
        errno = ETIMEDOUT;
        die("rank 0 of this job (SIRCL_MPI_JOB) did not accept this rank");
      }
      usleep(200000);
    }
    freeaddrinfo(found);
    /* Connect to every listener below this rank (rank 0's connection exists), then accept the ranks above. */
    for (int r = 1; r < world_rank; ++r) {
      struct sockaddr_in peer = {0};
      peer.sin_family = AF_INET;
      peer.sin_addr.s_addr = addresses[r][0];
      peer.sin_port = htons((uint16_t)addresses[r][1]);
      sockets[r] = connect_to(&peer, deadline);
      uint32_t mine[6] = {MAGIC, (uint32_t)world_rank, (uint32_t)world_size, 0, 0, job_hash};
      write_full(sockets[r], mine, sizeof mine);
    }
    for (int joined = world_rank + 1; joined < world_size;) {
      uint32_t peer_hello[6];
      int fd = accept_hello(listener, peer_hello, 6, deadline);
      if ((int)peer_hello[1] <= world_rank || sockets[peer_hello[1]] >= 0) {
        close(fd);
        continue;
      }
      sockets[peer_hello[1]] = fd;
      ++joined;
    }
    close(listener);
  }
  pthread_t thread;
  if (pthread_create(&thread, NULL, watchdog, NULL) == 0) pthread_detach(thread);
  pthread_mutex_unlock(&lock);
  return MPI_SUCCESS;
}

/* -- messages ---------------------------------------------------------------------------------------- */

static void send_message(int to, const comm_t *c, uint32_t round, const void *data, uint32_t bytes) {
  if (to == world_rank) {
    message *m = calloc(1, sizeof *m);
    m->comm = (int64_t)c->id;
    m->round = round;
    m->bytes = bytes;
    m->data = malloc(bytes ? bytes : 1);
    if (bytes) memcpy(m->data, data, bytes);
    message **tail = &pending[to];
    while (*tail) tail = &(*tail)->next;
    *tail = m;
    return;
  }
  uint32_t header[5] = {MAGIC, (uint32_t)c->id, (uint32_t)(c->id >> 32), round, bytes};
  write_full(sockets[to], header, sizeof header);
  if (bytes) write_full(sockets[to], data, bytes);
}

/* The message of collective `round` of `c` from world rank `from` (its payload, malloc'd; bytes in *bytes). */
static char *receive_message(int from, const comm_t *c, uint32_t round, uint32_t *bytes) {
  for (;;) {
    for (message **at = &pending[from]; *at; at = &(*at)->next) {
      message *m = *at;
      if (m->comm == (int64_t)c->id && m->round == round) {
        *at = m->next;
        char *data = m->data;
        *bytes = m->bytes;
        free(m);
        return data;
      }
    }
    if (from == world_rank) {
      errno = 0;
      die("a collective waited for a message to itself that was never sent");
    }
    uint32_t header[5];
    read_full(sockets[from], header, sizeof header);
    if (header[0] != MAGIC) {
      errno = 0;
      die("a peer sent an unframed message");
    }
    message *m = calloc(1, sizeof *m);
    m->comm = (int64_t)((uint64_t)header[1] | (uint64_t)header[2] << 32);
    m->round = header[3];
    m->bytes = header[4];
    m->data = malloc(m->bytes ? m->bytes : 1);
    if (m->bytes) read_full(sockets[from], m->data, m->bytes);
    message **tail = &pending[from];
    while (*tail) tail = &(*tail)->next;
    *tail = m;
  }
}

/* Every member's contribution in communicator rank order, back to back; sizes in `lengths`. */
static char *exchange(comm_t *c, const void *mine, uint32_t bytes, uint32_t *lengths) {
  uint32_t round = ++c->round;
  int leader = c->members[0];
  if (c->rank != 0) {
    send_message(leader, c, round, mine, bytes);
    uint32_t got;
    char *reply = receive_message(leader, c, round, &got);
    memcpy(lengths, reply, sizeof(uint32_t) * (size_t)c->size);
    uint64_t total = 0;
    for (int r = 0; r < c->size; ++r) total += lengths[r];
    if (got != sizeof(uint32_t) * (size_t)c->size + total) {
      errno = 0;
      die("a collective's reply has the wrong size");
    }
    char *all = malloc(total ? total : 1);
    memcpy(all, reply + sizeof(uint32_t) * (size_t)c->size, total);
    free(reply);
    return all;
  }
  char *parts[MAX_PROCS] = {0};
  uint64_t total = bytes;
  lengths[0] = bytes;
  for (int r = 1; r < c->size; ++r) {
    parts[r] = receive_message(c->members[r], c, round, &lengths[r]);
    total += lengths[r];
  }
  size_t head = sizeof(uint32_t) * (size_t)c->size;
  char *reply = malloc(head + total + 1);
  memcpy(reply, lengths, head);
  uint64_t at = head;
  for (int r = 0; r < c->size; ++r) {
    if (r == 0) {
      if (bytes) memcpy(reply + at, mine, bytes);
    } else {
      memcpy(reply + at, parts[r], lengths[r]);
      free(parts[r]);
    }
    at += lengths[r];
  }
  for (int r = 1; r < c->size; ++r) send_message(c->members[r], c, round, reply, (uint32_t)(head + total));
  char *all = malloc(total ? total : 1);
  memcpy(all, reply + head, total);
  free(reply);
  return all;
}

/* At communicator rank `root`: every member's contribution in rank order (sizes in `lengths`); NULL elsewhere. */
static char *gather_to(comm_t *c, int root, const void *mine, uint32_t bytes, uint32_t *lengths) {
  uint32_t round = ++c->round;
  if (c->rank != root) {
    send_message(c->members[root], c, round, mine, bytes);
    return NULL;
  }
  char *parts[MAX_PROCS] = {0};
  uint64_t total = 0;
  for (int r = 0; r < c->size; ++r) {
    if (r == root) {
      lengths[r] = bytes;
    } else {
      parts[r] = receive_message(c->members[r], c, round, &lengths[r]);
    }
    total += lengths[r];
  }
  char *all = malloc(total ? total : 1);
  uint64_t at = 0;
  for (int r = 0; r < c->size; ++r) {
    if (r == root) {
      if (bytes) memcpy(all + at, mine, bytes);
    } else {
      memcpy(all + at, parts[r], lengths[r]);
      free(parts[r]);
    }
    at += lengths[r];
  }
  return all;
}

static size_t type_size(MPI_Datatype t) {
  switch (t) {
    case MPI_BYTE: case MPI_CHAR: return 1;
    case MPI_INT: case MPI_UNSIGNED: return sizeof(int);
    case MPI_LONG: case MPI_UNSIGNED_LONG: return sizeof(long);
    case MPI_LONG_LONG: return sizeof(long long);
    case MPI_DOUBLE: return sizeof(double);
    case MPI_FLOAT: return sizeof(float);
    case MPI_INT64_T: case MPI_UINT64_T: return sizeof(int64_t);
    default: return 0;
  }
}

#define LOCKED(expression)              \
  do {                                  \
    pthread_mutex_lock(&lock);          \
    int result_ = (expression);         \
    pthread_mutex_unlock(&lock);        \
    return result_;                     \
  } while (0)

/* -- communicators ------------------------------------------------------------------------------------ */

static int comm_size_locked(MPI_Comm handle, int *size) {
  comm_t *c = comm_of(handle);
  if (!c || !size) return MPI_ERR_COMM;
  *size = c->size;
  return MPI_SUCCESS;
}
int MPI_Comm_size(MPI_Comm comm, int *size) { LOCKED(comm_size_locked(comm, size)); }

static int comm_rank_locked(MPI_Comm handle, int *rank) {
  comm_t *c = comm_of(handle);
  if (!c || !rank) return MPI_ERR_COMM;
  *rank = c->rank;
  return MPI_SUCCESS;
}
int MPI_Comm_rank(MPI_Comm comm, int *rank) { LOCKED(comm_rank_locked(comm, rank)); }

static uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

/* Every member of the parent sends (color, key); the members of each color, ordered by key and then by
 * parent rank, form a communicator whose id every member derives from the parent's id, the split's
 * collective number and the color. MPI_UNDEFINED joins none. */
static int comm_split_locked(MPI_Comm handle, int color, int key, MPI_Comm *out) {
  comm_t *parent = comm_of(handle);
  if (!parent || !out || (color < 0 && color != MPI_UNDEFINED)) return parent ? MPI_ERR_ARG : MPI_ERR_COMM;
  int mine[2] = {color, key};
  uint32_t lengths[MAX_PROCS];
  int *all = (int *)exchange(parent, mine, sizeof mine, lengths);
  uint32_t round = parent->round;
  *out = MPI_COMM_NULL;
  if (color == MPI_UNDEFINED) {
    free(all);
    return MPI_SUCCESS;
  }
  int slot = 2;
  while (slot < MAX_COMMS && comms[slot].used) ++slot;
  if (slot == MAX_COMMS) {
    free(all);
    return MPI_ERR_OTHER;
  }
  comm_t *c = &comms[slot];
  memset(c, 0, sizeof *c);
  int order[MAX_PROCS], n = 0;
  for (int r = 0; r < parent->size; ++r)
    if (all[2 * r] == color) order[n++] = r;
  for (int i = 1; i < n; ++i) /* stable insertion sort by key */
    for (int j = i; j > 0 && all[2 * order[j - 1] + 1] > all[2 * order[j] + 1]; --j) {
      int t = order[j];
      order[j] = order[j - 1];
      order[j - 1] = t;
    }
  c->used = 1;
  c->size = n;
  c->id = mix(mix(parent->id) ^ ((uint64_t)round << 32 | (uint32_t)color)) | 2u; /* never 1 (the world) */
  for (int i = 0; i < n; ++i) {
    c->members[i] = parent->members[order[i]];
    if (order[i] == parent->rank) c->rank = i;
  }
  free(all);
  *out = (MPI_Comm)(intptr_t)slot;
  return MPI_SUCCESS;
}
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *out) {
  LOCKED(comm_split_locked(comm, color, key, out));
}

static int comm_free_locked(MPI_Comm *handle) {
  if (!handle) return MPI_ERR_ARG;
  comm_t *c = comm_of(*handle);
  if (!c || *handle == MPI_COMM_WORLD) return MPI_ERR_COMM;
  c->used = 0;
  *handle = MPI_COMM_NULL;
  return MPI_SUCCESS;
}
int MPI_Comm_free(MPI_Comm *comm) { LOCKED(comm_free_locked(comm)); }

/* -- collectives ---------------------------------------------------------------------------------------- */

static int barrier_locked(MPI_Comm handle) {
  comm_t *c = comm_of(handle);
  if (!c) return MPI_ERR_COMM;
  uint32_t lengths[MAX_PROCS];
  free(exchange(c, NULL, 0, lengths));
  return MPI_SUCCESS;
}
int MPI_Barrier(MPI_Comm comm) { LOCKED(barrier_locked(comm)); }

static int bcast_locked(void *buffer, int count, MPI_Datatype type, int root, MPI_Comm handle) {
  comm_t *c = comm_of(handle);
  if (!c) return MPI_ERR_COMM;
  if (!type_size(type)) return MPI_ERR_TYPE;
  if (count < 0) return MPI_ERR_COUNT;
  if (root < 0 || root >= c->size) return MPI_ERR_ROOT;
  uint32_t bytes = (uint32_t)((size_t)count * type_size(type)), round = ++c->round;
  if (c->rank == root) {
    for (int r = 0; r < c->size; ++r)
      if (r != root) send_message(c->members[r], c, round, buffer, bytes);
    return MPI_SUCCESS;
  }
  uint32_t got;
  char *data = receive_message(c->members[root], c, round, &got);
  int ok = got == bytes;
  if (ok) memcpy(buffer, data, bytes);
  free(data);
  return ok ? MPI_SUCCESS : MPI_ERR_COUNT;
}
int MPI_Bcast(void *buffer, int count, MPI_Datatype type, int root, MPI_Comm comm) {
  LOCKED(bcast_locked(buffer, count, type, root, comm));
}

/* Place every member's contribution of `all` at its displacement; every length must equal its count. */
static int place(char *recv, const char *all, const uint32_t *lengths, int size, const int *counts, const int *displs,
                 size_t item) {
  uint64_t at = 0;
  for (int r = 0; r < size; ++r) {
    if (lengths[r] != (uint64_t)counts[r] * item) return MPI_ERR_COUNT;
    memcpy(recv + (size_t)displs[r] * item, all + at, lengths[r]);
    at += lengths[r];
  }
  return MPI_SUCCESS;
}

static int allgatherv_locked(const void *send, int send_count, MPI_Datatype send_type, void *recv,
                             const int *counts, const int *displs, MPI_Datatype recv_type, MPI_Comm handle) {
  comm_t *c = comm_of(handle);
  if (!c) return MPI_ERR_COMM;
  size_t item = type_size(recv_type);
  if (!item || !counts || !displs) return MPI_ERR_TYPE;
  const void *mine = send == MPI_IN_PLACE ? (const char *)recv + (size_t)displs[c->rank] * item : send;
  size_t bytes = (size_t)counts[c->rank] * item;
  if (send != MPI_IN_PLACE && (size_t)send_count * type_size(send_type) != bytes) return MPI_ERR_COUNT;
  uint32_t lengths[MAX_PROCS];
  char *all = exchange(c, mine, (uint32_t)bytes, lengths);
  int result = place(recv, all, lengths, c->size, counts, displs, item);
  free(all);
  return result;
}
int MPI_Allgatherv(const void *send, int send_count, MPI_Datatype send_type, void *recv, const int recv_counts[],
                   const int displacements[], MPI_Datatype recv_type, MPI_Comm comm) {
  LOCKED(allgatherv_locked(send, send_count, send_type, recv, recv_counts, displacements, recv_type, comm));
}

static int allgather_locked(const void *send, int send_count, MPI_Datatype send_type, void *recv, int recv_count,
                            MPI_Datatype recv_type, MPI_Comm handle) {
  comm_t *c = comm_of(handle);
  if (!c) return MPI_ERR_COMM;
  int counts[MAX_PROCS], displs[MAX_PROCS];
  for (int r = 0; r < c->size; ++r) {
    counts[r] = recv_count;
    displs[r] = r * recv_count;
  }
  return allgatherv_locked(send, send_count, send_type, recv, counts, displs, recv_type, handle);
}
int MPI_Allgather(const void *send, int send_count, MPI_Datatype send_type, void *recv, int recv_count,
                  MPI_Datatype recv_type, MPI_Comm comm) {
  LOCKED(allgather_locked(send, send_count, send_type, recv, recv_count, recv_type, comm));
}

static int gather_locked(const void *send, int send_count, MPI_Datatype send_type, void *recv, int recv_count,
                         MPI_Datatype recv_type, int root, MPI_Comm handle) {
  comm_t *c = comm_of(handle);
  if (!c) return MPI_ERR_COMM;
  if (root < 0 || root >= c->size) return MPI_ERR_ROOT;
  int in_place = send == MPI_IN_PLACE && c->rank == root;
  size_t bytes = in_place ? (size_t)recv_count * type_size(recv_type) : (size_t)send_count * type_size(send_type);
  if (!in_place && !type_size(send_type)) return MPI_ERR_TYPE;
  const void *mine = in_place ? (const char *)recv + (size_t)root * bytes : send;
  uint32_t lengths[MAX_PROCS];
  char *all = gather_to(c, root, mine, (uint32_t)bytes, lengths);
  if (c->rank != root) return MPI_SUCCESS;
  int result = MPI_SUCCESS;
  size_t item = type_size(recv_type);
  if (!item || (size_t)recv_count * item != bytes) {
    result = MPI_ERR_COUNT;
  } else {
    for (int r = 0; r < c->size; ++r)
      if (lengths[r] != bytes) result = MPI_ERR_COUNT;
    if (result == MPI_SUCCESS) memmove(recv, all, bytes * (size_t)c->size);
  }
  free(all);
  return result;
}
int MPI_Gather(const void *send, int send_count, MPI_Datatype send_type, void *recv, int recv_count,
               MPI_Datatype recv_type, int root, MPI_Comm comm) {
  LOCKED(gather_locked(send, send_count, send_type, recv, recv_count, recv_type, root, comm));
}

#define COMBINE(T)                                                                                   \
  for (int i = 0; i < count; ++i) {                                                                  \
    T acc = ((const T *)all)[i];                                                                     \
    for (int r = 1; r < n; ++r) {                                                                    \
      T v = ((const T *)(all + (size_t)r * bytes))[i];                                               \
      acc = op == MPI_SUM ? (T)(acc + v) : op == MPI_MIN ? (v < acc ? v : acc) : (v > acc ? v : acc); \
    }                                                                                                \
    ((T *)recv)[i] = acc;                                                                            \
  }

/* recv = the reduction of the n contributions of `all` (each `bytes` long) in order. */
static int combine(const char *all, void *recv, int count, MPI_Datatype type, MPI_Op op, int n, size_t bytes) {
  switch (type) {
    case MPI_INT: COMBINE(int) break;
    case MPI_UNSIGNED: COMBINE(unsigned) break;
    case MPI_LONG: COMBINE(long) break;
    case MPI_UNSIGNED_LONG: COMBINE(unsigned long) break;
    case MPI_LONG_LONG: COMBINE(long long) break;
    case MPI_INT64_T: COMBINE(int64_t) break;
    case MPI_UINT64_T: COMBINE(uint64_t) break;
    case MPI_DOUBLE: COMBINE(double) break;
    case MPI_FLOAT: COMBINE(float) break;
    default: return MPI_ERR_TYPE;
  }
  return MPI_SUCCESS;
}

static int reducible(MPI_Datatype type, MPI_Op op, int count) {
  if (count < 0) return MPI_ERR_COUNT;
  if (!type_size(type) || type == MPI_BYTE || type == MPI_CHAR) return MPI_ERR_TYPE;
  if (op != MPI_SUM && op != MPI_MIN && op != MPI_MAX) return MPI_ERR_OP;
  return MPI_SUCCESS;
}

static int allreduce_locked(const void *send, void *recv, int count, MPI_Datatype type, MPI_Op op, MPI_Comm handle) {
  comm_t *c = comm_of(handle);
  if (!c) return MPI_ERR_COMM;
  int check = reducible(type, op, count);
  if (check != MPI_SUCCESS) return check;
  size_t bytes = (size_t)count * type_size(type);
  uint32_t lengths[MAX_PROCS];
  char *all = exchange(c, send == MPI_IN_PLACE ? recv : send, (uint32_t)bytes, lengths);
  int result = MPI_SUCCESS;
  for (int r = 0; r < c->size; ++r)
    if (lengths[r] != bytes) result = MPI_ERR_COUNT;
  if (result == MPI_SUCCESS) result = combine(all, recv, count, type, op, c->size, bytes);
  free(all);
  return result;
}
int MPI_Allreduce(const void *send, void *recv, int count, MPI_Datatype type, MPI_Op op, MPI_Comm comm) {
  LOCKED(allreduce_locked(send, recv, count, type, op, comm));
}

static int reduce_locked(const void *send, void *recv, int count, MPI_Datatype type, MPI_Op op, int root,
                         MPI_Comm handle) {
  comm_t *c = comm_of(handle);
  if (!c) return MPI_ERR_COMM;
  if (root < 0 || root >= c->size) return MPI_ERR_ROOT;
  int check = reducible(type, op, count);
  if (check != MPI_SUCCESS) return check;
  size_t bytes = (size_t)count * type_size(type);
  const void *mine = send == MPI_IN_PLACE && c->rank == root ? recv : send;
  uint32_t lengths[MAX_PROCS];
  char *all = gather_to(c, root, mine, (uint32_t)bytes, lengths);
  if (c->rank != root) return MPI_SUCCESS;
  int result = MPI_SUCCESS;
  for (int r = 0; r < c->size; ++r)
    if (lengths[r] != bytes) result = MPI_ERR_COUNT;
  if (result == MPI_SUCCESS) result = combine(all, recv, count, type, op, c->size, bytes);
  free(all);
  return result;
}
int MPI_Reduce(const void *send, void *recv, int count, MPI_Datatype type, MPI_Op op, int root, MPI_Comm comm) {
  LOCKED(reduce_locked(send, recv, count, type, op, root, comm));
}

int MPI_Finalize(void) {
  pthread_mutex_lock(&lock);
  /* Peers close their connections only after the final barrier, which this process has then entered. */
  finalizing = 1;
  if (initialized && world_size > 1) {
    uint32_t lengths[MAX_PROCS];
    free(exchange(&comms[slot_of(MPI_COMM_WORLD)], NULL, 0, lengths));
  }
  for (int r = 0; r < MAX_PROCS; ++r)
    if (sockets[r] >= 0) close(sockets[r]);
  initialized = 0;
  pthread_mutex_unlock(&lock);
  return MPI_SUCCESS;
}

int MPI_Abort(MPI_Comm comm, int code) {
  (void)comm;
  fprintf(stderr, "sircl mpi shim: MPI_Abort(%d) on rank %d\n", code, world_rank);
  exit(code ? code : 1);
}

int MPI_Error_string(int code, char *text, int *length) {
  const char *what;
  switch (code) {
    case MPI_SUCCESS: what = "MPI_SUCCESS: no error"; break;
    case MPI_ERR_COUNT: what = "MPI_ERR_COUNT: invalid count, or contributions of different sizes"; break;
    case MPI_ERR_TYPE: what = "MPI_ERR_TYPE: invalid or unsupported datatype"; break;
    case MPI_ERR_COMM: what = "MPI_ERR_COMM: invalid communicator"; break;
    case MPI_ERR_ROOT: what = "MPI_ERR_ROOT: invalid root"; break;
    case MPI_ERR_OP: what = "MPI_ERR_OP: unsupported reduction op (MPI_SUM, MPI_MIN, MPI_MAX)"; break;
    case MPI_ERR_ARG: what = "MPI_ERR_ARG: invalid argument"; break;
    default: what = "MPI_ERR_OTHER: error in the sircl MPI shim"; break;
  }
  int n = snprintf(text, MPI_MAX_ERROR_STRING, "%s", what);
  if (length) *length = n < MPI_MAX_ERROR_STRING ? n : MPI_MAX_ERROR_STRING - 1;
  return MPI_SUCCESS;
}
