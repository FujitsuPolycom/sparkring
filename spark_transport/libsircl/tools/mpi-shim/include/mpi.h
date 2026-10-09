/* A launcher-free MPI subset for running nccl-tests (MPI=1) without an MPI
 * installation: every MPI call and constant nccl-tests v2.21.1 (commit afd59ab)
 * uses in its .cu sources, common.*, util.*, comm_ops.cu and the device-API GIN
 * tests, which were read for that list (nccl-tests is NVIDIA's BSD-3-Clause test
 * program, not NCCL's implementation), over TCP between the processes of one
 * job. The whole nccl-tests tree at that tag compiles and links against it.
 * Each process learns its place from the
 * environment:
 *   SIRCL_MPI_RANK  this process's rank, 0 to SIZE-1
 *   SIRCL_MPI_SIZE  the number of processes (1 to 64)
 *   SIRCL_MPI_ROOT  host:port where rank 0 listens and every rank connects
 *   SIRCL_MPI_TIMEOUT_S  seconds to wait for the group or a peer (default 300)
 * Communicators: MPI_COMM_WORLD and the communicators MPI_Comm_split makes, any
 * colors (MPI_UNDEFINED gives MPI_COMM_NULL), splits of splits. Collectives run
 * among a communicator's members only; reductions combine contributions in the
 * communicator's rank order, so every rank holds identical bits. Calls are
 * serialized within a process (MPI_THREAD_SERIALIZED). Not an MPI
 * implementation: no point-to-point calls, requests, groups or MPI-IO. */
#ifndef SIRCL_MPI_SHIM_H
#define SIRCL_MPI_SHIM_H
#ifdef __cplusplus
extern "C" {
#endif

/* Communicators are opaque handles (a pointer type, as in Open MPI, so code that prints one with %p is
 * well defined); datatypes and ops are integers. */
typedef struct sircl_mpi_comm *MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;

#define MPI_SUCCESS 0
#define MPI_ERR_COMM 5
#define MPI_ERR_TYPE 3
#define MPI_ERR_COUNT 2
#define MPI_ERR_ROOT 7
#define MPI_ERR_OP 9
#define MPI_ERR_ARG 12
#define MPI_ERR_OTHER 15
#define MPI_MAX_ERROR_STRING 256
#define MPI_UNDEFINED (-32766)
#define MPI_COMM_NULL ((MPI_Comm)0)
#define MPI_COMM_WORLD ((MPI_Comm)1)
#define MPI_IN_PLACE ((void *)1)
/* Datatypes; 0 is not a datatype (nccl-tests value-initializes unknown ones). */
#define MPI_DATATYPE_NULL 100
#define MPI_BYTE 101
#define MPI_INT 102
#define MPI_LONG 103
#define MPI_LONG_LONG 104
#define MPI_DOUBLE 105
#define MPI_INT64_T 106
#define MPI_CHAR 107
#define MPI_UNSIGNED 108
#define MPI_UNSIGNED_LONG 109
#define MPI_UINT64_T 110
#define MPI_FLOAT 111
/* Reduction ops; 0 is not an op. */
#define MPI_SUM 201
#define MPI_MIN 202
#define MPI_MAX 203

int MPI_Init(int *argc, char ***argv);
int MPI_Finalize(void);
int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *out);
int MPI_Comm_free(MPI_Comm *comm);
int MPI_Barrier(MPI_Comm comm);
int MPI_Bcast(void *buffer, int count, MPI_Datatype type, int root, MPI_Comm comm);
int MPI_Allgather(const void *send, int send_count, MPI_Datatype send_type, void *recv, int recv_count,
                  MPI_Datatype recv_type, MPI_Comm comm);
int MPI_Allgatherv(const void *send, int send_count, MPI_Datatype send_type, void *recv, const int recv_counts[],
                   const int displacements[], MPI_Datatype recv_type, MPI_Comm comm);
int MPI_Gather(const void *send, int send_count, MPI_Datatype send_type, void *recv, int recv_count,
               MPI_Datatype recv_type, int root, MPI_Comm comm);
int MPI_Reduce(const void *send, void *recv, int count, MPI_Datatype type, MPI_Op op, int root, MPI_Comm comm);
int MPI_Allreduce(const void *send, void *recv, int count, MPI_Datatype type, MPI_Op op, MPI_Comm comm);
int MPI_Abort(MPI_Comm comm, int code);
int MPI_Error_string(int code, char *text, int *length);

#ifdef __cplusplus
}
#endif
#endif
