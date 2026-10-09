/* The host entry points of NCCL's device API (NVIDIA NCCL's public header nccl_device/core.h and its
 * barrier and LL all-to-all headers, NCCL 2.28 and later), which programs built against current NCCL
 * headers import whether or not they use the device API (nccl-tests v2.21.1 does). libsircl has no
 * device API: every rank of a session is on its own host, so a rank's load/store-accessible (LSA) team
 * is the rank alone, and the device-side communicator, symmetric windows and GIN are not provided.
 *
 * - ncclCommQueryProperties reports the communicator's rank, size and device, no device-API support,
 *   no multimem, no GIN, no host RMA, and one LSA team per rank.
 * - ncclTeamWorld, ncclTeamLsa and ncclTeamRail describe those teams; ncclTeamRankToWorld and
 *   ncclTeamRankToLsa translate team ranks.
 * - ncclDevCommCreate, ncclDevCommDestroy, the device-pointer queries and the requirement builders refuse
 *   with ncclInvalidUsage (logged once per function); ncclLLA2ACalcSlots returns 0.
 * Every function has its pncclX twin. Declarations follow the public header's; the types below mirror
 * its definitions (struct layouts are ABI). */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "internal.h"

typedef struct ncclTeam {
  int nRanks, rank, stride;
} ncclTeam_t;
typedef struct ncclDevComm ncclDevComm_t;
typedef struct ncclDevCommRequirements ncclDevCommRequirements_t;
typedef struct ncclDevResourceRequirements ncclDevResourceRequirements_t;
typedef struct ncclGinBarrierHandle ncclGinBarrierHandle_t;
typedef struct ncclLsaBarrierHandle ncclLsaBarrierHandle_t;
typedef struct ncclLLA2AHandle ncclLLA2AHandle_t;
typedef struct ncclMultimemHandle {
  void *mcBasePtr;
} ncclMultimemHandle;
typedef enum { NCCL_GIN_TYPE_NONE = 0, NCCL_GIN_TYPE_PROXY = 2, NCCL_GIN_TYPE_GDAKI = 3 } ncclGinType_t;
typedef struct ncclCommProperties {
  size_t size;
  unsigned int magic;
  unsigned int version;
  int rank;
  int nRanks;
  int cudaDev;
  int nvmlDev;
  bool deviceApiSupport;
  bool multimemSupport;
  ncclGinType_t ginType;
  int nLsaTeams;
  bool hostRmaSupport;
  ncclGinType_t railedGinType;
} ncclCommProperties_t;

/* Each field is written only when the caller's struct (its `size`) holds it, so programs built against
 * an older or newer header with fields appended keep their layout. */
#define FILL(field, value)                                                                         \
  do {                                                                                             \
    if (props->size >= offsetof(ncclCommProperties_t, field) + sizeof(props->field)) props->field = (value); \
  } while (0)

ncclResult_t ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t *props) {
  if (!props || props->magic != NCCL_API_MAGIC || props->size < offsetof(ncclCommProperties_t, rank))
    return ncclInvalidArgument;
  int nranks = 0, rank = 0, device = -1;
  ncclResult_t result = ncclCommCount(comm, &nranks);
  if (result == ncclSuccess) result = ncclCommUserRank(comm, &rank);
  if (result == ncclSuccess) result = ncclCommCuDevice(comm, &device);
  if (result != ncclSuccess) return result;
  FILL(rank, rank);
  FILL(nRanks, nranks);
  FILL(cudaDev, device);
  FILL(nvmlDev, -1); /* not resolved: libsircl does not load NVML */
  FILL(deviceApiSupport, false);
  FILL(multimemSupport, false);
  FILL(ginType, NCCL_GIN_TYPE_NONE);
  FILL(nLsaTeams, nranks);
  FILL(hostRmaSupport, false);
  FILL(railedGinType, NCCL_GIN_TYPE_NONE);
  return ncclSuccess;
}
#undef FILL

/* World: every rank. LSA: the rank alone. Rail (the outer factor of the LSA team): every rank, one
 * apart. An invalid communicator gives the empty team. */
static ncclTeam_t team_of(ncclComm_t comm, int lsa) {
  ncclTeam_t team = {0, 0, 1};
  int nranks = 0, rank = 0;
  if (ncclCommCount(comm, &nranks) != ncclSuccess || ncclCommUserRank(comm, &rank) != ncclSuccess) return team;
  if (lsa) {
    team.nRanks = 1;
    return team;
  }
  team.nRanks = nranks;
  team.rank = rank;
  return team;
}

ncclTeam_t ncclTeamWorld(ncclComm_t comm) { return team_of(comm, 0); }
ncclTeam_t ncclTeamLsa(ncclComm_t comm) { return team_of(comm, 1); }
ncclTeam_t ncclTeamRail(ncclComm_t comm) { return team_of(comm, 0); }

/* The world rank of rank `r` of `team` (which contains this rank at team.rank); -1 outside it. */
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int r) {
  int nranks = 0, rank = 0;
  if (ncclCommCount(comm, &nranks) != ncclSuccess || ncclCommUserRank(comm, &rank) != ncclSuccess) return -1;
  if (r < 0 || r >= team.nRanks) return -1;
  int world = rank + (r - team.rank) * team.stride;
  return world >= 0 && world < nranks ? world : -1;
}

/* The LSA rank of rank `r` of `team`: 0 for this rank (its LSA team is itself), -1 for any other. */
int ncclTeamRankToLsa(ncclComm_t comm, ncclTeam_t team, int r) {
  int rank = 0, world = ncclTeamRankToWorld(comm, team, r);
  if (world < 0 || ncclCommUserRank(comm, &rank) != ncclSuccess) return -1;
  return world == rank ? 0 : -1;
}

ncclResult_t ncclDevCommCreate(ncclComm_t comm, const ncclDevCommRequirements_t *reqs, ncclDevComm_t *out) {
  (void)comm;
  (void)reqs;
  (void)out;
  return sccl_unsupported("ncclDevCommCreate");
}

ncclResult_t ncclDevCommDestroy(ncclComm_t comm, const ncclDevComm_t *devComm) {
  (void)comm;
  (void)devComm;
  return sccl_unsupported("ncclDevCommDestroy");
}

ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void **out) {
  (void)window;
  (void)offset;
  (void)out;
  return sccl_unsupported("ncclGetLsaMultimemDevicePointer");
}

ncclResult_t ncclGetMultimemDevicePointer(ncclWindow_t window, size_t offset, ncclMultimemHandle multimem,
                                          void **out) {
  (void)window;
  (void)offset;
  (void)multimem;
  (void)out;
  return sccl_unsupported("ncclGetMultimemDevicePointer");
}

ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void **out) {
  (void)window;
  (void)offset;
  (void)lsaRank;
  (void)out;
  return sccl_unsupported("ncclGetLsaDevicePointer");
}

ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void **out) {
  (void)window;
  (void)offset;
  (void)peer;
  (void)out;
  return sccl_unsupported("ncclGetPeerDevicePointer");
}

ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t comm, ncclTeam_t team, int nBarriers,
                                             ncclGinBarrierHandle_t *outHandle, ncclDevResourceRequirements_t *outReq) {
  (void)comm;
  (void)team;
  (void)nBarriers;
  (void)outHandle;
  (void)outReq;
  return sccl_unsupported("ncclGinBarrierCreateRequirement");
}

int ncclLLA2ACalcSlots(int maxElts, int maxEltSize) {
  (void)maxElts;
  (void)maxEltSize;
  (void)sccl_unsupported("ncclLLA2ACalcSlots");
  return 0;
}

ncclResult_t ncclLLA2ACreateRequirement(int nBlocks, int nSlots, ncclLLA2AHandle_t *outHandle,
                                        ncclDevResourceRequirements_t *outReq) {
  (void)nBlocks;
  (void)nSlots;
  (void)outHandle;
  (void)outReq;
  return sccl_unsupported("ncclLLA2ACreateRequirement");
}

ncclResult_t ncclLsaBarrierCreateRequirement(ncclTeam_t team, int nBarriers, ncclLsaBarrierHandle_t *outHandle,
                                             ncclDevResourceRequirements_t *outReq) {
  (void)team;
  (void)nBarriers;
  (void)outHandle;
  (void)outReq;
  return sccl_unsupported("ncclLsaBarrierCreateRequirement");
}

/* -- pnccl twins --------------------------------------------------------------------------------- */

ncclResult_t pncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t *props) {
  return ncclCommQueryProperties(comm, props);
}
ncclTeam_t pncclTeamWorld(ncclComm_t comm) { return ncclTeamWorld(comm); }
ncclTeam_t pncclTeamLsa(ncclComm_t comm) { return ncclTeamLsa(comm); }
ncclTeam_t pncclTeamRail(ncclComm_t comm) { return ncclTeamRail(comm); }
int pncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int r) { return ncclTeamRankToWorld(comm, team, r); }
int pncclTeamRankToLsa(ncclComm_t comm, ncclTeam_t team, int r) { return ncclTeamRankToLsa(comm, team, r); }
ncclResult_t pncclDevCommCreate(ncclComm_t comm, const ncclDevCommRequirements_t *reqs, ncclDevComm_t *out) {
  return ncclDevCommCreate(comm, reqs, out);
}
ncclResult_t pncclDevCommDestroy(ncclComm_t comm, const ncclDevComm_t *devComm) {
  return ncclDevCommDestroy(comm, devComm);
}
ncclResult_t pncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void **out) {
  return ncclGetLsaMultimemDevicePointer(window, offset, out);
}
ncclResult_t pncclGetMultimemDevicePointer(ncclWindow_t window, size_t offset, ncclMultimemHandle multimem,
                                           void **out) {
  return ncclGetMultimemDevicePointer(window, offset, multimem, out);
}
ncclResult_t pncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void **out) {
  return ncclGetLsaDevicePointer(window, offset, lsaRank, out);
}
ncclResult_t pncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void **out) {
  return ncclGetPeerDevicePointer(window, offset, peer, out);
}
ncclResult_t pncclGinBarrierCreateRequirement(ncclComm_t comm, ncclTeam_t team, int nBarriers,
                                              ncclGinBarrierHandle_t *outHandle, ncclDevResourceRequirements_t *outReq) {
  return ncclGinBarrierCreateRequirement(comm, team, nBarriers, outHandle, outReq);
}
int pncclLLA2ACalcSlots(int maxElts, int maxEltSize) { return ncclLLA2ACalcSlots(maxElts, maxEltSize); }
ncclResult_t pncclLLA2ACreateRequirement(int nBlocks, int nSlots, ncclLLA2AHandle_t *outHandle,
                                         ncclDevResourceRequirements_t *outReq) {
  return ncclLLA2ACreateRequirement(nBlocks, nSlots, outHandle, outReq);
}
ncclResult_t pncclLsaBarrierCreateRequirement(ncclTeam_t team, int nBarriers, ncclLsaBarrierHandle_t *outHandle,
                                              ncclDevResourceRequirements_t *outReq) {
  return ncclLsaBarrierCreateRequirement(team, nBarriers, outHandle, outReq);
}
