/* SIRCL's native proxy over the shared-memory verbs stand-in (emulation/shm_verbs.c);
 * compiled with fake_verbs/ ahead of the system headers. */
#define SCCL_PROXY_TAG emu
#define SCCL_RENAME_VERBS 1
#include "proxy_names.h"
#include "sircl_roce_proxy.c"
