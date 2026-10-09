/* Environment variable lookup. libsircl's own variables are named LIBSIRCL_*. Until release 0.7.0 each is
 * also read under its earlier name when the LIBSIRCL_* name is unset: SIRCL_CCL_<rest>, and
 * SIRCL_NCCLAPI_VERSION_CODE for LIBSIRCL_NCCL_API_VERSION. Every other name (SIRCL's own SIRCL_*
 * variables, NCCL_*) is read as given. */
#ifndef LIBSIRCL_ENV_NAMES_H
#define LIBSIRCL_ENV_NAMES_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline const char *sccl_env(const char *name) {
  const char *value = getenv(name);
  if (value || strncmp(name, "LIBSIRCL_", 9) != 0) return value;
  if (!strcmp(name, "LIBSIRCL_NCCL_API_VERSION")) return getenv("SIRCL_NCCLAPI_VERSION_CODE");
  char alias[128];
  snprintf(alias, sizeof alias, "SIRCL_CCL_%s", name + 9);
  return getenv(alias);
}
#endif
