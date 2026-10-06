#ifndef _ZZ_MEM_INCLUDE_
#define _ZZ_MEM_INCLUDE_

#include <stddef.h>
#include <stdlib.h>

#define ZZ_MEM_TRACKING     1
#define ZZ_MEM_THREADING    1
#define ZZ_MEM_LOGGING      0
#define ZZ_WORKSPACE_DEFAULT_LIMIT ((size_t) 128 * 1024 * 1024)

#ifdef __cplusplus
extern "C" {
#endif

extern long zz_mem_used;
extern long zz_mem_peak;
extern long zz_malloc_count;
extern long zz_free_count;

typedef struct zz_workspace_struct zz_workspace_t;

typedef struct
{
  size_t retained_limit_bytes;
  size_t live_bytes;
  size_t peak_live_bytes;
  size_t retained_bytes;
  size_t peak_retained_bytes;
  size_t requested_bytes;
  size_t cache_hits;
  size_t backing_allocations;
  size_t backing_frees;
}
zz_workspace_stats_t;

// Workspaces retain only freed ZZ allocations, up to the per-workspace limit.
// Every allocation made while bound must be freed before restoring the prior
// binding. A workspace may not be destroyed while it is bound or has live
// allocations, and one workspace must not be bound concurrently by multiple
// threads. The binding itself is thread-local.
zz_workspace_t *zz_workspace_create(size_t retained_limit_bytes);
void zz_workspace_destroy(zz_workspace_t *workspace);
size_t zz_workspace_trim(zz_workspace_t *workspace, size_t keep_bytes);
void zz_workspace_get_stats(zz_workspace_t *workspace,
                            zz_workspace_stats_t *stats);
void zz_workspace_reset_stats(zz_workspace_t *workspace);

// Bind a workspace to this thread and return the previous binding. Pass that
// return value to zz_workspace_restore() on every exit path.
zz_workspace_t *zz_workspace_bind(zz_workspace_t *workspace);
void zz_workspace_restore(zz_workspace_t *previous);

void *zz_malloc(size_t n);
void zz_free(void *p, size_t n);

#ifdef __cplusplus
}
#endif

#endif
