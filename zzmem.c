#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "zzmem.h"

long zz_mem_used;
long zz_mem_peak;
long zz_malloc_count;
long zz_free_count;

typedef union zz_block_header zz_block_header_t;

union zz_block_header
{
  max_align_t alignment;
  struct
  {
    zz_block_header_t *next;
    zz_workspace_t *owner;
    size_t capacity;
    size_t requested;
  }
  fields;
};

#define ZZ_POOL_BINS (sizeof(size_t) * CHAR_BIT)

struct zz_workspace_struct
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
  zz_block_header_t *bins[ZZ_POOL_BINS];
};

static _Thread_local zz_workspace_t *zz_bound_workspace;

static size_t size_add_saturated(size_t a, size_t b)
{
  return b > SIZE_MAX - a ? SIZE_MAX : a + b;
}

static unsigned zz_pool_bin(size_t capacity)
{
  unsigned bin = 0;
  while (capacity > 1)
    {
      capacity >>= 1;
      bin++;
    }
  return bin;
}

static void zz_track_alloc(size_t n)
{
#if ZZ_MEM_TRACKING
#if ZZ_MEM_THREADING
#pragma omp critical
#endif
  {
    zz_mem_used += (long) n;
    if (zz_mem_used > zz_mem_peak)
      zz_mem_peak = zz_mem_used;
    zz_malloc_count++;
  }
#else
  (void) n;
#endif
}

static void zz_track_free(size_t n)
{
#if ZZ_MEM_TRACKING
#if ZZ_MEM_THREADING
#pragma omp critical
#endif
  {
    zz_mem_used -= (long) n;
    zz_free_count++;
  }
#else
  (void) n;
#endif
}

static zz_block_header_t *zz_pool_find(zz_workspace_t *workspace, size_t n)
{
  unsigned first_bin = zz_pool_bin(n ? n : 1);
  for (unsigned bin = first_bin; bin < ZZ_POOL_BINS; bin++)
    {
      zz_block_header_t **link = &workspace->bins[bin];
      zz_block_header_t **best_link = NULL;
      size_t best_capacity = SIZE_MAX;
      while (*link)
        {
          zz_block_header_t *block = *link;
          if (block->fields.capacity >= n &&
              block->fields.capacity < best_capacity)
            {
              best_link = link;
              best_capacity = block->fields.capacity;
            }
          link = &block->fields.next;
        }
      if (best_link)
        {
          zz_block_header_t *block = *best_link;
          *best_link = block->fields.next;
          block->fields.next = NULL;
          workspace->retained_bytes -= block->fields.capacity;
          workspace->cache_hits++;
          return block;
        }
    }
  return NULL;
}

zz_workspace_t *zz_workspace_create(size_t retained_limit_bytes)
{
  zz_workspace_t *workspace =
    (zz_workspace_t *) calloc(1, sizeof(zz_workspace_t));
  if (workspace)
    workspace->retained_limit_bytes = retained_limit_bytes;
  return workspace;
}

zz_workspace_t *zz_workspace_bind(zz_workspace_t *workspace)
{
  zz_workspace_t *previous = zz_bound_workspace;
  zz_bound_workspace = workspace;
  return previous;
}

void zz_workspace_restore(zz_workspace_t *previous)
{
  zz_bound_workspace = previous;
}

void *zz_malloc(size_t n)
{
  zz_workspace_t *workspace = zz_bound_workspace;
  if (!workspace)
    {
      void *p = malloc(n);
      if (p)
        zz_track_alloc(n);
#if ZZ_MEM_LOGGING
      printf("zz_malloc %zu (%lx)\n", n, (unsigned long) p);
#endif
      return p;
    }

  size_t capacity = n ? n : 1;
  if (capacity > SIZE_MAX - sizeof(zz_block_header_t))
    return NULL;

  zz_block_header_t *block = NULL;
  int fresh = 0;
#if ZZ_MEM_THREADING
#pragma omp critical (zz_workspace_cache)
#endif
  {
    if (workspace->live_bytes <= SIZE_MAX - n)
      {
        block = zz_pool_find(workspace, n);
        if (!block)
          {
            block = (zz_block_header_t *)
              malloc(sizeof(zz_block_header_t) + capacity);
            if (block)
              {
                workspace->backing_allocations++;
                fresh = 1;
              }
          }
        if (block)
          {
            block->fields.owner = workspace;
            block->fields.requested = n;
            block->fields.next = NULL;
            if (fresh)
              block->fields.capacity = capacity;
            workspace->live_bytes += n;
            if (workspace->live_bytes > workspace->peak_live_bytes)
              workspace->peak_live_bytes = workspace->live_bytes;
            workspace->requested_bytes =
              size_add_saturated(workspace->requested_bytes, n);
          }
      }
  }

  if (block)
    zz_track_alloc(n);
#if ZZ_MEM_LOGGING
  printf("zz_malloc %zu (%lx)\n", n,
         (unsigned long) (block ? (void *) (block + 1) : NULL));
#endif
  return block ? (void *) (block + 1) : NULL;
}

void zz_free(void *p, size_t n)
{
  if (!p)
    {
      zz_track_free(n);
      return;
    }

  zz_workspace_t *workspace = zz_bound_workspace;
  if (!workspace)
    {
#if ZZ_MEM_LOGGING
      printf("zz_free %zu (%lx)\n", n, (unsigned long) p);
#endif
      free(p);
      zz_track_free(n);
      return;
    }

  zz_block_header_t *block = ((zz_block_header_t *) p) - 1;
  assert(block->fields.owner == workspace);
  assert(block->fields.requested == n);

  int retained = 0;
#if ZZ_MEM_THREADING
#pragma omp critical (zz_workspace_cache)
#endif
  {
    assert(workspace->live_bytes >= n);
    workspace->live_bytes -= n;
    if (block->fields.capacity <= workspace->retained_limit_bytes -
                                    workspace->retained_bytes)
      {
        unsigned bin = zz_pool_bin(block->fields.capacity);
        block->fields.next = workspace->bins[bin];
        workspace->bins[bin] = block;
        workspace->retained_bytes += block->fields.capacity;
        if (workspace->retained_bytes > workspace->peak_retained_bytes)
          workspace->peak_retained_bytes = workspace->retained_bytes;
        retained = 1;
      }
    else
      workspace->backing_frees++;
  }

#if ZZ_MEM_LOGGING
  printf("zz_free %zu (%lx)\n", n, (unsigned long) p);
#endif
  if (!retained)
    free(block);
  zz_track_free(n);
}

size_t zz_workspace_trim(zz_workspace_t *workspace, size_t keep_bytes)
{
  assert(workspace);
  assert(workspace != zz_bound_workspace);

#if ZZ_MEM_THREADING
#pragma omp critical (zz_workspace_cache)
#endif
  {
    if (keep_bytes > workspace->retained_bytes)
      keep_bytes = workspace->retained_bytes;
    for (unsigned bin = ZZ_POOL_BINS;
         bin > 0 && workspace->retained_bytes > keep_bytes; bin--)
      {
        zz_block_header_t **link = &workspace->bins[bin - 1];
        while (*link && workspace->retained_bytes > keep_bytes)
          {
            zz_block_header_t *block = *link;
            *link = block->fields.next;
            workspace->retained_bytes -= block->fields.capacity;
            workspace->backing_frees++;
            free(block);
          }
      }
  }
  return workspace->retained_bytes;
}

void zz_workspace_get_stats(zz_workspace_t *workspace,
                            zz_workspace_stats_t *stats)
{
  assert(workspace);
  assert(stats);
#if ZZ_MEM_THREADING
#pragma omp critical (zz_workspace_cache)
#endif
  {
    stats->retained_limit_bytes = workspace->retained_limit_bytes;
    stats->live_bytes = workspace->live_bytes;
    stats->peak_live_bytes = workspace->peak_live_bytes;
    stats->retained_bytes = workspace->retained_bytes;
    stats->peak_retained_bytes = workspace->peak_retained_bytes;
    stats->requested_bytes = workspace->requested_bytes;
    stats->cache_hits = workspace->cache_hits;
    stats->backing_allocations = workspace->backing_allocations;
    stats->backing_frees = workspace->backing_frees;
  }
}

void zz_workspace_reset_stats(zz_workspace_t *workspace)
{
  assert(workspace);
  assert(workspace != zz_bound_workspace);
#if ZZ_MEM_THREADING
#pragma omp critical (zz_workspace_cache)
#endif
  {
    workspace->peak_live_bytes = workspace->live_bytes;
    workspace->peak_retained_bytes = workspace->retained_bytes;
    workspace->requested_bytes = 0;
    workspace->cache_hits = 0;
    workspace->backing_allocations = 0;
    workspace->backing_frees = 0;
  }
}

void zz_workspace_destroy(zz_workspace_t *workspace)
{
  if (!workspace)
    return;
  assert(workspace != zz_bound_workspace);
  assert(workspace->live_bytes == 0);
  zz_workspace_trim(workspace, 0);
  free(workspace);
}
