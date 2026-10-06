#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include "hwmpz.h"
#include "rforest.h"

static void test_pool_reuse(void)
{
  zz_workspace_t *workspace = zz_workspace_create(16 * 1024);
  assert(workspace);

  zz_workspace_t *previous = zz_workspace_bind(workspace);
  void *large = zz_malloc(4096);
  assert(large);
  assert((uintptr_t) large % _Alignof(max_align_t) == 0);
  memset(large, 0x5a, 4096);
  zz_free(large, 4096);
  zz_workspace_restore(previous);

  zz_workspace_stats_t stats;
  zz_workspace_get_stats(workspace, &stats);
  assert(stats.live_bytes == 0);
  assert(stats.peak_live_bytes == 4096);
  assert(stats.retained_bytes == 4096);
  assert(stats.backing_allocations == 1);

  zz_workspace_reset_stats(workspace);
  previous = zz_workspace_bind(workspace);
  void *small = zz_malloc(1024);
  assert(small == large);
  memset(small, 0xa5, 1024);
  zz_free(small, 1024);
  zz_workspace_restore(previous);

  zz_workspace_get_stats(workspace, &stats);
  assert(stats.live_bytes == 0);
  assert(stats.retained_bytes == 4096);
  assert(stats.cache_hits == 1);
  assert(stats.backing_allocations == 0);

  zz_workspace_reset_stats(workspace);
  previous = zz_workspace_bind(workspace);
  void *grown = zz_malloc(8192);
  assert(grown);
  memset(grown, 0x3c, 8192);
  zz_free(grown, 8192);
  zz_workspace_restore(previous);

  zz_workspace_get_stats(workspace, &stats);
  assert(stats.retained_bytes == 12 * 1024);
  assert(stats.backing_allocations == 1);
  assert(zz_workspace_trim(workspace, 2048) <= 2048);
  zz_workspace_get_stats(workspace, &stats);
  assert(stats.retained_bytes <= 2048);
  assert(zz_workspace_trim(workspace, 0) == 0);
  zz_workspace_destroy(workspace);
}


static void test_pool_limit_and_contexts(void)
{
  zz_workspace_t *limited = zz_workspace_create(4096);
  zz_workspace_t *independent = zz_workspace_create(4096);
  assert(limited && independent);

  zz_workspace_t *previous = zz_workspace_bind(limited);
  void *p = zz_malloc(8192);
  assert(p);
  zz_free(p, 8192);
  zz_workspace_restore(previous);

  zz_workspace_stats_t limited_stats, independent_stats;
  zz_workspace_get_stats(limited, &limited_stats);
  assert(limited_stats.retained_limit_bytes == 4096);
  assert(limited_stats.retained_bytes == 0);
  assert(limited_stats.backing_frees == 1);

  previous = zz_workspace_bind(independent);
  p = zz_malloc(2048);
  assert(p);
  zz_free(p, 2048);
  zz_workspace_restore(previous);
  zz_workspace_get_stats(independent, &independent_stats);
  assert(independent_stats.retained_bytes == 2048);
  assert(independent_stats.cache_hits == 0);

  zz_workspace_get_stats(limited, &limited_stats);
  assert(limited_stats.retained_bytes == 0);
  zz_workspace_destroy(independent);
  zz_workspace_destroy(limited);
}


typedef struct
{
  zz_workspace_t *workspace;
  zz_workspace_t *previous;
}
worker_binding_args_t;

static void *worker_binding_test(void *opaque)
{
  worker_binding_args_t *args = (worker_binding_args_t *) opaque;
  args->previous = zz_workspace_bind(args->workspace);

  enum { BLOCKS = 32, ROUNDS = 2 };
  void *blocks[BLOCKS];
  for (int round = 0; round < ROUNDS; round++)
    {
      for (int i = 0; i < BLOCKS; i++)
        {
          blocks[i] = zz_malloc(2048);
          assert(blocks[i]);
          memset(blocks[i], i + round, 2048);
        }
      for (int i = 0; i < BLOCKS; i++)
        {
          unsigned char *bytes = (unsigned char *) blocks[i];
          assert(bytes[0] == (unsigned char) (i + round));
          assert(bytes[2047] == (unsigned char) (i + round));
          zz_free(blocks[i], 2048);
        }
    }

  zz_workspace_restore(args->previous);
  return NULL;
}


static void test_thread_local_binding(void)
{
  zz_workspace_t *outer = zz_workspace_create(4096);
  zz_workspace_t *worker = zz_workspace_create(64 * 1024);
  assert(outer && worker);

  zz_workspace_t *previous = zz_workspace_bind(outer);
  assert(previous == NULL);
  worker_binding_args_t args = { worker, (zz_workspace_t *) 1 };
  pthread_t thread;
  assert(pthread_create(&thread, NULL, worker_binding_test, &args) == 0);
  assert(pthread_join(thread, NULL) == 0);
  // A fresh worker thread must not inherit the caller's bound workspace.
  assert(args.previous == NULL);

  void *outer_block = zz_malloc(128);
  assert(outer_block);
  zz_free(outer_block, 128);
  zz_workspace_restore(previous);

  zz_workspace_stats_t outer_stats, worker_stats;
  zz_workspace_get_stats(outer, &outer_stats);
  zz_workspace_get_stats(worker, &worker_stats);
  assert(outer_stats.requested_bytes == 128);
  assert(outer_stats.backing_allocations == 1);
  assert(outer_stats.cache_hits == 0);
  assert(worker_stats.live_bytes == 0);
  assert(worker_stats.backing_allocations == 32);
  assert(worker_stats.cache_hits == 32);
  assert(worker_stats.retained_bytes == 32 * 2048);
  assert(zz_workspace_trim(worker, 0) == 0);
  assert(zz_workspace_trim(outer, 0) == 0);
  zz_workspace_destroy(worker);
  zz_workspace_destroy(outer);
}


static void init_mpz_array(mpz_t *values, size_t count)
{
  for (size_t i = 0; i < count; i++)
    mpz_init(values[i]);
}


static void clear_mpz_array(mpz_t *values, size_t count)
{
  for (size_t i = 0; i < count; i++)
    mpz_clear(values[i]);
}


static void run_small_forest(zz_workspace_t *workspace, int disable_fft,
                             mpz_t *outputs, mpz_t *final_v, mpz_t final_z)
{
  enum { DIM = 2, DEG = 1, N = 3, OUTPUT_COUNT = N * DIM };
  mpz_t matrix[DIM * DIM * (DEG + 1)];
  mpz_t moduli[N], vector[DIM], result[OUTPUT_COUNT], z;
  long endpoints[N] = { 1, 2, 3 };

  init_mpz_array(matrix, DIM * DIM * (DEG + 1));
  init_mpz_array(moduli, N);
  init_mpz_array(vector, DIM);
  init_mpz_array(result, OUTPUT_COUNT);
  mpz_init(z);

  // M(x) = [[1+x, 1], [x, 1]].
  mpz_set_ui(matrix[0], 1);
  mpz_set_ui(matrix[1], 1);
  mpz_set_ui(matrix[2], 1);
  mpz_set_ui(matrix[5], 1);
  mpz_set_ui(matrix[6], 1);
  mpz_set_ui(moduli[0], 3);
  mpz_set_ui(moduli[1], 5);
  mpz_set_ui(moduli[2], 7);
  mpz_set_ui(vector[0], 1);
  mpz_set_ui(vector[1], 2);
  mpz_mul(z, moduli[0], moduli[1]);
  mpz_mul(z, z, moduli[2]);

  int old_disable_fft = hw_disable_fft;
  hw_disable_fft = disable_fft;
  if (workspace)
    rforest_with_workspace(workspace, result, vector, 1, matrix, DEG, DIM,
                            moduli, 0, endpoints, N, z, 1);
  else
    rforest(result, vector, 1, matrix, DEG, DIM, moduli, 0, endpoints, N, z,
            1);
  hw_disable_fft = old_disable_fft;

  for (size_t i = 0; i < OUTPUT_COUNT; i++)
    mpz_set(outputs[i], result[i]);
  for (size_t i = 0; i < DIM; i++)
    mpz_set(final_v[i], vector[i]);
  mpz_set(final_z, z);

  mpz_clear(z);
  clear_mpz_array(result, OUTPUT_COUNT);
  clear_mpz_array(vector, DIM);
  clear_mpz_array(moduli, N);
  clear_mpz_array(matrix, DIM * DIM * (DEG + 1));
}


static void assert_mpz_arrays_equal(mpz_t *left, mpz_t *right, size_t count)
{
  for (size_t i = 0; i < count; i++)
    assert(mpz_cmp(left[i], right[i]) == 0);
}


static void test_forest_compatibility(void)
{
  enum { DIM = 2, N = 3, OUTPUT_COUNT = N * DIM };
  mpz_t reference_outputs[OUTPUT_COUNT], workspace_outputs[OUTPUT_COUNT];
  mpz_t reference_v[DIM], workspace_v[DIM];
  mpz_t reference_z, workspace_z;
  zz_workspace_t *workspace =
    zz_workspace_create(ZZ_WORKSPACE_DEFAULT_LIMIT);
  zz_workspace_t *outer = zz_workspace_create(4096);
  assert(workspace && outer);

  init_mpz_array(reference_outputs, OUTPUT_COUNT);
  init_mpz_array(workspace_outputs, OUTPUT_COUNT);
  init_mpz_array(reference_v, DIM);
  init_mpz_array(workspace_v, DIM);
  mpz_init(reference_z);
  mpz_init(workspace_z);

  run_small_forest(NULL, 0, reference_outputs, reference_v, reference_z);
  zz_workspace_t *previous = zz_workspace_bind(outer);
  assert(previous == NULL);
  run_small_forest(workspace, 0, workspace_outputs, workspace_v, workspace_z);
  void *outer_block = zz_malloc(128);
  assert(outer_block);
  zz_free(outer_block, 128);
  zz_workspace_restore(previous);

  zz_workspace_stats_t outer_stats;
  zz_workspace_get_stats(outer, &outer_stats);
  assert(outer_stats.requested_bytes == 128);
  assert(outer_stats.backing_allocations == 1);
  assert(outer_stats.cache_hits == 0);
  assert(zz_workspace_trim(outer, 0) == 0);
  zz_workspace_destroy(outer);

  assert_mpz_arrays_equal(reference_outputs, workspace_outputs, OUTPUT_COUNT);
  assert_mpz_arrays_equal(reference_v, workspace_v, DIM);
  assert(mpz_cmp(reference_z, workspace_z) == 0);
  assert(mpz_cmp_ui(workspace_z, 1) == 0);

  // Reuse the same context for another operation and the disabled-FFT path.
  run_small_forest(workspace, 0, workspace_outputs, workspace_v, workspace_z);
  assert_mpz_arrays_equal(reference_outputs, workspace_outputs, OUTPUT_COUNT);
  run_small_forest(workspace, 1, workspace_outputs, workspace_v, workspace_z);
  assert_mpz_arrays_equal(reference_outputs, workspace_outputs, OUTPUT_COUNT);
  assert_mpz_arrays_equal(reference_v, workspace_v, DIM);
  assert(mpz_cmp(reference_z, workspace_z) == 0);

  zz_workspace_stats_t stats;
  zz_workspace_get_stats(workspace, &stats);
  assert(stats.live_bytes == 0);
  assert(stats.retained_bytes == 0);
  assert(zz_workspace_trim(workspace, 0) == 0);
  zz_workspace_destroy(workspace);

  mpz_clear(workspace_z);
  mpz_clear(reference_z);
  clear_mpz_array(workspace_v, DIM);
  clear_mpz_array(reference_v, DIM);
  clear_mpz_array(workspace_outputs, OUTPUT_COUNT);
  clear_mpz_array(reference_outputs, OUTPUT_COUNT);
}


int main(void)
{
  test_pool_reuse();
  test_pool_limit_and_contexts();
  test_thread_local_binding();
  test_forest_compatibility();
  puts("ZZ FFT workspace tests passed");
  return 0;
}
