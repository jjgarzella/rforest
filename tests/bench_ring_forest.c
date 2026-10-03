#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmp.h>

#include "hwmpz.h"
#include "rforest.h"

#define BENCH_DIM 4
#define BENCH_ROWS 4
#define BENCH_DEG 1
#define BENCH_ENDPOINTS 64
#define BENCH_KAPPA 0
#define BENCH_SAMPLES 9

#define BLOCK_DIM (2 * BENCH_DIM)
#define RING_CELLS ((size_t)BENCH_DIM * (size_t)BENCH_DIM)
#define BLOCK_CELLS ((size_t)BLOCK_DIM * (size_t)BLOCK_DIM)
#define RING_VECTOR_CELLS (2 * RING_CELLS)
#define BLOCK_VECTOR_CELLS BLOCK_CELLS
#define RING_MATRIX_COUNT (2 * RING_CELLS * (BENCH_DEG + 1))
#define BLOCK_MATRIX_COUNT (BLOCK_CELLS * (BENCH_DEG + 1))

typedef struct {
    mpz_t *initial_v;
    mpz_t *working_v;
    mpz_t *outputs;
    mpz_t z;
} forest_state;

static mpz_t initial_z;

static mpz_t *new_values(size_t count)
{
    mpz_t *values = malloc(count * sizeof(*values));
    if (!values && count)
        abort();
    for (size_t i = 0; i < count; i++)
        mpz_init(values[i]);
    return values;
}

static void clear_values(mpz_t *values, size_t count)
{
    for (size_t i = 0; i < count; i++)
        mpz_clear(values[i]);
    free(values);
}

static void copy_values(mpz_t *destination, mpz_t *source, size_t count)
{
    for (size_t i = 0; i < count; i++)
        mpz_set(destination[i], source[i]);
}

static void initialize_ring_matrix(mpz_t *M)
{
    for (int row = 0; row < BENCH_DIM; row++) {
        for (int col = 0; col < BENCH_DIM; col++) {
            size_t entry = (size_t)row * BENCH_DIM + (size_t)col;
            size_t offset = entry * 2 * (BENCH_DEG + 1);
            mpz_set_si(M[offset], row == col ? 2 + row : (row + 2 * col) % 5 - 2);
            mpz_set_si(M[offset + 1], (row + 3 * col) % 5 - 2);
            mpz_set_si(M[offset + 2], (2 * row + col) % 7 - 3);
            mpz_set_si(M[offset + 3], (row + col) % 3 - 1);
        }
    }
}

static void embed_block_matrix(mpz_t *block_M, mpz_t *ring_M)
{
    for (int row = 0; row < BENCH_DIM; row++) {
        for (int col = 0; col < BENCH_DIM; col++) {
            size_t ring_entry = (size_t)row * BENCH_DIM + (size_t)col;
            size_t upper_left = (size_t)row * BLOCK_DIM + (size_t)col;
            size_t upper_right = upper_left + BENCH_DIM;
            size_t lower_left = (size_t)(BENCH_DIM + row) * BLOCK_DIM +
                                (size_t)col;
            size_t lower_right = lower_left + BENCH_DIM;
            for (int degree = 0; degree <= BENCH_DEG; degree++) {
                size_t ring_offset = ring_entry * 2 * (BENCH_DEG + 1);
                mpz_set(block_M[upper_left * (BENCH_DEG + 1) + degree],
                        ring_M[ring_offset + degree]);
                mpz_set(block_M[upper_right * (BENCH_DEG + 1) + degree],
                        ring_M[ring_offset + (BENCH_DEG + 1) + degree]);
                mpz_set_ui(block_M[lower_left * (BENCH_DEG + 1) + degree], 0);
                mpz_set(block_M[lower_right * (BENCH_DEG + 1) + degree],
                        ring_M[ring_offset + degree]);
            }
        }
    }
}

static void evaluate_ring_matrix(mpz_t *evaluated, mpz_t *M, long x)
{
    mpz_t value;
    mpz_init(value);
    for (size_t entry = 0; entry < RING_CELLS; entry++) {
        size_t source = entry * 2 * (BENCH_DEG + 1);
        mpz_mul_si(value, M[source + 1], x);
        mpz_add(value, value, M[source]);
        mpz_set(evaluated[entry], value);
        mpz_mul_si(value, M[source + 3], x);
        mpz_add(value, value, M[source + 2]);
        mpz_set(evaluated[RING_CELLS + entry], value);
    }
    mpz_clear(value);
}

static void naive_p2_multiply(mpz_t *C, mpz_t *A, mpz_t *B)
{
    mpz_t product;
    mpz_init(product);
    for (int row = 0; row < BENCH_ROWS; row++) {
        for (int col = 0; col < BENCH_DIM; col++) {
            size_t out = (size_t)row * BENCH_DIM + (size_t)col;
            mpz_set_ui(C[out], 0);
            mpz_set_ui(C[RING_CELLS + out], 0);
            for (int inner = 0; inner < BENCH_DIM; inner++) {
                size_t a = (size_t)row * BENCH_DIM + (size_t)inner;
                size_t b = (size_t)inner * BENCH_DIM + (size_t)col;
                mpz_mul(product, A[a], B[b]);
                mpz_add(C[out], C[out], product);
                mpz_mul(product, A[a], B[RING_CELLS + b]);
                mpz_add(C[RING_CELLS + out], C[RING_CELLS + out], product);
                mpz_mul(product, A[RING_CELLS + a], B[b]);
                mpz_add(C[RING_CELLS + out], C[RING_CELLS + out], product);
            }
        }
    }
    mpz_clear(product);
}

static void run_ring_forest(forest_state *state, mpz_t *M, mpz_t *moduli,
                            long *endpoints, long kbase, long n)
{
    copy_values(state->working_v, state->initial_v, RING_VECTOR_CELLS);
    mpz_set(state->z, initial_z);
    rforest_p2(state->outputs, state->working_v, BENCH_ROWS, M, BENCH_DEG,
               BENCH_DIM, moduli, kbase, endpoints, n, state->z, BENCH_KAPPA);
}

static void run_block_forest(forest_state *state, mpz_t *M, mpz_t *moduli,
                             long *endpoints, long kbase, long n)
{
    copy_values(state->working_v, state->initial_v, BLOCK_VECTOR_CELLS);
    mpz_set(state->z, initial_z);
    rforest(state->outputs, state->working_v, BLOCK_DIM, M, BENCH_DEG,
            BLOCK_DIM, moduli, kbase, endpoints, n, state->z, BENCH_KAPPA);
}

static void run_naive_adapter(forest_state *state, mpz_t *M, mpz_t *moduli,
                              long *endpoints, long kbase, long n,
                              mpz_t *evaluated, mpz_t *next_v)
{
    long k = kbase;
    copy_values(state->working_v, state->initial_v, RING_VECTOR_CELLS);
    mpz_set(state->z, initial_z);
    for (long endpoint = 0; endpoint < n; endpoint++) {
        while (k < endpoints[endpoint]) {
            evaluate_ring_matrix(evaluated, M, k++);
            naive_p2_multiply(next_v, state->working_v, evaluated);
            mpz_vec_mod_naive(state->working_v, next_v, RING_VECTOR_CELLS,
                              state->z);
        }
        mpz_vec_mod_naive(state->outputs + (size_t)endpoint * RING_VECTOR_CELLS,
                          state->working_v, RING_VECTOR_CELLS,
                          moduli[endpoint]);
        mpz_divexact(state->z, state->z, moduli[endpoint]);
        mpz_vec_mod_naive(state->working_v, state->working_v,
                          RING_VECTOR_CELLS, state->z);
    }
}

static uint64_t now_ns(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        abort();
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
           (uint64_t)now.tv_nsec;
}

static int compare_u64(const void *left, const void *right)
{
    uint64_t a = *(const uint64_t *)left;
    uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static uint64_t median(uint64_t *values, size_t count)
{
    qsort(values, count, sizeof(*values), compare_u64);
    return values[count / 2];
}

static void compare_regular_output(mpz_t *ring_A, mpz_t *block_A,
                                  mpz_t *ring_V, mpz_t *block_V)
{
    for (size_t endpoint = 0; endpoint < BENCH_ENDPOINTS; endpoint++) {
        mpz_t *coeff0 = ring_A + endpoint * RING_VECTOR_CELLS;
        mpz_t *coeff1 = coeff0 + RING_CELLS;
        mpz_t *block = block_A + endpoint * BLOCK_VECTOR_CELLS;
        for (int row = 0; row < BLOCK_DIM; row++) {
            for (int col = 0; col < BLOCK_DIM; col++) {
                size_t index = (size_t)row * BLOCK_DIM + (size_t)col;
                mpz_t *actual;
                if (row < BENCH_DIM && col < BENCH_DIM)
                    actual = coeff0 + (size_t)row * BENCH_DIM + (size_t)col;
                else if (row < BENCH_DIM)
                    actual = coeff1 + (size_t)row * BENCH_DIM +
                             (size_t)(col - BENCH_DIM);
                else if (col < BENCH_DIM)
                    actual = NULL;
                else
                    actual = coeff0 + (size_t)(row - BENCH_DIM) * BENCH_DIM +
                             (size_t)(col - BENCH_DIM);
                if (actual ? mpz_cmp(block[index], *actual) != 0
                           : mpz_sgn(block[index]) != 0)
                    abort();
            }
        }
    }
    for (int row = 0; row < BLOCK_DIM; row++) {
        for (int col = 0; col < BLOCK_DIM; col++) {
            size_t index = (size_t)row * BLOCK_DIM + (size_t)col;
            mpz_t *actual;
            if (row < BENCH_DIM && col < BENCH_DIM)
                actual = ring_V + (size_t)row * BENCH_DIM + (size_t)col;
            else if (row < BENCH_DIM)
                actual = ring_V + RING_CELLS + (size_t)row * BENCH_DIM +
                         (size_t)(col - BENCH_DIM);
            else if (col < BENCH_DIM)
                actual = NULL;
            else
                actual = ring_V + (size_t)(row - BENCH_DIM) * BENCH_DIM +
                         (size_t)(col - BENCH_DIM);
            if (actual ? mpz_cmp(block_V[index], *actual) != 0
                       : mpz_sgn(block_V[index]) != 0)
                abort();
        }
    }
}

static void compare_ring_results(forest_state *ring, forest_state *naive)
{
    for (size_t i = 0; i < BENCH_ENDPOINTS * RING_VECTOR_CELLS; i++)
        if (mpz_cmp(ring->outputs[i], naive->outputs[i]) != 0)
            abort();
    for (size_t i = 0; i < RING_VECTOR_CELLS; i++)
        if (mpz_cmp(ring->working_v[i], naive->working_v[i]) != 0)
            abort();
    if (mpz_cmp(ring->z, naive->z) != 0)
        abort();
}

static uint64_t time_ring(forest_state *state, mpz_t *M, mpz_t *moduli,
                          long *endpoints)
{
    uint64_t start = now_ns();
    run_ring_forest(state, M, moduli, endpoints, 1, BENCH_ENDPOINTS);
    return now_ns() - start;
}

static uint64_t time_block(forest_state *state, mpz_t *M, mpz_t *moduli,
                           long *endpoints)
{
    uint64_t start = now_ns();
    run_block_forest(state, M, moduli, endpoints, 1, BENCH_ENDPOINTS);
    return now_ns() - start;
}

static uint64_t time_naive(forest_state *state, mpz_t *M, mpz_t *moduli,
                           long *endpoints, mpz_t *evaluated,
                           mpz_t *next_v)
{
    uint64_t start = now_ns();
    run_naive_adapter(state, M, moduli, endpoints, 1, BENCH_ENDPOINTS,
                      evaluated, next_v);
    return now_ns() - start;
}

int main(void)
{
    mpz_t *ring_M = new_values(RING_MATRIX_COUNT);
    mpz_t *block_M = new_values(BLOCK_MATRIX_COUNT);
    mpz_t *ring_V = new_values(RING_VECTOR_CELLS);
    mpz_t *block_V = new_values(BLOCK_VECTOR_CELLS);
    mpz_t *moduli = new_values(BENCH_ENDPOINTS);
    mpz_t *ring_A = new_values(BENCH_ENDPOINTS * RING_VECTOR_CELLS);
    mpz_t *block_A = new_values(BENCH_ENDPOINTS * BLOCK_VECTOR_CELLS);
    mpz_t *naive_A = new_values(BENCH_ENDPOINTS * RING_VECTOR_CELLS);
    mpz_t *naive_V = new_values(RING_VECTOR_CELLS);
    mpz_t *evaluated = new_values(RING_VECTOR_CELLS);
    mpz_t *next_v = new_values(RING_VECTOR_CELLS);
    long endpoints[BENCH_ENDPOINTS];
    forest_state ring_state = {ring_V, new_values(RING_VECTOR_CELLS), ring_A, {0}};
    forest_state block_state = {block_V, new_values(BLOCK_VECTOR_CELLS), block_A, {0}};
    forest_state naive_state = {ring_V, naive_V, naive_A, {0}};
    uint64_t samples[3][BENCH_SAMPLES];
    uint64_t scratch[BENCH_SAMPLES];
    uint64_t medians[3];
    uint64_t deviations[3];

    mpz_init(initial_z);
    mpz_init(ring_state.z);
    mpz_init(block_state.z);
    mpz_init(naive_state.z);
    initialize_ring_matrix(ring_M);
    embed_block_matrix(block_M, ring_M);
    mpz_matrix_set_one(ring_V, BENCH_DIM);
    mpz_matrix_set_one(block_V, BLOCK_DIM);
    for (size_t i = 0; i < BENCH_ENDPOINTS; i++) {
        mpz_set_ui(moduli[i], 1009 + (unsigned long)i);
        endpoints[i] = (long)i + 2;
    }
    mpz_set_ui(initial_z, 13);
    for (size_t i = 0; i < BENCH_ENDPOINTS; i++)
        mpz_mul(initial_z, initial_z, moduli[i]);

    run_ring_forest(&ring_state, ring_M, moduli, endpoints, 1, BENCH_ENDPOINTS);
    run_block_forest(&block_state, block_M, moduli, endpoints, 1, BENCH_ENDPOINTS);
    run_naive_adapter(&naive_state, ring_M, moduli, endpoints, 1,
                      BENCH_ENDPOINTS, evaluated, next_v);
    compare_regular_output(ring_state.outputs, block_state.outputs,
                           ring_state.working_v, block_state.working_v);
    compare_ring_results(&ring_state, &naive_state);

    for (int algorithm = 0; algorithm < 3; algorithm++) {
        if (algorithm == 0)
            (void)time_ring(&ring_state, ring_M, moduli, endpoints);
        else if (algorithm == 1)
            (void)time_block(&block_state, block_M, moduli, endpoints);
        else
            (void)time_naive(&naive_state, ring_M, moduli, endpoints,
                             evaluated, next_v);
    }
    for (size_t sample = 0; sample < BENCH_SAMPLES; sample++) {
        for (size_t offset = 0; offset < 3; offset++) {
            int algorithm = (int)((sample + offset) % 3);
            if (algorithm == 0)
                samples[algorithm][sample] = time_ring(
                    &ring_state, ring_M, moduli, endpoints);
            else if (algorithm == 1)
                samples[algorithm][sample] = time_block(
                    &block_state, block_M, moduli, endpoints);
            else
                samples[algorithm][sample] = time_naive(
                    &naive_state, ring_M, moduli, endpoints,
                    evaluated, next_v);
        }
    }
    for (int algorithm = 0; algorithm < 3; algorithm++) {
        for (size_t sample = 0; sample < BENCH_SAMPLES; sample++)
            scratch[sample] = samples[algorithm][sample];
        medians[algorithm] = median(scratch, BENCH_SAMPLES);
        for (size_t sample = 0; sample < BENCH_SAMPLES; sample++)
            scratch[sample] = samples[algorithm][sample] > medians[algorithm]
                ? samples[algorithm][sample] - medians[algorithm]
                : medians[algorithm] - samples[algorithm][sample];
        deviations[algorithm] = median(scratch, BENCH_SAMPLES);
    }

    puts("case,algorithm,median_ns,mad_ns");
    printf("p2_dim4_rows4_degree1_n64_kappa0,p2-ring-forest,%llu,%llu\n",
           (unsigned long long)medians[0], (unsigned long long)deviations[0]);
    printf("p2_dim4_rows4_degree1_n64_kappa0,integer-block-forest,%llu,%llu\n",
           (unsigned long long)medians[1], (unsigned long long)deviations[1]);
    printf("p2_dim4_rows4_degree1_n64_kappa0,naive-ring-adapter,%llu,%llu\n",
           (unsigned long long)medians[2], (unsigned long long)deviations[2]);

    mpz_clear(initial_z);
    mpz_clear(ring_state.z);
    mpz_clear(block_state.z);
    mpz_clear(naive_state.z);
    clear_values(ring_M, RING_MATRIX_COUNT);
    clear_values(block_M, BLOCK_MATRIX_COUNT);
    clear_values(ring_V, RING_VECTOR_CELLS);
    clear_values(block_V, BLOCK_VECTOR_CELLS);
    clear_values(ring_state.working_v, RING_VECTOR_CELLS);
    clear_values(block_state.working_v, BLOCK_VECTOR_CELLS);
    clear_values(moduli, BENCH_ENDPOINTS);
    clear_values(ring_A, BENCH_ENDPOINTS * RING_VECTOR_CELLS);
    clear_values(block_A, BENCH_ENDPOINTS * BLOCK_VECTOR_CELLS);
    clear_values(naive_A, BENCH_ENDPOINTS * RING_VECTOR_CELLS);
    clear_values(naive_V, RING_VECTOR_CELLS);
    clear_values(evaluated, RING_VECTOR_CELLS);
    clear_values(next_v, RING_VECTOR_CELLS);
    return 0;
}
