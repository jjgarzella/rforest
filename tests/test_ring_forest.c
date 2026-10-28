#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gmp.h>

#include "rforest.h"
#include "hwmpz.h"

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

static size_t coeff_count(int np, int nq)
{
    return (size_t)np * (size_t)nq;
}

static void set_pattern(mpz_t *values, size_t count, int seed)
{
    for (size_t i = 0; i < count; i++) {
        long value = (long)((i * 5 + (size_t)seed) % 13) - 6;
        if (i % 7 == 0)
            value = 0;
        mpz_set_si(values[i], value);
    }
}

static void ring_step_mod(mpz_t *next, mpz_t *current,
                          mpz_t *evaluated, int np, int nq,
                          int rows, int dim, mpz_t modulus)
{
    const size_t coefficients = coeff_count(np, nq);
    const size_t rcells = (size_t)rows * (size_t)dim;
    const size_t dcells = (size_t)dim * (size_t)dim;
    mpz_t product;
    mpz_init(product);
    for (size_t i = 0; i < coefficients * rcells; i++)
        mpz_set_ui(next[i], 0);
    for (int ap = 0; ap < np; ap++) {
        for (int aq = 0; aq < nq; aq++) {
            size_t ai = (size_t)ap * nq + aq;
            for (int bp = 0; bp < np; bp++) {
                for (int bq = 0; bq < nq; bq++) {
                    int cp = ap + bp;
                    int cq = aq + bq;
                    size_t bi, ci;
                    if (cp >= np || cq >= nq)
                        continue;
                    bi = (size_t)bp * nq + bq;
                    ci = (size_t)cp * nq + cq;
                    for (int row = 0; row < rows; row++) {
                        for (int col = 0; col < dim; col++) {
                            size_t out = ci * rcells + (size_t)row * dim + col;
                            for (int inner = 0; inner < dim; inner++) {
                                size_t a = ai * rcells + (size_t)row * dim + inner;
                                size_t b = bi * dcells + (size_t)inner * dim + col;
                                mpz_mul(product, current[a], evaluated[b]);
                                mpz_add(next[out], next[out], product);
                            }
                            mpz_mod(next[out], next[out], modulus);
                        }
                    }
                }
            }
        }
    }
    mpz_clear(product);
}

static void reference_endpoint(mpz_t *result, mpz_t *initial_v,
                               mpz_t *M, int rows, int dim, int deg,
                               int np, int nq, long kbase, long endpoint,
                               mpz_t modulus)
{
    const size_t coefficients = coeff_count(np, nq);
    const size_t rcells = (size_t)rows * (size_t)dim;
    const size_t dcells = (size_t)dim * (size_t)dim;
    mpz_t *current = new_values(coefficients * rcells);
    mpz_t *next = new_values(coefficients * rcells);
    mpz_t *evaluated = new_values(coefficients * dcells);
    mpz_t value;

    for (size_t i = 0; i < coefficients * rcells; i++)
        mpz_mod(current[i], initial_v[i], modulus);
    mpz_init(value);
    for (long x = kbase; x < endpoint; x++) {
        for (size_t coeff = 0; coeff < coefficients; coeff++) {
            for (size_t cell = 0; cell < dcells; cell++) {
                size_t high = (cell * coefficients + coeff) * (size_t)(deg + 1) +
                              (size_t)deg;
                mpz_set(value, M[high]);
                for (int power = deg - 1; power >= 0; power--) {
                    size_t index = (cell * coefficients + coeff) *
                                   (size_t)(deg + 1) + (size_t)power;
                    mpz_mul_si(value, value, x);
                    mpz_add(value, value, M[index]);
                }
                mpz_mod(evaluated[coeff * dcells + cell], value, modulus);
            }
        }
        ring_step_mod(next, current, evaluated, np, nq, rows, dim, modulus);
        for (size_t i = 0; i < coefficients * rcells; i++)
            mpz_set(current[i], next[i]);
    }
    for (size_t i = 0; i < coefficients * rcells; i++)
        mpz_set(result[i], current[i]);
    mpz_clear(value);
    clear_values(current, coefficients * rcells);
    clear_values(next, coefficients * rcells);
    clear_values(evaluated, coefficients * dcells);
}

static int equal_values(mpz_t *left, mpz_t *right, size_t count)
{
    for (size_t i = 0; i < count; i++)
        if (mpz_cmp(left[i], right[i]) != 0)
            return 0;
    return 1;
}

static void check_p2_block_embedding(mpz_t *M, mpz_t *initial_v,
                                    mpz_t *ring_outputs,
                                    mpz_t *ring_final_v,
                                    mpz_t *moduli, long *endpoints,
                                    long kbase, int rows, int dim, int deg,
                                    int kappa, mpz_t ring_final_z)
{
    const int block_dim = 2 * dim;
    const size_t block_matrix_cells = (size_t)block_dim * (size_t)block_dim;
    const size_t ring_vector_cells = (size_t)rows * (size_t)dim;
    const size_t block_vector_cells = (size_t)rows * (size_t)block_dim;
    const size_t endpoint_count = 3;
    mpz_t *block_M = new_values((size_t)(deg + 1) * block_matrix_cells);
    mpz_t *block_V = new_values(block_vector_cells);
    mpz_t *block_A = new_values(endpoint_count * block_vector_cells);
    mpz_t block_z;

    /* Standard P^2 regular representation [[M0,M1],[0,M0]]. */
    for (int x_degree = 0; x_degree <= deg; x_degree++) {
        for (int row = 0; row < dim; row++) {
            for (int col = 0; col < dim; col++) {
                size_t entry = (size_t)row * (size_t)dim + (size_t)col;
                size_t source0 = (entry * 2) * (size_t)(deg + 1) +
                                 (size_t)x_degree;
                size_t source1 = (entry * 2 + 1) * (size_t)(deg + 1) +
                                 (size_t)x_degree;
                size_t upper = ((size_t)row * (size_t)block_dim +
                                (size_t)col) * (size_t)(deg + 1) +
                               (size_t)x_degree;
                size_t cross = ((size_t)row * (size_t)block_dim +
                                (size_t)(dim + col)) * (size_t)(deg + 1) +
                               (size_t)x_degree;
                size_t lower = ((size_t)(dim + row) * (size_t)block_dim +
                                (size_t)(dim + col)) * (size_t)(deg + 1) +
                               (size_t)x_degree;
                mpz_set(block_M[upper], M[source0]);
                mpz_set(block_M[cross], M[source1]);
                mpz_set(block_M[lower], M[source0]);
            }
        }
    }
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < dim; col++) {
            mpz_set(block_V[(size_t)row * block_dim + (size_t)col],
                    initial_v[(size_t)row * dim + (size_t)col]);
            mpz_set(block_V[(size_t)row * block_dim + (size_t)(dim + col)],
                    initial_v[ring_vector_cells + (size_t)row * dim + (size_t)col]);
        }
    }
    mpz_init_set_ui(block_z, 13);
    for (size_t i = 0; i < endpoint_count; i++)
        mpz_mul(block_z, block_z, moduli[i]);
    rforest(block_A, block_V, rows, block_M, deg, block_dim, moduli,
            kbase, endpoints, (long)endpoint_count, block_z, kappa);

    for (size_t endpoint = 0; endpoint < endpoint_count; endpoint++) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < dim; col++) {
                size_t ring_offset = endpoint * 2 * ring_vector_cells +
                                     (size_t)row * (size_t)dim + (size_t)col;
                size_t block_offset = endpoint * block_vector_cells +
                                      (size_t)row * (size_t)block_dim + (size_t)col;
                size_t block_cross = block_offset + (size_t)dim;
                if (mpz_cmp(ring_outputs[ring_offset], block_A[block_offset]) != 0 ||
                    mpz_cmp(ring_outputs[ring_offset + ring_vector_cells],
                            block_A[block_cross]) != 0) {
                    fputs("P^2 forest differs from integer block embedding\n", stderr);
                    abort();
                }
            }
        }
    }
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < dim; col++) {
            if (mpz_cmp(ring_final_v[(size_t)row * dim + (size_t)col],
                        block_V[(size_t)row * block_dim + (size_t)col]) != 0 ||
                mpz_cmp(ring_final_v[ring_vector_cells + (size_t)row * dim +
                                     (size_t)col],
                        block_V[(size_t)row * block_dim + (size_t)(dim + col)]) != 0) {
                fputs("P^2 final V differs from integer block embedding\n", stderr);
                abort();
            }
        }
    }
    if (mpz_cmp(block_z, ring_final_z) != 0) {
        fputs("P^2 final z differs from integer block embedding\n", stderr);
        abort();
    }
    mpz_clear(block_z);
    clear_values(block_M, (size_t)(deg + 1) * block_matrix_cells);
    clear_values(block_V, block_vector_cells);
    clear_values(block_A, endpoint_count * block_vector_cells);
}

static void one_family(int family, int np, int nq)
{
    const int rows = 1; /* rectangular initial V, rows < dim */
    const int dim = 2;
    const int deg = 1; /* transition polynomial x is separate from P/Q */
    const long kbase = 1;
    const long endpoint_values[] = {1, 3, 4}; /* exclusive endpoints */
    const size_t count = coeff_count(np, nq);
    const size_t rcells = (size_t)rows * (size_t)dim;
    const size_t dcells = (size_t)dim * (size_t)dim;
    const size_t vcount = count * rcells;
    const size_t mcount = (size_t)(deg + 1) * count * dcells;
    const size_t outcount = 3 * vcount;
    mpz_t *initial_v = new_values(vcount);
    mpz_t *working_v = new_values(vcount);
    mpz_t *M = new_values(mcount);
    mpz_t *expected = new_values(outcount);
    mpz_t *outputs = new_values(outcount);
    mpz_t *expected_v = new_values(vcount);
    mpz_t *matrix_copy = new_values(mcount);
    mpz_t *moduli = new_values(3);
    mpz_t *moduli_copy = new_values(3);
    long endpoints[3] = {1, 3, 4};
    long endpoints_copy[3] = {1, 3, 4};
    mpz_t z, expected_z, final_modulus;
    int kappas[] = {0, 1, 3};

    set_pattern(initial_v, vcount, 2);
    set_pattern(M, mcount, 5);
    mpz_set_ui(moduli[0], 5);
    mpz_set_ui(moduli[1], 7);
    mpz_set_ui(moduli[2], 11);
    for (size_t i = 0; i < mcount; i++)
        mpz_set(matrix_copy[i], M[i]);
    for (size_t i = 0; i < 3; i++)
        mpz_set(moduli_copy[i], moduli[i]);

    mpz_init_set_ui(z, 13);
    for (size_t i = 0; i < 3; i++)
        mpz_mul(z, z, moduli[i]);
    mpz_init_set(expected_z, z);
    for (size_t i = 0; i < 3; i++)
        mpz_divexact(expected_z, expected_z, moduli[i]);
    mpz_init_set(final_modulus, expected_z);

    for (size_t i = 0; i < 3; i++)
        reference_endpoint(expected + i * vcount, initial_v, M, rows, dim, deg,
                           np, nq, kbase, endpoint_values[i], moduli[i]);
    reference_endpoint(expected_v, initial_v, M, rows, dim, deg, np, nq,
                       kbase, endpoint_values[2], final_modulus);

    for (size_t repeat = 0; repeat < sizeof(kappas) / sizeof(kappas[0]); repeat++) {
        for (size_t i = 0; i < vcount; i++)
            mpz_set(working_v[i], initial_v[i]);
        for (size_t i = 0; i < outcount; i++)
            mpz_set_ui(outputs[i], 0);
        mpz_set(z, expected_z);
        /* Restore z to the product consumed by this forest call. */
        mpz_mul(z, z, moduli[0]);
        mpz_mul(z, z, moduli[1]);
        mpz_mul(z, z, moduli[2]);
        if (family == 0)
            rforest_p2(outputs, working_v, rows, M, deg, dim, moduli,
                       kbase, endpoints, 3, z, kappas[repeat]);
        else if (family == 1)
            rforest_pn(outputs, working_v, rows, M, deg, dim, np, moduli,
                       kbase, endpoints, 3, z, kappas[repeat]);
        else
            rforest_pnq(outputs, working_v, rows, M, deg, dim, np, moduli,
                        kbase, endpoints, 3, z, kappas[repeat]);
        if (!equal_values(outputs, expected, outcount) ||
            !equal_values(working_v, expected_v, vcount) ||
            mpz_cmp(z, expected_z) != 0 ||
            !equal_values(M, matrix_copy, mcount) ||
            !equal_values(moduli, moduli_copy, 3) ||
            memcmp(endpoints, endpoints_copy, sizeof(endpoints)) != 0) {
            fprintf(stderr, "ring forest family %d failed for kappa=%d\n",
                    family, kappas[repeat]);
            abort();
        }
        if (family == 0)
            check_p2_block_embedding(M, initial_v, outputs, working_v, moduli,
                                     endpoints, kbase, rows, dim, deg,
                                     kappas[repeat], z);
    }

    mpz_clear(z);
    mpz_clear(expected_z);
    mpz_clear(final_modulus);
    clear_values(initial_v, vcount);
    clear_values(working_v, vcount);
    clear_values(M, mcount);
    clear_values(expected, outcount);
    clear_values(outputs, outcount);
    clear_values(expected_v, vcount);
    clear_values(matrix_copy, mcount);
    clear_values(moduli, 3);
    clear_values(moduli_copy, 3);
}

#ifdef RFOREST_TEST_SOFT_TRANSFER_DISPATCH
int rforest_test_soft_transfer_active;

static unsigned long soft_dispatch_counts[5];
static int soft_expected_first_dispatch;
static unsigned long soft_dispatch_events;
static int soft_capture_skip;
static int soft_skip_negative_noncanonical_seen;
static int soft_require_next_boundary;
static int soft_boundary_seen;
static mpz_t *soft_saved_skip_values;
static size_t soft_saved_skip_count;

static void reset_soft_dispatch(int expected_first)
{
    memset(soft_dispatch_counts, 0, sizeof(soft_dispatch_counts));
    soft_expected_first_dispatch = expected_first;
    soft_dispatch_events = 0;
    soft_capture_skip = 0;
    soft_skip_negative_noncanonical_seen = 0;
    soft_require_next_boundary = 0;
    soft_boundary_seen = 0;
    if (soft_saved_skip_values) {
        clear_values(soft_saved_skip_values, soft_saved_skip_count);
        soft_saved_skip_values = NULL;
        soft_saved_skip_count = 0;
    }
}

void rforest_test_soft_transfer_dispatch(int path, mpz_t *output,
                                         mpz_t *input, long n, mpz_t modulus)
{
    mpz_t expected;
    assert(path >= RFOREST_TEST_SOFT_MOD_SKIP &&
           path <= RFOREST_TEST_SOFT_MOD_FFT_DISABLED);
    if (soft_dispatch_events == 0 && soft_expected_first_dispatch)
        assert(path == soft_expected_first_dispatch);
    soft_dispatch_events++;
    soft_dispatch_counts[path]++;
    mpz_init(expected);
    for (long i = 0; i < n; i++) {
        if (path == RFOREST_TEST_SOFT_MOD_SKIP) {
            assert(mpz_cmp(output[i], input[i]) == 0);
            if (mpz_sgn(input[i]) < 0 &&
                mpz_cmpabs(input[i], modulus) >= 0)
                soft_skip_negative_noncanonical_seen = 1;
            if (soft_capture_skip && !soft_saved_skip_values) {
                soft_saved_skip_values = new_values((size_t)n);
                soft_saved_skip_count = (size_t)n;
                for (long j = 0; j < n; j++)
                    mpz_set(soft_saved_skip_values[j], output[j]);
            }
        } else {
            mpz_fdiv_r(expected, input[i], modulus);
            assert(mpz_cmp(output[i], expected) == 0);
            assert(mpz_sgn(output[i]) >= 0);
            assert(mpz_cmp(output[i], modulus) < 0);
        }
    }
    mpz_clear(expected);
}

void rforest_test_soft_subtree_boundary(long start, mpz_t *incoming,
                                        mpz_t *reduced, long n, mpz_t modulus)
{
    mpz_t expected;
    int saw_negative_noncanonical = 0;
    if (!soft_require_next_boundary)
        return;
    assert(start == 2);
    assert(soft_saved_skip_values != NULL);
    assert(soft_saved_skip_count == (size_t)n);
    assert(equal_values(incoming, soft_saved_skip_values, (size_t)n));
    mpz_init(expected);
    for (long i = 0; i < n; i++) {
        mpz_fdiv_r(expected, incoming[i], modulus);
        assert(mpz_cmp(reduced[i], expected) == 0);
        assert(mpz_sgn(reduced[i]) >= 0);
        assert(mpz_cmp(reduced[i], modulus) < 0);
        if (mpz_sgn(incoming[i]) < 0 &&
            mpz_cmp(incoming[i], reduced[i]) != 0)
            saw_negative_noncanonical = 1;
    }
    assert(saw_negative_noncanonical);
    mpz_clear(expected);
    soft_boundary_seen = 1;
    soft_require_next_boundary = 0;
}

static void test_soft_skip_is_hard_reduced_at_next_subtree(void)
{
    const int rows = 1, dim = 1, deg = 0;
    const long n = 4, kbase = 1;
    const long endpoint_values[] = {1, 1, 3, 4};
    const size_t vector_cells = 2;
    mpz_t *M = new_values(2);
    mpz_t *M_copy = new_values(2);
    mpz_t *initial_v = new_values(vector_cells);
    mpz_t *working_v = new_values(vector_cells);
    mpz_t *expected_outputs = new_values((size_t)n * vector_cells);
    mpz_t *outputs = new_values((size_t)n * vector_cells);
    mpz_t *expected_final_v = new_values(vector_cells);
    mpz_t *moduli = new_values((size_t)n);
    mpz_t *moduli_copy = new_values((size_t)n);
    long endpoints[4] = {1, 1, 3, 4};
    long endpoints_copy[4] = {1, 1, 3, 4};
    mpz_t z, expected_z, final_modulus;

    mpz_set_ui(M[0], 1);
    mpz_set_ui(M[1], 0);
    mpz_set(M_copy[0], M[0]);
    mpz_set(M_copy[1], M[1]);
    mpz_set_si(initial_v[0], -1000);
    mpz_set_si(initial_v[1], -777);
    for (long i = 0; i < n; i++) {
        mpz_set_ui(moduli[i], 2);
        mpz_set(moduli_copy[i], moduli[i]);
    }
    for (long i = 0; i < n; i++)
        reference_endpoint(expected_outputs + (size_t)i * vector_cells,
                           initial_v, M, rows, dim, deg, 2, 1, kbase,
                           endpoint_values[i], moduli[i]);
    mpz_init_set_ui(z, 1);
    for (long i = 0; i < n; i++)
        mpz_mul(z, z, moduli[i]);
    mpz_init_set_ui(expected_z, 1);
    mpz_init_set_ui(final_modulus, 1);
    reference_endpoint(expected_final_v, initial_v, M, rows, dim, deg,
                       2, 1, kbase, endpoint_values[n - 1], final_modulus);
    for (size_t i = 0; i < vector_cells; i++)
        mpz_set(working_v[i], initial_v[i]);

    reset_soft_dispatch(RFOREST_TEST_SOFT_MOD_SKIP);
    soft_capture_skip = 1;
    soft_require_next_boundary = 1;
    rforest_p2(outputs, working_v, rows, M, deg, dim, moduli, kbase,
               endpoints, n, z, 2);

    assert(soft_boundary_seen);
    assert(soft_skip_negative_noncanonical_seen);
    assert(soft_dispatch_counts[RFOREST_TEST_SOFT_MOD_SKIP] > 0);
    assert(soft_dispatch_counts[RFOREST_TEST_SOFT_MOD_GMP] == 0);
    assert(soft_dispatch_counts[RFOREST_TEST_SOFT_MOD_FFT] == 0);
    assert(equal_values(outputs, expected_outputs, (size_t)n * vector_cells));
    assert(equal_values(working_v, expected_final_v, vector_cells));
    assert(mpz_cmp(z, expected_z) == 0);
    assert(equal_values(M, M_copy, 2));
    assert(equal_values(moduli, moduli_copy, (size_t)n));
    assert(memcmp(endpoints, endpoints_copy, sizeof(endpoints)) == 0);
    printf("{\"case\":\"skip-carry\",\"path\":\"skip\",\"transfer_events\":%lu,\"negative_noncanonical\":true,\"next_subtree_start\":2,\"hard_reduced\":true,\"final_z\":\"1\"}\n",
           soft_dispatch_events);

    mpz_clear(z);
    mpz_clear(expected_z);
    mpz_clear(final_modulus);
    clear_values(M, 2);
    clear_values(M_copy, 2);
    clear_values(initial_v, vector_cells);
    clear_values(working_v, vector_cells);
    clear_values(expected_outputs, (size_t)n * vector_cells);
    clear_values(outputs, (size_t)n * vector_cells);
    clear_values(expected_final_v, vector_cells);
    clear_values(moduli, (size_t)n);
    clear_values(moduli_copy, (size_t)n);
    reset_soft_dispatch(0);
}

static void set_power_of_two_limbs(mpz_t value, size_t limbs)
{
    assert(limbs > 0);
    assert(limbs <= (size_t)ULONG_MAX / GMP_NUMB_BITS);
    mpz_set_ui(value, 1);
    mpz_mul_2exp(value, value,
                 (mp_bitcnt_t)(limbs * GMP_NUMB_BITS - 1));
}

static void fill_transfer_values(mpz_t *values, size_t count,
                                 size_t limbs)
{
    assert(limbs > 0);
    assert(limbs <= (size_t)ULONG_MAX / GMP_NUMB_BITS);
    for (size_t i = 0; i < count; i++) {
        mpz_set_ui(values[i], (unsigned long)(i + 1));
        mpz_setbit(values[i],
                   (mp_bitcnt_t)(limbs * GMP_NUMB_BITS - 1));
        if (i & 1)
            mpz_neg(values[i], values[i]);
    }
}

static void run_large_transfer_dispatch(int family, int parameter,
                                       int primary_path, int disable_fft)
{
    const int rows = 81, dim = 1, deg = 0;
    const long n = 2, kbase = 0;
    const long endpoint_values[] = {1, 2};
    const size_t coefficients = family == 0 ? 2 :
                                family == 1 ? (size_t)parameter :
                                (size_t)parameter * (size_t)parameter;
    const size_t vector_cells = coefficients * (size_t)rows;
    const size_t matrix_cells = coefficients;
    mpz_t *M = new_values(matrix_cells);
    mpz_t *M_copy = new_values(matrix_cells);
    mpz_t *initial_v = new_values(vector_cells);
    mpz_t *working_v = new_values(vector_cells);
    mpz_t *expected_outputs = new_values((size_t)n * vector_cells);
    mpz_t *outputs = new_values((size_t)n * vector_cells);
    mpz_t *expected_final_v = new_values(vector_cells);
    mpz_t *moduli = new_values((size_t)n);
    mpz_t *moduli_copy = new_values((size_t)n);
    long endpoints[2] = {1, 2};
    long endpoints_copy[2] = {1, 2};
    mpz_t residual, z, expected_z;
    size_t input_limbs;
    int previous_hw_disable_fft = hw_disable_fft;
    long crossover = 0;
    size_t residual_limbs;
    size_t first_transfer_modulus_limbs;
    const char *family_name = family == 0 ? "p2" :
                              family == 1 ? "pn" : "pnq";
    const char *path_name = primary_path == RFOREST_TEST_SOFT_MOD_GMP ? "gmp" :
                            primary_path == RFOREST_TEST_SOFT_MOD_FFT ? "fft" :
                            "fft-disabled-fallback";

    for (size_t i = 0; i < matrix_cells; i++)
        mpz_set_ui(M[i], 0);
    mpz_set_ui(M[0], 1); /* identity in the constant ring coefficient */
    for (size_t i = 0; i < matrix_cells; i++)
        mpz_set(M_copy[i], M[i]);
    mpz_set_ui(moduli[0], 3);
    mpz_set_ui(moduli[1], 5);
    mpz_set(moduli_copy[0], moduli[0]);
    mpz_set(moduli_copy[1], moduli[1]);
    mpz_init(residual);

    if (primary_path == RFOREST_TEST_SOFT_MOD_GMP) {
        mpz_set_ui(residual, 1);
        mpz_setbit(residual, 40);
        residual_limbs = (size_t)mpz_size(residual);
        input_limbs = 2;
    } else {
        crossover = mpz_mod_fft_crossover((long)vector_cells);
        assert(crossover > 0);
        size_t modulus_limbs = (size_t)crossover + 64;
        residual_limbs = modulus_limbs;
        input_limbs = 2 * modulus_limbs;
        set_power_of_two_limbs(residual, modulus_limbs);
        assert(input_limbs > (size_t)crossover);
    }
    mpz_t first_transfer_modulus;
    mpz_init_set(first_transfer_modulus, residual);
    mpz_mul(first_transfer_modulus, first_transfer_modulus, moduli[1]);
    first_transfer_modulus_limbs = (size_t)mpz_size(first_transfer_modulus);
    fill_transfer_values(initial_v, vector_cells, input_limbs);
    for (size_t i = 0; i < vector_cells; i++)
        mpz_set(working_v[i], initial_v[i]);
    for (size_t i = 0; i < 2; i++)
        reference_endpoint(expected_outputs + i * vector_cells,
                           initial_v, M, rows, dim, deg,
                           family == 2 ? parameter :
                           family == 1 ? parameter : 2,
                           family == 2 ? parameter : 1,
                           kbase, endpoint_values[i], moduli[i]);
    reference_endpoint(expected_final_v, initial_v, M, rows, dim, deg,
                       family == 2 ? parameter :
                       family == 1 ? parameter : 2,
                       family == 2 ? parameter : 1,
                       kbase, endpoint_values[1], residual);

    mpz_init_set(z, residual);
    mpz_mul(z, z, moduli[0]);
    mpz_mul(z, z, moduli[1]);
    mpz_init_set(expected_z, residual);
    hw_disable_fft = disable_fft;
    reset_soft_dispatch(primary_path);
    if (family == 0)
        rforest_p2(outputs, working_v, rows, M, deg, dim, moduli,
                   kbase, endpoints, n, z, 1);
    else if (family == 1)
        rforest_pn(outputs, working_v, rows, M, deg, dim, parameter,
                   moduli, kbase, endpoints, n, z, 1);
    else
        rforest_pnq(outputs, working_v, rows, M, deg, dim, parameter,
                    moduli, kbase, endpoints, n, z, 1);

    assert(soft_dispatch_events == 2);
    assert(soft_dispatch_counts[primary_path] == 1);
    assert(soft_dispatch_counts[RFOREST_TEST_SOFT_MOD_SKIP] == 1);
    assert(equal_values(outputs, expected_outputs, (size_t)n * vector_cells));
    assert(equal_values(working_v, expected_final_v, vector_cells));
    assert(mpz_cmp(z, expected_z) == 0);
    assert(equal_values(M, M_copy, matrix_cells));
    assert(equal_values(moduli, moduli_copy, (size_t)n));
    assert(memcmp(endpoints, endpoints_copy, sizeof(endpoints)) == 0);
    printf("{\"case\":\"large-transfer\",\"family\":\"%s\",\"primary_path\":\"%s\",\"fft_disabled\":%s,\"vector_cells\":%zu,\"max_input_limbs\":%zu,\"first_transfer_modulus_limbs\":%zu,\"fft_crossover_limbs\":%ld,\"primary_events\":%lu,\"skip_events\":%lu}\n",
           family_name, path_name, disable_fft ? "true" : "false",
           vector_cells, input_limbs, first_transfer_modulus_limbs,
           crossover, soft_dispatch_counts[primary_path],
           soft_dispatch_counts[RFOREST_TEST_SOFT_MOD_SKIP]);
    hw_disable_fft = previous_hw_disable_fft;

    mpz_clear(first_transfer_modulus);
    mpz_clear(residual);
    mpz_clear(z);
    mpz_clear(expected_z);
    clear_values(M, matrix_cells);
    clear_values(M_copy, matrix_cells);
    clear_values(initial_v, vector_cells);
    clear_values(working_v, vector_cells);
    clear_values(expected_outputs, (size_t)n * vector_cells);
    clear_values(outputs, (size_t)n * vector_cells);
    clear_values(expected_final_v, vector_cells);
    clear_values(moduli, (size_t)n);
    clear_values(moduli_copy, (size_t)n);
    reset_soft_dispatch(0);
}

static void test_scalar_case(void)
{
    const int rows = 1, dim = 1, deg = 0;
    long endpoints[1] = {1};
    mpz_t *M = new_values(2);
    mpz_t *V = new_values(2);
    mpz_t *A = new_values(2);
    mpz_t *m = new_values(1);
    mpz_t z;
    mpz_set_si(M[0], 2);
    mpz_set_si(M[1], 3);
    mpz_set_si(V[0], 5);
    mpz_set_si(V[1], -7);
    mpz_set_ui(m[0], 11);
    mpz_init_set_ui(z, 11 * 17);
    rforest_p2(A, V, rows, M, deg, dim, m, 0, endpoints, 1, z, 0);
    assert(mpz_cmp_ui(A[0], 10) == 0);
    assert(mpz_cmp_ui(A[1], 1) == 0);
    assert(mpz_cmp_ui(V[0], 10) == 0);
    assert(mpz_cmp_ui(V[1], 1) == 0);
    assert(mpz_cmp_ui(z, 17) == 0);
    mpz_clear(z);
    clear_values(M, 2);
    clear_values(V, 2);
    clear_values(A, 2);
    clear_values(m, 1);
}

static void test_empty_input(void)
{
    mpz_t *A = new_values(2);
    mpz_t *V = new_values(2);
    mpz_t *M = new_values(2);
    mpz_t *m = new_values(1);
    long endpoint = 1;
    mpz_t z;
    mpz_set_si(V[0], -5);
    mpz_set_ui(V[1], 7);
    mpz_set_ui(M[0], 1);
    mpz_set_ui(M[1], 2);
    mpz_set_ui(m[0], 3);
    mpz_init_set_ui(z, 19);
    rforest_p2(A, V, 1, M, 0, 1, m, 0, &endpoint, 0, z, 0);
    assert(mpz_cmp_si(V[0], -5) == 0);
    assert(mpz_cmp_ui(V[1], 7) == 0);
    assert(mpz_sgn(A[0]) == 0 && mpz_sgn(A[1]) == 0);
    assert(mpz_cmp_ui(M[0], 1) == 0 && mpz_cmp_ui(M[1], 2) == 0);
    assert(mpz_cmp_ui(m[0], 3) == 0);
    assert(mpz_cmp_ui(z, 19) == 0);
    mpz_clear(z);
    clear_values(A, 2);
    clear_values(V, 2);
    clear_values(M, 2);
    clear_values(m, 1);
}
#endif

int main(void)
{
    int previous_hw_disable_fft = hw_disable_fft;
    one_family(0, 2, 1); /* P^2, with coefficientwise residues */
    one_family(1, 3, 1); /* independent direct P^n reference */
    one_family(2, 2, 2); /* bivariate P/Q box, not total degree */
    one_family(1, 1, 1); /* P^1 agrees with integer forest arithmetic */
    one_family(2, 1, 1); /* P^1,Q^1 agrees with integer arithmetic */
    hw_disable_fft = 1;
    one_family(0, 2, 1);
    one_family(1, 3, 1);
    one_family(2, 2, 2);
    one_family(1, 1, 1);
    one_family(2, 1, 1);
    hw_disable_fft = previous_hw_disable_fft;
#ifdef RFOREST_TEST_SOFT_TRANSFER_DISPATCH
    test_soft_skip_is_hard_reduced_at_next_subtree();
    test_scalar_case();
    test_empty_input();
    run_large_transfer_dispatch(0, 2, RFOREST_TEST_SOFT_MOD_GMP, 0);
    run_large_transfer_dispatch(0, 2, RFOREST_TEST_SOFT_MOD_FFT, 0);
    run_large_transfer_dispatch(0, 2,
                                RFOREST_TEST_SOFT_MOD_FFT_DISABLED, 1);
    run_large_transfer_dispatch(1, 3, RFOREST_TEST_SOFT_MOD_GMP, 0);
    run_large_transfer_dispatch(1, 3, RFOREST_TEST_SOFT_MOD_FFT, 0);
    run_large_transfer_dispatch(1, 3,
                                RFOREST_TEST_SOFT_MOD_FFT_DISABLED, 1);
    run_large_transfer_dispatch(2, 2, RFOREST_TEST_SOFT_MOD_GMP, 0);
    run_large_transfer_dispatch(2, 2, RFOREST_TEST_SOFT_MOD_FFT, 0);
    run_large_transfer_dispatch(2, 2,
                                RFOREST_TEST_SOFT_MOD_FFT_DISABLED, 1);
    reset_soft_dispatch(0);
#endif
    puts("PASS ring forest exact sequential references");
    return 0;
}
