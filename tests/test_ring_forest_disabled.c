#include <stdio.h>

#ifndef RFOREST_ENABLE_RING_FOREST_TESTS

int main(void)
{
    puts("DISABLED ring forest tests: activate with ring forest PR 5");
    return 0;
}

#else

#include <stdlib.h>
#include <string.h>

#include <gmp.h>

#include "rforest.h"

/* Proposed facades and coefficient layouts are documented in ring_api_proposal.md. */
void rforest_p2(mpz_t *A, mpz_t *V, int rows, mpz_t *M, int deg, int dim,
                mpz_t *m, long kbase, long *k, long n, mpz_t z, int kappa);
void rforest_pn(mpz_t *A, mpz_t *V, int rows, mpz_t *M, int deg, int dim,
                int nP, mpz_t *m, long kbase, long *k, long n, mpz_t z,
                int kappa);
void rforest_pnq(mpz_t *A, mpz_t *V, int rows, mpz_t *M, int deg, int dim,
                 int N, mpz_t *m, long kbase, long *k, long n, mpz_t z,
                 int kappa);

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

static void ring_step_mod(mpz_t *next, const mpz_t *current,
                          const mpz_t *evaluated, int np, int nq,
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

static void reference_endpoint(mpz_t *result, const mpz_t *initial_v,
                               const mpz_t *M, int rows, int dim, int deg,
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

static int equal_values(const mpz_t *left, const mpz_t *right, size_t count)
{
    for (size_t i = 0; i < count; i++)
        if (mpz_cmp(left[i], right[i]) != 0)
            return 0;
    return 1;
}

static void check_p2_block_embedding(const mpz_t *M, const mpz_t *initial_v,
                                    const mpz_t *ring_outputs,
                                    const mpz_t *ring_final_v,
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

int main(void)
{
    one_family(0, 2, 1); /* P^2, with coefficientwise residues */
    one_family(1, 3, 1); /* independent direct P^n reference */
    one_family(2, 2, 2); /* bivariate P/Q box, not total degree */
    puts("PASS ring forest exact sequential references");
    return 0;
}

#endif
