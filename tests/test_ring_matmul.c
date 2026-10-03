#include <stdio.h>

#if !defined(RFOREST_ENABLE_P2_MATMUL_TESTS) && \
    !defined(RFOREST_ENABLE_PN_MATMUL_TESTS) && \
    !defined(RFOREST_ENABLE_RING_MATMUL_TESTS)

int main(void)
{
    puts("DISABLED ring matrix tests: activate with ring matmul PRs 2-4");
    return 0;
}

#else

#include <stdlib.h>
#include <string.h>

#include <gmp.h>

#include "hwmpz.h"
#include "mpzfft.h"

/* Coefficient-major API details are documented in ring_api.md. */
#if defined(RFOREST_ENABLE_PN_MATMUL_TESTS) || \
    defined(RFOREST_ENABLE_RING_MATMUL_TESTS)
mpz_t *mpz_rmatrix_mult_pn(mpz_t *C, mpz_t *A, int r, mpz_t *B, int d,
                           int n, mpz_t w);
#endif
#ifdef RFOREST_ENABLE_RING_MATMUL_TESTS
mpz_t *mpz_rmatrix_mult_pnq(mpz_t *C, mpz_t *A, int r, mpz_t *B, int d,
                            int N, mpz_t w);
#endif

static unsigned observed_forward_transforms;
static unsigned observed_inverse_transforms;
static unsigned observed_fourier_matrix_products;

void __real_mpzfft_fft(mpzfft_t rop, mpz_t op, int threads);
void __wrap_mpzfft_fft(mpzfft_t rop, mpz_t op, int threads)
{
    observed_forward_transforms++;
    __real_mpzfft_fft(rop, op, threads);
}

void __real_mpzfft_ifft(mpz_t rop, mpzfft_t op, int threads);
void __wrap_mpzfft_ifft(mpz_t rop, mpzfft_t op, int threads)
{
    observed_inverse_transforms++;
    __real_mpzfft_ifft(rop, op, threads);
}

void __real_zz_mpnfft_poly_matrix_mul(zz_mpnfft_poly_t *rop,
                                     zz_mpnfft_poly_t *op1,
                                     zz_mpnfft_poly_t *op2,
                                     unsigned dim1, unsigned dim2,
                                     unsigned dim3, int threads);
void __wrap_zz_mpnfft_poly_matrix_mul(zz_mpnfft_poly_t *rop,
                                      zz_mpnfft_poly_t *op1,
                                      zz_mpnfft_poly_t *op2,
                                      unsigned dim1, unsigned dim2,
                                      unsigned dim3, int threads)
{
    observed_fourier_matrix_products++;
    __real_zz_mpnfft_poly_matrix_mul(rop, op1, op2, dim1, dim2, dim3,
                                    threads);
}

enum { FIXTURE_LIMIT = 100000 };

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

static size_t ring_coeff_count(int n, int nq)
{
    return (size_t)n * (size_t)nq;
}

static size_t cells(int rows, int dim)
{
    return (size_t)rows * (size_t)dim;
}

static void fill_matrix(mpz_t *values, size_t count, int mode, unsigned seed)
{
    for (size_t i = 0; i < count; i++) {
        long value = (long)((i * 7 + seed * 3) % 19) - 9;
        if (mode == 1 && i % 5 != 0)
            value = 0; /* sparse */
        if (mode == 2)
            value = 0; /* zero */
        mpz_set_si(values[i], value);
        if (mode == 3 && value)
            mpz_mul_2exp(values[i], values[i], 2048); /* large signed operands */
    }
}

static void reference_product(mpz_t *C, mpz_t *A, mpz_t *B,
                              int rows, int dim, int np, int nq, int bivariate)
{
    const size_t acells = cells(rows, dim);
    const size_t bcells = (size_t)dim * (size_t)dim;
    mpz_t product;
    mpz_init(product);
    for (size_t i = 0; i < (size_t)np * (size_t)nq * acells; i++)
        mpz_set_ui(C[i], 0);

    for (int ap = 0; ap < np; ap++) {
        for (int aq = 0; aq < nq; aq++) {
            size_t ai = (size_t)ap * (size_t)nq + (size_t)aq;
            for (int bp = 0; bp < np; bp++) {
                for (int bq = 0; bq < nq; bq++) {
                    int cp = ap + bp;
                    int cq = aq + bq;
                    size_t bi, ci;
                    if (cp >= np || cq >= nq)
                        continue; /* box truncation: discard either overflow */
                    bi = (size_t)bp * (size_t)nq + (size_t)bq;
                    ci = (size_t)cp * (size_t)nq + (size_t)cq;
                    for (int row = 0; row < rows; row++) {
                        for (int col = 0; col < dim; col++) {
                            size_t out = ci * acells + (size_t)row * dim + col;
                            for (int inner = 0; inner < dim; inner++) {
                                size_t left = ai * acells + (size_t)row * dim + inner;
                                size_t right = bi * bcells + (size_t)inner * dim + col;
                                mpz_mul(product, A[left], B[right]);
                                mpz_add(C[out], C[out], product);
                            }
                        }
                    }
                }
            }
        }
    }
    mpz_clear(product);
    (void)bivariate;
}

static int equal_values(mpz_t *left, mpz_t *right, size_t count)
{
    for (size_t i = 0; i < count; i++)
        if (mpz_cmp(left[i], right[i]) != 0)
            return 0;
    return 1;
}

static void run_case(const char *name, int np, int nq, int rows, int dim,
                     int mode, int bivariate)
{
    const size_t coeffs = ring_coeff_count(np, nq);
    const size_t acells = cells(rows, dim);
    const size_t bcells = (size_t)dim * (size_t)dim;
    const size_t left_count = coeffs * acells;
    const size_t right_count = coeffs * bcells;
    const size_t output_count = coeffs * acells;
    mpz_t *A = new_values(left_count);
    mpz_t *B = new_values(right_count);
    mpz_t *C = new_values(output_count);
    mpz_t *expected = new_values(output_count);
#if defined(RFOREST_ENABLE_P2_MATMUL_TESTS) || \
    defined(RFOREST_ENABLE_PN_MATMUL_TESTS) || \
    defined(RFOREST_ENABLE_RING_MATMUL_TESTS)
    mpz_t *saved_A = new_values(left_count);
    mpz_t *saved_B = new_values(right_count);
#endif
    mpz_t work;

    fill_matrix(A, left_count, mode, 2);
    fill_matrix(B, right_count, mode, 7);
#if defined(RFOREST_ENABLE_P2_MATMUL_TESTS) || \
    defined(RFOREST_ENABLE_PN_MATMUL_TESTS) || \
    defined(RFOREST_ENABLE_RING_MATMUL_TESTS)
    mpz_vec_set(saved_A, A, (long)left_count);
    mpz_vec_set(saved_B, B, (long)right_count);
#endif
    mpz_init(work);
    reference_product(expected, A, B, rows, dim, np, nq, bivariate);
#if defined(RFOREST_ENABLE_PN_MATMUL_TESTS) || \
    defined(RFOREST_ENABLE_RING_MATMUL_TESTS)
    if (bivariate) {
#ifdef RFOREST_ENABLE_RING_MATMUL_TESTS
        mpz_rmatrix_mult_pnq(C, A, rows, B, dim, np, work);
#else
        fputs("bivariate test is not enabled in this stage\n", stderr);
        abort();
#endif
    } else {
        mpz_rmatrix_mult_pn(C, A, rows, B, dim, np, work);
    }
#else
    if (bivariate || np != 2) {
        fputs("P^2 test requested a ring family not enabled in this stage\n",
              stderr);
        abort();
    }
    mpz_rmatrix_mult_p2(C, A, rows, B, dim, work);
#endif
    if (!equal_values(C, expected, output_count)) {
        fprintf(stderr, "%s: ring product differs from direct GMP reference\n", name);
        abort();
    }
#if defined(RFOREST_ENABLE_P2_MATMUL_TESTS) || \
    defined(RFOREST_ENABLE_PN_MATMUL_TESTS) || \
    defined(RFOREST_ENABLE_RING_MATMUL_TESTS)
    if (!equal_values(saved_A, A, left_count) ||
        !equal_values(saved_B, B, right_count)) {
        fprintf(stderr, "%s: ring product modified an input coefficient\n", name);
        abort();
    }
#endif
    mpz_clear(work);
    clear_values(A, left_count);
    clear_values(B, right_count);
    clear_values(C, output_count);
    clear_values(expected, output_count);
#if defined(RFOREST_ENABLE_P2_MATMUL_TESTS) || \
    defined(RFOREST_ENABLE_PN_MATMUL_TESTS) || \
    defined(RFOREST_ENABLE_RING_MATMUL_TESTS)
    clear_values(saved_A, left_count);
    clear_values(saved_B, right_count);
#endif
}

static void expect_token(FILE *input, const char *expected)
{
    char token[256];
    if (fscanf(input, "%255s", token) != 1 || strcmp(token, expected) != 0) {
        fprintf(stderr, "AWS ring fixture: expected %s\n", expected);
        abort();
    }
}

static void read_values(FILE *input, mpz_t *values, size_t count)
{
    char token[4096];
    for (size_t i = 0; i < count; i++) {
        if (fscanf(input, "%4095s", token) != 1 || mpz_set_str(values[i], token, 10))
            abort();
    }
}

static void test_aws_block_equivalence(mpz_t *A, mpz_t *B,
                                       mpz_t *expected, int dim)
{
    size_t cells = (size_t)dim * (size_t)dim;
    size_t block_cells = 4 * cells;
    mpz_t *left = new_values(block_cells);
    mpz_t *right = new_values(block_cells);
    mpz_t *product = new_values(block_cells);
    mpz_t work;
    int block_dim = 2 * dim;

    for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
            size_t source = (size_t)row * (size_t)dim + (size_t)col;
            mpz_set(left[(size_t)row * (size_t)block_dim + (size_t)col],
                    A[source]);
            mpz_set(left[(size_t)row * (size_t)block_dim + (size_t)dim +
                          (size_t)col], A[cells + source]);
            mpz_set(left[((size_t)dim + (size_t)row) * (size_t)block_dim +
                         (size_t)dim + (size_t)col], A[source]);
            mpz_set(right[(size_t)row * (size_t)block_dim + (size_t)col],
                    B[source]);
            mpz_set(right[(size_t)row * (size_t)block_dim + (size_t)dim +
                          (size_t)col], B[cells + source]);
            mpz_set(right[((size_t)dim + (size_t)row) * (size_t)block_dim +
                          (size_t)dim + (size_t)col], B[source]);
        }
    }

    mpz_init(work);
    mpz_rmatrix_mult(product, left, block_dim, right, block_dim, work);
    for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
            size_t expected_index = (size_t)row * (size_t)dim + (size_t)col;
            size_t top_left = (size_t)row * (size_t)block_dim + (size_t)col;
            size_t top_right = top_left + (size_t)dim;
            size_t bottom_left = ((size_t)dim + (size_t)row) *
                                 (size_t)block_dim + (size_t)col;
            size_t bottom_right = bottom_left + (size_t)dim;
            if (mpz_cmp(product[top_left], expected[expected_index]) != 0 ||
                mpz_cmp(product[top_right], expected[cells + expected_index]) != 0 ||
                mpz_sgn(product[bottom_left]) != 0 ||
                mpz_cmp(product[bottom_right], expected[expected_index]) != 0) {
                fputs("AWS P^2 product disagreed with integer block embedding\n",
                      stderr);
                abort();
            }
        }
    }
    mpz_clear(work);
    clear_values(left, block_cells);
    clear_values(right, block_cells);
    clear_values(product, block_cells);
}

static void test_aws_products(const char *path)
{
    FILE *input = fopen(path, "r");
    char token[256];
    size_t case_count;
    if (!input) {
        perror(path);
        abort();
    }
    expect_token(input, "AWS_RING_P2_PRODUCTS");
    if (fscanf(input, "%255s", token) != 1 || strcmp(token, "1") != 0)
        abort();
    expect_token(input, "AWS_COMMIT");
    if (fscanf(input, "%255s", token) != 1 ||
        strcmp(token, "a1fc50dd667d262b5d83d1d4ceb1499bdbead288") != 0)
        abort();
    expect_token(input, "ZETA_SUITE_COMMIT");
    if (fscanf(input, "%255s", token) != 1 ||
        strcmp(token, "621107b12200a3c234c2e3dd6a4ba9dc87be9bea") != 0)
        abort();
    expect_token(input, "AWS_SOURCE");
    if (fscanf(input, "%255s %255s", token, token) != 2)
        abort();
    expect_token(input, "GENERIC_UNIVARIATE_SOURCE");
    if (fscanf(input, "%255s", token) != 1)
        abort();
    expect_token(input, "P8_GENERIC_REFERENCE_CHECKS");
    if (fscanf(input, "%255s", token) != 1)
        abort();
    expect_token(input, "CASE_COUNT");
    if (fscanf(input, "%zu", &case_count) != 1 || case_count > FIXTURE_LIMIT)
        abort();

    for (size_t test = 0; test < case_count; test++) {
        int dim;
        size_t count;
        mpz_t *A, *B, *C, *expected;
        mpz_t work;
        expect_token(input, "CASE");
        if (fscanf(input, "%255s %d", token, &dim) != 2 || dim <= 0)
            abort();
        count = (size_t)dim * (size_t)dim;
        A = new_values(2 * count);
        B = new_values(2 * count);
        C = new_values(2 * count);
        expected = new_values(2 * count);
        for (int k = 0; k < 2; k++) {
            expect_token(input, k == 0 ? "A0" : "A1");
            read_values(input, A + (size_t)k * count, count);
        }
        for (int k = 0; k < 2; k++) {
            expect_token(input, k == 0 ? "B0" : "B1");
            read_values(input, B + (size_t)k * count, count);
        }
        for (int k = 0; k < 2; k++) {
            expect_token(input, k == 0 ? "C0" : "C1");
            read_values(input, expected + (size_t)k * count, count);
        }
        expect_token(input, "END_CASE");
        mpz_init(work);
        mpz_rmatrix_mult_p2(C, A, dim, B, dim, work);
        if (!equal_values(C, expected, 2 * count)) {
            fprintf(stderr, "AWS P^2 exact product mismatch: %s\n", token);
            abort();
        }
        test_aws_block_equivalence(A, B, expected, dim);
        mpz_clear(work);
        clear_values(A, 2 * count);
        clear_values(B, 2 * count);
        clear_values(C, 2 * count);
        clear_values(expected, 2 * count);
    }
    fclose(input);
}

static void test_p2_order_and_cancellation(void)
{
    mpz_t *A = new_values(8);
    mpz_t *B = new_values(8);
    mpz_t *AB = new_values(8);
    mpz_t *BA = new_values(8);
    mpz_t work;

    /* A0 and B0 do not commute; the P coefficient also checks product order. */
    mpz_set_si(A[0], 1); mpz_set_si(A[1], 1);
    mpz_set_si(A[2], 0); mpz_set_si(A[3], 1);
    mpz_set_si(B[0], 1); mpz_set_si(B[1], 0);
    mpz_set_si(B[2], 1); mpz_set_si(B[3], 1);
    mpz_set_si(A[4], 0); mpz_set_si(A[5], 1);
    mpz_set_si(A[6], 1); mpz_set_si(A[7], 0);
    mpz_set_si(B[4], 1); mpz_set_si(B[5], 1);
    mpz_set_si(B[6], 0); mpz_set_si(B[7], 1);
    mpz_init(work);
    mpz_rmatrix_mult_p2(AB, A, 2, B, 2, work);
    mpz_rmatrix_mult_p2(BA, B, 2, A, 2, work);
    if (equal_values(AB, BA, 8)) {
        fputs("P^2 product lost matrix order\n", stderr);
        abort();
    }

    /* (I + I*P)(I - I*P) has an exactly cancelling P coefficient. */
    for (size_t i = 0; i < 8; i++) {
        mpz_set_ui(A[i], 0);
        mpz_set_ui(B[i], 0);
    }
    mpz_set_ui(A[0], 1); mpz_set_ui(A[3], 1);
    mpz_set_ui(A[4], 1); mpz_set_ui(A[7], 1);
    mpz_set_ui(B[0], 1); mpz_set_ui(B[3], 1);
    mpz_set_ui(B[4], 1); mpz_set_ui(B[7], 1);
    mpz_neg(B[4], B[4]); mpz_neg(B[7], B[7]);
    mpz_rmatrix_mult_p2(AB, A, 2, B, 2, work);
    for (size_t i = 4; i < 8; i++)
        if (mpz_sgn(AB[i]) != 0) {
            fputs("P^2 cancellation coefficient was nonzero\n", stderr);
            abort();
        }
    mpz_clear(work);
    clear_values(A, 8);
    clear_values(B, 8);
    clear_values(AB, 8);
    clear_values(BA, 8);
}

#ifdef RFOREST_ENABLE_P2_MATMUL_TESTS
static void test_shared_fourier_dispatch(void)
{
    const int rows = 8;
    const int dim = 8;
    const size_t acells = (size_t)rows * (size_t)dim;
    const size_t bcells = (size_t)dim * (size_t)dim;
    const size_t input_entries = 2 * acells + 2 * bcells;
    size_t threshold = ((size_t)rows + (size_t)dim) * (size_t)dim *
                       (size_t)mpz_mat_fft_crossover(dim);
    size_t limbs = threshold / input_entries + 1;
    mpz_t *A = new_values(2 * acells);
    mpz_t *B = new_values(2 * bcells);
    mpz_t *C = new_values(2 * acells);
    mpz_t *expected = new_values(2 * acells);
    mpz_t work;

    if (limbs == 0 || limbs > (SIZE_MAX / GMP_NUMB_BITS)) {
        fputs("invalid P^2 Fourier dispatch test size\n", stderr);
        abort();
    }
    for (size_t i = 0; i < 2 * acells; i++) {
        mpz_set_ui(A[i], (unsigned long)(i % 31) + 1);
        mpz_setbit(A[i], (mp_bitcnt_t)(limbs * GMP_NUMB_BITS - 1));
        if (i & 1)
            mpz_neg(A[i], A[i]);
    }
    for (size_t i = 0; i < 2 * bcells; i++) {
        mpz_set_ui(B[i], (unsigned long)(i % 29) + 1);
        mpz_setbit(B[i], (mp_bitcnt_t)(limbs * GMP_NUMB_BITS - 1));
        if (i & 1)
            mpz_neg(B[i], B[i]);
    }
    mpz_init(work);
    reference_product(expected, A, B, rows, dim, 2, 1, 0);
    observed_forward_transforms = 0;
    observed_inverse_transforms = 0;
    observed_fourier_matrix_products = 0;
    hw_disable_fft = 0;
    mpz_rmatrix_mult_p2(C, A, rows, B, dim, work);

    if (!equal_values(C, expected, 2 * acells)) {
        fputs("P^2 Fourier result disagreed with direct GMP reference\n", stderr);
        abort();
    }
    if (observed_forward_transforms != input_entries ||
        observed_inverse_transforms != 2 * acells ||
        observed_fourier_matrix_products != 3) {
        fprintf(stderr,
                "P^2 Fourier reuse counts: forward=%u inverse=%u matrix=%u\n",
                observed_forward_transforms, observed_inverse_transforms,
                observed_fourier_matrix_products);
        abort();
    }

    observed_forward_transforms = 0;
    observed_inverse_transforms = 0;
    observed_fourier_matrix_products = 0;
    hw_disable_fft = 1;
    mpz_rmatrix_mult_p2(C, A, rows, B, dim, work);
    if (!equal_values(C, expected, 2 * acells) ||
        observed_forward_transforms != 0 ||
        observed_inverse_transforms != 0 ||
        observed_fourier_matrix_products != 0) {
        fputs("P^2 hw_disable_fft fallback was incorrect or used Fourier transforms\n",
              stderr);
        abort();
    }
    hw_disable_fft = 0;
    mpz_clear(work);
    clear_values(A, 2 * acells);
    clear_values(B, 2 * bcells);
    clear_values(C, 2 * acells);
    clear_values(expected, 2 * acells);
}
#endif

#ifdef RFOREST_ENABLE_PN_MATMUL_TESTS
static void test_pn_compatibility(void)
{
    const int rows = 2;
    const int dim = 3;
    const size_t acells = (size_t)rows * (size_t)dim;
    const size_t bcells = (size_t)dim * (size_t)dim;
    mpz_t *A1 = new_values(acells);
    mpz_t *B1 = new_values(bcells);
    mpz_t *pn1 = new_values(acells);
    mpz_t *integer = new_values(acells);
    mpz_t *A2 = new_values(2 * acells);
    mpz_t *B2 = new_values(2 * bcells);
    mpz_t *pn2 = new_values(2 * acells);
    mpz_t *specialized = new_values(2 * acells);
    mpz_t work;

    fill_matrix(A1, acells, 0, 3);
    fill_matrix(B1, bcells, 0, 5);
    fill_matrix(A2, 2 * acells, 0, 7);
    fill_matrix(B2, 2 * bcells, 0, 11);
    mpz_init(work);
    mpz_rmatrix_mult_pn(pn1, A1, rows, B1, dim, 1, work);
    mpz_rmatrix_mult(integer, A1, rows, B1, dim, work);
    if (!equal_values(pn1, integer, acells)) {
        fputs("P^1 product differed from integer matrix multiplication\n",
              stderr);
        abort();
    }
    mpz_rmatrix_mult_pn(pn2, A2, rows, B2, dim, 2, work);
    mpz_rmatrix_mult_p2(specialized, A2, rows, B2, dim, work);
    if (!equal_values(pn2, specialized, 2 * acells)) {
        fputs("P^2 general facade differed from the specialized path\n",
              stderr);
        abort();
    }
    mpz_clear(work);
    clear_values(A1, acells);
    clear_values(B1, bcells);
    clear_values(pn1, acells);
    clear_values(integer, acells);
    clear_values(A2, 2 * acells);
    clear_values(B2, 2 * bcells);
    clear_values(pn2, 2 * acells);
    clear_values(specialized, 2 * acells);
}

static void test_shared_fourier_pn_dispatch(void)
{
    const int n = 3;
    const int rows = 8;
    const int dim = 8;
    const size_t acells = (size_t)rows * (size_t)dim;
    const size_t bcells = (size_t)dim * (size_t)dim;
    const size_t left_count = (size_t)n * acells;
    const size_t right_count = (size_t)n * bcells;
    const size_t input_entries = left_count + right_count;
    size_t threshold = ((size_t)rows + (size_t)dim) * (size_t)dim *
                       (size_t)mpz_mat_fft_crossover(dim);
    size_t limbs = threshold / input_entries + 1;
    mpz_t *A = new_values(left_count);
    mpz_t *B = new_values(right_count);
    mpz_t *C = new_values(left_count);
    mpz_t *expected = new_values(left_count);
    mpz_t work;

    if (limbs == 0 || limbs > SIZE_MAX / GMP_NUMB_BITS) {
        fputs("invalid P^n Fourier dispatch test size\n", stderr);
        abort();
    }
    for (size_t i = 0; i < acells; i++) {
        mpz_set_ui(A[i], (unsigned long)(i % 31) + 1);
        mpz_setbit(A[i], (mp_bitcnt_t)(limbs * GMP_NUMB_BITS - 1));
        if (i & 1)
            mpz_neg(A[i], A[i]);
        mpz_set(A[acells + i], A[i]);
        mpz_set(A[2 * acells + i], A[i]);

        mpz_set_ui(B[i], (unsigned long)(i % 29) + 1);
        mpz_setbit(B[i], (mp_bitcnt_t)(limbs * GMP_NUMB_BITS - 1));
        if (i & 1)
            mpz_neg(B[i], B[i]);
        mpz_mul_2exp(B[bcells + i], B[i], 1);
        mpz_neg(B[bcells + i], B[bcells + i]);
        mpz_set(B[2 * bcells + i], B[i]);
    }

    mpz_init(work);
    reference_product(expected, A, B, rows, dim, n, 1, 0);
    observed_forward_transforms = 0;
    observed_inverse_transforms = 0;
    observed_fourier_matrix_products = 0;
    hw_disable_fft = 0;
    mpz_rmatrix_mult_pn(C, A, rows, B, dim, n, work);
    if (!equal_values(C, expected, left_count)) {
        fputs("P^n Fourier result disagreed with direct GMP reference\n",
              stderr);
        abort();
    }
    for (size_t i = 2 * acells; i < 3 * acells; i++)
        if (mpz_sgn(C[i]) != 0) {
            fputs("P^n full signed cancellation coefficient was nonzero\n",
                  stderr);
            abort();
        }
    if (observed_forward_transforms != input_entries ||
        observed_inverse_transforms != left_count ||
        observed_fourier_matrix_products != (unsigned)(n * (n + 1) / 2)) {
        fprintf(stderr,
                "P^n Fourier reuse counts: forward=%u inverse=%u matrix=%u\n",
                observed_forward_transforms, observed_inverse_transforms,
                observed_fourier_matrix_products);
        abort();
    }

    observed_forward_transforms = 0;
    observed_inverse_transforms = 0;
    observed_fourier_matrix_products = 0;
    hw_disable_fft = 1;
    mpz_rmatrix_mult_pn(C, A, rows, B, dim, n, work);
    if (!equal_values(C, expected, left_count) ||
        observed_forward_transforms != 0 ||
        observed_inverse_transforms != 0 ||
        observed_fourier_matrix_products != 0) {
        fputs("P^n hw_disable_fft fallback was incorrect or used Fourier transforms\n",
              stderr);
        abort();
    }
    hw_disable_fft = 0;
    mpz_clear(work);
    clear_values(A, left_count);
    clear_values(B, right_count);
    clear_values(C, left_count);
    clear_values(expected, left_count);
}

#endif

#ifdef RFOREST_ENABLE_RING_MATMUL_TESTS
static void test_bivariate_noncommuting_coefficients(void)
{
    const int N = 2;
    const int dim = 2;
    const size_t count = (size_t)N * (size_t)N * (size_t)dim * (size_t)dim;
    mpz_t *A = new_values(count);
    mpz_t *B = new_values(count);
    mpz_t *C = new_values(count);
    mpz_t *expected = new_values(count);
    mpz_t work;

    fill_matrix(A, count, 0, 2);
    fill_matrix(B, count, 0, 7);
    mpz_set_ui(A[0], 1);
    mpz_set_ui(A[1], 2);
    mpz_set_ui(A[2], 3);
    mpz_set_ui(A[3], 4);
    mpz_set_ui(B[0], 0);
    mpz_set_ui(B[1], 1);
    mpz_set_ui(B[2], 1);
    mpz_set_ui(B[3], 0);

    mpz_init(work);
    reference_product(expected, A, B, dim, dim, N, N, 1);
    mpz_rmatrix_mult_pnq(C, A, dim, B, dim, N, work);
    if (!equal_values(C, expected, count) || mpz_cmp_ui(C[0], 2) != 0 ||
        mpz_cmp_ui(C[1], 1) != 0 || mpz_cmp_ui(C[2], 4) != 0 ||
        mpz_cmp_ui(C[3], 3) != 0) {
        fputs("bivariate matrix coefficients were not multiplied in A*B order\n",
              stderr);
        abort();
    }
    mpz_clear(work);
    clear_values(A, count);
    clear_values(B, count);
    clear_values(C, count);
    clear_values(expected, count);
}

static void test_bivariate_input_alias(void)
{
    const int N = 2;
    const int dim = 3;
    const size_t count = (size_t)N * (size_t)N * (size_t)dim * (size_t)dim;
    mpz_t *A = new_values(count);
    mpz_t *saved = new_values(count);
    mpz_t *C = new_values(count);
    mpz_t *expected = new_values(count);
    mpz_t work;

    fill_matrix(A, count, 0, 11);
    mpz_vec_set(saved, A, (long)count);
    mpz_init(work);
    reference_product(expected, A, A, dim, dim, N, N, 1);
    mpz_rmatrix_mult_pnq(C, A, dim, A, dim, N, work);
    if (!equal_values(C, expected, count) ||
        !equal_values(A, saved, count)) {
        fputs("bivariate multiplication mishandled aliased inputs\n", stderr);
        abort();
    }
    mpz_clear(work);
    clear_values(A, count);
    clear_values(saved, count);
    clear_values(C, count);
    clear_values(expected, count);
}

static void test_shared_fourier_pnq_dispatch(void)
{
    const int N = 3;
    const int rows = 8;
    const int dim = 8;
    const size_t coefficients = (size_t)N * (size_t)N;
    const size_t acells = (size_t)rows * (size_t)dim;
    const size_t bcells = (size_t)dim * (size_t)dim;
    const size_t left_count = coefficients * acells;
    const size_t right_count = coefficients * bcells;
    const size_t input_entries = left_count + right_count;
    size_t threshold = ((size_t)rows + (size_t)dim) * (size_t)dim *
                       (size_t)mpz_mat_fft_crossover(dim);
    size_t active_entries = 2 * (acells + bcells);
    size_t limbs = threshold / active_entries + 1;
    mpz_t *A = new_values(left_count);
    mpz_t *B = new_values(right_count);
    mpz_t *C = new_values(left_count);
    mpz_t *expected = new_values(left_count);
    mpz_t work;

    if (limbs == 0 || limbs > SIZE_MAX / GMP_NUMB_BITS) {
        fputs("invalid bivariate Fourier dispatch test size\n", stderr);
        abort();
    }
    for (size_t i = 0; i < acells; i++) {
        mpz_set_ui(A[i], (unsigned long)(i % 31) + 1);
        mpz_setbit(A[i], (mp_bitcnt_t)(limbs * GMP_NUMB_BITS - 1));
        if (i & 1)
            mpz_neg(A[i], A[i]);
        mpz_set(A[(coefficients - 1) * acells + i], A[i]);

        mpz_set_ui(B[i], (unsigned long)(i % 29) + 1);
        mpz_setbit(B[i], (mp_bitcnt_t)(limbs * GMP_NUMB_BITS - 1));
        if (i & 1)
            mpz_neg(B[i], B[i]);
        mpz_neg(B[(coefficients - 1) * bcells + i], B[i]);
    }

    mpz_init(work);
    reference_product(expected, A, B, rows, dim, N, N, 1);
    observed_forward_transforms = 0;
    observed_inverse_transforms = 0;
    observed_fourier_matrix_products = 0;
    hw_disable_fft = 0;
    mpz_rmatrix_mult_pnq(C, A, rows, B, dim, N, work);
    if (!equal_values(C, expected, left_count)) {
        fputs("bivariate Fourier result disagreed with direct GMP reference\n",
              stderr);
        abort();
    }
    for (size_t i = (coefficients - 1) * acells; i < left_count; i++)
        if (mpz_sgn(C[i]) != 0) {
            fputs("bivariate signed cancellation coefficient was nonzero\n",
                  stderr);
            abort();
        }
    {
        size_t pairs = (size_t)N * (size_t)(N + 1) / 2;
        if (observed_forward_transforms != input_entries ||
            observed_inverse_transforms != left_count ||
            observed_fourier_matrix_products != pairs * pairs) {
            fprintf(stderr,
                    "bivariate Fourier reuse counts: forward=%u inverse=%u matrix=%u\n",
                    observed_forward_transforms, observed_inverse_transforms,
                    observed_fourier_matrix_products);
            abort();
        }
    }

    observed_forward_transforms = 0;
    observed_inverse_transforms = 0;
    observed_fourier_matrix_products = 0;
    hw_disable_fft = 1;
    mpz_rmatrix_mult_pnq(C, A, rows, B, dim, N, work);
    if (!equal_values(C, expected, left_count) ||
        observed_forward_transforms != 0 ||
        observed_inverse_transforms != 0 ||
        observed_fourier_matrix_products != 0) {
        fputs("bivariate hw_disable_fft fallback was incorrect or used Fourier transforms\n",
              stderr);
        abort();
    }
    hw_disable_fft = 0;
    mpz_clear(work);
    clear_values(A, left_count);
    clear_values(B, right_count);
    clear_values(C, left_count);
    clear_values(expected, left_count);
}

static void test_bivariate_box_corner(void)
{
    mpz_t *A = new_values(9);
    mpz_t *B = new_values(9);
    mpz_t *C = new_values(9);
    mpz_t *expected = new_values(9);
    mpz_t work;
    /* Scalar 1x1 matrices make the two discarded overflow terms explicit. */
    mpz_set_ui(A[8], 2);  /* P^2 Q^2 */
    mpz_set_ui(B[0], 3);  /* 1 */
    mpz_set_ui(A[6], 5);  /* P^2 */
    mpz_set_ui(B[3], 7);  /* P: P^3 overflows */
    mpz_set_ui(A[2], 11); /* Q^2 */
    mpz_set_ui(B[1], 13); /* Q: Q^3 overflows */
    mpz_init(work);
    reference_product(expected, A, B, 1, 1, 3, 3, 1);
    mpz_rmatrix_mult_pnq(C, A, 1, B, 1, 3, work);
    if (!equal_values(C, expected, 9) || mpz_cmp_ui(C[8], 6) != 0) {
        fputs("bivariate box truncation dropped the highest corner or kept overflow\n",
              stderr);
        abort();
    }
    mpz_clear(work);
    clear_values(A, 9);
    clear_values(B, 9);
    clear_values(C, 9);
    clear_values(expected, 9);
}
#endif

int main(int argc, char **argv)
{
    const char *aws_fixture = argc > 1 ? argv[1]
        : "tests/fixtures/aws_ring/p2_aws_products.txt";
    hw_disable_fft = 0;
    hw_mpz_setup();
    test_aws_products(aws_fixture);
    /* Base matrix API forbids C overlapping A or B; this ring API keeps that contract. */
    run_case("p2-signed-dense-rectangular", 2, 1, 2, 3, 0, 0);
    run_case("p2-sparse", 2, 1, 1, 4, 1, 0);
    run_case("p2-zero", 2, 1, 2, 3, 2, 0);
    run_case("p2-large", 2, 1, 2, 3, 3, 0);
#ifdef RFOREST_ENABLE_PN_MATMUL_TESTS
    run_case("pn-one-is-integer-matmul", 1, 1, 2, 3, 0, 0);
    run_case("pn-one-unit-dimension-multiple-rows", 1, 1, 2, 1, 0, 0);
    run_case("pn-two-general-facade", 2, 1, 2, 3, 0, 0);
    run_case("pn-three", 3, 1, 2, 3, 0, 0);
    run_case("pn-four-large-signed", 4, 1, 2, 3, 3, 0);
    run_case("pn-five-cancellation-shapes", 5, 1, 2, 3, 1, 0);
#endif
#ifdef RFOREST_ENABLE_RING_MATMUL_TESTS
    run_case("pnq-one", 1, 1, 2, 3, 0, 1);
    run_case("pnq-two-signed-sparse", 2, 2, 2, 3, 1, 1);
    run_case("pnq-three-box-and-high-corner", 3, 3, 2, 3, 0, 1);
    run_case("pnq-three-sparse", 3, 3, 2, 4, 1, 1);
    run_case("pnq-three-large-signed", 3, 3, 2, 3, 3, 1);
#endif
    test_p2_order_and_cancellation();
#ifdef RFOREST_ENABLE_P2_MATMUL_TESTS
    test_shared_fourier_dispatch();
#endif
#ifdef RFOREST_ENABLE_PN_MATMUL_TESTS
    test_pn_compatibility();
    test_shared_fourier_pn_dispatch();
#endif
#ifdef RFOREST_ENABLE_RING_MATMUL_TESTS
    test_bivariate_noncommuting_coefficients();
    test_bivariate_input_alias();
    test_bivariate_box_corner();
    test_shared_fourier_pnq_dispatch();
#else
    puts("DISABLED bivariate ring matrix tests: activate in PR 4");
#endif
    puts("PASS ring matrix API exact references");
    hw_mpz_clear();
    return 0;
}

#endif
