#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <gmp.h>

#include "hwmpz.h"

enum ring_kind { RING_UNIVARIATE, RING_BIVARIATE };

typedef struct {
    char name[96];
    enum ring_kind kind;
    int truncation;
    int rows;
    int dim;
    unsigned bits;
    int sparse;
    size_t coefficient_count;
    mpz_t *left;
    mpz_t *right;
    mpz_t *aws_expected;
} ring_case;

static const unsigned warmup_count = 2;
static const unsigned repeat_count = 9;

static mpz_t *mpz_array_new(size_t count)
{
    mpz_t *values = malloc(count * sizeof(*values));
    if (!values && count) {
        fputs("ring baseline: allocation failed\n", stderr);
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < count; i++)
        mpz_init(values[i]);
    return values;
}

static void mpz_array_clear(mpz_t *values, size_t count)
{
    if (!values)
        return;
    for (size_t i = 0; i < count; i++)
        mpz_clear(values[i]);
    free(values);
}

static size_t checked_product(size_t left, size_t right)
{
    if (right && left > SIZE_MAX / right) {
        fputs("ring baseline: grid size overflow\n", stderr);
        exit(EXIT_FAILURE);
    }
    return left * right;
}

static size_t matrix_cells(const ring_case *test)
{
    return checked_product((size_t)test->rows, (size_t)test->dim);
}

static uint64_t splitmix64(uint64_t *state)
{
    uint64_t value = (*state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static void random_signed_mpz(mpz_t value, uint64_t *state, unsigned bits)
{
    const size_t words_count = (bits + 63U) / 64U;
    uint64_t *words = malloc(words_count * sizeof(*words));
    unsigned leading_bits = bits % 64U;
    if (!words && words_count) {
        fputs("ring baseline: random operand allocation failed\n", stderr);
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < words_count; i++)
        words[i] = splitmix64(state);
    if (leading_bits) {
        words[0] &= (UINT64_C(1) << leading_bits) - 1;
        words[0] |= UINT64_C(1) << (leading_bits - 1);
    } else if (words_count) {
        words[0] |= UINT64_C(1) << 63;
    }
    mpz_import(value, words_count, 1, sizeof(*words), 0, 0, words);
    if (splitmix64(state) & 1)
        mpz_neg(value, value);
    free(words);
}

static size_t coefficient_index(const ring_case *test, int p, int q)
{
    return test->kind == RING_BIVARIATE
               ? (size_t)p * (size_t)test->truncation + (size_t)q
               : (size_t)p;
}

static ring_case make_case(const char *name, enum ring_kind kind, int truncation,
                           int rows, int dim, unsigned bits, int sparse,
                           unsigned long seed)
{
    ring_case test;
    size_t cells;
    uint64_t state = (uint64_t)seed;

    memset(&test, 0, sizeof(test));
    snprintf(test.name, sizeof(test.name), "%s", name);
    test.kind = kind;
    test.truncation = truncation;
    test.rows = rows;
    test.dim = dim;
    test.bits = bits;
    test.sparse = sparse;
    test.coefficient_count = kind == RING_BIVARIATE
                                 ? checked_product((size_t)truncation,
                                                   (size_t)truncation)
                                 : (size_t)truncation;
    cells = matrix_cells(&test);
    test.left = mpz_array_new(checked_product(test.coefficient_count, cells));
    test.right = mpz_array_new(checked_product(test.coefficient_count,
                                               (size_t)dim * (size_t)dim));

    for (size_t i = 0; i < test.coefficient_count * cells; i++) {
        if (sparse && (splitmix64(&state) & UINT64_C(255)) < 205) {
            mpz_set_ui(test.left[i], 0);
            continue;
        }
        random_signed_mpz(test.left[i], &state, bits);
    }
    for (size_t i = 0; i < test.coefficient_count * (size_t)dim * (size_t)dim; i++) {
        if (sparse && (splitmix64(&state) & UINT64_C(255)) < 205) {
            mpz_set_ui(test.right[i], 0);
            continue;
        }
        random_signed_mpz(test.right[i], &state, bits);
    }
    return test;
}

static void clear_case(ring_case *test)
{
    size_t cells = matrix_cells(test);
    mpz_array_clear(test->left, test->coefficient_count * cells);
    mpz_array_clear(test->right, test->coefficient_count * (size_t)test->dim *
                                    (size_t)test->dim);
    mpz_array_clear(test->aws_expected,
                    test->aws_expected ? 2 * cells : 0);
    memset(test, 0, sizeof(*test));
}

static void zero_result(mpz_t *result, size_t count)
{
    for (size_t i = 0; i < count; i++)
        mpz_set_ui(result[i], 0);
}

static void reference_product(const ring_case *test, mpz_t *result)
{
    const size_t left_cells = matrix_cells(test);
    const size_t right_cells = (size_t)test->dim * (size_t)test->dim;
    const int n = test->truncation;
    mpz_t product;

    zero_result(result, test->coefficient_count * left_cells);
    mpz_init(product);
    for (int ap = 0; ap < n; ap++) {
        for (int aq = 0; aq < (test->kind == RING_BIVARIATE ? n : 1); aq++) {
            size_t a_coeff = coefficient_index(test, ap, aq);
            for (int bp = 0; bp < n; bp++) {
                for (int bq = 0; bq < (test->kind == RING_BIVARIATE ? n : 1); bq++) {
                    int cp = ap + bp;
                    int cq = aq + bq;
                    size_t b_coeff;
                    size_t c_coeff;
                    if (cp >= n || cq >= (test->kind == RING_BIVARIATE ? n : 1))
                        continue;
                    b_coeff = coefficient_index(test, bp, bq);
                    c_coeff = coefficient_index(test, cp, cq);
                    for (int row = 0; row < test->rows; row++) {
                        for (int col = 0; col < test->dim; col++) {
                            size_t out = c_coeff * left_cells +
                                         (size_t)row * (size_t)test->dim +
                                         (size_t)col;
                            for (int inner = 0; inner < test->dim; inner++) {
                                size_t a = a_coeff * left_cells +
                                           (size_t)row * (size_t)test->dim +
                                           (size_t)inner;
                                size_t b = b_coeff * right_cells +
                                           (size_t)inner * (size_t)test->dim +
                                           (size_t)col;
                                mpz_mul(product, test->left[a], test->right[b]);
                                mpz_add(result[out], result[out], product);
                            }
                        }
                    }
                }
            }
        }
    }
    mpz_clear(product);
}

static void adapter_product(const ring_case *test, mpz_t *result)
{
    const size_t left_cells = matrix_cells(test);
    const size_t right_cells = (size_t)test->dim * (size_t)test->dim;
    const size_t result_cells = test->coefficient_count * left_cells;
    mpz_t *product = mpz_array_new(left_cells);
    mpz_t work;

    mpz_init(work);
    zero_result(result, result_cells);
    for (int ap = 0; ap < test->truncation; ap++) {
        for (int aq = 0; aq < (test->kind == RING_BIVARIATE ? test->truncation : 1); aq++) {
            size_t a_coeff = coefficient_index(test, ap, aq);
            for (int bp = 0; bp < test->truncation; bp++) {
                for (int bq = 0; bq < (test->kind == RING_BIVARIATE ? test->truncation : 1); bq++) {
                    int cp = ap + bp;
                    int cq = aq + bq;
                    size_t b_coeff;
                    size_t c_coeff;
                    if (cp >= test->truncation ||
                        cq >= (test->kind == RING_BIVARIATE ? test->truncation : 1))
                        continue;
                    b_coeff = coefficient_index(test, bp, bq);
                    c_coeff = coefficient_index(test, cp, cq);
                    mpz_rmatrix_mult(product, test->left + a_coeff * left_cells,
                                     test->rows,
                                     test->right + b_coeff * right_cells,
                                     test->dim, work);
                    for (size_t i = 0; i < left_cells; i++)
                        mpz_add(result[c_coeff * left_cells + i],
                                result[c_coeff * left_cells + i], product[i]);
                }
            }
        }
    }
    mpz_clear(work);
    mpz_array_clear(product, left_cells);
}

static void block_product(const ring_case *test, mpz_t *result)
{
    const int n = test->truncation;
    const int bivar = test->kind == RING_BIVARIATE;
    const size_t basis = bivar ? (size_t)n * (size_t)n : (size_t)n;
    const size_t rows = checked_product(basis, (size_t)test->rows);
    const size_t dim = checked_product(basis, (size_t)test->dim);
    const size_t left_cells = matrix_cells(test);
    const size_t right_cells = (size_t)test->dim * (size_t)test->dim;
    const size_t left_embedding_cells = checked_product(rows, dim);
    const size_t right_embedding_cells = checked_product(dim, dim);
    mpz_t *left_embedding;
    mpz_t *right_embedding;
    mpz_t *product;
    mpz_t work;

    if (rows > INT_MAX || dim > INT_MAX) {
        fputs("ring baseline: block embedding exceeds int dimensions\n", stderr);
        exit(EXIT_FAILURE);
    }
    left_embedding = mpz_array_new(left_embedding_cells);
    right_embedding = mpz_array_new(right_embedding_cells);
    product = mpz_array_new(left_embedding_cells);
    mpz_init(work);

    for (int ap = 0; ap < n; ap++) {
        for (int aq = 0; aq < (bivar ? n : 1); aq++) {
            size_t coeff = coefficient_index(test, ap, aq);
            for (int ip = 0; ip < n; ip++) {
                for (int iq = 0; iq < (bivar ? n : 1); iq++) {
                    int op = ap + ip;
                    int oq = aq + iq;
                    size_t input_basis;
                    size_t output_basis;
                    if (op >= n || oq >= (bivar ? n : 1))
                        continue;
                    input_basis = bivar ? (size_t)ip * (size_t)n + (size_t)iq
                                        : (size_t)ip;
                    output_basis = bivar ? (size_t)op * (size_t)n + (size_t)oq
                                         : (size_t)op;
                    for (int row = 0; row < test->rows; row++) {
                        for (int col = 0; col < test->dim; col++) {
                            size_t from = coeff * left_cells +
                                          (size_t)row * (size_t)test->dim +
                                          (size_t)col;
                            size_t to = (output_basis * (size_t)test->rows +
                                         (size_t)row) * dim +
                                        input_basis * (size_t)test->dim +
                                        (size_t)col;
                            mpz_set(left_embedding[to], test->left[from]);
                        }
                    }
                    for (int row = 0; row < test->dim; row++) {
                        for (int col = 0; col < test->dim; col++) {
                            size_t from = coeff * right_cells +
                                          (size_t)row * (size_t)test->dim +
                                          (size_t)col;
                            size_t to = (output_basis * (size_t)test->dim +
                                         (size_t)row) * dim +
                                        input_basis * (size_t)test->dim +
                                        (size_t)col;
                            mpz_set(right_embedding[to], test->right[from]);
                        }
                    }
                }
            }
        }
    }

    mpz_rmatrix_mult(product, left_embedding, (int)rows, right_embedding,
                     (int)dim, work);
    zero_result(result, test->coefficient_count * left_cells);
    for (int p = 0; p < n; p++) {
        for (int q = 0; q < (bivar ? n : 1); q++) {
            size_t coeff = coefficient_index(test, p, q);
            size_t output_basis = bivar ? (size_t)p * (size_t)n + (size_t)q
                                        : (size_t)p;
            for (int row = 0; row < test->rows; row++) {
                for (int col = 0; col < test->dim; col++) {
                    size_t from = (output_basis * (size_t)test->rows +
                                   (size_t)row) * dim + (size_t)col;
                    size_t to = coeff * left_cells +
                                (size_t)row * (size_t)test->dim + (size_t)col;
                    mpz_set(result[to], product[from]);
                }
            }
        }
    }

    mpz_clear(work);
    mpz_array_clear(left_embedding, left_embedding_cells);
    mpz_array_clear(right_embedding, right_embedding_cells);
    mpz_array_clear(product, left_embedding_cells);
}

static int equal_results(mpz_t *left, mpz_t *right, size_t count)
{
    for (size_t i = 0; i < count; i++)
        if (mpz_cmp(left[i], right[i]) != 0)
            return 0;
    return 1;
}

static void fail_case(const ring_case *test, const char *algorithm)
{
    fprintf(stderr, "ring baseline: %s disagreed for case %s\n",
            algorithm, test->name);
    exit(EXIT_FAILURE);
}

static void expect_word(FILE *input, const char *expected)
{
    char word[128];
    if (fscanf(input, "%127s", word) != 1 || strcmp(word, expected) != 0) {
        fprintf(stderr, "ring baseline: expected fixture token %s\n", expected);
        exit(EXIT_FAILURE);
    }
}

static void read_mpz_values(FILE *input, mpz_t *values, size_t count)
{
    char word[4096];
    for (size_t i = 0; i < count; i++) {
        if (fscanf(input, "%4095s", word) != 1 || mpz_set_str(values[i], word, 10)) {
            fputs("ring baseline: malformed AWS integer fixture\n", stderr);
            exit(EXIT_FAILURE);
        }
    }
}

static void read_aws_cases(const char *path, ring_case **cases_out, size_t *count_out)
{
    FILE *input = fopen(path, "r");
    ring_case *cases;
    char ignored[256];
    char case_id[96];
    size_t count;

    if (!input) {
        fprintf(stderr, "ring baseline: cannot open %s: %s\n", path, strerror(errno));
        exit(EXIT_FAILURE);
    }
    expect_word(input, "AWS_RING_P2_PRODUCTS");
    if (fscanf(input, "%255s", ignored) != 1 || strcmp(ignored, "1") != 0)
        exit(EXIT_FAILURE);
    expect_word(input, "AWS_COMMIT");
    if (fscanf(input, "%255s", ignored) != 1 ||
        strcmp(ignored, "a1fc50dd667d262b5d83d1d4ceb1499bdbead288") != 0)
        exit(EXIT_FAILURE);
    expect_word(input, "ZETA_SUITE_COMMIT");
    if (fscanf(input, "%255s", ignored) != 1 ||
        strcmp(ignored, "621107b12200a3c234c2e3dd6a4ba9dc87be9bea") != 0)
        exit(EXIT_FAILURE);
    expect_word(input, "AWS_SOURCE");
    if (fscanf(input, "%255s %255s", ignored, ignored) != 2)
        exit(EXIT_FAILURE);
    expect_word(input, "GENERIC_UNIVARIATE_SOURCE");
    if (fscanf(input, "%255s", ignored) != 1)
        exit(EXIT_FAILURE);
    expect_word(input, "P8_GENERIC_REFERENCE_CHECKS");
    if (fscanf(input, "%255s", ignored) != 1)
        exit(EXIT_FAILURE);
    expect_word(input, "CASE_COUNT");
    if (fscanf(input, "%zu", &count) != 1 || count == 0 || count > 100)
        exit(EXIT_FAILURE);
    cases = calloc(count, sizeof(*cases));
    if (!cases)
        exit(EXIT_FAILURE);

    for (size_t i = 0; i < count; i++) {
        int dim;
        size_t cells;
        expect_word(input, "CASE");
        if (fscanf(input, "%95s %d", case_id, &dim) != 2 || dim <= 0)
            exit(EXIT_FAILURE);
        cases[i] = make_case(case_id, RING_UNIVARIATE, 2, dim, dim, 0, 0,
                             (unsigned long)i + 1);
        cells = (size_t)dim * (size_t)dim;
        cases[i].aws_expected = mpz_array_new(2 * cells);
        for (int coefficient = 0; coefficient < 2; coefficient++) {
            expect_word(input, coefficient == 0 ? "A0" : "A1");
            read_mpz_values(input, cases[i].left + (size_t)coefficient * cells, cells);
        }
        for (int coefficient = 0; coefficient < 2; coefficient++) {
            expect_word(input, coefficient == 0 ? "B0" : "B1");
            read_mpz_values(input, cases[i].right + (size_t)coefficient * cells, cells);
        }
        for (int coefficient = 0; coefficient < 2; coefficient++) {
            expect_word(input, coefficient == 0 ? "C0" : "C1");
            read_mpz_values(input,
                            cases[i].aws_expected + (size_t)coefficient * cells,
                            cells);
        }
        expect_word(input, "END_CASE");
    }
    expect_word(input, "END");
    if (fscanf(input, "%255s", ignored) == 1) {
        fputs("ring baseline: trailing AWS fixture content\n", stderr);
        exit(EXIT_FAILURE);
    }
    fclose(input);
    *cases_out = cases;
    *count_out = count;
}

static uint64_t now_ns(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
           (uint64_t)now.tv_nsec;
}

static int compare_u64(const void *left, const void *right)
{
    uint64_t a = *(const uint64_t *)left;
    uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

typedef void (*product_function)(const ring_case *, mpz_t *);

static void measure(product_function product, const ring_case *test,
                    uint64_t *median, uint64_t *mad)
{
    const size_t count = test->coefficient_count * matrix_cells(test);
    uint64_t samples[repeat_count];
    uint64_t deviations[repeat_count];
    mpz_t *result;

    for (unsigned i = 0; i < warmup_count; i++) {
        result = mpz_array_new(count);
        product(test, result);
        mpz_array_clear(result, count);
    }
    for (unsigned i = 0; i < repeat_count; i++) {
        uint64_t start;
        start = now_ns();
        result = mpz_array_new(count);
        product(test, result);
        samples[i] = now_ns() - start;
        mpz_array_clear(result, count);
    }
    qsort(samples, repeat_count, sizeof(samples[0]), compare_u64);
    *median = samples[repeat_count / 2];
    for (unsigned i = 0; i < repeat_count; i++)
        deviations[i] = samples[i] > *median ? samples[i] - *median
                                             : *median - samples[i];
    qsort(deviations, repeat_count, sizeof(deviations[0]), compare_u64);
    *mad = deviations[repeat_count / 2];
}

static void run_case(ring_case *test)
{
    const size_t result_count = test->coefficient_count * matrix_cells(test);
    mpz_t *expected = mpz_array_new(result_count);
    mpz_t *adapter = mpz_array_new(result_count);
    mpz_t *block = mpz_array_new(result_count);
    uint64_t adapter_median;
    uint64_t adapter_mad;
    uint64_t block_median;
    uint64_t block_mad;
    double speedup;

    reference_product(test, expected);
    if (test->aws_expected &&
        !equal_results(expected, test->aws_expected, result_count))
        fail_case(test, "independent exact reference vs AWS output");
    adapter_product(test, adapter);
    block_product(test, block);
    if (!equal_results(expected, adapter, result_count))
        fail_case(test, "classical ring adapter vs direct reference");
    if (!equal_results(expected, block, result_count))
        fail_case(test, "integer block embedding vs direct reference");

    measure(adapter_product, test, &adapter_median, &adapter_mad);
    measure(block_product, test, &block_median, &block_mad);
    speedup = block_median ? (double)adapter_median / (double)block_median : 0.0;
    printf("%s,%s,%d,%d,%d,%u,%s,%llu,%llu,%llu,%llu,%.3f\n",
           test->name,
           test->kind == RING_BIVARIATE ? "ZPQ" :
               (test->truncation == 2 ? "P2" : "PN"),
           test->truncation, test->rows, test->dim, test->bits,
           test->sparse ? "sparse" : "dense",
           (unsigned long long)adapter_median,
           (unsigned long long)adapter_mad,
           (unsigned long long)block_median,
           (unsigned long long)block_mad,
           speedup);
    mpz_array_clear(expected, result_count);
    mpz_array_clear(adapter, result_count);
    mpz_array_clear(block, result_count);
}

static void run_grid_case(const char *name, enum ring_kind kind, int truncation,
                          int rows, int dim, unsigned bits, int sparse,
                          unsigned long seed)
{
    ring_case test = make_case(name, kind, truncation, rows, dim, bits, sparse, seed);
    run_case(&test);
    clear_case(&test);
}

int main(int argc, char **argv)
{
    ring_case *aws_cases;
    size_t aws_count;
    unsigned long seed = 1000;

    if (argc != 2) {
        fprintf(stderr, "usage: %s tests/fixtures/aws_ring/p2_aws_products.txt\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    hw_disable_fft = 0;
    hw_mpz_setup();
    read_aws_cases(argv[1], &aws_cases, &aws_count);
    puts("case,ring,truncation,rows,dim,bits,density,adapter_median_ns,adapter_MAD_ns,block_median_ns,block_MAD_ns,adapter_over_block");
    for (size_t i = 0; i < aws_count; i++) {
        run_case(&aws_cases[i]);
        clear_case(&aws_cases[i]);
    }
    free(aws_cases);

    for (int dim_index = 0; dim_index < 2; dim_index++) {
        int dim = dim_index == 0 ? 2 : 8;
        for (int row_index = 0; row_index < 2; row_index++) {
            int rows = row_index == 0 ? 1 : dim;
            for (int bits_index = 0; bits_index < 2; bits_index++) {
                unsigned bits = bits_index == 0 ? 64 : 512;
                for (int sparse = 0; sparse <= 1; sparse++) {
                    char name[96];
                    snprintf(name, sizeof(name), "p2-d%d-r%d-b%u-%s", dim,
                             rows, bits, sparse ? "sparse" : "dense");
                    run_grid_case(name, RING_UNIVARIATE, 2, rows, dim, bits,
                                  sparse, seed++);
                }
            }
        }
    }

    for (int truncation_index = 0; truncation_index < 3; truncation_index++) {
        int truncation = truncation_index == 0 ? 1 :
                         (truncation_index == 1 ? 3 : 5);
        for (int dim_index = 0; dim_index < 2; dim_index++) {
            int dim = dim_index == 0 ? 2 : 6;
            for (int row_index = 0; row_index < 2; row_index++) {
                int rows = row_index == 0 ? 1 : (dim > 2 ? dim / 2 : dim);
                for (int bits_index = 0; bits_index < 2; bits_index++) {
                    unsigned bits = bits_index == 0 ? 32 : 256;
                    for (int sparse = 0; sparse <= 1; sparse++) {
                        char name[96];
                        snprintf(name, sizeof(name), "pn-n%d-d%d-r%d-b%u-%s",
                                 truncation, dim, rows, bits,
                                 sparse ? "sparse" : "dense");
                        run_grid_case(name, RING_UNIVARIATE, truncation, rows,
                                      dim, bits, sparse, seed++);
                    }
                }
            }
        }
    }

    for (int truncation_index = 0; truncation_index < 3; truncation_index++) {
        int truncation = truncation_index + 1;
        for (int dim_index = 0; dim_index < 2; dim_index++) {
            int dim = dim_index == 0 ? 2 : 4;
            for (int row_index = 0; row_index < 2; row_index++) {
                int rows = row_index == 0 ? 1 : dim;
                for (int bits_index = 0; bits_index < 2; bits_index++) {
                    unsigned bits = bits_index == 0 ? 32 : 256;
                    for (int sparse = 0; sparse <= 1; sparse++) {
                        char name[96];
                        snprintf(name, sizeof(name), "zpq-n%d-d%d-r%d-b%u-%s",
                                 truncation, dim, rows, bits,
                                 sparse ? "sparse" : "dense");
                        run_grid_case(name, RING_BIVARIATE, truncation, rows,
                                      dim, bits, sparse, seed++);
                    }
                }
            }
        }
    }
    hw_mpz_clear();
    return EXIT_SUCCESS;
}
