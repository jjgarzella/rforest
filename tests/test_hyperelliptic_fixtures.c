#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gmp.h>

#include "rforest.h"
#include "hwmpz.h"

#define MAX_FIXTURE_CELLS 1000000U
#define FIXTURE_REPEAT_RUNS 2

static unsigned long fft_matrix_multiply_calls;

mpz_t *__real_mpz_rmatrix_mult_fft(mpz_t *C, mpz_t *A, int r, mpz_t *B,
                                   int d, mpz_t w);

mpz_t *__wrap_mpz_rmatrix_mult_fft(mpz_t *C, mpz_t *A, int r, mpz_t *B,
                                   int d, mpz_t w)
{
    fft_matrix_multiply_calls++;
    return __real_mpz_rmatrix_mult_fft(C, A, r, B, d, w);
}

typedef struct {
    FILE *file;
    const char *path;
    size_t line;
    char *token;
    size_t token_capacity;
} fixture_reader;

typedef struct {
    char *case_name;
    char *call_name;
    long reference_prime;
    int dim;
    int rows;
    int deg;
    long n;
    long kbase;
    int kappa;
    int aws_default_kappa;
    int expect_fft_matrix_mul;
    size_t matrix_cells;
    size_t input_cells;
    size_t coefficient_cells;
    size_t output_cells;
    mpz_t initial_z;
    mpz_t *moduli;
    long *primes;
    long *endpoints;
    mpz_t *initial_v;
    mpz_t *matrix;
    mpz_t *expected;
} fixture;

static void reader_error(fixture_reader *reader, const char *format, ...)
{
    va_list args;
    fprintf(stderr, "%s:%zu: malformed fixture: ", reader->path, reader->line);
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fputc('\n', stderr);
    exit(EXIT_FAILURE);
}

static int next_token(fixture_reader *reader, const char **token_out)
{
    int ch;
    size_t length = 0;

    do {
        ch = fgetc(reader->file);
        if (ch == '\n')
            reader->line++;
    } while (ch != EOF && isspace((unsigned char)ch));

    if (ch == EOF) {
        if (ferror(reader->file))
            reader_error(reader, "error while reading input");
        return 0;
    }

    do {
        if (length + 1 >= reader->token_capacity) {
            size_t capacity = reader->token_capacity ? reader->token_capacity * 2 : 64;
            char *grown = realloc(reader->token, capacity);
            if (!grown)
                reader_error(reader, "out of memory while reading token");
            reader->token = grown;
            reader->token_capacity = capacity;
        }
        reader->token[length++] = (char)ch;
        ch = fgetc(reader->file);
    } while (ch != EOF && !isspace((unsigned char)ch));

    if (ch == '\n')
        reader->line++;
    if (ch == EOF && ferror(reader->file))
        reader_error(reader, "error while reading input");

    reader->token[length] = '\0';
    *token_out = reader->token;
    return 1;
}

static const char *required_token(fixture_reader *reader, const char *description)
{
    const char *token;
    if (!next_token(reader, &token))
        reader_error(reader, "expected %s before end of file", description);
    return token;
}

static void expect_token(fixture_reader *reader, const char *expected)
{
    const char *token = required_token(reader, expected);
    if (strcmp(token, expected) != 0)
        reader_error(reader, "expected '%s', got '%s'", expected, token);
}

static char *read_text(fixture_reader *reader, const char *description)
{
    const char *token = required_token(reader, description);
    char *copy = strdup(token);
    if (!copy)
        reader_error(reader, "out of memory while reading %s", description);
    return copy;
}

static void skip_text(fixture_reader *reader, const char *description)
{
    (void)required_token(reader, description);
}

static long read_long(fixture_reader *reader, const char *description)
{
    const char *token = required_token(reader, description);
    char *end = NULL;
    long value;

    errno = 0;
    value = strtol(token, &end, 10);
    if (errno == ERANGE || end == token || *end != '\0')
        reader_error(reader, "expected a decimal integer for %s, got '%s'", description, token);
    return value;
}

static int read_int(fixture_reader *reader, const char *description)
{
    long value = read_long(reader, description);
    if (value < INT_MIN || value > INT_MAX)
        reader_error(reader, "%s is outside the supported int range", description);
    return (int)value;
}

static void read_mpz(fixture_reader *reader, mpz_t value, const char *description)
{
    const char *token = required_token(reader, description);
    if (mpz_set_str(value, token, 10) != 0)
        reader_error(reader, "expected a decimal integer for %s, got '%s'", description, token);
}

static size_t checked_count(fixture_reader *reader, size_t left, size_t right,
                            const char *description)
{
    if (right && left > MAX_FIXTURE_CELLS / right)
        reader_error(reader, "%s exceeds the supported fixture size", description);
    return left * right;
}

static mpz_t *new_mpz_array(fixture_reader *reader, size_t count, const char *description)
{
    mpz_t *values;
    size_t i;

    if (count > MAX_FIXTURE_CELLS) {
        if (reader)
            reader_error(reader, "%s exceeds the supported fixture size", description);
        fprintf(stderr, "fixture runner: %s exceeds the supported fixture size\n", description);
        exit(EXIT_FAILURE);
    }
    values = malloc(count * sizeof(*values));
    if (!values && count) {
        if (reader)
            reader_error(reader, "out of memory allocating %s", description);
        fprintf(stderr, "fixture runner: out of memory allocating %s\n", description);
        exit(EXIT_FAILURE);
    }
    for (i = 0; i < count; i++)
        mpz_init(values[i]);
    return values;
}

static void clear_mpz_array(mpz_t *values, size_t count)
{
    size_t i;
    if (!values)
        return;
    for (i = 0; i < count; i++)
        mpz_clear(values[i]);
    free(values);
}

static mpz_t *copy_mpz_array(mpz_t *source, size_t count)
{
    mpz_t *copy = malloc(count * sizeof(*copy));
    size_t i;
    if (!copy && count) {
        fputs("fixture runner: out of memory copying GMP values\n", stderr);
        exit(EXIT_FAILURE);
    }
    for (i = 0; i < count; i++) {
        mpz_init(copy[i]);
        mpz_set(copy[i], source[i]);
    }
    return copy;
}

static void read_labeled_long(fixture_reader *reader, const char *label,
                              long *value, const char *description)
{
    expect_token(reader, label);
    *value = read_long(reader, description);
}

static void read_labeled_int(fixture_reader *reader, const char *label,
                             int *value, const char *description)
{
    expect_token(reader, label);
    *value = read_int(reader, description);
}

static void fixture_read(const char *path, fixture *data)
{
    fixture_reader reader = {0};
    const char *token;
    long version;
    long n;
    size_t n_cells;
    size_t i;
    int found_reference_prime = 0;
    mpz_t product;
    mpz_t squared_prime;

    memset(data, 0, sizeof(*data));
    reader.path = path;
    reader.line = 1;
    reader.file = fopen(path, "r");
    if (!reader.file) {
        fprintf(stderr, "%s: cannot open fixture: %s\n", path, strerror(errno));
        exit(EXIT_FAILURE);
    }

    expect_token(&reader, "RFOREST_FIXTURE");
    version = read_long(&reader, "fixture version");
    if (version != 1)
        reader_error(&reader, "unsupported fixture version %ld", version);

    expect_token(&reader, "CASE");
    data->case_name = read_text(&reader, "case name");
    expect_token(&reader, "CALL");
    data->call_name = read_text(&reader, "call name");
    expect_token(&reader, "AWS_COMMIT");
    skip_text(&reader, "AWS source revision");
    expect_token(&reader, "ZETA_SUITE_COMMIT");
    skip_text(&reader, "zeta suite revision");
    expect_token(&reader, "PYRFOREST_WRAPPER_COMMIT");
    skip_text(&reader, "pyrforest wrapper revision");
    read_labeled_long(&reader, "REFERENCE_PRIME", &data->reference_prime,
                      "reference prime");
    if (data->reference_prime <= 1)
        reader_error(&reader, "reference prime must be greater than one");

    read_labeled_int(&reader, "DIM", &data->dim, "DIM");
    read_labeled_int(&reader, "ROWS", &data->rows, "ROWS");
    read_labeled_int(&reader, "DEG", &data->deg, "DEG");
    read_labeled_long(&reader, "N", &data->n, "N");
    read_labeled_long(&reader, "KBASE", &data->kbase, "KBASE");
    read_labeled_int(&reader, "KAPPA", &data->kappa, "KAPPA");
    read_labeled_int(&reader, "AWS_DEFAULT_KAPPA", &data->aws_default_kappa,
                     "AWS wrapper default KAPPA");
    read_labeled_int(&reader, "EXPECT_FFT_MATRIX_MUL", &data->expect_fft_matrix_mul,
                     "FFT matrix multiplication expectation");

    if (data->dim <= 0 || data->rows <= 0 || data->deg < 0 ||
        data->n <= 0 || data->kappa < 0 || data->aws_default_kappa < 0)
        reader_error(&reader, "DIM/ROWS/N must be positive and DEG/KAPPA must be nonnegative");
    if (data->expect_fft_matrix_mul != 0 && data->expect_fft_matrix_mul != 1)
        reader_error(&reader, "EXPECT_FFT_MATRIX_MUL must be zero or one");

    n = data->n;
    if ((unsigned long)n > MAX_FIXTURE_CELLS)
        reader_error(&reader, "N exceeds the supported fixture size");
    n_cells = (size_t)n;
    data->matrix_cells = checked_count(&reader, (size_t)data->dim,
                                       (size_t)data->dim, "matrix dimensions");
    data->input_cells = checked_count(&reader, (size_t)data->rows,
                                      (size_t)data->dim, "initial V dimensions");
    data->output_cells = checked_count(&reader, n_cells, data->input_cells,
                                       "expected output matrices");
    data->coefficient_cells = checked_count(&reader, data->matrix_cells,
                                             (size_t)data->deg + 1,
                                             "matrix polynomial coefficients");

    mpz_init(data->initial_z);
    expect_token(&reader, "INITIAL_Z");
    read_mpz(&reader, data->initial_z, "INITIAL_Z");
    if (mpz_sgn(data->initial_z) <= 0)
        reader_error(&reader, "INITIAL_Z must be positive");

    data->moduli = new_mpz_array(&reader, n_cells, "moduli");
    data->primes = calloc(n_cells, sizeof(*data->primes));
    data->endpoints = calloc(n_cells, sizeof(*data->endpoints));
    if (!data->primes || !data->endpoints)
        reader_error(&reader, "out of memory allocating modulus metadata");

    expect_token(&reader, "MODULI");
    mpz_init(product);
    mpz_init(squared_prime);
    mpz_set_ui(product, 1);
    for (i = 0; i < n_cells; i++) {
        long prime = read_long(&reader, "modulus prime");
        long endpoint;

        if (prime <= 1)
            reader_error(&reader, "moduli[%zu] prime must be greater than one", i);
        if (i && prime <= data->primes[i - 1])
            reader_error(&reader, "modulus primes must be strictly increasing");
        data->primes[i] = prime;
        if (prime == data->reference_prime)
            found_reference_prime = 1;
        read_mpz(&reader, data->moduli[i], "squared modulus");
        if (mpz_sgn(data->moduli[i]) <= 0)
            reader_error(&reader, "moduli[%zu] must be positive", i);
        mpz_set_ui(squared_prime, (unsigned long)prime);
        mpz_mul_ui(squared_prime, squared_prime, (unsigned long)prime);
        if (mpz_cmp(squared_prime, data->moduli[i]) != 0)
            reader_error(&reader, "moduli[%zu] is not the square of its recorded prime", i);
        endpoint = read_long(&reader, "exclusive endpoint");
        if ((i == 0 && endpoint < data->kbase) ||
            (i > 0 && endpoint < data->endpoints[i - 1]))
            reader_error(&reader, "exclusive endpoints must be nondecreasing from KBASE");
        data->endpoints[i] = endpoint;
        mpz_mul(product, product, data->moduli[i]);
    }

    if (!found_reference_prime)
        reader_error(&reader, "REFERENCE_PRIME is not present in MODULI");
    if (mpz_cmp(product, data->initial_z) != 0)
        reader_error(&reader, "INITIAL_Z does not equal the product of MODULI");
    mpz_clear(product);
    mpz_clear(squared_prime);

    data->initial_v = new_mpz_array(&reader, data->input_cells, "initial V");
    expect_token(&reader, "INITIAL_V");
    for (i = 0; i < data->input_cells; i++)
        read_mpz(&reader, data->initial_v[i], "INITIAL_V entry");

    data->matrix = new_mpz_array(&reader, data->coefficient_cells,
                                 "matrix polynomial coefficients");
    expect_token(&reader, "MATRIX_COEFFICIENTS");
    for (i = 0; i < data->coefficient_cells; i++)
        read_mpz(&reader, data->matrix[i], "matrix polynomial coefficient");

    data->expected = new_mpz_array(&reader, data->output_cells,
                                   "expected matrices");
    expect_token(&reader, "EXPECTED_MATRICES");
    for (i = 0; i < n_cells; i++) {
        long expected_prime;
        size_t j;
        expect_token(&reader, "PRIME");
        expected_prime = read_long(&reader, "expected matrix prime");
        if (expected_prime != data->primes[i])
            reader_error(&reader, "expected matrix %zu is for prime %ld, expected %ld",
                         i, expected_prime, data->primes[i]);
        for (j = 0; j < data->input_cells; j++) {
            mpz_t *entry = &data->expected[i * data->input_cells + j];
            read_mpz(&reader, *entry, "expected matrix entry");
            if (mpz_sgn(*entry) < 0 || mpz_cmp(*entry, data->moduli[i]) >= 0)
                reader_error(&reader, "expected matrix %zu entry %zu is outside [0, modulus)",
                             i, j);
        }
    }

    expect_token(&reader, "END");
    if (next_token(&reader, &token))
        reader_error(&reader, "unexpected trailing token '%s'", token);
    fclose(reader.file);
    free(reader.token);
}

static void fixture_clear(fixture *data)
{
    clear_mpz_array(data->moduli, (size_t)data->n);
    clear_mpz_array(data->initial_v, data->input_cells);
    clear_mpz_array(data->matrix, data->coefficient_cells);
    clear_mpz_array(data->expected, data->output_cells);
    free(data->primes);
    free(data->endpoints);
    free(data->case_name);
    free(data->call_name);
    mpz_clear(data->initial_z);
}

static void expected_final_v(const fixture *data, const mpz_t final_z,
                             mpz_t *result)
{
    size_t vector_cells = data->input_cells;
    mpz_t *next_v = new_mpz_array(NULL, vector_cells, "reference final V");
    mpz_t *evaluated = new_mpz_array(NULL, data->matrix_cells, "evaluated transition matrix");
    mpz_t product;
    mpz_t term;
    long value;
    int row;
    int col;

    for (size_t i = 0; i < vector_cells; i++)
        mpz_set(result[i], data->initial_v[i]);

    /* rforest returns V modulo the residual z; modulo one, every entry is zero. */
    if (mpz_cmp_ui(final_z, 1) == 0) {
        for (size_t i = 0; i < vector_cells; i++)
            mpz_set_ui(result[i], 0);
        clear_mpz_array(next_v, vector_cells);
        clear_mpz_array(evaluated, data->matrix_cells);
        return;
    }

    mpz_init(product);
    mpz_init(term);
    for (value = data->kbase; value < data->endpoints[data->n - 1]; value++) {
        for (row = 0; row < data->dim; row++) {
            for (col = 0; col < data->dim; col++) {
                size_t entry = (size_t)row * (size_t)data->dim + (size_t)col;
                mpz_t *out = &evaluated[entry];
                int power;
                mpz_set(*out, data->matrix[entry * ((size_t)data->deg + 1) +
                                           (size_t)data->deg]);
                for (power = data->deg - 1; power >= 0; power--) {
                    mpz_mul_si(*out, *out, value);
                    mpz_add(*out, *out,
                            data->matrix[entry * ((size_t)data->deg + 1) +
                                         (size_t)power]);
                }
                mpz_mod(*out, *out, final_z);
            }
        }

        for (row = 0; row < data->rows; row++) {
            for (col = 0; col < data->dim; col++) {
                int inner;
                mpz_set_ui(product, 0);
                for (inner = 0; inner < data->dim; inner++) {
                    mpz_mul(term, result[(size_t)row * (size_t)data->dim +
                                          (size_t)inner],
                            evaluated[(size_t)inner * (size_t)data->dim +
                                      (size_t)col]);
                    mpz_add(product, product, term);
                }
                mpz_mod(next_v[(size_t)row * (size_t)data->dim + (size_t)col],
                        product, final_z);
            }
        }
        for (size_t i = 0; i < vector_cells; i++)
            mpz_set(result[i], next_v[i]);
    }
    mpz_clear(product);
    mpz_clear(term);
    clear_mpz_array(next_v, vector_cells);
    clear_mpz_array(evaluated, data->matrix_cells);
}

static int mpz_arrays_equal(mpz_t *left, mpz_t *right, size_t count)
{
    size_t i;
    for (i = 0; i < count; i++)
        if (mpz_cmp(left[i], right[i]) != 0)
            return 0;
    return 1;
}

static int tree_height(long n, int kappa)
{
    int bits = 0;
    while (n > 0) {
        bits++;
        n >>= 1;
    }
    return bits > kappa ? bits - kappa : 0;
}

static void check_fixture(const char *path, const fixture *data, int kappa,
                          int disable_fft, int enforce_fft_expectation,
                          int use_workspace)
{
    mpz_t *matrix_snapshot = copy_mpz_array(data->matrix, data->coefficient_cells);
    mpz_t *moduli_snapshot = copy_mpz_array(data->moduli, (size_t)data->n);
    long *endpoints_snapshot = malloc((size_t)data->n * sizeof(*endpoints_snapshot));
    mpz_t *reference_v = new_mpz_array(NULL, data->input_cells, "reference final V");
    mpz_t working_z;
    mpz_t expected_z;
    size_t i;
    size_t j;
    int repeat;
    unsigned long fft_calls_per_run = 0;
    int previous_hw_disable_fft = hw_disable_fft;
    zz_workspace_t *workspace = use_workspace
        ? zz_workspace_create(ZZ_WORKSPACE_DEFAULT_LIMIT) : NULL;

    if (use_workspace && !workspace) {
        fputs("fixture runner: out of memory creating ZZ workspace\n", stderr);
        exit(EXIT_FAILURE);
    }

    if (!endpoints_snapshot) {
        fprintf(stderr, "fixture runner: out of memory copying endpoints\n");
        exit(EXIT_FAILURE);
    }
    memcpy(endpoints_snapshot, data->endpoints,
           (size_t)data->n * sizeof(*endpoints_snapshot));

    mpz_init_set(expected_z, data->initial_z);
    for (i = 0; i < (size_t)data->n; i++) {
        if (!mpz_divisible_p(expected_z, data->moduli[i])) {
            fprintf(stderr, "%s (%s): INITIAL_Z is not divisible by modulus %zu\n",
                    path, data->case_name, i);
            exit(EXIT_FAILURE);
        }
        mpz_divexact(expected_z, expected_z, data->moduli[i]);
    }
    expected_final_v(data, expected_z, reference_v);

    hw_disable_fft = disable_fft;
    for (repeat = 0; repeat < FIXTURE_REPEAT_RUNS; repeat++) {
        mpz_t *outputs = new_mpz_array(NULL, data->output_cells, "rforest outputs");
        mpz_t *working_v = copy_mpz_array(data->initial_v, data->input_cells);
        unsigned long fft_calls_before = fft_matrix_multiply_calls;
        unsigned long fft_calls;

        mpz_init_set(working_z, data->initial_z);
        if (workspace) {
            zz_workspace_reset_stats(workspace);
            rforest_with_workspace(workspace, outputs, working_v, data->rows,
                                   data->matrix, data->deg, data->dim,
                                   data->moduli, data->kbase, data->endpoints,
                                   data->n, working_z, kappa);
        } else {
            rforest(outputs, working_v, data->rows, data->matrix, data->deg,
                    data->dim, data->moduli, data->kbase, data->endpoints,
                    data->n, working_z, kappa);
        }
        fft_calls = fft_matrix_multiply_calls - fft_calls_before;

        if (repeat == 0)
            fft_calls_per_run = fft_calls;
        else if (fft_calls != fft_calls_per_run) {
            fprintf(stderr, "%s (%s): repeat %d changed FFT matrix multiply count from %lu to %lu\n",
                    path, data->case_name, repeat + 1, fft_calls_per_run, fft_calls);
            exit(EXIT_FAILURE);
        }

        if (enforce_fft_expectation && data->expect_fft_matrix_mul && fft_calls == 0) {
            fprintf(stderr,
                    "%s (%s): expected natural mpz_rmatrix_mult_fft dispatch, observed none\n",
                    path, data->case_name);
            exit(EXIT_FAILURE);
        }
        if (disable_fft && fft_calls != 0) {
            fprintf(stderr,
                    "%s (%s): hw_disable_fft set but observed %lu matrix FFT multiplies\n",
                    path, data->case_name, fft_calls);
            exit(EXIT_FAILURE);
        }
        if (workspace) {
            zz_workspace_stats_t stats;
            zz_workspace_get_stats(workspace, &stats);
            if (stats.live_bytes != 0) {
                fprintf(stderr, "%s (%s): workspace retained live bytes after rforest\n",
                        path, data->case_name);
                exit(EXIT_FAILURE);
            }
            if (enforce_fft_expectation && data->expect_fft_matrix_mul &&
                fft_calls > 0 && stats.cache_hits == 0) {
                fprintf(stderr,
                        "%s (%s): FFT-dispatched workspace run recorded no cache reuse\n",
                        path, data->case_name);
                exit(EXIT_FAILURE);
            }
        }

        for (i = 0; i < (size_t)data->n; i++) {
            for (j = 0; j < data->input_cells; j++) {
                size_t offset = i * data->input_cells + j;
                size_t row = j / (size_t)data->dim;
                size_t col = j % (size_t)data->dim;
                if (mpz_sgn(outputs[offset]) < 0 ||
                    mpz_cmp(outputs[offset], data->moduli[i]) >= 0) {
                    fprintf(stderr, "%s (%s, %s): output at modulus[%zu] is not a canonical residue\n",
                            path, data->case_name, data->call_name, i);
                    exit(EXIT_FAILURE);
                }
                if (mpz_cmp(outputs[offset], data->expected[offset]) != 0) {
                    fprintf(stderr, "%s (%s, %s): output mismatch at modulus[%zu] ",
                            path, data->case_name, data->call_name, i);
                    gmp_fprintf(stderr, "prime %ld (mod %Zd), matrix coordinate (%zu,%zu): expected %Zd, got %Zd\n",
                                data->primes[i], data->moduli[i], row, col,
                                data->expected[offset], outputs[offset]);
                    exit(EXIT_FAILURE);
                }
            }
        }

        if (mpz_cmp(working_z, expected_z) != 0) {
            fprintf(stderr, "%s (%s): final z mismatch: expected ", path, data->case_name);
            gmp_fprintf(stderr, "%Zd, got %Zd\n", expected_z, working_z);
            exit(EXIT_FAILURE);
        }
        for (j = 0; j < data->input_cells; j++) {
            size_t row = j / (size_t)data->dim;
            size_t col = j % (size_t)data->dim;
            if (mpz_cmp(working_v[j], reference_v[j]) != 0) {
                fprintf(stderr, "%s (%s): final V mismatch at coordinate (%zu,%zu): expected ",
                        path, data->case_name, row, col);
                gmp_fprintf(stderr, "%Zd, got %Zd with final z %Zd\n",
                            reference_v[j], working_v[j], expected_z);
                exit(EXIT_FAILURE);
            }
        }

        if (!mpz_arrays_equal(data->matrix, matrix_snapshot, data->coefficient_cells) ||
            !mpz_arrays_equal(data->moduli, moduli_snapshot, (size_t)data->n) ||
            memcmp(data->endpoints, endpoints_snapshot,
                   (size_t)data->n * sizeof(*endpoints_snapshot)) != 0) {
            fprintf(stderr, "%s (%s): rforest mutated reusable fixture inputs\n",
                    path, data->case_name);
            exit(EXIT_FAILURE);
        }

        mpz_clear(working_z);
        clear_mpz_array(outputs, data->output_cells);
        clear_mpz_array(working_v, data->input_cells);
    }

    hw_disable_fft = previous_hw_disable_fft;
    mpz_clear(expected_z);
    clear_mpz_array(reference_v, data->input_cells);
    clear_mpz_array(matrix_snapshot, data->coefficient_cells);
    clear_mpz_array(moduli_snapshot, (size_t)data->n);
    free(endpoints_snapshot);

    if (workspace) {
        zz_workspace_stats_t stats;
        zz_workspace_get_stats(workspace, &stats);
        if (stats.live_bytes != 0) {
            fprintf(stderr, "%s (%s): workspace still has live bytes before destroy\n",
                    path, data->case_name);
            exit(EXIT_FAILURE);
        }
        if (enforce_fft_expectation && data->expect_fft_matrix_mul &&
            !disable_fft && stats.retained_bytes == 0) {
            fprintf(stderr, "%s (%s): FFT workspace retained no reusable storage\n",
                    path, data->case_name);
            exit(EXIT_FAILURE);
        }
        if (zz_workspace_trim(workspace, 0) != 0) {
            fprintf(stderr, "%s (%s): workspace trim did not release retained storage\n",
                    path, data->case_name);
            exit(EXIT_FAILURE);
        }
        zz_workspace_destroy(workspace);
    }

    printf("PASS %s (%s, %s): %ld matrices, dim=%d rows=%d, kappa=%d, tree height=%d, repeats=%d, FFT matrix multiplies/run=%lu%s%s\n",
           path, data->case_name, data->call_name, data->n, data->dim, data->rows,
           kappa, tree_height(data->n, kappa), FIXTURE_REPEAT_RUNS,
           fft_calls_per_run, disable_fft ? " (FFT disabled)" : "",
           use_workspace ? " (workspace)" : "");
}

static int parse_kappa(const char *text)
{
    char *end = NULL;
    long value;

    errno = 0;
    value = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value < 0 || value > INT_MAX) {
        fprintf(stderr, "fixture runner: invalid nonnegative kappa '%s'\n", text);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

int main(int argc, char **argv)
{
    int arg;
    int have_fixture = 0;
    int disable_fft = 0;
    int use_workspace = 0;
    int kappa_override_set = 0;
    int kappa_override = 0;
    int fixture_count = 0;

    for (arg = 1; arg < argc; arg++) {
        if (strcmp(argv[arg], "--disable-fft") == 0) {
            disable_fft = 1;
        } else if (strcmp(argv[arg], "--workspace") == 0) {
            use_workspace = 1;
        } else if (strcmp(argv[arg], "--kappa") == 0) {
            if (++arg >= argc) {
                fprintf(stderr, "fixture runner: --kappa requires a value\n");
                return EXIT_FAILURE;
            }
            if (kappa_override_set) {
                fprintf(stderr, "fixture runner: --kappa may be specified only once\n");
                return EXIT_FAILURE;
            }
            kappa_override = parse_kappa(argv[arg]);
            kappa_override_set = 1;
        } else if (argv[arg][0] == '-') {
            fprintf(stderr, "fixture runner: unknown option '%s'\n", argv[arg]);
            return EXIT_FAILURE;
        } else {
            have_fixture = 1;
        }
    }

    if (!have_fixture) {
        fprintf(stderr, "usage: %s [--disable-fft] [--workspace] [--kappa K] fixture.rf [fixture.rf ...]\n", argv[0]);
        return EXIT_FAILURE;
    }

    for (arg = 1; arg < argc; arg++) {
        fixture data;
        int kappa;
        if (strcmp(argv[arg], "--disable-fft") == 0)
            continue;
        if (strcmp(argv[arg], "--workspace") == 0)
            continue;
        if (strcmp(argv[arg], "--kappa") == 0) {
            arg++;
            continue;
        }
        fixture_read(argv[arg], &data);
        kappa = kappa_override_set ? kappa_override : data.kappa;
        check_fixture(argv[arg], &data, kappa, disable_fft,
                      !disable_fft && !kappa_override_set, use_workspace);
        fixture_clear(&data);
        fixture_count++;
    }

    printf("PASS all %d native rforest fixture(s)\n", fixture_count);
    return EXIT_SUCCESS;
}
