#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <gmp.h>
#include "mpzfft.h"
#include "zzmem.h"
#include "hwmem.h"
#include "hwmpz.h"

#define HW_ZZ_PRIMES            4

int hw_disable_fft;
int mpzfft_threads = 1; // setting this to a value other than 1 is typically not all that helpful (better to parallelize at a higher level)

static zz_moduli_t zz_moduli;
static int mpzfft_initialized;

static inline void hw_mpzfft_setup ()
{
    if ( hw_disable_fft || mpzfft_initialized ) return;
    zz_moduli_init (&zz_moduli, ZZ_MAX_PRIMES);
#if HW_MEM_TRACKING
    extern unsigned long zz_overhead;
    zz_overhead = zz_mem_peak;
#endif
    mpzfft_initialized = 1;
}

static inline void hw_mpzfft_clear ()
{
    if ( hw_disable_fft || ! mpzfft_initialized ) return;
    zz_moduli_clear (&zz_moduli);
    mpzfft_initialized = 0;
}

void hw_mpz_setup (void) { hw_mpzfft_setup ();}
void hw_mpz_clear (void) { hw_mpzfft_clear (); }

static int mpz_rmatrix_size_mul(size_t left, size_t right, size_t *product);
static int mpz_rmatrix_size_add(size_t left, size_t right, size_t *sum);
static void mpz_rmatrix_bad_dimensions(void);
static void mpz_rmatrix_pn_sizes(int r, int d, int n,
                                 size_t *acells, size_t *bcells,
                                 size_t *left_count, size_t *right_count,
                                 size_t *terms);
static int mpz_rmatrix_fft_shape_safe(int r, int d);
static int mpz_rmatrix_sampled_small(mpz_t *A, size_t acells,
                                     mpz_t *B, size_t bcells,
                                     size_t planes, size_t threshold);

mpz_t *mpz_vec_mod_fft (mpz_t *A, mpz_t *B, long n, mpz_t m)
{
    mpzfft_mod_t mod;
    long b, c;

    assert ( mpzfft_initialized );
    b = mpz_sizeinbase (m, 2);
    for ( long i = 0 ; i < n ; i++ ) { c = mpz_sizeinbase (B[i], 2);  if ( c > b ) b = c; }
    mpzfft_mod_init (&mod, b, m, HW_ZZ_PRIMES, &zz_moduli, mpzfft_threads);
    for ( long i = 0 ; i < n ; i++ ) { mpzfft_mod_mod (&mod, A[i], B[i], 1); }
    mpzfft_mod_clear (&mod);
    return A;
}

mpz_t *mpz_vec_mod_init_fft (mpz_t *A, mpz_t *B, long n, mpz_t m, mpz_t w)
{
    mpzfft_mod_t mod;
    long b, c;

    assert ( mpzfft_initialized );
    b = mpz_sizeinbase (m, 2);
    for ( long i = 0 ; i < n ; i++ ) { c = mpz_sizeinbase (B[i], 2);  if ( c > b ) b = c; }
    mpzfft_mod_init (&mod, b, m, HW_ZZ_PRIMES, &zz_moduli, mpzfft_threads);
    for ( long i = 0 ; i < n ; i++ ) { mpzfft_mod_mod (&mod, w, B[i], 1);  mpz_init_set (A[i], w); }
    mpzfft_mod_clear (&mod);
    return A;
}

static inline size_t mpz_rmatrix_product_max_bits (mpz_t *A, int r, mpz_t *B, int d)
{
    register int i;
    register size_t m, n;
    
    m = 0;
    for ( i = 0 ; i < r*d ; i++ ) { n = mpz_size(A[i]); if ( n > m ) m = n; }
    for ( i = 0 ; i < d*d ; i++ ) { n = mpz_size(B[i]); if ( n > m ) m = n; }
    return GMP_NUMB_BITS*(2*m+1);   // note that the +1 limb more than covers the extra log(d) bits due to additions
}

mpz_t *mpz_rmatrix_mult_fft (mpz_t *C, mpz_t *A, int r, mpz_t *B, int d, mpz_t w)
{
    mpzfft_params_t params;
    mpzfft_t *AT, *BT;
    assert ( mpzfft_initialized );
    
    mpzfft_params_init (&params, mpz_rmatrix_product_max_bits (A, r, B, d), d, HW_ZZ_PRIMES, &zz_moduli);

    // transform input matrices
    AT = hw_malloc (r*d*sizeof(mpzfft_t));
    for ( int i = 0 ; i < r*d; i++) { mpzfft_init(AT[i], &params);  mpzfft_fft (AT[i], A[i], mpzfft_threads); }
    BT = hw_malloc (d*d*sizeof(mpzfft_t));
    for ( int i = 0 ; i < d*d; i++) { mpzfft_init(BT[i], &params);  mpzfft_fft (BT[i], B[i], mpzfft_threads); }

    // multiply matrices of Fourier coefficients
    mpzfft_matrix_mul(AT, AT, BT, r, d, d, mpzfft_threads);

    // inverse transform results and cleanup
    for ( int i = 0 ; i < d*d ; i++) { mpzfft_clear (BT[i]); }
    for ( int i = 0 ; i < r*d ; i++) { mpzfft_ifft (C[i], AT[i], mpzfft_threads); mpzfft_clear (AT[i]); }

    hw_free (AT, r*d*sizeof(mpzfft_t));
    hw_free (BT, d*d*sizeof(mpzfft_t));

    mpzfft_params_clear (&params);
    return C;
}

static void mpz_rmatrix_mult_p2_naive(mpz_t *C, mpz_t *A, int r,
                                     mpz_t *B, int d, mpz_t w)
{
    size_t acells = (size_t)r * (size_t)d;
    size_t bcells = (size_t)d * (size_t)d;

    for (int i = 0; i < r; i++) {
        size_t row = (size_t)i * (size_t)d;
        for (int j = 0; j < d; j++) {
            size_t out = row + (size_t)j;
            mpz_mul(C[out], A[row], B[j]);
            mpz_mul(C[acells + out], A[row], B[bcells + (size_t)j]);
            mpz_addmul(C[acells + out], A[acells + row], B[j]);
        }
        for (int k = 1; k < d; k++) {
            size_t a0 = row + (size_t)k;
            size_t a1 = acells + a0;
            size_t b0 = (size_t)k * (size_t)d;
            size_t b1 = bcells + b0;
            for (int j = 0; j < d; j++) {
                size_t out = row + (size_t)j;
                mpz_addmul(C[out], A[a0], B[b0 + (size_t)j]);
                mpz_addmul(C[acells + out], A[a0], B[b1 + (size_t)j]);
                mpz_addmul(C[acells + out], A[a1], B[b0 + (size_t)j]);
            }
        }
    }
    (void)w;
}

static int mpz_rmatrix_p2_product_bits(mpz_t *A, size_t acells,
                                      mpz_t *B, size_t bcells, int d,
                                      size_t *bits)
{
    size_t max_limbs = 0;
    size_t terms = (size_t)2 * (size_t)d;
    size_t sum_bits = 0;
    size_t limb_bits;

    for (size_t i = 0; i < 2 * acells; i++) {
        size_t limbs = mpz_size(A[i]);
        if (limbs > max_limbs)
            max_limbs = limbs;
    }
    for (size_t i = 0; i < 2 * bcells; i++) {
        size_t limbs = mpz_size(B[i]);
        if (limbs > max_limbs)
            max_limbs = limbs;
    }

    /* ceil(log2(2*d)) bounds the complete P coefficient's signed sum. */
    for (size_t count = terms; count > 1; count = count / 2 + count % 2)
        sum_bits++;
    if (!mpz_rmatrix_size_mul(max_limbs, 2, &max_limbs) ||
        !mpz_rmatrix_size_mul(max_limbs, (size_t)GMP_NUMB_BITS,
                              &limb_bits) ||
        !mpz_rmatrix_size_add(limb_bits, sum_bits, bits))
        return 0;
    return *bits > 0 && *bits <= SIZE_MAX - 2;
}

static void mpz_rmatrix_sample_max_limbs(mpz_t *values, size_t plane_cells,
                                         size_t planes, size_t *max_limbs)
{
    for (size_t plane = 0; plane < planes; plane++) {
        size_t base = plane * plane_cells;
        size_t samples[3] = {base, base + plane_cells / 2,
                             base + plane_cells - 1};
        for (size_t i = 0; i < 3; i++) {
            size_t limbs = mpz_size(values[samples[i]]);
            if (limbs > *max_limbs)
                *max_limbs = limbs;
        }
    }
}

static int mpz_rmatrix_sampled_small(mpz_t *A, size_t acells,
                                     mpz_t *B, size_t bcells,
                                     size_t planes, size_t threshold)
{
    size_t acount;
    size_t bcount;
    size_t input_count;
    size_t max_limbs = 0;
    size_t estimated_total;

    /* Small samples skip a full input scan; a missed large sparse entry only
       selects the exact classical path and cannot weaken the FFT bound. */
    if (!mpz_rmatrix_size_mul(planes, acells, &acount) ||
        !mpz_rmatrix_size_mul(planes, bcells, &bcount) ||
        !mpz_rmatrix_size_add(acount, bcount, &input_count))
        return 0;
    mpz_rmatrix_sample_max_limbs(A, acells, planes, &max_limbs);
    mpz_rmatrix_sample_max_limbs(B, bcells, planes, &max_limbs);
    if (max_limbs == 0)
        return 1;
    if (!mpz_rmatrix_size_mul(max_limbs, input_count, &estimated_total))
        return 0;
    return estimated_total <= threshold;
}

static int mpz_rmatrix_p2_use_fft(mpz_t *A, size_t acells, mpz_t *B,
                                  size_t bcells, int r, int d)
{
    size_t total_limbs = 0;
    size_t crossover;
    size_t rows_plus_dim;
    size_t matrix_scale;
    size_t threshold;
    long tuned_crossover;

    if (hw_disable_fft || d <= 2 || !mpz_rmatrix_fft_shape_safe(r, d) ||
        acells > SIZE_MAX / (2 * sizeof(mpzfft_t)) ||
        bcells > SIZE_MAX / (2 * sizeof(mpzfft_t)))
        return 0;

    tuned_crossover = mpz_mat_fft_crossover(d);
    if (tuned_crossover <= 0)
        return 0;
    crossover = (size_t)tuned_crossover;
    if (!mpz_rmatrix_size_add((size_t)r, (size_t)d, &rows_plus_dim) ||
        !mpz_rmatrix_size_mul(rows_plus_dim, (size_t)d, &matrix_scale) ||
        !mpz_rmatrix_size_mul(matrix_scale, crossover, &threshold))
        return 0;
    if (mpz_rmatrix_sampled_small(A, acells, B, bcells, 2, threshold))
        return 0;

    for (size_t i = 0; i < 2 * acells; i++) {
        size_t limbs = mpz_size(A[i]);
        if (total_limbs > SIZE_MAX - limbs)
            total_limbs = SIZE_MAX;
        else
            total_limbs += limbs;
    }
    for (size_t i = 0; i < 2 * bcells; i++) {
        size_t limbs = mpz_size(B[i]);
        if (total_limbs > SIZE_MAX - limbs)
            total_limbs = SIZE_MAX;
        else
            total_limbs += limbs;
    }
    return total_limbs > threshold;
}

static mpz_t *mpz_rmatrix_mult_p2_fft(mpz_t *C, mpz_t *A, int r,
                                      mpz_t *B, int d, size_t bits)
{
    mpzfft_params_t params;
    mpzfft_t *AT, *BT, *CT;
    size_t acells = (size_t)r * (size_t)d;
    size_t bcells = (size_t)d * (size_t)d;

    assert(mpzfft_initialized);
    mpzfft_params_init(&params, bits, 2 * (size_t)d, HW_ZZ_PRIMES,
                       &zz_moduli);

    AT = hw_malloc(2 * acells * sizeof(mpzfft_t));
    for (size_t i = 0; i < 2 * acells; i++) {
        mpzfft_init(AT[i], &params);
        mpzfft_fft(AT[i], A[i], mpzfft_threads);
    }
    BT = hw_malloc(2 * bcells * sizeof(mpzfft_t));
    for (size_t i = 0; i < 2 * bcells; i++) {
        mpzfft_init(BT[i], &params);
        mpzfft_fft(BT[i], B[i], mpzfft_threads);
    }
    CT = hw_malloc(2 * acells * sizeof(mpzfft_t));
    for (size_t i = 0; i < 2 * acells; i++)
        mpzfft_init(CT[i], &params);

    /* C0=A0*B0. The P coefficient is accumulated before one inverse
       transform per output entry. The A1 transform array becomes scratch
       only after its sole matrix product has consumed it. */
    mpzfft_matrix_mul(CT, AT, BT, (unsigned)r, (unsigned)d, (unsigned)d,
                      mpzfft_threads);
    mpzfft_matrix_mul(CT + acells, AT, BT + bcells,
                      (unsigned)r, (unsigned)d, (unsigned)d, mpzfft_threads);
    mpzfft_matrix_mul(AT + acells, AT + acells, BT,
                      (unsigned)r, (unsigned)d, (unsigned)d, mpzfft_threads);
    for (size_t i = 0; i < acells; i++)
        mpzfft_add(CT[acells + i], CT[acells + i], AT[acells + i],
                   mpzfft_threads);

    for (size_t i = 0; i < acells; i++)
        mpzfft_ifft(C[i], CT[i], mpzfft_threads);
    for (size_t i = 0; i < acells; i++)
        mpzfft_ifft(C[acells + i], CT[acells + i], mpzfft_threads);

    for (size_t i = 0; i < 2 * acells; i++)
        mpzfft_clear(CT[i]);
    for (size_t i = 0; i < 2 * bcells; i++)
        mpzfft_clear(BT[i]);
    for (size_t i = 0; i < 2 * acells; i++)
        mpzfft_clear(AT[i]);
    hw_free(CT, 2 * acells * sizeof(mpzfft_t));
    hw_free(BT, 2 * bcells * sizeof(mpzfft_t));
    hw_free(AT, 2 * acells * sizeof(mpzfft_t));
    mpzfft_params_clear(&params);
    return C;
}

mpz_t *mpz_rmatrix_mult_p2(mpz_t *C, mpz_t *A, int r, mpz_t *B,
                           int d, mpz_t w)
{
    size_t acells, bcells, left_count, right_count, terms;

    mpz_rmatrix_pn_sizes(r, d, 2, &acells, &bcells, &left_count,
                         &right_count, &terms);
    if (mpz_rmatrix_p2_use_fft(A, acells, B, bcells, r, d)) {
        size_t bits;
        if (mpz_rmatrix_p2_product_bits(A, acells, B, bcells, d, &bits))
            return mpz_rmatrix_mult_p2_fft(C, A, r, B, d, bits);
    }
    mpz_rmatrix_mult_p2_naive(C, A, r, B, d, w);
    (void)left_count;
    (void)right_count;
    (void)terms;
    return C;
}

static int mpz_rmatrix_size_mul(size_t left, size_t right, size_t *product)
{
    if (right && left > SIZE_MAX / right)
        return 0;
    *product = left * right;
    return 1;
}

static int mpz_rmatrix_size_add(size_t left, size_t right, size_t *sum)
{
    if (left > SIZE_MAX - right)
        return 0;
    *sum = left + right;
    return 1;
}

static int mpz_rmatrix_fft_shape_safe(int r, int d)
{
    size_t acells;
    size_t bcells;
    size_t total_cells;

    if (r <= 0 || d <= 0 ||
        !mpz_rmatrix_size_mul((size_t)r, (size_t)d, &acells) ||
        !mpz_rmatrix_size_mul((size_t)d, (size_t)d, &bcells) ||
        !mpz_rmatrix_size_add(acells, bcells, &total_cells) ||
        !mpz_rmatrix_size_add(total_cells, acells, &total_cells))
        return 0;
    return total_cells <= UINT_MAX;
}

static void mpz_rmatrix_bad_dimensions(void)
{
    fputs("ring matrix multiplication: invalid or unrepresentable dimensions\n",
          stderr);
    abort();
}

static void mpz_rmatrix_pn_sizes(int r, int d, int n,
                                 size_t *acells, size_t *bcells,
                                 size_t *left_count, size_t *right_count,
                                 size_t *terms)
{
    if (r <= 0 || d <= 0 || n <= 0 ||
        !mpz_rmatrix_size_mul((size_t)r, (size_t)d, acells) ||
        !mpz_rmatrix_size_mul((size_t)d, (size_t)d, bcells) ||
        *acells > INT_MAX || *bcells > INT_MAX ||
        !mpz_rmatrix_size_mul((size_t)n, *acells, left_count) ||
        !mpz_rmatrix_size_mul((size_t)n, *bcells, right_count) ||
        !mpz_rmatrix_size_mul((size_t)n, (size_t)d, terms) ||
        *left_count > SIZE_MAX / sizeof(mpz_t) ||
        *right_count > SIZE_MAX / sizeof(mpz_t))
        mpz_rmatrix_bad_dimensions();
}

static size_t mpz_rmatrix_pn_max_limbs(mpz_t *A, size_t acount,
                                       mpz_t *B, size_t bcount)
{
    size_t max_limbs = 0;

    for (size_t i = 0; i < acount; i++) {
        size_t limbs = mpz_size(A[i]);
        if (limbs > max_limbs)
            max_limbs = limbs;
    }
    for (size_t i = 0; i < bcount; i++) {
        size_t limbs = mpz_size(B[i]);
        if (limbs > max_limbs)
            max_limbs = limbs;
    }
    return max_limbs;
}

static int mpz_rmatrix_pn_product_bits(mpz_t *A, size_t acount,
                                       mpz_t *B, size_t bcount,
                                       size_t terms, size_t *bits)
{
    size_t max_limbs = mpz_rmatrix_pn_max_limbs(A, acount, B, bcount);
    size_t sum_bits = 0;
    size_t limb_bits;

    /* A coefficient has at most n*d signed scalar products. */
    for (size_t count = terms; count > 1; count = count / 2 + count % 2)
        sum_bits++;
    if (!mpz_rmatrix_size_mul(max_limbs, 2, &max_limbs) ||
        !mpz_rmatrix_size_mul(max_limbs, (size_t)GMP_NUMB_BITS,
                              &limb_bits) ||
        !mpz_rmatrix_size_add(limb_bits, sum_bits, bits))
        return 0;
    return *bits > 0 && *bits <= SIZE_MAX - 2;
}

static void mpz_rmatrix_pn_add_limbs(size_t *total, size_t limbs)
{
    if (*total > SIZE_MAX - limbs)
        *total = SIZE_MAX;
    else
        *total += limbs;
}

static int mpz_rmatrix_pn_scalar_dispatch_safe(int r, int d)
{
    size_t rows_plus_dim;
    size_t matrix_scale;
    size_t threshold;
    long tuned_crossover;

    if (d == 1)
        return 1;
    if (!mpz_rmatrix_fft_shape_safe(r, d))
        return 0;
    tuned_crossover = mpz_mat_fft_crossover(d);
    if (tuned_crossover <= 0 ||
        !mpz_rmatrix_size_add((size_t)r, (size_t)d, &rows_plus_dim) ||
        !mpz_rmatrix_size_mul(rows_plus_dim, (size_t)d, &matrix_scale) ||
        !mpz_rmatrix_size_mul(matrix_scale, (size_t)tuned_crossover,
                              &threshold))
        return 0;
    return threshold <= (size_t)LONG_MAX;
}

static int mpz_rmatrix_pn_scalar_inputs_safe(mpz_t *A, size_t acount,
                                             mpz_t *B, size_t bcount)
{
    size_t total_limbs = 0;
    size_t max_limbs = 0;
    const size_t max_product_limbs =
        (SIZE_MAX / (size_t)GMP_NUMB_BITS - 1) / 2;

    for (size_t i = 0; i < acount; i++) {
        size_t limbs = mpz_size(A[i]);
        if (limbs > (size_t)LONG_MAX ||
            total_limbs > (size_t)LONG_MAX - limbs)
            return 0;
        total_limbs += limbs;
        if (limbs > max_limbs)
            max_limbs = limbs;
    }
    for (size_t i = 0; i < bcount; i++) {
        size_t limbs = mpz_size(B[i]);
        if (limbs > (size_t)LONG_MAX ||
            total_limbs > (size_t)LONG_MAX - limbs)
            return 0;
        total_limbs += limbs;
        if (limbs > max_limbs)
            max_limbs = limbs;
    }
    return max_limbs <= max_product_limbs;
}

static int mpz_rmatrix_pn_use_fft(mpz_t *A, size_t acount,
                                  mpz_t *B, size_t bcount,
                                  size_t acells, size_t bcells,
                                  size_t planes, int r, int d,
                                  size_t terms)
{
    size_t total_limbs = 0;
    size_t rows_plus_dim;
    size_t matrix_scale;
    size_t threshold;
    size_t crossover;
    long tuned_crossover;

    if (hw_disable_fft || d <= 2 || !mpz_rmatrix_fft_shape_safe(r, d) ||
        terms > UINT_MAX ||
        acount > SIZE_MAX / sizeof(mpzfft_t) ||
        bcount > SIZE_MAX / sizeof(mpzfft_t))
        return 0;

    tuned_crossover = mpz_mat_fft_crossover(d);
    if (tuned_crossover <= 0)
        return 0;
    crossover = (size_t)tuned_crossover;
    if (!mpz_rmatrix_size_add((size_t)r, (size_t)d, &rows_plus_dim) ||
        !mpz_rmatrix_size_mul(rows_plus_dim, (size_t)d, &matrix_scale) ||
        !mpz_rmatrix_size_mul(matrix_scale, crossover, &threshold))
        return 0;
    if (mpz_rmatrix_sampled_small(A, acells, B, bcells, planes, threshold))
        return 0;

    for (size_t i = 0; i < acount; i++)
        mpz_rmatrix_pn_add_limbs(&total_limbs, mpz_size(A[i]));
    for (size_t i = 0; i < bcount; i++)
        mpz_rmatrix_pn_add_limbs(&total_limbs, mpz_size(B[i]));
    return total_limbs > threshold;
}

static void *mpz_rmatrix_workspace_alloc(size_t bytes)
{
    void *memory = hw_malloc(bytes);
    if (!memory) {
        fputs("ring matrix multiplication: workspace allocation failed\n",
              stderr);
        abort();
    }
    return memory;
}

static mpz_t *mpz_rmatrix_mult_pn_fft(mpz_t *C, mpz_t *A, int r,
                                      mpz_t *B, int d, int n,
                                      size_t acells, size_t left_count,
                                      size_t right_count, size_t terms,
                                      size_t bits)
{
    mpzfft_params_t params;
    mpzfft_t *AT, *BT, *CT, *product;
    size_t left_bytes = left_count * sizeof(mpzfft_t);
    size_t right_bytes = right_count * sizeof(mpzfft_t);
    size_t product_bytes = acells * sizeof(mpzfft_t);

    assert(mpzfft_initialized);
    assert(terms <= UINT_MAX);
    mpzfft_params_init(&params, bits, (unsigned)terms, HW_ZZ_PRIMES,
                       &zz_moduli);

    AT = mpz_rmatrix_workspace_alloc(left_bytes);
    for (size_t i = 0; i < left_count; i++) {
        mpzfft_init(AT[i], &params);
        mpzfft_fft(AT[i], A[i], mpzfft_threads);
    }
    BT = mpz_rmatrix_workspace_alloc(right_bytes);
    for (size_t i = 0; i < right_count; i++) {
        mpzfft_init(BT[i], &params);
        mpzfft_fft(BT[i], B[i], mpzfft_threads);
    }
    CT = mpz_rmatrix_workspace_alloc(left_bytes);
    for (size_t i = 0; i < left_count; i++)
        mpzfft_init(CT[i], &params);
    product = mpz_rmatrix_workspace_alloc(product_bytes);
    for (size_t i = 0; i < acells; i++)
        mpzfft_init(product[i], &params);

    for (int ai = 0; ai < n; ai++) {
        for (int bi = 0; bi < n - ai; bi++) {
            size_t output_offset = (size_t)(ai + bi) * acells;
            mpzfft_matrix_mul(product, AT + (size_t)ai * acells,
                              BT + (size_t)bi * (size_t)d * (size_t)d,
                              (unsigned)r, (unsigned)d, (unsigned)d,
                              mpzfft_threads);
            for (size_t i = 0; i < acells; i++)
                mpzfft_add(CT[output_offset + i], CT[output_offset + i],
                           product[i], mpzfft_threads);
        }
    }

    for (size_t i = 0; i < left_count; i++)
        mpzfft_ifft(C[i], CT[i], mpzfft_threads);

    for (size_t i = 0; i < acells; i++)
        mpzfft_clear(product[i]);
    for (size_t i = 0; i < left_count; i++)
        mpzfft_clear(CT[i]);
    for (size_t i = 0; i < right_count; i++)
        mpzfft_clear(BT[i]);
    for (size_t i = 0; i < left_count; i++)
        mpzfft_clear(AT[i]);
    hw_free(product, product_bytes);
    hw_free(CT, left_bytes);
    hw_free(BT, right_bytes);
    hw_free(AT, left_bytes);
    mpzfft_params_clear(&params);
    return C;
}

static void mpz_rmatrix_mult_pn_classical(mpz_t *C, mpz_t *A, int r,
                                          mpz_t *B, int d, int n,
                                          size_t acells, size_t bcells,
                                          mpz_t w)
{
    for (int ai = 0; ai < n; ai++) {
        size_t a_offset = (size_t)ai * acells;
        for (int bi = 0; bi < n - ai; bi++) {
            size_t b_offset = (size_t)bi * bcells;
            size_t c_offset = (size_t)(ai + bi) * acells;
            int first_inner = ai == 0 ? 1 : 0;
            if (ai == 0) {
                for (int row = 0; row < r; row++) {
                    size_t left = a_offset + (size_t)row * (size_t)d;
                    for (int col = 0; col < d; col++) {
                        size_t out = c_offset + (size_t)row * (size_t)d +
                                     (size_t)col;
                        size_t right = b_offset + (size_t)col;
                        mpz_mul(C[out], A[left], B[right]);
                    }
                }
            }
            for (int inner = first_inner; inner < d; inner++) {
                size_t right = b_offset + (size_t)inner * (size_t)d;
                for (int row = 0; row < r; row++) {
                    size_t left = a_offset + (size_t)row * (size_t)d +
                                  (size_t)inner;
                    size_t out_row = c_offset + (size_t)row * (size_t)d;
                    for (int col = 0; col < d; col++) {
                        mpz_addmul(C[out_row + (size_t)col], A[left],
                                   B[right + (size_t)col]);
                    }
                }
            }
        }
    }
    (void)w;
}

mpz_t *mpz_rmatrix_mult_pn(mpz_t *C, mpz_t *A, int r, mpz_t *B,
                           int d, int n, mpz_t w)
{
    size_t acells, bcells, left_count, right_count, terms;

    mpz_rmatrix_pn_sizes(r, d, n, &acells, &bcells, &left_count,
                         &right_count, &terms);
    if (n == 1) {
        if (d == 1 || left_count > SIZE_MAX / sizeof(mpzfft_t) ||
            right_count > SIZE_MAX / sizeof(mpzfft_t) ||
            !mpz_rmatrix_pn_scalar_dispatch_safe(r, d) ||
            (!hw_disable_fft &&
             !mpz_rmatrix_pn_scalar_inputs_safe(A, left_count, B,
                                                right_count))) {
            mpz_rmatrix_mult_naive(C, A, r, B, d, w);
            return C;
        }
        mpz_rmatrix_mult(C, A, r, B, d, w);
        return C;
    }
    if (n == 2)
        return mpz_rmatrix_mult_p2(C, A, r, B, d, w);

    if (mpz_rmatrix_pn_use_fft(A, left_count, B, right_count, acells, bcells,
                               (size_t)n, r, d, terms)) {
        size_t bits;
        if (mpz_rmatrix_pn_product_bits(A, left_count, B, right_count,
                                        terms, &bits))
            return mpz_rmatrix_mult_pn_fft(C, A, r, B, d, n, acells,
                                            left_count, right_count, terms,
                                            bits);
    }
    mpz_rmatrix_mult_pn_classical(C, A, r, B, d, n, acells, bcells, w);
    return C;
}
