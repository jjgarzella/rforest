#include <assert.h>
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
        for (int j = 0; j < d; j++) {
            size_t out = (size_t)i * (size_t)d + (size_t)j;
            mpz_set_zero(C[out]);
            mpz_set_zero(C[acells + out]);
            for (int k = 0; k < d; k++) {
                mpz_addmul(C[out], A[(size_t)i * (size_t)d + (size_t)k],
                           B[(size_t)k * (size_t)d + (size_t)j]);
                mpz_addmul(C[acells + out],
                           A[(size_t)i * (size_t)d + (size_t)k],
                           B[bcells + (size_t)k * (size_t)d + (size_t)j]);
                mpz_addmul(C[acells + out],
                           A[acells + (size_t)i * (size_t)d + (size_t)k],
                           B[(size_t)k * (size_t)d + (size_t)j]);
            }
        }
    }
    (void)w;
}

static size_t mpz_rmatrix_p2_product_bits(mpz_t *A, size_t acells,
                                         mpz_t *B, size_t bcells, int d)
{
    size_t max_limbs = 0;
    size_t terms = (size_t)2 * (size_t)d;
    size_t sum_bits = 0;

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
    while (terms > 1) {
        terms = terms / 2 + terms % 2;
        sum_bits++;
    }
    assert(max_limbs <= (SIZE_MAX - sum_bits) /
                        (2 * (size_t)GMP_NUMB_BITS));
    return 2 * max_limbs * (size_t)GMP_NUMB_BITS + sum_bits;
}

static int mpz_rmatrix_p2_use_fft(mpz_t *A, size_t acells, mpz_t *B,
                                  size_t bcells, int r, int d)
{
    size_t total_limbs = 0;
    size_t crossover;

    if (hw_disable_fft || d <= 2)
        return 0;
    for (size_t i = 0; i < 2 * acells; i++)
        total_limbs += mpz_size(A[i]);
    for (size_t i = 0; i < 2 * bcells; i++)
        total_limbs += mpz_size(B[i]);

    crossover = ((size_t)r + (size_t)d) * (size_t)d *
                (size_t)mpz_mat_fft_crossover(d);
    return total_limbs > crossover;
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
    size_t acells = (size_t)r * (size_t)d;
    size_t bcells = (size_t)d * (size_t)d;

    assert(r > 0 && d > 0);
    if (mpz_rmatrix_p2_use_fft(A, acells, B, bcells, r, d)) {
        size_t bits = mpz_rmatrix_p2_product_bits(A, acells, B, bcells, d);
        if (bits > 0)
            return mpz_rmatrix_mult_p2_fft(C, A, r, B, d, bits);
    }
    mpz_rmatrix_mult_p2_naive(C, A, r, B, d, w);
    return C;
}
