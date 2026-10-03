#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include "hwmem.h"
#include "hwmpz.h"
#include "rtree.h"
#include "rforest.h"

#define _max(a,b) ((a)>(b)?(a):(b))

void rforest (mpz_t *A, mpz_t *V, int rows, mpz_t *M, int deg, int dim, mpz_t *m, long kbase, long *ks, long n, mpz_t z, int kappa)
{
    assert ( A && V && rows > 0 && M && deg >= 0 && dim > 0 && m && ks && n >= 0 && kappa >= 0 );
    if ( !n ) return;

    int ell = _ui_len(n) - kappa; // we will use <= 2^kappa trees of height ell.
    if ( ell < 0 ) ell = 0;

    hw_mem_init(0);
    hw_mpz_setup();

    // working space
    mpz_t *w = mpz_vec_alloc_and_init2 (_max(deg,rows*dim)+1, mpz_bits(z));

    if ( !ell ) {

        mpz_t *Mk = mpz_vec_alloc_and_init (dim*dim);
        for ( long k = kbase, i = 0 ; i < n ; i++ ) {
            long nextk = ks[i];  assert(nextk >= k);
            while ( k < nextk ) {
                mpz_poly_matrix_eval_si (Mk, M, dim, deg, k++, w);
                mpz_rmatrix_mult_mod_inplace (V, rows, Mk, dim, z, w);
            }
            mpz_vec_mod_hard (A+i*rows*dim, V, rows*dim, m[i]);
            mpz_divexact (z, z, m[i]);
            mpz_rmatrix_mod (V, V, rows, dim, z);
        }
        mpz_vec_clear_and_free (Mk, dim*dim);

    } else {

        mpz_t *W0 = mpz_matrix_alloc_and_init (dim), *W1 = mpz_matrix_alloc_and_init (dim);
        mpz_t **mtree = rtree_alloc (ell, 1), **Mtree = rtree_alloc (ell, dim*dim), **Rtree = rtree_alloc (ell, rows*dim);
        long i, s, k = kbase, t = 1L << ell;
        for ( s = 0 ; s <= n ; s += t ) {   // we need to insert 1 at the start of the modulus array so we have n+1 moduli
            long i0 = s;
            long i1 = i0+t;

            // set moduli for current subtree, pad with 1's if necessary, note that i shifts by 1
            if ( !i0 ) mpz_set_one(mtree[ell][0]);  // insert 1 at the begining
            for ( long i = i0?i0:1 ; i < i1 ; i++ ) if ( i <= n ) mpz_set (mtree[ell][i-i0],m[i-1]); else mpz_set_one(mtree[ell][i-i0]);

            // set leaves of matrix tree using transition matrix
            // note that i is not shifted by 1
            mpz_t *Mk = Mtree[ell];
            for ( i = i0 ; i < i1 && i < n ; i++, Mk += dim*dim ) {
                long nextk = ks[i];  assert(nextk >= k);
                if ( k == nextk ) mpz_matrix_set_one (Mk, dim); else mpz_poly_matrix_eval_si (Mk, M, dim, deg, k++, w);
                while ( k < nextk ) {
                    mpz_matrix_set (W0, Mk, dim);
                    mpz_poly_matrix_eval_si (W1, M, dim, deg, k++, w);
                    mpz_matrix_mult (Mk, W0, W1, dim, w[0]);
                }
            }
            for ( ; i < i1 ; i++, Mk += dim*dim ) mpz_matrix_set_one (Mk, dim); // pad with the identity matrix

            // Run the remainder tree algorithm: build/build/reduce
            rtree_build (mtree, ell, 1);  rtree_build (Mtree, ell, dim);
            rtree_reduce_rows (Rtree, V, Mtree, mtree, ell, dim, rows);

            // copy leaves of current tree to output -- use mpz_vec_mod_naive to force a hard mod (leaf values should be small)
            // note that i shifts by 1
            for ( i = i0?i0:1 ; i < i1 && i <= n ; i++ ) {
                mpz_vec_mod_naive (A+(i-1)*rows*dim, Rtree[ell] + (i-i0)*(rows*dim), rows*dim, m[i-1]);
            }
            // update the modulus and transfer vector
            mpz_divexact (z, z, mtree[0][0]);
            mpz_rmatrix_mult_mod_inplace (V, rows, Mtree[0], dim, z, w);
        }
        mpz_matrix_clear_and_free (W0, dim); mpz_matrix_clear_and_free (W1, dim);
        rtree_free (mtree, ell, 1);  rtree_free (Mtree, ell, dim*dim);  rtree_free (Rtree, ell, rows*dim);
    }

    mpz_vec_clear_and_free (w, _max(deg,rows*dim)+1);
    hw_mpz_clear();
    hw_mem_clear();

    // for consistency reduce V = V mod z before returning (note mpz_rmatrix_mod does a soft reduction)
    mpz_vec_mod_naive (V, V, rows*dim, z);
}

typedef enum {
    RING_FOREST_P2,
    RING_FOREST_PN,
    RING_FOREST_PNQ
} ring_forest_kind;

typedef struct {
    ring_forest_kind kind;
    int parameter;
    size_t coefficients;
    size_t matrix_cells;
    size_t vector_cells;
} ring_forest_shape;

static size_t ring_checked_mul(size_t left, size_t right)
{
    if (right && left > SIZE_MAX / right)
        abort();
    return left * right;
}

static size_t ring_checked_add(size_t left, size_t right)
{
    if (left > SIZE_MAX - right)
        abort();
    return left + right;
}

static long ring_checked_long(size_t value)
{
    if (value > (size_t)LONG_MAX)
        abort();
    return (long)value;
}

static void ring_check_vector_size(size_t count)
{
    if (count > (size_t)LONG_MAX || count > SIZE_MAX / sizeof(mpz_t))
        abort();
}

static void ring_matrix_multiply(const ring_forest_shape *shape,
                                 mpz_t *C, mpz_t *A, int rows,
                                 mpz_t *B, int dim, mpz_t work)
{
    switch (shape->kind) {
    case RING_FOREST_P2:
        mpz_rmatrix_mult_p2(C, A, rows, B, dim, work);
        break;
    case RING_FOREST_PN:
        mpz_rmatrix_mult_pn(C, A, rows, B, dim, shape->parameter, work);
        break;
    case RING_FOREST_PNQ:
        mpz_rmatrix_mult_pnq(C, A, rows, B, dim, shape->parameter, work);
        break;
    default:
        abort();
    }
}

static void ring_set_identity(mpz_t *matrix,
                              const ring_forest_shape *shape, int dim)
{
    mpz_vec_set_zero(matrix, ring_checked_long(shape->matrix_cells));
    mpz_matrix_set_one(matrix, dim);
}

/* M is [entry][ring coefficient][ascending x degree]. */
static void ring_poly_matrix_eval_si(mpz_t *evaluated, mpz_t *M,
                                     const ring_forest_shape *shape,
                                     int dim, int deg, long x)
{
    size_t degree_count = (size_t)deg + 1;
    size_t plane_cells = (size_t)dim * (size_t)dim;
    size_t entry_stride = ring_checked_mul(shape->coefficients, degree_count);
    mpz_t value;

    mpz_init(value);
    for (size_t entry = 0; entry < plane_cells; entry++) {
        size_t source = entry * entry_stride;
        for (size_t coefficient = 0; coefficient < shape->coefficients;
             coefficient++) {
            size_t source_coefficient = source + coefficient * degree_count;
            size_t output = coefficient * plane_cells + entry;
            mpz_set(value, M[source_coefficient + (size_t)deg]);
            for (int power = deg - 1; power >= 0; power--) {
                mpz_mul_si(value, value, x);
                mpz_add(value, value, M[source_coefficient + (size_t)power]);
            }
            mpz_set(evaluated[output], value);
        }
    }
    mpz_clear(value);
}

static void ring_tree_build(mpz_t **tree, int ell, int dim,
                            const ring_forest_shape *shape, mpz_t work)
{
    for (int level = ell - 1; level >= 0; level--) {
        long width = 1L << level;
        for (long node = 0; node < width; node++) {
            size_t output_offset = (size_t)node * shape->matrix_cells;
            size_t left_offset = (size_t)(2 * node) * shape->matrix_cells;
            size_t right_offset = left_offset + shape->matrix_cells;
            ring_matrix_multiply(shape, tree[level] + output_offset,
                                 tree[level + 1] + left_offset, dim,
                                 tree[level + 1] + right_offset, dim, work);
        }
    }
}

static void ring_tree_reduce_rows(mpz_t **Rtree, mpz_t *V,
                                  mpz_t **Mtree, mpz_t **mtree,
                                  int ell, int dim, int rows,
                                  const ring_forest_shape *shape,
                                  mpz_t *work)
{
    mpz_t *reduced_vector = mpz_vec_alloc_and_init(
        ring_checked_long(shape->vector_cells));
    mpz_t *reduced_matrix = mpz_vec_alloc_and_init(
        ring_checked_long(shape->matrix_cells));
    long vector_cells = ring_checked_long(shape->vector_cells);
    long matrix_cells = ring_checked_long(shape->matrix_cells);

    mpz_vec_mod_naive(Rtree[0], V, vector_cells, mtree[0][0]);
    for (int level = 1; level <= ell; level++) {
        long width = 1L << level;
        for (long node = 0; node < width; node++) {
            mpz_t *output = Rtree[level] + (size_t)node * shape->vector_cells;
            mpz_t *parent = Rtree[level - 1] +
                            (size_t)(node / 2) * shape->vector_cells;
            if ((node & 1L) == 0) {
                mpz_vec_mod_naive(output, parent, vector_cells,
                                  mtree[level][node]);
            } else {
                mpz_t *sibling = Mtree[level] +
                                 (size_t)(node - 1) * shape->matrix_cells;
                mpz_vec_mod_naive(reduced_vector, parent, vector_cells,
                                  mtree[level][node]);
                mpz_vec_mod_naive(reduced_matrix, sibling, matrix_cells,
                                  mtree[level][node]);
                ring_matrix_multiply(shape, output, reduced_vector, rows,
                                     reduced_matrix, dim, work[0]);
                mpz_vec_mod_naive(output, output, vector_cells,
                                  mtree[level][node]);
            }
        }
    }
    mpz_vec_clear_and_free(reduced_vector, vector_cells);
    mpz_vec_clear_and_free(reduced_matrix, matrix_cells);
}

static void rforest_ring(mpz_t *A, mpz_t *V, int rows, mpz_t *M,
                         int deg, int dim, mpz_t *m, long kbase,
                         long *ks, long n, mpz_t z, int kappa,
                         ring_forest_kind kind, int parameter)
{
    ring_forest_shape shape;
    size_t row_dim;
    size_t degree_count;
    size_t matrix_input_count;
    size_t output_count;
    int ell;
    mpz_t modulus_product;

    if (!A || !V || rows <= 0 || !M || deg < 0 || dim <= 0 || !m || !ks ||
        n < 0 || kappa < 0 || parameter <= 0)
        abort();
    if (!n)
        return;

    shape.kind = kind;
    shape.parameter = parameter;
    if (kind == RING_FOREST_P2)
        shape.coefficients = 2;
    else if (kind == RING_FOREST_PN)
        shape.coefficients = (size_t)parameter;
    else
        shape.coefficients = ring_checked_mul((size_t)parameter,
                                              (size_t)parameter);
    shape.matrix_cells = ring_checked_mul(shape.coefficients,
                             ring_checked_mul((size_t)dim, (size_t)dim));
    row_dim = ring_checked_mul((size_t)rows, (size_t)dim);
    shape.vector_cells = ring_checked_mul(shape.coefficients, row_dim);
    degree_count = (size_t)deg + 1;
    matrix_input_count = ring_checked_mul(
        shape.matrix_cells, degree_count);
    output_count = ring_checked_mul((size_t)n, shape.vector_cells);
    ring_check_vector_size(shape.matrix_cells);
    ring_check_vector_size(shape.vector_cells);
    ring_check_vector_size(matrix_input_count);
    ring_check_vector_size(output_count);
    if (mpz_sgn(z) <= 0)
        abort();
    mpz_init_set_ui(modulus_product, 1);
    for (long i = 0; i < n; i++) {
        if (mpz_sgn(m[i]) <= 0)
            abort();
        mpz_mul(modulus_product, modulus_product, m[i]);
    }
    if (!mpz_divisible_p(z, modulus_product))
        abort();
    mpz_clear(modulus_product);

    ell = _ui_len((uint64_t)n) - kappa;
    if (ell < 0)
        ell = 0;
    if (ell > RTREE_MAX_LEVELS)
        abort();

    hw_mem_init(0);
    hw_mpz_setup();
    mpz_t *work = mpz_vec_alloc_and_init(1);
    mpz_t *evaluated = mpz_vec_alloc_and_init(
        ring_checked_long(shape.matrix_cells));
    mpz_t *next_v = mpz_vec_alloc_and_init(
        ring_checked_long(shape.vector_cells));

    if (!ell) {
        for (long k = kbase, i = 0; i < n; i++) {
            long nextk = ks[i];
            assert(nextk >= k);
            while (k < nextk) {
                ring_poly_matrix_eval_si(evaluated, M, &shape, dim, deg, k++);
                ring_matrix_multiply(&shape, next_v, V, rows, evaluated,
                                     dim, work[0]);
                mpz_vec_mod_naive(V, next_v,
                                  ring_checked_long(shape.vector_cells), z);
            }
            mpz_vec_mod_naive(A + (size_t)i * shape.vector_cells, V,
                              ring_checked_long(shape.vector_cells), m[i]);
            mpz_divexact(z, z, m[i]);
            mpz_vec_mod_naive(V, V, ring_checked_long(shape.vector_cells), z);
        }
    } else {
        long t = 1L << ell;
        size_t tree_matrix_cells;
        size_t tree_vector_cells;
        size_t tree_modulus_cells;
        size_t tree_depth_cells;
        size_t matrix_tree_bytes;
        size_t vector_tree_bytes;
        size_t modulus_tree_bytes;
        size_t total_tree_bytes;
        mpz_t **mtree;
        mpz_t **Mtree;
        mpz_t **Rtree;
        mpz_t *W0;
        mpz_t *W1;
        long k = kbase;

        if (n > LONG_MAX - t || shape.matrix_cells > INT_MAX ||
            shape.vector_cells > INT_MAX)
            abort();
        tree_matrix_cells = ring_checked_mul(shape.matrix_cells, (size_t)t);
        tree_vector_cells = ring_checked_mul(shape.vector_cells, (size_t)t);
        tree_modulus_cells = (size_t)t;
        tree_depth_cells = ring_checked_mul((size_t)2, (size_t)t) - 1;
        ring_checked_long(tree_matrix_cells);
        ring_checked_long(tree_vector_cells);
        ring_checked_long(tree_modulus_cells);
        matrix_tree_bytes = ring_checked_mul(
            ring_checked_mul(shape.matrix_cells, tree_depth_cells),
            sizeof(mpz_t));
        vector_tree_bytes = ring_checked_mul(
            ring_checked_mul(shape.vector_cells, tree_depth_cells),
            sizeof(mpz_t));
        modulus_tree_bytes = ring_checked_mul(tree_depth_cells, sizeof(mpz_t));
        total_tree_bytes = ring_checked_add(
            ring_checked_add(matrix_tree_bytes, vector_tree_bytes),
            modulus_tree_bytes);
        if (total_tree_bytes > SIZE_MAX -
                ring_checked_mul((size_t)(ell + 1),
                                 3 * sizeof(mpz_t *)))
            abort();

        mtree = rtree_alloc(ell, 1);
        Mtree = rtree_alloc(ell, (int)shape.matrix_cells);
        Rtree = rtree_alloc(ell, (int)shape.vector_cells);
        W0 = mpz_vec_alloc_and_init(ring_checked_long(shape.matrix_cells));
        W1 = mpz_vec_alloc_and_init(ring_checked_long(shape.matrix_cells));

        for (long start = 0; start <= n; start += t) {
            long stop = start + t;
            long leaf;
            if (!start)
                mpz_set_one(mtree[ell][0]);
            for (leaf = 0; leaf < t; leaf++) {
                long modulus_index = start + leaf - 1;
                if (!start && !leaf)
                    continue;
                if (modulus_index >= 0 && modulus_index < n)
                    mpz_set(mtree[ell][leaf], m[modulus_index]);
                else
                    mpz_set_one(mtree[ell][leaf]);
            }

            for (leaf = 0; leaf < t; leaf++) {
                long interval = start + leaf;
                mpz_t *leaf_matrix = Mtree[ell] +
                                     (size_t)leaf * shape.matrix_cells;
                if (interval >= n) {
                    ring_set_identity(leaf_matrix, &shape, dim);
                    continue;
                }
                {
                    long nextk = ks[interval];
                    assert(nextk >= k);
                    if (k == nextk) {
                        ring_set_identity(leaf_matrix, &shape, dim);
                    } else {
                        ring_poly_matrix_eval_si(leaf_matrix, M, &shape,
                                                 dim, deg, k++);
                    }
                    while (k < nextk) {
                        mpz_vec_set(W0, leaf_matrix,
                                    ring_checked_long(shape.matrix_cells));
                        ring_poly_matrix_eval_si(W1, M, &shape, dim, deg, k++);
                        ring_matrix_multiply(&shape, leaf_matrix, W0, dim,
                                             W1, dim, work[0]);
                    }
                }
            }

            rtree_build(mtree, ell, 1);
            ring_tree_build(Mtree, ell, dim, &shape, work[0]);
            ring_tree_reduce_rows(Rtree, V, Mtree, mtree, ell, dim, rows,
                                  &shape, work);

            for (long i = start ? start : 1; i < stop && i <= n; i++) {
                size_t output_offset = (size_t)(i - 1) * shape.vector_cells;
                size_t leaf_offset = (size_t)(i - start) * shape.vector_cells;
                mpz_vec_mod_naive(A + output_offset,
                                  Rtree[ell] + leaf_offset,
                                  ring_checked_long(shape.vector_cells),
                                  m[i - 1]);
            }

            mpz_divexact(z, z, mtree[0][0]);
            ring_matrix_multiply(&shape, next_v, V, rows, Mtree[0], dim,
                                 work[0]);
            mpz_vec_mod_naive(V, next_v,
                              ring_checked_long(shape.vector_cells), z);
        }

        mpz_vec_clear_and_free(W0, ring_checked_long(shape.matrix_cells));
        mpz_vec_clear_and_free(W1, ring_checked_long(shape.matrix_cells));
        rtree_free(mtree, ell, 1);
        rtree_free(Mtree, ell, (int)shape.matrix_cells);
        rtree_free(Rtree, ell, (int)shape.vector_cells);
        (void)tree_matrix_cells;
        (void)tree_vector_cells;
    }

    mpz_vec_clear_and_free(evaluated,
                           ring_checked_long(shape.matrix_cells));
    mpz_vec_clear_and_free(next_v,
                           ring_checked_long(shape.vector_cells));
    mpz_vec_clear_and_free(work, 1);
    hw_mpz_clear();
    hw_mem_clear();
    mpz_vec_mod_naive(V, V, ring_checked_long(shape.vector_cells), z);
}

void rforest_p2(mpz_t *A, mpz_t *V, int rows, mpz_t *M, int deg, int dim,
                mpz_t *m, long kbase, long *ks, long n, mpz_t z, int kappa)
{
    rforest_ring(A, V, rows, M, deg, dim, m, kbase, ks, n, z, kappa,
                 RING_FOREST_P2, 2);
}

void rforest_pn(mpz_t *A, mpz_t *V, int rows, mpz_t *M, int deg, int dim,
                int nP, mpz_t *m, long kbase, long *ks, long n, mpz_t z,
                int kappa)
{
    rforest_ring(A, V, rows, M, deg, dim, m, kbase, ks, n, z, kappa,
                 RING_FOREST_PN, nP);
}

void rforest_pnq(mpz_t *A, mpz_t *V, int rows, mpz_t *M, int deg, int dim,
                 int N, mpz_t *m, long kbase, long *ks, long n, mpz_t z,
                 int kappa)
{
    rforest_ring(A, V, rows, M, deg, dim, m, kbase, ks, n, z, kappa,
                 RING_FOREST_PNQ, N);
}


void mproduct (mpz_t z, mpz_t *m, long n)
{
    mpz_t *w = mpz_vec_alloc_and_init (n);
    mpz_vec_set (w, m, n);
    mpz_vec_product (z, w, n);
}
