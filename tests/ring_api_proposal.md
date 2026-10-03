# Ring matrix and forest API proposal

This proposal keeps the existing `mpz_rmatrix_mult` and `rforest` interfaces
unchanged. Ring coefficients are exact GMP integers; a ring product is not
reduced modulo an integer unless a forest endpoint requests a modulus.

## Coefficient layout

Matrix coefficient arrays are coefficient-major and row-major within each
coefficient plane:

- `Z[P]/(P^n)`: `[P exponent][row][column]`, with `0 <= P exponent < n`.
- `Z[P,Q]/(P^N,Q^N)`: `[P exponent][Q exponent][row][column]`, with each
  exponent in `[0,N)`.
- The left matrix is `rows x dim`; the right matrix is `dim x dim`; output is
  `rows x dim`, matching `mpz_rmatrix_mult`.

The bivariate truncation is a box. A product term is dropped if either output
exponent is at least `N`; in particular `P^(N-1)Q^(N-1)` is retained. `n=1`
and `N=1` each reduce to ordinary integer matrix multiplication.

## Proposed facades

```c
void mpz_rmatrix_mult_p2(mpz_t *C, mpz_t *A, int rows,
                         mpz_t *B, int dim, mpz_t work);
void mpz_rmatrix_mult_pn(mpz_t *C, mpz_t *A, int rows,
                         mpz_t *B, int dim, int n, mpz_t work);
void mpz_rmatrix_mult_pnq(mpz_t *C, mpz_t *A, int rows,
                          mpz_t *B, int dim, int N, mpz_t work);

void rforest_p2(mpz_t *A, mpz_t *V, int rows, mpz_t *M,
                int deg, int dim, mpz_t *m, long kbase,
                long *k, long n, mpz_t z, int kappa);
void rforest_pn(mpz_t *A, mpz_t *V, int rows, mpz_t *M,
                int deg, int dim, int nP, mpz_t *m, long kbase,
                long *k, long n, mpz_t z, int kappa);
void rforest_pnq(mpz_t *A, mpz_t *V, int rows, mpz_t *M,
                 int deg, int dim, int N, mpz_t *m, long kbase,
                 long *k, long n, mpz_t z, int kappa);
```

For forests, each matrix entry stores its ring coefficients, each followed by
its ascending transition-polynomial coefficients:
`M[row][column][ring coefficient][x degree]`. This preserves the existing
`rforest` row/column/degree convention when the ring has one coefficient. `V`
uses the ring matrix layout above. Outputs are
`[endpoint][ring coefficient][row][column]`. The endpoint `k[i]` is exclusive,
as in `rforest`; each output plane is reduced coefficient by coefficient
modulo `m[i]`. On return, `z` has each consumed modulus divided out and `V` is
the complete product modulo the residual `z`.

The multiplication facades preserve the base API's shape and aliasing
contract: `C` must not overlap either input. Existing in-place calls use a
separate work array before copying back; ring-specific in-place wrappers can
follow that pattern if a later caller needs them. The `hw_disable_fft` switch
selects the same classical fallback behavior as the integer matrix API.

## Disabled test milestones

`make test` keeps the original AWS forest suite enabled and prints explicit
disabled status for the new ring tests. The guarded C/GMP test bodies are
activated as their APIs land:

1. PR 2 enables P² matmul tests and checks the exact P7 source products in
   `tests/fixtures/aws_ring/p2_aws_products.txt`, signed/dense/sparse/zero and
   large operands, noncommuting order, cancellation, rectangular shape, and
   nonaliasing output semantics.
2. PR 3 adds P^n cases for `n=1`, `2`, `3`, and `5`, each checked by an
   independent direct coefficient-ring GMP product.
3. PR 4 enables bivariate `N=1`, `2`, and `3` cases. The `N=3` fixture retains
   the highest corner and separately exercises overflow in P and Q, which
   distinguishes box truncation from total-degree truncation.
4. PR 5 enables forest comparisons against a direct sequential coefficient
   ring reference, with the transition polynomial `x` kept separate from P/Q,
   rectangular initial `V`, exclusive endpoints, three `kappa` values,
   coefficientwise residues, final `z`, final `V`, and immutable inputs. P²
   also gets block-embedding equivalence cases based on the captured AWS forest
   fixtures in `tests/fixtures/aws/`. General P^n and bivariate expected values
   come from direct GMP loops rather than a production backend.

The P², P^n, and bivariate multiplication facades and their constructed exact
references are enabled together on the stage-4 branch. The test also checks
that each coefficient entry is transformed once, every truncated Fourier-side
matrix product is counted, each output is reconstructed once, aliased input
pointers remain valid, and `hw_disable_fft` takes the exact classical path.
Forest tests remain disabled until the final forest integration stage.
