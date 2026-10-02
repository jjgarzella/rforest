# AWS-derived remainder forest fixtures

These fixtures preserve calls made by the existing Sage implementation in
AWS-2026, along with every expected matrix returned for its ordered prime
indices. They are plain decimal text so a C reader can parse them with standard
file I/O and GMP.

## Cases

All calls use primes `3, 5, ..., 41`, modulus `p^2`, exclusive endpoint `p`,
and `kbase=1`. The Sage wrapper's default is `kappa=3` for these 12 indices.
The wrapper also defaults `V` to the identity matrix. `INITIAL_Z` is the exact
product of all 12 moduli.

| Fixture | AWS call | Matrix dimension | Expected matrices |
| --- | --- | ---: | ---: |
| `p41_g1_d4_001_factorial_i0.rf` | 2x2 factorial call, `i=0` | 2 | 12 |
| `p41_g1_d4_001_block_i0.rf` | `construct_T_bar_poly`, `i=0` | 8 | 12 |
| `p41_g3_d8_001_block_i1.rf` | `construct_T_bar_poly`, `i=1` | 16 | 12 |
| `p41_g8_d18_001_block_i0.rf` | `construct_T_bar_poly`, `i=0` | 36 | 12 |

The block dimensions are `2*d`, where `d` is the degree of the selected
curve's `f(x)`. The fixture keeps every matrix coefficient and every complete
`rows x dim` result matrix; it does not reduce the output to traces or
characteristic polynomials.

## Text format

Whitespace separates tokens; line breaks are for readability. A reader can
consume the labels and then the fixed number of decimal integers described by
the header:

- `RFOREST_FIXTURE 1`, followed by case, call, source revisions, and reference
  prime metadata.
- `DIM`, `ROWS`, `DEG`, `N`, `KBASE`, `KAPPA`, and `INITIAL_Z` give the native
  call dimensions and initial state.
- `MODULI` has `N` ordered triples: `p p_squared exclusive_endpoint`.
- `INITIAL_V` has `ROWS` row-major rows of `DIM` integers.
- `MATRIX_COEFFICIENTS` has `DIM*DIM` entries in row-major matrix order. Each
  entry has `DEG+1` signed integers in ascending polynomial degree order
  (`c0 c1 ...`).
- `EXPECTED_MATRICES` has `N` records. Each starts with `PRIME p`, followed by
  `ROWS` row-major rows of `DIM` integers reduced modulo that record's `p^2`.
- `END` terminates the fixture.

All stored numbers are exact decimal integers. Polynomial coefficients retain
their signs; expected residues are in `[0, p^2)`. For the factorial fixture,
the source matrix is `[[x, 0], [0, x]]`, so each result is `(p-1)!` times the
identity modulo `p^2`.

## Provenance and reproduction

Source revisions:

- AWS-2026: `a1fc50dd667d262b5d83d1d4ceb1499bdbead288`
- hyperell_suite / zeta_test_suite: `621107b12200a3c234c2e3dd6a4ba9dc87be9bea`
- pyrforest Sage wrapper interface used to record effective defaults:
  `8317488b07a8f9519c0af384f83e1c807dcee5d9`

The capture runner symlinks the pinned AWS Sage files and suite loader into a
temporary layout, leaving both source checkouts untouched. It replaces only
the imported `remainder_forest` boundary with direct sequential Sage matrix
multiplication over `Z/(p^2)`. This computes each expected output as
`V*M(kbase)*...*M(p-1)` without calling rforest, mpzfft, or a compiled
`pyrforest` backend. It loads the original AWS p=41 test and runs its
L-polynomial comparison for the three listed cases. The resulting four
fixtures were also replayed with a separate pure-Python integer triple loop,
which checked each full output matrix, modulus ordering, and `INITIAL_Z`.

To reproduce, clone the suite at the pinned commit and point the runner at a
read-only AWS-2026 checkout at its pinned commit:

```sh
scratch=$(mktemp -d)
git clone --filter=blob:none https://github.com/jjgarzella/hyperell_suite.git "$scratch/zeta_test_suite"
git -C "$scratch/zeta_test_suite" checkout 621107b12200a3c234c2e3dd6a4ba9dc87be9bea
AWS_SOURCE=/path/to/AWS-2026 \
ZETA_SUITE="$scratch/zeta_test_suite" \
FIXTURE_DIR="$PWD/tests/fixtures/aws" \
sage "$PWD/tests/fixtures/aws/capture_aws_fixtures.sage"
```

The runner checks both source revisions before execution. On Sage 10.9, the
three original p=41 comparisons and four fixture captures took 1.310 seconds.
No zeta suite datasets are included here.
