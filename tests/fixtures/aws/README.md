# AWS-derived remainder forest fixtures

These fixtures preserve calls made by the existing Sage implementation in
AWS-2026, along with every expected matrix returned for its ordered prime
indices. They are plain decimal text so a C reader can parse them with standard
file I/O and GMP.

## Cases

The four original cases use primes `3, 5, ..., 41`, modulus `p^2`, exclusive
endpoint `p`, and `kbase=1`. The Sage wrapper's default is `kappa=3` for these
12 indices. The wrapper also defaults `V` to the identity matrix.
`INITIAL_Z` is the exact product of all moduli.

| Fixture | AWS call | Matrix dimension | Expected matrices |
| --- | --- | ---: | ---: |
| `p41_g1_d4_001_factorial_i0.rf` | 2x2 factorial call, `i=0` | 2 | 12 |
| `p41_g1_d4_001_block_i0.rf` | `construct_T_bar_poly`, `i=0` | 8 | 12 |
| `p41_g3_d8_001_block_i1.rf` | `construct_T_bar_poly`, `i=1` | 16 | 12 |
| `p41_g8_d18_001_block_i0.rf` | `construct_T_bar_poly`, `i=0` | 36 | 12 |
| `p41_g1_d4_001_bound2683_kappa0_block_i1.rf` | `construct_T_bar_poly`, `i=1` | 8 | 388 |

The block dimensions are `2*d`, where `d` is the degree of the selected
curve's `f(x)`. The fixture keeps every matrix coefficient and every complete
`rows x dim` result matrix; it does not reduce the output to traces or
characteristic polynomials. The expanded case uses the genus-one curve's real
AWS `i=1` full block matrix, primes through 2683, modulus `p^2`, endpoint `p`,
and `kbase=1`. The source call omits `kappa`, whose AWS wrapper default is 5
for 388 indices; the native fixture explicitly replays with `kappa=0`, the
smallest valid value, to exercise the normal matrix FFT cutoff.

## Text format

Whitespace separates tokens; line breaks are for readability. A reader can
consume the labels and then the fixed number of decimal integers described by
the header:

- `RFOREST_FIXTURE 1`, followed by case, call, source revisions, and reference
  prime metadata.
- `DIM`, `ROWS`, `DEG`, `N`, `KBASE`, `KAPPA`, `AWS_DEFAULT_KAPPA`,
  `EXPECT_FFT_MATRIX_MUL`, and `INITIAL_Z` give the native call dimensions,
  replay settings, and initial state. `KAPPA` is passed to native `rforest`;
  `AWS_DEFAULT_KAPPA` records the source wrapper's effective default.
  `EXPECT_FFT_MATRIX_MUL 1` makes the fixture fail unless the linked
  `mpz_rmatrix_mult_fft` matrix-multiplication function is reached.
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
the imported `remainder_forest` boundary with sequential reference
multiplication; it does not call rforest, mpzfft, or a compiled `pyrforest`
backend. The original AWS p=41 test and its L-polynomial comparison run for
the three source cases. The four p=41 fixtures were also replayed with a
separate pure-Python integer triple loop, checking every output matrix,
modulus ordering, and `INITIAL_Z`.

Set `FFT_CAPTURE_BOUND` and `FFT_CAPTURE_KAPPA` to capture an expanded `i=1`
block call. The script obtains the matrix, moduli, and endpoints from the
original AWS function at that bound. It skips the preceding `i=0` result only
to reach the desired call, then stops at the real `i=1` input. Expected
matrices come from one exact sequential matrix product over `ZZ`; the full
matrix is reduced at every `p^2` endpoint. This is equivalent to sequential
multiplication modulo each `p^2` and remains independent of the tested
library. `FFT_CAPTURE_CASE` defaults to `p41_g1_d4_001`.

Calibration found no matrix FFT dispatch through bound 2677 (387 matrices),
and one dispatch at bound 2683 (388 matrices). The native test executable uses
the GNU linker `--wrap` option to count calls to `mpz_rmatrix_mult_fft` only,
so FFT reduction cannot satisfy the assertion. The wrapper is test-only and
does not add a public library API or alter production crossover thresholds.
The measured clean `make test` build and run took 5.58 seconds; replay alone
took about 0.09 seconds.

To reproduce, clone the suite at the pinned commit and point the runner at a
read-only AWS-2026 checkout at its pinned commit:

```sh
scratch=$(mktemp -d)
git clone --filter=blob:none https://github.com/jjgarzella/hyperell_suite.git "$scratch/zeta_test_suite"
git -C "$scratch/zeta_test_suite" checkout 621107b12200a3c234c2e3dd6a4ba9dc87be9bea
AWS_SOURCE=/path/to/AWS-2026 \
ZETA_SUITE="$scratch/zeta_test_suite" \
FIXTURE_DIR="$PWD/tests/fixtures/aws" \
FFT_CAPTURE_BOUND=2683 \
FFT_CAPTURE_KAPPA=0 \
sage "$PWD/tests/fixtures/aws/capture_aws_fixtures.sage"
```

The runner checks both source revisions before execution. On Sage 10.9, the
three original p=41 comparisons and four fixture captures took 1.310 seconds.
No zeta suite datasets are included here.
