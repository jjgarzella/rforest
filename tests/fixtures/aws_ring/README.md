# Captured P² products and ring baselines

> **Upstream cleanup required:** This fork-specific source reproduction guide
> and its recorded benchmark results must be rewritten for upstream or removed
> before merging into upstream rforest.

`p2_aws_products.txt` records exact P² matrix multiplication triples from the
existing AWS-2026 Sage implementation. The capture script uses
`good-code/P7-utils.sage` and `good-code/P7-avg-poly.sage` to construct and
multiply `ZP2Matrix` values for the actual p=41 genus-one, genus-three, and
genus-eight cases. It stores every entry of both operands and both output
coefficient matrices. The capture checks AWS's `ZP2Matrix.__mul__` result
against a separate direct integer coefficient-ring loop before writing it.

The capture also loads `P8/P8-utils.sage`, its generic `ZPmuMatrix` source, and
checks three coefficients of an independent truncation-three example against
a direct sum of integer matrix products. General P^n native expectations and
bivariate fixtures use direct GMP references; no P8 output is treated as the
oracle for production code.

Pinned source revisions:

- AWS-2026: `a1fc50dd667d262b5d83d1d4ceb1499bdbead288`
- zeta test suite cases: `621107b12200a3c234c2e3dd6a4ba9dc87be9bea`

AWS-2026 is a private source repository; reproduction requires GitHub access
to `JaneShi99/AWS-2026`.

To reproduce in a temporary source layout:

```sh
git clone https://github.com/JaneShi99/AWS-2026.git /tmp/AWS-2026
git -C /tmp/AWS-2026 checkout a1fc50dd667d262b5d83d1d4ceb1499bdbead288
git clone https://github.com/jjgarzella/zeta_test_suite.git /tmp/zeta_test_suite
git -C /tmp/zeta_test_suite checkout 621107b12200a3c234c2e3dd6a4ba9dc87be9bea
AWS_SOURCE=/tmp/AWS-2026 \
ZETA_SUITE=/tmp/zeta_test_suite \
OUTPUT="$PWD/tests/fixtures/aws_ring/p2_aws_products.txt" \
sage tests/fixtures/aws_ring/capture_aws_ring_products.sage
```

The source and suite checkouts are read-only inputs to the script. Only the
three selected curve cases are represented in this small fixture; no zeta
datasets are copied into rforest.

## Benchmark command and grid

Build and run the benchmark with:

```sh
make clean bench_ring_baselines
./bench_ring_baselines tests/fixtures/aws_ring/p2_aws_products.txt \
  > tests/fixtures/aws_ring/baseline_results.csv
```

`baseline_results.csv` is the recorded initial run. The grid has 115 fixed
cases: the three captured Sage P² products; synthetic P² cases at dimensions 2
and 8; P^n with `n=1,3,5` at dimensions 2 and 6; and bivariate boxes with
`N=1,2,3` at dimensions 2 and 4. Synthetic cases include one-row and
multirow inputs, 32/64/256/512-bit signed operands, and both dense and sparse
coefficients. Synthetic values are generated from the local SplitMix64
implementation and fixed seed sequence, independent of GMP's random
generator, and stay identical between baselines and repeats.

The `adapter` baseline calls the existing `mpz_rmatrix_mult` once for every
valid coefficient pair and accumulates with GMP. The `block` baseline builds
the integer regular representation, calls the same integer matrix API, and
extracts every output coefficient. Each timed run allocates the ring output
and algorithm work arrays. The adapter run includes every integer matrix call
and GMP accumulation; the block run also includes embedding assembly and
coefficient extraction. Inputs are generated once and held fixed. Before
timing, both are checked coefficient by coefficient against a separate direct
GMP matrix-coefficient reference; the direct GMP result is additionally
checked against the stored Sage products for those three cases. Each case has
two warmups and nine timed repetitions. CSV times are per-case medians and
median absolute deviations in nanoseconds; `adapter_over_block` is adapter median divided by block median,
so values below 1 indicate the adapter was faster.

The recorded run was made with GCC 13.3.0, GMP 6.3.0, Linux 6.12.76-linuxkit
on aarch64 on 2026-10-03. Across all 115 cases, the current adapter was faster
than integer block embedding. These are pre-optimization reference
measurements, not claims about a future ring FFT implementation. Future PRs
must compare each identical input case against both baselines and investigate
any measurable slowdown; wall-clock values are not normal test gates.
