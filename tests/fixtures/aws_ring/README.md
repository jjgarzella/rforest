# AWS P² products and ring baselines

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
cases: the three captured AWS P² products; synthetic P² cases at dimensions 2
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
GMP matrix-coefficient reference; AWS rows are additionally checked against
the captured Sage result. Each case has two warmups and nine timed
repetitions. CSV times are per-case medians and median absolute deviations in
nanoseconds; `adapter_over_block` is adapter median divided by block median,
so values below 1 indicate the adapter was faster.

The recorded run was made with GCC 13.3.0, GMP 6.3.0, Linux 6.12.76-linuxkit
on aarch64 on 2026-10-03. Across all 115 cases, the current adapter was faster
than integer block embedding. These are pre-optimization reference
measurements, not claims about a future ring FFT implementation. Future PRs
must compare each identical input case against both baselines and investigate
any measurable slowdown; wall-clock values are not normal test gates.

## P² implementation comparison

Reproduce the stage-2 comparison with:

```sh
make clean bench_ring_baselines
./bench_ring_baselines tests/fixtures/aws_ring/p2_aws_products.txt \
  --p2-optimized > tests/fixtures/aws_ring/p2_optimized_results.csv
```

This mode uses the same three AWS fixtures and the same 16 synthetic P² cases,
fixed inputs, two warmups, and nine timed repetitions as the initial baseline
run. It checks the native P² result against the direct GMP reference on every
case, then reports per-case medians, MADs, and native-to-baseline ratios for
both the adapter and integer block embedding. A ratio below 1 means the native
P² operation was faster. The recorded stage-2 run has no per-case median
regression against either baseline: native/adapter ratios range from 0.445 to
0.970, and native/block ratios range from 0.165 to 0.640. The closest adapter
case is the dense 8x8, 512-bit grid input at 0.970; its 2.29 microsecond median
advantage exceeds the measured MADs (0.46 microseconds native, 0.29 adapter).
The CSV retains all case-level timing variability. The separate native test
also verifies that ordinary dispatch observes one forward transform per input
coefficient entry, three Fourier matrix products, and one inverse transform
per output coefficient entry, and that `hw_disable_fft` takes the exact
classical fallback.

## Arbitrary univariate implementation comparison

Reproduce the stage-3 univariate comparison with:

```sh
make bench_ring_baselines
./bench_ring_baselines tests/fixtures/aws_ring/p2_aws_products.txt \
  --pn-optimized > /tmp/pn_run_1.csv
```

The 67 cases include all three captured AWS P² products, the 16 synthetic P²
cases, and the complete univariate baseline grid for `n=1,3,5`. The native
facade also uses the existing specialized P² implementation for the P² rows.
For each case, the command checks the adapter, block embedding, and native
result against the direct GMP coefficient reference. It then records two
warmup batches and 27 repeated samples per algorithm. Each sample averages 16
complete operations, including output allocation. Adapter, block, and native
algorithms run in a rotating interleaved order. The adapter makes
`n(n+1)/2` integer matrix calls for an univariate truncation `n`. The original
nine-sample stage-3 report is preserved in
`pn_optimized_results_initial.csv`; it used GCC 13.3.0, GMP 6.3.0, Linux
6.12.76-linuxkit on aarch64, on 2026-10-03.

The initial native/block ratios ranged from 0.088 to 0.671, while
native/adapter ratios ranged from 0.500 to 1.178. Three dense cases exceeded
the combined per-run MADs: P² 8x8 at 512 bits, P^5 3x6 at 32 bits, and P^5
3x6 at 256 bits. To repeat the acceptance follow-up and aggregate three
independent runs:

```sh
for run in 1 2 3; do
  ./bench_ring_baselines tests/fixtures/aws_ring/p2_aws_products.txt \
    --pn-optimized > "/tmp/pn_run_${run}.csv"
done
python3 tests/fixtures/aws_ring/aggregate_pn_benchmarks.py \
  /tmp/pn_run_1.csv /tmp/pn_run_2.csv /tmp/pn_run_3.csv \
  > tests/fixtures/aws_ring/pn_optimized_results.csv
```

The aggregator reports the median of the three process medians and the MAD
across those process medians for each algorithm, then computes both ratios
from the aggregated medians. All 67 native case medians beat both baselines:
native/adapter ranges from 0.319 to 0.976, and native/block from 0.086 to
0.633. The three follow-up cases have adapter/native medians and across-run
MADs of 69,216/782 ns and 67,546/765 ns (P², ratio 0.976); 17,135/52 ns and
13,838/49 ns (P^5 at 32 bits, ratio 0.808); and 40,434/656 ns and 38,401/691
ns (P^5 at 256 bits, ratio 0.950). The second process run had a noisy P²
native/adapter median ratio of 1.010, but the 734 ns difference was below the
combined within-run MAD of 9,964 ns. The aggregate table has no per-case median
regressions against either baseline. The original
`baseline_results.csv` remains the pre-optimization reference.

## Bivariate implementation comparison

Reproduce the stage-4 bivariate comparison with three independent processes:

```sh
make bench_ring_baselines
for run in 1 2 3; do
  ./bench_ring_baselines tests/fixtures/aws_ring/p2_aws_products.txt \
    --pnq-optimized > "/tmp/pnq_run_${run}.csv"
done
python3 tests/fixtures/aws_ring/aggregate_pn_benchmarks.py --family pnq \
  /tmp/pnq_run_1.csv /tmp/pnq_run_2.csv /tmp/pnq_run_3.csv \
  > tests/fixtures/aws_ring/pnq_optimized_results.csv
```

The 48 fixed bivariate cases cover `N=1,2,3`, dimensions 2 and 4, one-row
and square left operands, 32/256-bit signed inputs, and dense/sparse
coefficients. `--pnq-optimized` starts at the same deterministic seed position
as the bivariate section of the original 115-case grid. Each case checks the
direct GMP reference, the integer block embedding, the naive coefficient-pair
adapter, and the bivariate facade before timing. The three algorithms are
measured in rotating paired order with two warmups, 27 samples per process,
and 16 complete operations per sample; allocation, transforms, reconstruction,
embedding, and ring assembly stay inside the timed operation. The final CSV
reports the median of the three process medians and the MAD across those
medians for each algorithm.

The one-process exploratory result is retained in
`pnq_optimized_results_initial.csv`. Its two dense, full-row 256-bit cases at
`N=2` and `N=3` had native/adapter ratios of 1.013 and 1.037, both within the
combined per-process MAD. The classical path was tightened to initialize each
output from its first matrix product and accumulate the remaining products.
The three-process aggregate in `pnq_optimized_results.csv` has no per-case
median regression: all 48 cases beat both baselines, with native/adapter ratios
from 0.375 to 0.955 and native/block ratios from 0.043 to 0.604. The closest
adapter case is dense `N=3`, 4x4, 256-bit input at 0.955. The independent exact
test also confirms one forward transform per input entry, one inverse per
output entry, and `[N(N+1)/2]^2` Fourier-side matrix products; for `N=3` these
counts are 1152, 576, and 36 respectively. `hw_disable_fft` is checked against
the same direct GMP reference.

The existing arbitrary univariate facade was benchmarked again on this stage
branch as a stack check. All 67 current univariate cases still beat both
baselines: native/adapter ratios range from 0.270 to 0.998 and native/block
ratios from 0.084 to 0.646. The accepted stage-3 result remains in
`pn_optimized_results.csv`; the three-process stage-4 stack check is recorded
separately in `pn_stack_check_results.csv`.
