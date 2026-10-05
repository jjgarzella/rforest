# rforest


This directory contains the content of https://math.mit.edu/~drew/rforest_v1.0.tar (with some minor edits to silence some compiler warnings).

The author is [Andrew V. Sutherland](https://math.mit.edu/~drew/)

Algorithm details are described in
 - [Computing Haase-Witt matrices of hyperelliptic curves in average polynomial-time](https://arxiv.org/abs/1402.3246)
 - [Computing Haase-Witt matrices of hyperelliptic curves in average polynomial-time, II](https://arxiv.org/abs/1410.5222)

## Native regression tests

Run `make test` (or `make check`) to build the native fixture runner and replay
all captured AWS cases in `tests/fixtures/aws/*.rf`. The command also replays
the four small p=41 cases with `kappa=0` (tree regime) and `kappa=4` (sequential
regime), then replays every case with `hw_disable_fft` enabled. Each fixture
replay runs twice from fresh `V` and `z` state, checks every exact output
residue and the final `z` and `V`, and verifies that the reusable matrix,
modulus, and endpoint inputs were not changed. The ordinary pass asserts the
confirmed matrix FFT dispatch; the FFT-disabled pass verifies no matrix FFT
calls occur. The runner links directly to this checkout's `librforest.a` and
uses C, GMP, and the checked-in fixture files; Sage, Python, and zeta data are
not needed at test runtime. Fixture provenance and the text schema are
documented in `tests/fixtures/aws/README.md`.

## Ring matrix forests

> **Upstream cleanup required:** This PR-stack status and its linked planning
> and benchmark documents must be rewritten for upstream or removed before
> merging into upstream rforest. Update or remove this README section at the
> same time.

The public `rforest_p2`, `rforest_pn`, and `rforest_pnq` APIs run the existing
remainder forest over `Z[P]/(P^2)`, `Z[P]/(P^n)`, and
`Z[P,Q]/(P^N,Q^N)`. The polynomial variable `x` used to evaluate transition
matrices is independent of P and Q. The ring matrix multiplication facades in
`hwmpz.h` provide the FFT and classical paths; each forest product-tree node
calls one facade, so transforms are reused only inside that matrix
multiplication call.

For P², `V` and `A` store `[coefficient][row][column]` with coefficients
`1,P`. For P^n they store `[P exponent][row][column]`; for the bivariate ring
they store `[P exponent][Q exponent][row][column]`, with Q varying fastest.
`M` is entry-major: `[matrix row][matrix column][ring coefficient][x degree]`,
and x degrees are ascending. The matrix dimension `dim` is square for each
transition matrix, while `V` may have any positive `rows` by `dim` shape.
`A` contains `n` endpoint matrices in the same coefficient-major layout, with
each coefficient reduced modulo its endpoint modulus. Endpoint values `k[i]`
are exclusive. On return, `z` has the consumed moduli divided out and `V` is
the complete product modulo the remaining `z`; `M`, `m`, and `k` are
unchanged. Caller arrays must not overlap one another.

The caller allocates and initializes `M`, `V`, `A`, `m`, and `k`; the forest
allocates and frees its internal product and remainder trees. These APIs cover
the three fixed rings above with GMP integer coefficients. They do not expose
caller-defined rings or other quotient ideals. The truncation and matrix
dimensions must be positive, and internal size products are checked before
tree allocation.

`make test` enables the direct sequential GMP forest references for all three
ring families, checks coefficientwise residues, carried state, rectangular
initial `V`, multiple kappa regimes and repeated runs, and repeats those
checks with `hw_disable_fft`. The fixture runner also converts each captured
AWS block matrix `[[M0,M1],[0,M0]]` to P², runs both forest implementations,
and compares every full output matrix and final state against the captured AWS
results. Fixture provenance is in `tests/fixtures/aws/README.md`; ring layouts,
matrix multiplication references and benchmark grids are described in
`tests/ring_api.md` and `tests/fixtures/aws_ring/README.md`.

Run `make bench-ring` for the P² and univariate matrix acceptance grid,
`make bench-ring-pnq` for the bivariate grid, or pass `--pn-optimized` to
`bench_ring_baselines` to replay the recorded univariate stack grid. These are
review measurements; they do not impose timing assertions on normal tests.
