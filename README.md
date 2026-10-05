# rforest


This directory contains the content of https://math.mit.edu/~drew/rforest_v1.0.tar (with some minor edits to silence some compiler warnings).

The author is [Andrew V. Sutherland](https://math.mit.edu/~drew/)

Algorithm details are described in
 - [Computing Haase-Witt matrices of hyperelliptic curves in average polynomial-time](https://arxiv.org/abs/1402.3246)
 - [Computing Haase-Witt matrices of hyperelliptic curves in average polynomial-time, II](https://arxiv.org/abs/1410.5222)

## Native regression tests

Run `make test` (or `make check`) to build the native hyperelliptic fixture
runner and replay all captured curve cases stored under the source-provenance
path `tests/fixtures/aws/*.rf`. The command also replays the four small p=41
cases with `kappa=0` (tree regime) and `kappa=4` (sequential regime), then
replays every case with `hw_disable_fft` enabled. Each fixture
replay runs twice from fresh `V` and `z` state, checks every exact output
residue and the final `z` and `V`, and verifies that the reusable matrix,
modulus, and endpoint inputs were not changed. The ordinary pass asserts the
confirmed matrix FFT dispatch; the FFT-disabled pass verifies no matrix FFT
calls occur. The runner links directly to this checkout's `librforest.a` and
uses C, GMP, and the checked-in fixture files; Sage, Python, and zeta data are
not needed at test runtime. Fixture provenance and the text schema are
documented in `tests/fixtures/aws/README.md`.
