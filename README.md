# rforest


This directory contains the content of https://math.mit.edu/~drew/rforest_v1.0.tar (with some minor edits to silence some compiler warnings).

The author is [Andrew V. Sutherland](https://math.mit.edu/~drew/)

Algorithm details are described in
 - [Computing Haase-Witt matrices of hyperelliptic curves in average polynomial-time](https://arxiv.org/abs/1402.3246)
 - [Computing Haase-Witt matrices of hyperelliptic curves in average polynomial-time, II](https://arxiv.org/abs/1410.5222)

## Native regression tests

Run `make test` (or `make check`) to build the native fixture runner and replay
all captured AWS cases in `tests/fixtures/aws/*.rf`. The runner links directly
to this checkout's `librforest.a` and uses C, GMP, and the checked-in fixture
files; Sage, Python, and zeta data are not needed at test runtime. Fixture
provenance and the text schema are documented in `tests/fixtures/aws/README.md`.
