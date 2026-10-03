CC = gcc
##### using -fPIC slows things down by a few percent, not a big deal
CFLAGS ?= -O3 -fPIC -fomit-frame-pointer -funroll-loops -pedantic -std=gnu11
LDFLAGS ?=
INCLUDES ?= -I/usr/local/include
LIBS ?= -L/usr/local/lib -lgmp -lm
INSTALL_ROOT = /usr/local

MPZFFTHEADERS = zzcrt.h zzmem.h zzmisc.h mpzfft_moduli.h mpnfft.h mpnfft_mod.h fermat.h split.h reduce.h split_reduce.h crt.h recompose.h crt_recompose.h  fft62/arith128.h fft62/mod62.h fft62/fft62.h
MPZFFTOBJECTS = zzmisc.o moduli.o split.o reduce.o split_reduce.o crt.o recompose.o crt_recompose.o mpnfft.o fermat.o mpnfft_mod.o mpzfft.o fft62/mod62.o fft62/fft62.o zzmem.o
RFORESTHEADERS = hwmpz.h hwmpz_tune.h hwmem.h rtree.h
RFORESTOBJECTS = hwmpz.o hwmpz_tune.o hwmem.o rtree.o rforest.o
HEADERS = $(MPZFFTHEADERS) $(RFORESTHEADERS)
OBJECTS = $(MPZFFTOBJECTS) $(RFORESTOBJECTS)
PROGRAMS = test_rforest
TEST_PROGRAMS = test_rforest_fixtures test_ring_matmul_disabled test_ring_forest_disabled
BENCH_PROGRAMS = bench_ring_baselines
FIXTURE_LDFLAGS ?= -Wl,--wrap=mpz_rmatrix_mult_fft
FIXTURES = $(sort $(wildcard tests/fixtures/aws/*.rf))
KAPPA_FIXTURES = \
	tests/fixtures/aws/p41_g1_d4_001_factorial_i0.rf \
	tests/fixtures/aws/p41_g1_d4_001_block_i0.rf \
	tests/fixtures/aws/p41_g3_d8_001_block_i1.rf \
	tests/fixtures/aws/p41_g8_d18_001_block_i0.rf

all: librforest.a $(PROGRAMS)

.PHONY: all clean install test check

clean:
	rm -f *.o
	rm -f fft62/*.o
	rm -f tests/*.o
	rm -f librforest.a $(PROGRAMS) $(TEST_PROGRAMS) $(BENCH_PROGRAMS)

install: all
	cp -v rforest.h $(INSTALL_ROOT)/include
	cp -v librforest.a $(INSTALL_ROOT)/lib

##### rforest library

librforest.a: $(OBJECTS)
	ar -r librforest.a $(OBJECTS)
	ranlib librforest.a
	
##### executables

test_rforest: test_rforest.o librforest.a rforest.h
	$(CC) $(LDFLAGS) -o $@ $< librforest.a $(LIBS)

test_rforest_fixtures: tests/test_rforest_fixtures.o librforest.a rforest.h
	$(CC) $(LDFLAGS) $(FIXTURE_LDFLAGS) -o $@ $< librforest.a $(LIBS)

test_ring_matmul_disabled: tests/test_ring_matmul_disabled.o librforest.a
	$(CC) $(LDFLAGS) -o $@ $< librforest.a $(LIBS)

test_ring_forest_disabled: tests/test_ring_forest_disabled.o librforest.a
	$(CC) $(LDFLAGS) -o $@ $< librforest.a $(LIBS)

bench_ring_baselines: tests/bench_ring_baselines.o librforest.a hwmpz.h
	$(CC) $(LDFLAGS) -o $@ $< librforest.a $(LIBS)

.PHONY: bench-ring
bench-ring: bench_ring_baselines
	./bench_ring_baselines tests/fixtures/aws_ring/p2_aws_products.txt

tests/test_rforest_fixtures.o: tests/test_rforest_fixtures.c rforest.h
	$(CC) $(CFLAGS) $(INCLUDES) -I. -o $@ -c $<

tests/bench_ring_baselines.o: tests/bench_ring_baselines.c hwmpz.h
	$(CC) $(CFLAGS) $(INCLUDES) -I. -o $@ -c $<

tests/test_ring_matmul_disabled.o: tests/test_ring_matmul_disabled.c
	$(CC) $(CFLAGS) $(INCLUDES) -I. -o $@ -c $<

tests/test_ring_forest_disabled.o: tests/test_ring_forest_disabled.c
	$(CC) $(CFLAGS) $(INCLUDES) -I. -o $@ -c $<

test: test_rforest_fixtures test_ring_matmul_disabled test_ring_forest_disabled
	./test_rforest_fixtures $(FIXTURES)
	./test_rforest_fixtures --kappa 0 $(KAPPA_FIXTURES)
	./test_rforest_fixtures --kappa 4 $(KAPPA_FIXTURES)
	./test_rforest_fixtures --disable-fft $(FIXTURES)
	./test_ring_matmul_disabled
	./test_ring_forest_disabled

check: test

##### hwlpoly modules

hwmem.o: hwmem.c hwmem.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

hwmpz.o: hwmpz.c hwmpz.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

hwmpz_tune.o: hwmpz_tune.c hwmem.h hwmpz.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

rtree.o: rtree.c rtree.h hwmem.h hwmpz.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

rforest.o: rforest.c rforest.h hwmem.h hwmpz.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

test_rforest.o: test_rforest.c rforest.h hwmem.h hwmpz.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

##### mpzfft C modules

zzmisc.o: zzmisc.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

moduli.o : moduli.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

split.o : split.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

reduce.o : reduce.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

split_reduce.o : split_reduce.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

recompose.o : recompose.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

crt.o : crt.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

crt_recompose.o : crt_recompose.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

mpnfft.o : mpnfft.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

fermat.o : fermat.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

mpnfft_mod.o : mpnfft_mod.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

mpzfft.o : mpzfft.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

zzmem.o : zzmem.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

fft62/mod62.o : fft62/mod62.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

fft62/fft62.o : fft62/fft62.c mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

##### master header files

mpzfft.h: $(MPZFFTHEADERS)
	touch mpzfft.h

rforest.h: $(HEADERS)
	touch rforest.h
