#!/usr/bin/env sage
"""Capture real P^2 matrix products from the pinned AWS-2026 implementation.

The source and zeta-suite checkouts are read-only inputs.  The output is a
small native C/GMP fixture; no suite datasets are copied into rforest.
"""

import os
import subprocess
import sys


AWS_REVISION = "a1fc50dd667d262b5d83d1d4ceb1499bdbead288"
ZETA_SUITE_REVISION = "621107b12200a3c234c2e3dd6a4ba9dc87be9bea"
CASE_IDS = [
    "p41_g1_d4_001",
    "p41_g3_d8_001",
    "p41_g8_d18_001",
]


def revision(path):
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()


def rows_as_integers(matrix_value):
    return [int(value) for value in matrix_value.list()]


def independent_p2_product(left, right):
    """Direct integer coefficient-ring product, independent of Sage classes."""
    dim = left["dim"]
    result = [[0] * (dim * dim), [0] * (dim * dim)]
    for row in range(dim):
        for col in range(dim):
            offset = row * dim + col
            for inner in range(dim):
                a = row * dim + inner
                b = inner * dim + col
                result[0][offset] += left["a0"][a] * right["b0"][b]
                result[1][offset] += (
                    left["a0"][a] * right["b1"][b]
                    + left["a1"][a] * right["b0"][b]
                )
    return result


def matrix_data(value):
    return {
        "dim": value.nrows(),
        "a0": rows_as_integers(value.m0),
        "a1": rows_as_integers(value.m1),
    }


aws_source = os.path.abspath(os.environ["AWS_SOURCE"])
zeta_suite = os.path.abspath(os.environ["ZETA_SUITE"])
output_path = os.path.abspath(os.environ["OUTPUT"])
if revision(aws_source) != AWS_REVISION:
    raise RuntimeError("AWS_SOURCE is not at pinned revision " + AWS_REVISION)
if revision(zeta_suite) != ZETA_SUITE_REVISION:
    raise RuntimeError("ZETA_SUITE is not at pinned revision " + ZETA_SUITE_REVISION)

os.chdir(aws_source)
load("good-code/P7-utils.sage")
load("good-code/P7-avg-poly.sage")
load("P8/P8-utils.sage")
load(os.path.join(zeta_suite, "sage/loader.sage"))

# Exercise the generic P^mu implementation from P8 on independently assembled
# coefficient matrices.  The integer loop below is the expected result.
p8_a = ZPmuMatrix(
    [matrix(ZZ, 2, 2, [1, -2, 3, 4]), matrix(ZZ, 2, 2, [5, 6, -7, 8]),
     matrix(ZZ, 2, 2, [-9, 10, 11, -12])], 3
)
p8_b = ZPmuMatrix(
    [matrix(ZZ, 2, 2, [-2, 1, 0, 3]), matrix(ZZ, 2, 2, [4, -5, 6, 1]),
     matrix(ZZ, 2, 2, [7, 2, -3, 8])], 3
)
p8_product = p8_a * p8_b
for degree in range(3):
    expected = matrix(ZZ, 2, 2)
    for left_degree in range(degree + 1):
        expected += p8_a.mats[left_degree] * p8_b.mats[degree - left_degree]
    if p8_product.mats[degree] != expected:
        raise AssertionError("P8 generic univariate matrix multiplication mismatch")

case_path = os.path.join(
    zeta_suite, "cases/hyperelliptic_p41_even_degree.json"
)
records = []
for case_id in CASE_IDS:
    case = load_case_by_id(case_path, case_id)
    polynomial, _ = case.curve.hyperelliptic_polynomials()
    coefficients = [ZZ(coefficient) for coefficient in polynomial]
    dimension = int(polynomial.degree())

    left = construct_T_bar_ijp(0, 1, dimension, coefficients)
    right = construct_T_bar_ijp(0, 2, dimension, coefficients)
    aws_product = left * right

    left_data = matrix_data(left)
    right_data = {"b0": rows_as_integers(right.m0),
                  "b1": rows_as_integers(right.m1)}
    aws_output = [rows_as_integers(aws_product.m0),
                  rows_as_integers(aws_product.m1)]
    expected_output = independent_p2_product(left_data, right_data)
    if aws_output != expected_output:
        raise AssertionError(case_id + ": AWS P^2 product differs from direct integer reference")

    records.append({
        "case_id": case_id,
        "dim": dimension,
        "a0": left_data["a0"],
        "a1": left_data["a1"],
        "b0": right_data["b0"],
        "b1": right_data["b1"],
        "c0": aws_output[0],
        "c1": aws_output[1],
    })

with open(output_path, "w") as output:
    output.write("AWS_RING_P2_PRODUCTS 1\n")
    output.write("AWS_COMMIT " + AWS_REVISION + "\n")
    output.write("ZETA_SUITE_COMMIT " + ZETA_SUITE_REVISION + "\n")
    output.write("AWS_SOURCE good-code/P7-utils.sage good-code/P7-avg-poly.sage\n")
    output.write("GENERIC_UNIVARIATE_SOURCE P8/P8-utils.sage\n")
    output.write("P8_GENERIC_REFERENCE_CHECKS 3\n")
    output.write("CASE_COUNT %d\n" % len(records))
    for record in records:
        output.write("CASE %s %d\n" % (record["case_id"], record["dim"]))
        for label in ("a0", "a1", "b0", "b1", "c0", "c1"):
            output.write(label.upper() + "\n")
            output.write(" ".join(str(value) for value in record[label]) + "\n")
        output.write("END_CASE\n")
    output.write("END\n")

print("Captured %d exact P^2 matrix products from AWS %s" %
      (len(records), AWS_REVISION))
