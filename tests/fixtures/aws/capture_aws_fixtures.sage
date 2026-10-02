#!/usr/bin/env sage
"""Capture small exact fixtures from the pinned AWS Sage implementation.

Run from any directory with AWS_SOURCE, ZETA_SUITE, and FIXTURE_DIR set.
The checked-in AWS test and implementation files are symlinked into a fresh
scratch layout so their relative Sage load paths resolve without modifying
either source checkout.
"""

import os
import subprocess
import sys
import tempfile
import time
from types import ModuleType


AWS_REVISION = "a1fc50dd667d262b5d83d1d4ceb1499bdbead288"
ZETA_REVISION = "621107b12200a3c234c2e3dd6a4ba9dc87be9bea"
PYRFOREST_WRAPPER_REVISION = "8317488b07a8f9519c0af384f83e1c807dcee5d9"
CASE_IDS = ["p41_g1_d4_001", "p41_g3_d8_001", "p41_g8_d18_001"]
CAPTURED_CALLS = {
    ("p41_g1_d4_001", "factorial_i0"),
    ("p41_g1_d4_001", "block_i0"),
    ("p41_g3_d8_001", "block_i1"),
    ("p41_g8_d18_001", "block_i0"),
}
CALL_NAMES = ["block_i0", "block_i1", "factorial_i0", "factorial_i1"]


def git_revision(path):
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()


def require_link(source, destination):
    source = os.path.abspath(source)
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    if os.path.lexists(destination):
        if os.path.realpath(destination) != source:
            raise RuntimeError("scratch path already exists with different content: " + destination)
    else:
        os.symlink(source, destination)


aws_source = os.path.abspath(os.environ["AWS_SOURCE"])
zeta_suite = os.path.abspath(os.environ["ZETA_SUITE"])
fixture_dir = os.path.abspath(os.environ["FIXTURE_DIR"])
if git_revision(aws_source) != AWS_REVISION:
    raise RuntimeError("AWS_SOURCE is not at the pinned revision " + AWS_REVISION)
if git_revision(zeta_suite) != ZETA_REVISION:
    raise RuntimeError("ZETA_SUITE is not at the pinned revision " + ZETA_REVISION)

scratch = os.path.abspath(
    os.environ.get("AWS_CAPTURE_SCRATCH")
    or tempfile.mkdtemp(prefix="aws-rforest-fixtures-")
)
require_link(
    os.path.join(aws_source, "pyrforest", "P7-avg-poly.sage"),
    os.path.join(scratch, "pyrforest", "P7-avg-poly.sage"),
)
require_link(
    os.path.join(aws_source, "pyrforest", "P7-utils.sage"),
    os.path.join(scratch, "pyrforest", "P7-utils.sage"),
)
require_link(
    os.path.join(aws_source, "tests", "test_avg_poly_pyrforest_p41.sage"),
    os.path.join(scratch, "tests", "test_avg_poly_pyrforest_p41.sage"),
)
require_link(
    os.path.join(zeta_suite, "sage"),
    os.path.join(scratch, "tests", "hyperell_suite", "sage"),
)
require_link(
    os.path.join(zeta_suite, "cases"),
    os.path.join(scratch, "tests", "hyperell_suite", "cases"),
)
os.makedirs(fixture_dir, exist_ok=True)
os.chdir(scratch)


# The AWS source imports this module directly. This replacement follows the
# documented product V*M(kbase)*...*M(k(p)-1) with Sage modular matrices,
# recording the actual call arguments before it returns each full matrix.
capture_module = ModuleType("pyrforest")
capture_records = []
active_case_id = None
active_call_index = 0


def sequential_remainder_forest(M, m, k, kbase=0, indices=None, V=None, ans=None, kappa=None, **kwargs):
    global active_call_index
    if indices is None:
        raise RuntimeError("AWS call did not supply its ordered indices")
    indices = list(indices)
    call_name = CALL_NAMES[active_call_index]
    active_call_index += 1

    dim = M.nrows()
    if M.ncols() != dim:
        raise RuntimeError("AWS call supplied a non-square matrix")
    rows = dim if V is None else V.nrows()
    deg = max(
        [0]
        + [int(M[i, j].degree()) for i in range(dim) for j in range(dim) if M[i, j] not in ZZ]
    )
    initial_v = [
        int(V[i, j]) if V is not None else int(i == j)
        for i in range(rows)
        for j in range(dim)
    ]
    moduli = [int(m(index) if callable(m) else m[index]) for index in indices]
    endpoints = [int(k(index) if callable(k) else k[index]) for index in indices]
    if kappa is None:
        effective_kappa = 1 if len(indices) <= 1 else int(ceil(log(log(len(indices), 2), 2)) + 1)
    else:
        effective_kappa = int(kappa)

    matrix_coefficients = []
    for i in range(dim):
        for j in range(dim):
            entry = M[i, j]
            for coefficient_index in range(deg + 1):
                if entry in ZZ:
                    coefficient = int(entry) if coefficient_index == 0 else 0
                else:
                    coefficient = int(entry[coefficient_index])
                matrix_coefficients.append(coefficient)

    result = {}
    outputs = []
    for index, modulus, endpoint in zip(indices, moduli, endpoints):
        ring = Integers(modulus)
        accumulator = matrix(ring, rows, dim, initial_v)
        for argument in range(int(kbase), endpoint):
            evaluated_entries = []
            for i in range(dim):
                for j in range(dim):
                    entry = M[i, j]
                    value = entry if entry in ZZ else entry(argument)
                    evaluated_entries.append(int(value) % modulus)
            evaluated_matrix = matrix(ring, dim, dim, evaluated_entries)
            accumulator *= evaluated_matrix
        output = [int(value) % modulus for value in accumulator.list()]
        result[index] = matrix(ZZ, rows, dim, output)
        outputs.append({"index": int(index), "modulus": modulus, "endpoint": endpoint, "values": output})

    capture_records.append(
        {
            "case_id": active_case_id,
            "call_name": call_name,
            "dim": dim,
            "rows": rows,
            "deg": deg,
            "kbase": int(kbase),
            "kappa": effective_kappa,
            "initial_z": _product(moduli),
            "moduli": moduli,
            "endpoints": endpoints,
            "initial_v": initial_v,
            "matrix_coefficients": matrix_coefficients,
            "outputs": outputs,
        }
    )

    if ans is not None:
        for index in indices:
            ans[index] *= result[index]
        return None
    return result


def _product(values):
    product_value = 1
    for value in values:
        product_value *= int(value)
    return product_value


capture_module.remainder_forest = sequential_remainder_forest
sys.modules["pyrforest"] = capture_module

# Load the original AWS p=41 unittest unchanged. Its per-case L-polynomial
# comparison is the final zeta reference check for each selected curve.
load("tests/test_avg_poly_pyrforest_p41.sage")


def write_fixture(record):
    case_id = record["case_id"]
    call_name = record["call_name"]
    fixture_name = case_id + "_" + call_name + ".rf"
    path = os.path.join(fixture_dir, fixture_name)
    with open(path, "w", encoding="ascii") as fixture:
        fixture.write("RFOREST_FIXTURE 1\n")
        fixture.write("CASE " + case_id + "\n")
        fixture.write("CALL " + call_name + "\n")
        fixture.write("AWS_COMMIT " + AWS_REVISION + "\n")
        fixture.write("ZETA_SUITE_COMMIT " + ZETA_REVISION + "\n")
        fixture.write("PYRFOREST_WRAPPER_COMMIT " + PYRFOREST_WRAPPER_REVISION + "\n")
        fixture.write("REFERENCE_PRIME 41\n")
        fixture.write("DIM {}\n".format(record["dim"]))
        fixture.write("ROWS {}\n".format(record["rows"]))
        fixture.write("DEG {}\n".format(record["deg"]))
        fixture.write("N {}\n".format(len(record["moduli"])))
        fixture.write("KBASE {}\n".format(record["kbase"]))
        fixture.write("KAPPA {}\n".format(record["kappa"]))
        fixture.write("INITIAL_Z {}\n".format(record["initial_z"]))
        fixture.write("MODULI\n")
        for output, modulus, endpoint in zip(
            record["outputs"], record["moduli"], record["endpoints"]
        ):
            fixture.write("{} {} {}\n".format(output["index"], modulus, endpoint))
        fixture.write("INITIAL_V\n")
        for row in range(record["rows"]):
            start = row * record["dim"]
            end = start + record["dim"]
            fixture.write(" ".join(str(value) for value in record["initial_v"][start:end]) + "\n")
        fixture.write("MATRIX_COEFFICIENTS\n")
        stride = record["deg"] + 1
        for entry in range(record["dim"] * record["dim"]):
            start = entry * stride
            end = start + stride
            fixture.write(
                " ".join(str(value) for value in record["matrix_coefficients"][start:end]) + "\n"
            )
        fixture.write("EXPECTED_MATRICES\n")
        for output in record["outputs"]:
            fixture.write("PRIME {}\n".format(output["index"]))
            for row in range(record["rows"]):
                start = row * record["dim"]
                end = start + record["dim"]
                fixture.write(" ".join(str(value) for value in output["values"][start:end]) + "\n")
        fixture.write("END\n")
    print("wrote " + path)


started = time.perf_counter()
for case_id in CASE_IDS:
    active_case_id = case_id
    active_call_index = 0

    def selected_cases(path, selected_id=case_id):
        return [load_case_by_id(path, selected_id)]

    load_cases = selected_cases
    suite = unittest.TestSuite([TestAvgPolyPyrforestP41EvenDegree("test_cases")])
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("original AWS p=41 comparison failed for " + case_id)
    if active_call_index != 4:
        raise RuntimeError("expected four actual AWS remainder_forest calls for " + case_id)

written = set()
for record in capture_records:
    key = (record["case_id"], record["call_name"])
    if key in CAPTURED_CALLS:
        write_fixture(record)
        written.add(key)
if written != CAPTURED_CALLS:
    raise RuntimeError("capture set mismatch: wrote {!r}, expected {!r}".format(written, CAPTURED_CALLS))

print("validated {} selected AWS p=41 L-polynomial comparisons".format(len(CASE_IDS)))
print("captured {} fixture calls with independent sequential modular products".format(len(written)))
print("capture and comparison runtime: {:.3f}s".format(time.perf_counter() - started))
