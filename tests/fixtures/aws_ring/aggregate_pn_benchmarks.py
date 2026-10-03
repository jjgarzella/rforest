#!/usr/bin/env python3
"""Aggregate repeated --pn-optimized CSV runs by case."""

import argparse
import csv
import statistics
import sys
from pathlib import Path


TIMING_COLUMNS = (
    "adapter_median_ns",
    "adapter_MAD_ns",
    "block_median_ns",
    "block_MAD_ns",
    "pn_median_ns",
    "pn_MAD_ns",
)


def read_run(path):
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        rows = list(reader)
    return reader.fieldnames, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.runs) < 3 or len(args.runs) % 2 == 0:
        parser.error("provide an odd number of at least three run CSVs")

    try:
        fieldnames, first_rows = read_run(args.runs[0])
        case_names = [row["case"] for row in first_rows]
        runs = [first_rows]
        for path in args.runs[1:]:
            headers, rows = read_run(path)
            if (headers != fieldnames or
                    [row["case"] for row in rows] != case_names):
                raise ValueError(f"{path}: CSV columns or case order differ")
            runs.append(rows)

        aggregated = []
        for row_index, case in enumerate(case_names):
            row = dict(first_rows[row_index])
            medians = {}
            for column in TIMING_COLUMNS[::2]:
                samples = [int(run[row_index][column]) for run in runs]
                center = int(statistics.median(samples))
                deviations = [abs(sample - center) for sample in samples]
                row[column] = str(center)
                row[column.replace("median", "MAD")] = str(
                    int(statistics.median(deviations))
                )
                medians[column] = center

            row["pn_over_adapter"] = (
                f"{medians['pn_median_ns'] / medians['adapter_median_ns']:.3f}"
            )
            row["pn_over_block"] = (
                f"{medians['pn_median_ns'] / medians['block_median_ns']:.3f}"
            )
            aggregated.append(row)

        writer = csv.DictWriter(
            sys.stdout, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(aggregated)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
