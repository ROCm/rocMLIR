#!/usr/bin/env python3
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compare tuning result files produced by tuningRunner.py.

Operates on the winner ``.tsv`` files (one best perfConfig per problem). The
comparison is operation agnostic: it keys problems by the ``testVector`` column
and compares the ``TFlops`` (performance) and ``durationSec`` (tuning wall time)
columns. Any number of files can be compared at once; one file is chosen as the
baseline (the first one by default) and every other file is reported relative to
it, along with absolute per-file distributions.

Example::

    ./compareTuningResults.py \\
        tier1-gemm-configs-quick.tsv \\
        tier1-gemm-configs-prob-map.tsv \\
        tier1-gemm-configs-smart.tsv
"""

import argparse
import math
import statistics
import sys
from pathlib import Path

# Columns we need from the winner .tsv. Everything else is ignored, which keeps
# the script op-agnostic (the problem key is the opaque testVector string).
KEY_COL = "testVector"
PERF_COL = "TFlops"
TIME_COL = "durationSec"


class Result:
    """Parsed winner data for a single tuning result file."""

    def __init__(self, path, label):
        self.path = path
        self.label = label
        # testVector -> TFlops
        self.perf = {}
        # testVector -> durationSec
        self.time = {}

    @property
    def problems(self):
        return set(self.perf)


def parse_header(line):
    """Split a header line into column names, tolerating a leading '#'."""
    line = line.lstrip("#").strip("\n")
    return line.split("\t")


def load_file(path, label):
    """Load a winner .tsv into a Result keyed by testVector."""
    result = Result(path, label)
    header = None
    idx = {}
    with open(path, "r") as f:
        for raw in f:
            if not raw.strip():
                continue
            if header is None:
                header = parse_header(raw)
                for col in (KEY_COL, PERF_COL, TIME_COL):
                    if col not in header:
                        raise ValueError(f"{path}: missing required column '{col}'. "
                                         f"Found columns: {header}")
                idx = {col: header.index(col) for col in (KEY_COL, PERF_COL, TIME_COL)}
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) <= max(idx.values()):
                continue
            key = fields[idx[KEY_COL]].strip()
            if not key:
                continue
            result.perf[key] = _to_float(fields[idx[PERF_COL]])
            result.time[key] = _to_float(fields[idx[TIME_COL]])
    if header is None:
        raise ValueError(f"{path}: file has no header row")
    return result


def _to_float(text):
    text = text.strip()
    if not text:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def geomean(values):
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def percentile(sorted_vals, pct):
    """Linear-interpolation percentile over a pre-sorted list."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_vals[low]
    frac = rank - low
    return sorted_vals[low] * (1 - frac) + sorted_vals[high] * frac


def distribution(values):
    """Return a dict of summary statistics for a list of values."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return {
        "n": len(vals),
        "min": vals[0],
        "p10": percentile(vals, 10),
        "p25": percentile(vals, 25),
        "median": percentile(vals, 50),
        "p75": percentile(vals, 75),
        "p90": percentile(vals, 90),
        "max": vals[-1],
        "mean": statistics.fmean(vals),
        "geomean": geomean(vals),
        "sum": sum(vals),
    }


def fmt(value, width=10, prec=3):
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:>{width}.{prec}f}"


def ascii_histogram(values, bins=12, width=40):
    """Build a small ASCII histogram for a list of numeric values."""
    vals = [v for v in values if v is not None]
    if not vals:
        return ["  (no data)"]
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return [f"  {lo:.3f} | {'#' * width} ({len(vals)})"]
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        b = int((v - lo) / step)
        if b == bins:
            b = bins - 1
        counts[b] += 1
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        edge_lo = lo + i * step
        edge_hi = edge_lo + step
        bar = "#" * int(round(width * c / peak))
        lines.append(f"  [{edge_lo:7.3f},{edge_hi:7.3f}) | {bar:<{width}} {c}")
    return lines


def print_distribution_table(title, header_label, results, selector):
    print(f"\n{title}")
    cols = ["label", "n", "min", "p25", "median", "mean", "geomean", "p75", "max", "sum"]
    print("  " + "  ".join(f"{c:>10}" if c != "label" else f"{c:<28}" for c in cols))
    for r in results:
        d = distribution([selector(r, k) for k in r.problems])
        if d is None:
            print(f"  {r.label:<28}  (no data)")
            continue
        print("  " + f"{r.label:<28}" + "  " + "  ".join([
            fmt(d["n"], prec=0),
            fmt(d["min"]),
            fmt(d["p25"]),
            fmt(d["median"]),
            fmt(d["mean"]),
            fmt(d["geomean"]),
            fmt(d["p75"]),
            fmt(d["max"]),
            fmt(d["sum"], prec=1),
        ]))


def compare_to_baseline(results, baseline, common, selector, higher_is_better, unit):
    """Print per-problem ratio distributions of each file vs the baseline."""
    print(f"\nPer-problem ratio vs baseline '{baseline.label}' "
          f"(over {len(common)} shared problems)")
    if higher_is_better:
        print("  ratio = file / baseline   (>1 means file is better)")
    else:
        print("  ratio = file / baseline   (<1 means file is faster/cheaper)")
    for r in results:
        if r is baseline:
            continue
        ratios = []
        wins = ties = losses = 0
        for k in common:
            b = selector(baseline, k)
            v = selector(r, k)
            if b is None or v is None or b == 0:
                continue
            ratio = v / b
            ratios.append(ratio)
            # "Win" always means the better outcome for this metric.
            better = ratio > 1.0 if higher_is_better else ratio < 1.0
            worse = ratio < 1.0 if higher_is_better else ratio > 1.0
            if math.isclose(ratio, 1.0, rel_tol=1e-9):
                ties += 1
            elif better:
                wins += 1
            elif worse:
                losses += 1
        d = distribution(ratios)
        print(f"\n  {r.label}  ({unit})")
        if d is None:
            print("    (no comparable problems)")
            continue
        print(f"    geomean ratio : {d['geomean']:.4f}")
        print(f"    median  ratio : {d['median']:.4f}")
        print(f"    min/max ratio : {d['min']:.4f} / {d['max']:.4f}")
        print(f"    p10/p90 ratio : {d['p10']:.4f} / {d['p90']:.4f}")
        print(f"    better/equal/worse : {wins} / {ties} / {losses}  (of {len(ratios)})")
        print("    ratio distribution:")
        for line in ascii_histogram(ratios):
            print("    " + line)


def report_coverage(results):
    all_problems = set()
    for r in results:
        all_problems |= r.problems
    common = set(results[0].problems)
    for r in results[1:]:
        common &= r.problems
    print("Problem coverage")
    print(f"  union of problems across all files : {len(all_problems)}")
    print(f"  intersection (shared by all files) : {len(common)}")
    for r in results:
        missing = len(all_problems - r.problems)
        print(f"    {r.label:<28} problems={len(r.problems):<6} missing={missing}")
    return common


def derive_label(path):
    name = Path(path).name
    for suffix in (".tsv.debug", ".tsv"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def main(argv=None):
    parser = argparse.ArgumentParser(prog="compareTuningResults.py",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument("files",
                        nargs="+",
                        type=Path,
                        help="Winner .tsv files to compare (>=1, op-agnostic)")
    parser.add_argument("-b",
                        "--baseline",
                        type=int,
                        default=0,
                        help="Index (0-based) of the baseline file (default: 0)")
    parser.add_argument("-l",
                        "--labels",
                        nargs="+",
                        default=None,
                        help="Custom labels, one per input file")
    args = parser.parse_args(argv)

    if args.labels and len(args.labels) != len(args.files):
        parser.error("number of --labels must match number of files")
    if not 0 <= args.baseline < len(args.files):
        parser.error("--baseline index out of range")

    results = []
    for i, path in enumerate(args.files):
        if not path.exists():
            parser.error(f"file not found: {path}")
        label = args.labels[i] if args.labels else derive_label(path)
        results.append(load_file(path, label))

    baseline = results[args.baseline]

    print("=" * 78)
    print("Tuning result comparison")
    print("=" * 78)
    for r in results:
        tag = "  [baseline]" if r is baseline else ""
        print(f"  {r.label:<28} <- {r.path}{tag}")

    common = report_coverage(results)

    print("\n" + "=" * 78)
    print("PERFORMANCE  (TFlops, higher is better)")
    print("=" * 78)
    print_distribution_table("Absolute TFlops distribution per file:", "TFlops", results,
                             lambda r, k: r.perf.get(k))
    compare_to_baseline(results,
                        baseline,
                        common,
                        lambda r, k: r.perf.get(k),
                        higher_is_better=True,
                        unit="TFlops ratio")

    print("\n" + "=" * 78)
    print("TUNING WALL TIME  (durationSec, lower is better)")
    print("=" * 78)
    print_distribution_table("Per-problem durationSec distribution per file:", "durationSec",
                             results, lambda r, k: r.time.get(k))
    print("\nTotal tuning wall time per file:")
    base_total = sum(v for v in baseline.time.values() if v is not None)
    for r in results:
        total = sum(v for v in r.time.values() if v is not None)
        ratio = (total / base_total) if base_total else float("nan")
        rel = "" if r is baseline else f"   ({ratio:.3f}x baseline)"
        print(f"  {r.label:<28} {total:12.1f} s ({total / 60:.1f} min){rel}")
    compare_to_baseline(results,
                        baseline,
                        common,
                        lambda r, k: r.time.get(k),
                        higher_is_better=False,
                        unit="durationSec ratio")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
