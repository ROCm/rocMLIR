#!/usr/bin/env python3
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Inspect the perfConfigs measured during (smart) tuning.

Operates on the ``.tsv.debug`` files produced by tuningRunner.py, which contain
*every* measured perfConfig (not just the winner). For each file it reports:

  * how many measured perfConfigs were non-applicable (the kernel could not be
    compiled/run, so it has no TFlops/Stats), and
  * how many measured perfConfigs request Split-K > 1.

The Split-K extraction mirrors ``get_splitk_value`` in quickTuningGen.py, which
itself tracks the serialization in RockAttrDefs.td, so it is op-agnostic across
gemm / conv / attention perfConfig layouts.
"""

import argparse
import sys
from pathlib import Path

PERF_COL = "PerfConfig"
PERF_VAL_COL = "TFlops"
STATS_COL = "Stats"


def parse_perfconfig(perfconfig):
    """Parse a perfconfig string into (format, version, params).

    Kept in sync with parse_perfconfig() in quickTuningGen.py.
    """
    parts = perfconfig.split(":")
    if len(parts) == 3:
        return parts[0], int(parts[1][1:]), parts[2].split(",")
    if len(parts) == 2:
        if parts[0].startswith("v"):
            return None, int(parts[0][1:]), parts[1].split(",")
        return parts[0], 1, parts[1].split(",")
    return None, 1, perfconfig.split(",")


def get_splitk_value(perfconfig):
    """Extract the Split-K value from a perfconfig string.

    Kept in sync with get_splitk_value() in quickTuningGen.py.
    """
    fmt, version, params = parse_perfconfig(perfconfig)

    idx = None
    if fmt == "attn":
        if version >= 3:
            idx = 8
        elif version >= 2:
            idx = 7
    else:
        if version >= 4:
            idx = 7
        elif version >= 2:
            idx = 6

    if idx is not None and idx < len(params):
        try:
            return int(params[idx])
        except ValueError:
            return None
    return None


def is_applicable(perf_text, stats_text):
    """A measurement is applicable iff it produced a TFlops number.

    Non-applicable configs (compile/launch rejected) come back with empty TFlops
    and empty Stats in the .debug file.
    """
    if perf_text.strip():
        return True
    if stats_text.strip():
        return True
    return False


class FileStats:

    def __init__(self, label):
        self.label = label
        self.total = 0
        self.non_applicable = 0
        self.splitk_gt1 = 0
        self.splitk_gt1_applicable = 0
        self.unparsed_splitk = 0


def analyze_file(path, label):
    stats = FileStats(label)
    header = None
    idx = {}
    with open(path, "r") as f:
        for raw in f:
            if header is None:
                header = raw.rstrip("\n").split("\t")
                for col in (PERF_COL, PERF_VAL_COL, STATS_COL):
                    if col not in header:
                        raise ValueError(f"{path}: missing required column '{col}'. "
                                         f"Found columns: {header}")
                idx = {c: header.index(c) for c in (PERF_COL, PERF_VAL_COL, STATS_COL)}
                continue
            if not raw.strip():
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) <= max(idx.values()):
                # Trailing empty (non-applicable) cells may be dropped; pad them.
                fields += [""] * (max(idx.values()) + 1 - len(fields))

            stats.total += 1

            perf_text = fields[idx[PERF_VAL_COL]]
            stats_text = fields[idx[STATS_COL]]
            applicable = is_applicable(perf_text, stats_text)
            if not applicable:
                stats.non_applicable += 1

            perfconfig = fields[idx[PERF_COL]].strip()
            splitk = get_splitk_value(perfconfig)
            if splitk is None:
                stats.unparsed_splitk += 1
            elif splitk > 1:
                stats.splitk_gt1 += 1
                if applicable:
                    stats.splitk_gt1_applicable += 1
    if header is None:
        raise ValueError(f"{path}: file has no header row")
    return stats


def pct(num, denom):
    return (100.0 * num / denom) if denom else 0.0


def report(all_stats):
    print("=" * 78)
    print("Smart-tuning measured-perfConfig statistics (.tsv.debug)")
    print("=" * 78)
    for s in all_stats:
        applicable = s.total - s.non_applicable
        print(f"\n{s.label}")
        print(f"  total measured perfConfigs : {s.total}")
        print(f"  non-applicable             : {s.non_applicable:>8} "
              f"({pct(s.non_applicable, s.total):5.2f}%)")
        print(f"  applicable                 : {applicable:>8} "
              f"({pct(applicable, s.total):5.2f}%)")
        print(f"  Split-K > 1 (all)          : {s.splitk_gt1:>8} "
              f"({pct(s.splitk_gt1, s.total):5.2f}%)")
        print(f"  Split-K > 1 (applicable)   : {s.splitk_gt1_applicable:>8} "
              f"({pct(s.splitk_gt1_applicable, s.total):5.2f}% of total)")
        if s.unparsed_splitk:
            print(f"  perfConfigs w/o parseable Split-K : {s.unparsed_splitk}")

    if len(all_stats) > 1:
        print("\n" + "-" * 78)
        print("Combined")
        tot = sum(s.total for s in all_stats)
        na = sum(s.non_applicable for s in all_stats)
        sk = sum(s.splitk_gt1 for s in all_stats)
        print(f"  total measured perfConfigs : {tot}")
        print(f"  non-applicable             : {na} ({pct(na, tot):.2f}%)")
        print(f"  Split-K > 1                : {sk} ({pct(sk, tot):.2f}%)")
    print()


def derive_label(path):
    return Path(path).name


def main(argv=None):
    parser = argparse.ArgumentParser(prog="smartTuningConfigStats.py",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument("files",
                        nargs="+",
                        type=Path,
                        help="One or more .tsv.debug files to analyze")
    args = parser.parse_args(argv)

    all_stats = []
    for path in args.files:
        if not path.exists():
            parser.error(f"file not found: {path}")
        all_stats.append(analyze_file(path, derive_label(path)))

    report(all_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
