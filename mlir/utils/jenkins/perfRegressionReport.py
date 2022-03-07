#!/usr/bin/evn python3

import reportUtils

from pathlib import PurePath
import sys
from typing import Tuple

import pandas as pd

def loadMlirData(filename: str):
    df = pd.read_csv(filename, sep=',', header=0, index_col=False)
    COLUMNS_DROPPED = ['MIOpen TFlops (no MLIR Kernels)', 'MLIR/MIOpen', 'MIOpen TFlops (Tuned MLIR Kernels)',
                       'MIOpen TFlops (Untuned MLIR Kernels)', 'Tuned/Untuned', 'Tuned/MIOpen']
    df.drop(columns=COLUMNS_DROPPED, inplace=True, errors='ignore')
    return df

def summarizeStat(grouped, func, data):
    ret = grouped.agg(func)
    ret.loc[("All", "All"),:] = data.agg(func)
    return ret

def computePerfStats(oldDf: pd.DataFrame, newDf: pd.DataFrame, oldLabel: str, newLabel: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        data = newDf.merge(oldDf, on=reportUtils.TEST_PARAMETERS, suffixes=('_new', '_old'))
    except KeyError as e:
        print("Missing columns in data, forcing copy: ", e, file=sys.stderr)
        return computePerfStats(newDf.copy(), newDf, "forced copy", newLabel)
    if len(data) == 0:
        print("Old and new data have come from disjoint performance runs, ignoring old data",
            file=sys.stderr)
        return computePerfStats(newDf.copy(), newDf, "forced copy", newLabel)

    if (oldLabel == newLabel):
        oldLabel += "_old"
        newLabel += "_new"
    oldLabel = f"MLIR TFlops ({oldLabel})"
    newLabel = f"MLIR TFlops ({newLabel})"
    data.rename(columns={'MLIR TFlops_old': oldLabel,
                         'MLIR TFlops_new': newLabel}, inplace=True)
    data['Current/Previous'] = data[newLabel] / data[oldLabel]

    columnsToAverage = [oldLabel, newLabel, 'Current/Previous']
    STATISTICS = [("Geo. mean", reportUtils.geoMean),
        ("Arith. mean", "mean")]
    grouped = data.groupby(["Direction", "DataType"])[columnsToAverage]
    stats = pd.concat({name: summarizeStat(grouped, func, data[columnsToAverage])
            for name, func in STATISTICS}, axis=0).unstack(level=0)

    return data, stats

def getPerfDate(statsPath: PurePath, default="???"):
    path = statsPath.with_name('perf-run-date')
    try:
        with open(str(path), "r") as f:
            return f.readline().rstrip()
    except FileNotFoundError: # Shouldn't happen once things get running
        return default

if __name__ == '__main__':
    oldDataPath = PurePath(sys.argv[1]) if len(sys.argv) >= 2\
        else PurePath('./', 'oldData/', reportUtils.PERF_REPORT_FILE)
    newDataPath = PurePath(sys.argv[2]) if len(sys.argv) >= 3\
        else PurePath('./', reportUtils.PERF_REPORT_FILE)
    outputPath = PurePath(sys.argv[3]) if len(sys.argv) >= 4\
        else PurePath('./', 'MLIR_Performance_Changes.html')

    try:
        newDf = loadMlirData(str(newDataPath))
        newLabel = getPerfDate(newDataPath, "new")
    except FileNotFoundError:
        print("Could not load current performance data: run ./MIOpenDriver.py or provide a path", file=sys.stderr)
        sys.exit(1)
    try:
        oldDf = loadMlirData(str(oldDataPath))
        oldLabel = getPerfDate(oldDataPath, "old")
    except FileNotFoundError:
        print("Warning: No old performance data, reusing new one", file=sys.stderr)
        oldDf = newDf.copy()
        oldLabel = "copy"

    data, summary = computePerfStats(oldDf, newDf, oldLabel, newLabel)
    with open(outputPath, "w") as outputStream:
        reportUtils.htmlReport(data, summary, "MLIR Performance Changes", ["Current/Previous"], outputStream)
