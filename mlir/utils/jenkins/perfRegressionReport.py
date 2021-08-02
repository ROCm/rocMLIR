#!/usr/bin/evn python3

import reportUtils

from pathlib import PurePath
import sys
from typing import Tuple

import pandas as pd

def loadMlirData(filename: str):
    df = pd.read_csv(filename, sep=',', header=0, index_col=False)
    df.drop(columns=['MIOpen TFlops'], inplace=True, errors='ignore')
    return df

def computePerfStats(oldDf: pd.DataFrame, newDf: pd.DataFrame, oldLabel: str, newLabel: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    oldLabel = f"MLIR TFlops ({oldLabel})"
    newLabel = f"MLIR TFlops ({newLabel})"

    data = newDf.merge(oldDf, on=reportUtils.TEST_PARAMETERS, suffixes=('_new', '_old'))
    data.rename(columns={'MLIR TFlops_old': oldLabel, 'MLIR TFlops_new': newLabel}, inplace=True)
    data.drop(columns=['MLIR/MIOpen_old', 'MLIR/MIOpen_new'], inplace=True)
    data['Current/Previous'] = data[newLabel] / data[oldLabel]

    COLUMNS_TO_AVERAGE = [oldLabel, newLabel, 'Current/Previous']
    means = reportUtils.geoMean(data[COLUMNS_TO_AVERAGE])
    means = pd.Series(means, index=COLUMNS_TO_AVERAGE)
    means.name = "Geo. mean"
    arithMeans = data[COLUMNS_TO_AVERAGE].mean(axis=0)
    arithMeans.name = "Arith. mean"
    stdDevs = data[COLUMNS_TO_AVERAGE].std(axis = 0)
    stdDevs.name = "Std. dev"

    stats = pd.DataFrame([means, arithMeans, stdDevs])

    return data, stats

def getPerfDate(statsPath: PurePath, default="???"):
    path = statsPath.with_name('perf-run-date')
    try:
        with open(str(path), "r") as f:
            return f.readline().rstrip()
    except FileNotFoundError: # Shouldn't happen once things get running
        return default

if __name__ == '__main__':
    oldDataPath = PurePath(sys.argv[1]) if len(sys.argv) >= 2 else PurePath('./', 'oldData/', reportUtils.PERF_REPORT_FILE)
    newDataPath = PurePath(sys.argv[2]) if len(sys.argv) >= 3 else PurePath('./', reportUtils.PERF_REPORT_FILE)
    outputPath = PurePath(sys.argv[3]) if len(sys.argv) >= 4 else PurePath('./', 'MLIR_Performance_Changes.html')

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
        reportUtils.htmlReport(data, summary, "MLIR Performance Changes", "Current/Previous", outputStream)
