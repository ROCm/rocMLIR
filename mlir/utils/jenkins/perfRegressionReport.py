#!/usr/bin/evn python3

import reportUtils

from pathlib import PurePath
import sys
from typing import Tuple

import pandas as pd

def loadMlirData(filename):
    df = pd.read_csv(filename, sep=',', header=0, index_col=False)
    df.drop(columns=['MIOpen TFlops'], inplace=True, errors='ignore')
    return df

def computePerfStats(oldDf: pd.DataFrame, newDf: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = newDf.merge(oldDf, on=reportUtils.TEST_PARAMETERS, suffixes=('_new', '_old'))
    data.rename(columns={'TFlops_old': 'TFlops (old)', 'TFlops_new': 'TFlops (new)'}, inplace=True)
    data.drop(columns=['Speedup_old', 'Speedup_new'], inplace=True)
    data['Speedup'] = data['TFlops (new)'] / data['TFlops (old)']

    COLUMNS_TO_AVERAGE = ['TFlops (old)', 'TFlops (new)', 'Speedup']
    means = reportUtils.geoMean(data[COLUMNS_TO_AVERAGE])
    means = pd.Series(means, index=COLUMNS_TO_AVERAGE)
    means.name = "Geo. mean"
    arithMeans = data[COLUMNS_TO_AVERAGE].mean(axis=0)
    arithMeans.name = "Arith. mean"
    stdDevs = data[COLUMNS_TO_AVERAGE].std(axis = 0)
    stdDevs.name = "Std. dev"

    stats = pd.DataFrame([means, arithMeans, stdDevs])

    return data, stats

if __name__ == '__main__':
    oldDataPath = PurePath(sys.argv[1]) if len(sys.argv) >= 2 else PurePath('./', 'oldData/', reportUtils.PERF_REPORT_FILE)
    newDataPath = PurePath(sys.argv[2]) if len(sys.argv) >= 3 else PurePath('./', reportUtils.PERF_REPORT_FILE)
    outputPath = PurePath(sys.argv[3]) if len(sys.argv) >= 4 else PurePath('./', 'MLIR_Performance_Changes.html')

    try:
        newDf = loadMlirData(str(newDataPath))
    except FileNotFoundError:
        print("Could not load current performance data: run ./MIOpenDriver.py or provide a path", file=sys.stderr)
        sys.exit(1)
    try:
        oldDf = loadMlirData(str(oldDataPath))
    except FileNotFoundError:
        print("Warning: No old performance data, reusing new one", file=sys.stderr)
        oldDf = newDf.copy()
    
    data, summary = computePerfStats(oldDf, newDf)
    with open(outputPath, "w") as outputStream:
        reportUtils.htmlReport(data, summary, "MLIR Performance Changes", outputStream)
