#!/usr/bin/evn python3

import reportUtils

from pathlib import PurePath
import sys
from typing import Tuple

import pandas as pd

def loadMlirData(filename: str):
    df = pd.read_csv(filename, sep=',', header=0, index_col=False)
    COLUMNS_DROPPED = ['MIOpen TFlops (no MLIR Kernels)', 'MLIR/MIOpen', 'MIOpen TFlops (Tuned MLIR Kernels)',
                       'MIOpen TFlops (Untuned MLIR Kernels)', 'Tuned/Untuned', 'Tuned/MIOpen',
                       'rocBLAS TFlops (no MLIR kernels)', 'MLIR/rocBLAS', 'Tuned/rocBLAS']
    df.drop(columns=COLUMNS_DROPPED, inplace=True, errors='ignore')
    # Work around empty PerfConfig field whin migrating from no tuning to yes tuning
    # Can be removed next time we touch this
    if 'PerfConfig' in df:
        df['PerfConfig'] = df['PerfConfig'].fillna('None')
    return df

def mergePerfConfigs(v: Tuple[str, str]) -> str:
    v1, v2 = v
    if v1 == v2:
        return v1
    return f"{v1} -> {v2}"

def summarizeStat(grouped, func, data):
    ret = grouped.agg(func)
    ret.loc[("All", "All", "All"),:] = data.agg(func)
    return ret

def computePerfStats(oldDf: pd.DataFrame, newDf: pd.DataFrame, oldLabel: str, newLabel: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    isGemm = "TransA" in newDf
    # Ignore perf config in join
    joinCols = reportUtils.GEMM_TEST_PARAMETERS[:-1] if isGemm else reportUtils.CONV_TEST_PARAMETERS[:-1]
    try:
        data = newDf.merge(oldDf, on=joinCols, suffixes=('_new', '_old'))
    except KeyError as e:
        print("Missing columns in data, forcing copy: ", e, file=sys.stderr)
        return computePerfStats(newDf.copy(), newDf, "forced copy", newLabel)
    if len(data) == 0:
        print("Old and new data have come from disjoint performance runs, ignoring old data",
            file=sys.stderr)
        return computePerfStats(newDf.copy(), newDf, "forced copy", newLabel)

    # Clean up PerfConfig columns, as the report generator wants a single PerfConfig
    if "PerfConfig_old" in data and "PerfConfig_new" in data:
        perfConfigColPos = data.columns.get_loc("PerfConfig_old")
        zipped = list(map(mergePerfConfigs, zip(data["PerfConfig_old"], data["PerfConfig_new"])))
        data.insert(perfConfigColPos, "PerfConfig", zipped)
        data.drop(columns=["PerfConfig_old", "PerfConfig_new"], inplace=True)

    if (oldLabel == newLabel):
        oldLabel += "_old"
        newLabel += "_new"
    oldLabel = f"MLIR TFlops ({oldLabel})"
    newLabel = f"MLIR TFlops ({newLabel})"
    oldLabelTuned = f"Tuned TFlops ({oldLabel})"
    newLabelTuned = f"Tuned TFlops ({newLabel})"
    data.rename(columns={'MLIR TFlops_old': oldLabel,
                         'MLIR TFlops_new': newLabel,
                         'TFlops_old': oldLabel,
                         'TFlops_new': newLabel,
                         'Tuned MLIR TFlops_old': oldLabelTuned,
                         'Tuned MLIR TFlops_new': newLabelTuned}, inplace=True)
    data['Current/Previous'] = data[newLabel] / data[oldLabel]
    hasTuning = False
    if oldLabelTuned in data and newLabelTuned in data:
        data['Tuned Current/Previous'] = data[newLabelTuned] / data[oldLabelTuned]
        hasTuning = True

    columnsToAverage = ['Current/Previous', oldLabel, newLabel]
    if hasTuning:
        columnsToAverage += ['Tuned Current/Previous', oldLabelTuned, newLabelTuned]
    STATISTICS = [("Geo. mean", reportUtils.geoMean),
        ("Arith. mean", "mean")]
    groups = ["DataType", "TransA", "TransB"] if isGemm else ["Direction", "DataType", "InputLayout"]
    grouped = data.groupby(groups)[columnsToAverage]
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
    chip = sys.argv[1]
    oldDataPath = PurePath(sys.argv[2]) if len(sys.argv) >= 3\
        else PurePath('./', 'oldData/', chip + '_' + reportUtils.PERF_REPORT_FILE)
    newDataPath = PurePath(sys.argv[3]) if len(sys.argv) >= 4\
        else PurePath('./', chip + '_' + reportUtils.PERF_REPORT_FILE)
    outputPath = PurePath(sys.argv[4]) if len(sys.argv) >= 5\
        else PurePath('./', chip + '_' + 'MLIR_Performance_Changes.html')

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
    isGemm = ("TransA" in data)
    hasTuning = ("Tuned Current/Previous" in data)
    if isGemm and len(sys.argv) < 5:
        outputPath = PurePath('./', chip + '_' + 'MLIR_Performance_Changes_Gemm.html')
    with open(outputPath, "w") as outputStream:
        toHighlight = ["Current/Previous", "Tuned Current/Previous"] if hasTuning \
            else ["Current/Previous"]
        reportUtils.htmlReport(data, summary,
            "MLIR Performance Changes, " + ("GEMM" if isGemm else "Conv"),
            toHighlight, outputStream)
