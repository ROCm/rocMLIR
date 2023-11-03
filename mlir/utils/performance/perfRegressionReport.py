#!/usr/bin/evn python3

import reportUtils
import perfRunner

from pathlib import PurePath
from perfRunner import getNumCU
import sys
from typing import Tuple

import pandas as pd

def loadMlirData(filename: str):
    df = pd.read_csv(filename, sep=',', header=0, index_col=False)
    COLUMNS_DROPPED = ['MIOpen TFlops (no MLIR Kernels)', 'MLIR/MIOpen', 'MIOpen TFlops (Tuned MLIR Kernels)',
                       'MIOpen TFlops (Untuned MLIR Kernels)', 'Tuned/Untuned', 'Tuned/MIOpen',
                       'rocBLAS TFlops (no MLIR Kernels)', 'MLIR/rocBLAS', 'Tuned/rocBLAS', 'Quick Tuned/rocBLAS',
                       'Quick Tuned/MIOpen', 'Quick Tuned/Untuned', 'Quick Tuned/Tuned' ]
    df.drop(columns=COLUMNS_DROPPED, inplace=True, errors='ignore')
    # Work around empty PerfConfig field whin migrating from no tuning to yes tuning
    # Can be removed next time we touch this
    if 'PerfConfig' in df:
        df['PerfConfig'] = df['PerfConfig'].fillna('None')
    if 'numCU' not in df:
        df.insert(4, 'numCU', getNumCU(df['Chip'][0]))
    return df

def mergePerfConfigs(v: Tuple[str, str]) -> str:
    v1, v2 = v
    if v1 == v2:
        return v1
    return f"{v1} -> {v2}"

def summarizeStat(grouped, func, data):
    ret = grouped.agg(func)
    if ret.index.nlevels == 1:
        ret.loc["All"] = data.agg(func)
    else:
        ret.loc[("All",) * ret.index.nlevels,:] = data.agg(func)
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

    if "PerfConfig (quick tuned)_old" in data and "PerfConfig (quick tuned)_new" in data:
        perfConfigColPos = data.columns.get_loc("PerfConfig (quick tuned)_old")
        zipped = list(map(mergePerfConfigs, zip(data["PerfConfig (quick tuned)_old"], data["PerfConfig (quick tuned)_new"])))
        data.insert(perfConfigColPos, "PerfConfig (quick tuned)", zipped)
        data.drop(columns=["PerfConfig (quick tuned)_old", "PerfConfig (quick tuned)_new"], inplace=True)

    if (oldLabel == newLabel):
        oldLabel += "_old"
        newLabel += "_new"
    oldLabel = f"MLIR TFlops ({oldLabel})"
    newLabel = f"MLIR TFlops ({newLabel})"
    oldLabelTuned = f"Tuned TFlops ({oldLabel})"
    newLabelTuned = f"Tuned TFlops ({newLabel})"
    oldLabelQuickTuned = f"Quick Tuned TFlops ({oldLabel})"
    newLabelQuickTuned = f"Quick Tuned TFlops ({newLabel})"
    data.rename(columns={'MLIR TFlops_old': oldLabel,
                         'MLIR TFlops_new': newLabel,
                         'TFlops_old': oldLabel,
                         'TFlops_new': newLabel,
                         'Tuned MLIR TFlops_old': oldLabelTuned,
                         'Tuned MLIR TFlops_new': newLabelTuned,
                         "Quick Tuned MLIR TFlops_old": oldLabelQuickTuned,
                         "Quick Tuned MLIR TFlops_new": newLabelQuickTuned}, inplace=True)
    data['% change'] = 100.0 * (data[newLabel] - data[oldLabel]) / data[oldLabel]
    hasTuning = False
    hasQuickTuning = False
    if oldLabelTuned in data and newLabelTuned in data:
        data['% change (tuned)'] = 100.0 * (data[newLabelTuned] - data[oldLabelTuned]) / data[oldLabelTuned]
        hasTuning = True
    if oldLabelQuickTuned in data and newLabelQuickTuned in data:
        data['% change (quick tuned)'] = 100.0 * (data[newLabelQuickTuned] - data[oldLabelQuickTuned]) / data[oldLabelQuickTuned]
        hasQuickTuning = True
    columnsToAverage = ['% change', oldLabel, newLabel]
    if hasTuning:
        columnsToAverage += ['% change (tuned)', oldLabelTuned, newLabelTuned]
    if hasQuickTuning:
         columnsToAverage += ['% change (quick tuned)', oldLabelQuickTuned, newLabelQuickTuned]
    STATISTICS = [("Geo. mean", reportUtils.geoMean),
        ("Arith. mean", "mean")]
    groups = ["DataType"] if isGemm else ["Direction", "DataType", "InputLayout"]
    grouped = data.groupby(groups)[columnsToAverage]
    stats = pd.concat({name: summarizeStat(grouped, func, data[columnsToAverage])
            for name, func in STATISTICS}, axis=0).unstack(level=0)
    stats.drop(columns=[('% change', 'Geo. mean'), ('% change (tuned)', 'Geo. mean'), ('% change (quick tuned)', 'Geo. mean')],
        errors='ignore', inplace=True)

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
        else PurePath('./', 'oldData/', chip + '_' + reportUtils.PERF_REPORT_FILE['MIOpen'])
    newDataPath = PurePath(sys.argv[3]) if len(sys.argv) >= 4\
        else PurePath('./', chip + '_' + reportUtils.PERF_REPORT_FILE['MIOpen'])
    outputPath = PurePath(sys.argv[4]) if len(sys.argv) >= 5\
        else PurePath('./', chip + '_' + 'MLIR_Performance_Changes.html')

    try:
        newDf = loadMlirData(str(newDataPath))
        newLabel = getPerfDate(newDataPath, "new")
    except FileNotFoundError:
        print("Could not load current performance data: run perf or provide a path", file=sys.stderr)
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
    hasTuning = ("% change (tuned)" in data)
    if isGemm and len(sys.argv) < 5:
        outputPath = PurePath('./', chip + '_' + 'MLIR_Performance_Changes_Gemm.html')
    with open(outputPath, "w") as outputStream:
        toHighlight = ["% change", "% change (tuned)"] if hasTuning \
            else ["% change"]
        reportUtils.htmlReport(data, summary,
            "MLIR Performance Changes, " + ("GEMM" if isGemm else "Conv"),
            toHighlight, reportUtils.colorForChanges, outputStream)
