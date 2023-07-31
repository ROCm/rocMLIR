#!/usr/bin/env python3

import reportUtils

import csv
import numpy as np
import pandas as pd
import sys

#Create html reports from .csv files
def printAllPerformance(chip, lib='rocBLAS'):
    perfReportFound = False

    try:
        df = pd.read_csv(chip + '_' + reportUtils.PERF_REPORT_FILE[lib])
        perfReportFound = True
        if 'Tuned MLIR TFlops' in df:
            COLUMNS_TO_AVERAGE = ['MLIR TFlops', 'Tuned MLIR TFlops',
                f'{lib} TFlops (no MLIR Kernels)', 'Tuned/Untuned',
                f'MLIR/{lib}', f'Tuned/{lib}']
        else:
            COLUMNS_TO_AVERAGE = ['MLIR TFlops',
                f'{lib} TFlops (no MLIR Kernels)',
                f'MLIR/{lib}']
    except FileNotFoundError:
        print('Perf report not found.')
        return

    # Only plot the actual averages, not the ratios
    # (This conveniently keeps the old behavior for the no tuning DB case)
    plotMean = df[COLUMNS_TO_AVERAGE[:3]].agg(reportUtils.geoMean)
    plotMean.name = "Geo. mean"
    plotMean = pd.DataFrame(plotMean).T
    plotMean[['MLIR TFlops', f'{lib} TFlops (no MLIR Kernels)']]\
        .to_csv(chip + '_' + reportUtils.PERF_PLOT_REPORT_FILE[lib], index=False)

    if lib == 'MIOpen':
        means = df.groupby(["Direction", "DataType", "InputLayout"])[COLUMNS_TO_AVERAGE]\
            .agg(reportUtils.geoMean)
        means.loc["All", "ALL","ALL"] = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
    else:
        means = df.groupby(["DataType"])[COLUMNS_TO_AVERAGE]\
            .agg(reportUtils.geoMean)
        means.loc["All"] = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
    means.to_csv(chip + '_' + reportUtils.PERF_STATS_REPORT_FILE[lib])

    toHighlight = [f"MLIR/{lib}"]
    if "Tuned/Untuned" in df:
        toHighlight += [f"Tuned/{lib}", f"Tuned/Untuned"]

    with open(chip + "_" + f"MLIR_vs_{lib}.html", 'w') as htmlOutput:
        reportUtils.htmlReport(df, means, f"MLIR vs. {lib} performance",
        toHighlight, reportUtils.colorForSpeedups, htmlOutput)

# Main function.
if __name__ == '__main__':
    lib = sys.argv[2] if len(sys.argv) > 2 else 'rocBLAS'
    printAllPerformance(sys.argv[1], lib)
