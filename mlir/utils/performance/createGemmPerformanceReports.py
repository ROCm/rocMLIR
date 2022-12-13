#!/usr/bin/env python3

import reportUtils

import csv
import numpy as np
import pandas as pd
import sys

#Create html reports from .csv files
def printAllPerformance(chip):
    perfReportFound = False

    try:
        df = pd.read_csv(chip + '_' + reportUtils.PERF_REPORT_GEMM_FILE)
        perfReportFound = True
        if 'Tuned MLIR TFlops' in df:
            COLUMNS_TO_AVERAGE = ['MLIR TFlops', 'Tuned MLIR TFlops',
                'rocBLAS TFlops (no MLIR Kernels)', 'Tuned/Untuned',
                'MLIR/rocBLAS', 'Tuned/rocBLAS']
        else:
            COLUMNS_TO_AVERAGE = ['MLIR TFlops',
                'rocBLAS TFlops (no MLIR Kernels)',
                'MLIR/rocBLAS']
    except FileNotFoundError:
        print('Perf report not found.')
        return

    # Only plot the actual averages, not the ratios
    # (This conveniently keeps the old behavior for the no tuning DB case)
    plotMean = df[COLUMNS_TO_AVERAGE[:3]].agg(reportUtils.geoMean)
    plotMean.name = "Geo. mean"
    plotMean = pd.DataFrame(plotMean).T

    plotMean[['MLIR TFlops', 'rocBLAS TFlops (no MLIR Kernels)']]\
        .to_csv(chip + '_' + reportUtils.PERF_PLOT_REPORT_GEMM_FILE, index=False)

    means = df.groupby(["DataType", "TransA", "TransB"])[COLUMNS_TO_AVERAGE]\
        .agg(reportUtils.geoMean)
    means.loc[("All", "All", "All"),:] = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
    means.to_csv(chip + '_' + reportUtils.PERF_STATS_REPORT_GEMM_FILE)

    toHighlight = ["MLIR/rocBLAS"]
    if "Tuned/Untuned" in df:
        toHighlight += ["Tuned/rocBLAS", "Tuned/Untuned"]

    with open(chip + "_" + "MLIR_vs_rocBLAS.html", 'w') as htmlOutput:
        reportUtils.htmlReport(df, means, "MLIR vs. rocBLAS performance",
        toHighlight, htmlOutput)

# Main function.
if __name__ == '__main__':
    printAllPerformance(sys.argv[1])
