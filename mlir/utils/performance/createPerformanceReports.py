#!/usr/bin/env python3

import reportUtils

import csv
import numpy as np
import pandas as pd
import sys

#Create html reports from .csv files
def printAllPerformance(chip):
    def readReport(file, kind):
        try:
            return pd.read_csv(chip + '_' + file)
        except FileNotFoundError:
            print(f'${kind} report not found.')
            return None

    COLUMNS_TO_AVERAGE = ['MLIR TFlops', 'MIOpen TFlops (no MLIR Kernels)', 'MLIR/MIOpen']

    df = readReport(reportUtils.PERF_REPORT_FILE, 'Perf')
    if df.empty:
        return
    tuned_df = readReport(reportUtils.MIOPEN_TUNED_REPORT_FILE,
                          'MIOpen with tuned MLIR')
    untuned_df = readReport(reportUtils.MIOPEN_UNTUNED_REPORT_FILE,
                            'MIOpen with untuned MLIR')

    # Add tuned and untuned performance to the existing performance table
    if not tuned_df.empty and not untuned_df.empty:
        df['MIOpen TFlops (Tuned MLIR Kernels)'] = tuned_df['TFlops']
        df['MIOpen TFlops (Untuned MLIR Kernels)'] = untuned_df['TFlops']
        df['Tuned/Untuned'] = df['MIOpen TFlops (Tuned MLIR Kernels)']/df['MIOpen TFlops (Untuned MLIR Kernels)']
        df['Tuned/MIOpen'] = df['MIOpen TFlops (Tuned MLIR Kernels)']/df['MIOpen TFlops (no MLIR Kernels)']
        df.to_csv(chip + '_' + reportUtils.PERF_REPORT_FILE, index=False)
        COLUMNS_TO_AVERAGE = ['MLIR TFlops', 'MIOpen TFlops (no MLIR Kernels)', 'MLIR/MIOpen',
                              'MIOpen TFlops (Tuned MLIR Kernels)', 'MIOpen TFlops (Untuned MLIR Kernels)',
                              'Tuned/Untuned', 'Tuned/MIOpen']

    plotMean = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
    plotMean.name = "Geo. mean"
    plotMean = pd.DataFrame(plotMean).T

    plotMean[['MLIR TFlops', 'MIOpen TFlops (no MLIR Kernels)']]\
        .to_csv(chip + '_' + reportUtils.PERF_PLOT_REPORT_FILE, index=False)

    means = df.groupby(["Direction", "DataType", "InputLayout"])[COLUMNS_TO_AVERAGE]\
        .agg(reportUtils.geoMean)
    means.loc[("All", "All", "All"),:] = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
    means.to_csv(chip + '_' + reportUtils.PERF_STATS_REPORT_FILE)

    with open(chip + "_" + "MLIR_vs_MIOpen.html", 'w') as htmlOutput:
        reportUtils.htmlReport(df, means, "MLIR vs. MIOpen performance",
          ["MLIR/MIOpen", "Tuned/Untuned", "Tuned/MIOpen"],
          reportUtils.colorForSpeedups,
          htmlOutput)

# Main function.
if __name__ == '__main__':
    printAllPerformance(sys.argv[1])
