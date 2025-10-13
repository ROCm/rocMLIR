#!/usr/bin/env python3

import reportUtils

import csv
import numpy as np
import pandas as pd
import sys


#Create html reports from .csv files
def printAllPerformance(chip, op):

    COLUMNS_TO_AVERAGE = ['Fusion TFlops', 'MLIR TFlops', 'Fusion/MLIR']

    df = pd.read_csv(chip + '_' + op + '_' +
                     reportUtils.PERF_REPORT_FUSION_FILE)

    plotMean = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
    plotMean.name = "Geo. mean"
    plotMean = pd.DataFrame(plotMean).T

    plotMean[['Fusion TFlops']]\
        .to_csv(chip + '_' + op + '_' + reportUtils.PERF_PLOT_REPORT_FUSION_FILE, index=False)

    if (op == 'conv'):
        means = df.groupby(["Direction", "DataType", "InputLayout"])[COLUMNS_TO_AVERAGE]\
            .agg(reportUtils.geoMean)
        means.loc[("All", "All",
                   "All"), :] = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
        means.to_csv(chip + '_' + op + '_' +
                     reportUtils.PERF_STATS_REPORT_FUSION_FILE)
    else:
        means = df.groupby(["DataType"])[COLUMNS_TO_AVERAGE]\
            .agg(reportUtils.geoMean)
        means.loc["All"] = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
        means.to_csv(chip + '_' + op + '_' +
                     reportUtils.PERF_STATS_REPORT_FUSION_FILE)

    toHighlight = ['Fusion/MLIR']

    with open(chip + "_" + op + '_' + f"fusion.html", 'w') as htmlOutput:
        reportUtils.htmlReport(df, means, f"Fusion performance", toHighlight,
                               reportUtils.colorForSpeedups, htmlOutput)


# Main function.
if __name__ == '__main__':
    try:
        printAllPerformance(sys.argv[1], 'conv')
        printAllPerformance(sys.argv[1], 'gemm')
    except FileNotFoundError:
        print(f'Error: No performance report found for {sys.argv[1]}')
        sys.exit(1)
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)
