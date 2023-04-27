#!/usr/bin/env python3

import reportUtils

import csv
import numpy as np
import pandas as pd
import sys

#Create html reports from .csv files
def printAllPerformance(chip, op):
    perfReportFound = False

    COLUMNS_TO_AVERAGE = ['TFlops']
    try:
        df = pd.read_csv(chip + '_' + op + '_' + reportUtils.PERF_REPORT_FUSION_FILE)
        perfReportFound = True
    except FileNotFoundError:
        print('Perf report not found.')
        return

    plotMean = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
    plotMean.name = "Geo. mean"
    plotMean = pd.DataFrame(plotMean).T

    plotMean[['TFlops']]\
        .to_csv(chip + '_' + op + '_' + reportUtils.PERF_PLOT_REPORT_FUSION_FILE, index=False)

    if (op == 'conv'):
        means = df.groupby(["Direction", "DataType", "InputLayout"])[COLUMNS_TO_AVERAGE]\
            .agg(reportUtils.geoMean)
        means.loc[("All", "All", "All"),:] = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
        means.to_csv(chip + '_' + op + '_' + reportUtils.PERF_STATS_REPORT_FUSION_FILE)
    else:
        means = df.groupby(["DataType"])[COLUMNS_TO_AVERAGE]\
            .agg(reportUtils.geoMean)
        means.loc["All"] = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
        means.to_csv(chip + '_' + op + '_' + reportUtils.PERF_STATS_REPORT_FUSION_FILE)

    toHighlight = []

    with open(chip + "_" + op + '_' + f"fusion.html", 'w') as htmlOutput:
        reportUtils.htmlReport(df, means, f"Fusion performance",
        toHighlight, reportUtils.colorForSpeedups, htmlOutput)

# Main function.
if __name__ == '__main__':
    printAllPerformance(sys.argv[1], 'conv')
    printAllPerformance(sys.argv[1], 'gemm')
