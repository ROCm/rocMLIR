#!/usr/bin/env python3

import reportUtils

import csv
import numpy as np
import pandas as pd

#Create html reports from .csv files
def printAllPerformance():
    perfReportFound = False
    tunedReportFound = False
    untunedReportFound = False
    try:
        df = pd.read_csv(reportUtils.PERF_REPORT_FILE)
        perfReportFound = True
        COLUMNS_TO_AVERAGE = ['MLIR TFlops', 'MIOpen TFlops', 'MLIR/MIOpen']
    except FileNotFoundError:
        print('Perf report not found.')
        return
    
    try:
        tuned_df = pd.read_csv(reportUtils.MIOPEN_TUNED_REPORT_FILE)
        tunedReportFound = True
    except FileNotFoundError:
        print('MIOpen with turned MLIR report not found.')
    
    try:
        untuned_df = pd.read_csv(reportUtils.MIOPEN_UNTUNED_REPORT_FILE)
        untunedReportFound = True
    except FileNotFoundError:
        print('MIOpen with untuned MLIR report not found.')

    # Add tuned and untuned performance to the existing performance table 
    if tunedReportFound == True and untunedReportFound == True :
        df['MIOpen TFlops (Tuned MLIR Kernels)'] = tuned_df['TFlops']
        df['MIOpen TFlops (Untuned MLIR Kernels)'] = untuned_df['TFlops']
        df['Tuned/Untuned'] = df['MIOpen TFlops (Tuned MLIR Kernels)']/df['MIOpen TFlops (Untuned MLIR Kernels)']
        df.to_csv(reportUtils.PERF_REPORT_FILE, index=False)
        COLUMNS_TO_AVERAGE = ['MLIR TFlops', 'MIOpen TFlops', 'MLIR/MIOpen',
                              'MIOpen TFlops (Tuned MLIR Kernels)', 'MIOpen TFlops (Untuned MLIR Kernels)', 'Tuned/Untuned']

    plotMean = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
    plotMean.name = "Geo. mean"
    plotMean = pd.DataFrame(plotMean).T

    plotMean[['MLIR TFlops', 'MIOpen TFlops']]\
        .to_csv(reportUtils.PERF_PLOT_REPORT_FILE, index=False)

    means = df.groupby(["Direction", "DataType"])[COLUMNS_TO_AVERAGE]\
        .agg(reportUtils.geoMean)
    means.loc[("All", "All"),:] = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
    means.to_csv(reportUtils.PERF_STATS_REPORT_FILE)

    with open("MLIR_vs_MIOpen.html", 'w') as htmlOutput:
        reportUtils.htmlReport(df, means, "MLIR vs. MIOpen performance", ["MLIR/MIOpen", "Tuned/Untuned"], htmlOutput)

# Main function.
if __name__ == '__main__':
    printAllPerformance()
