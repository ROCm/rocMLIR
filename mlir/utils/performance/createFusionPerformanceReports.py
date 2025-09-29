#!/usr/bin/env python3

import reportUtils

import pandas as pd
import sys


# Create html reports from .csv files
def print_all_performance(chip, op):

    columns_to_average = ['Fusion TFlops', 'MLIR TFlops', 'Fusion/MLIR']
    try:
        df = pd.read_csv(chip + '_' + op + '_' +
                         reportUtils.PERF_REPORT_FUSION_FILE)
    except FileNotFoundError:
        print('Perf report not found.')
        return

    plot_mean = df[columns_to_average].agg(reportUtils.geo_mean)
    plot_mean.name = "Geo. mean"
    plot_mean = pd.DataFrame(plot_mean).T

    plot_mean[['Fusion TFlops']]\
        .to_csv(chip + '_' + op + '_' + reportUtils.PERF_PLOT_REPORT_FUSION_FILE, index=False)

    if (op == 'conv'):
        means = df.groupby(["Direction", "DataType", "InputLayout"])[columns_to_average]\
            .agg(reportUtils.geo_mean)
        means.loc[("All", "All", "All"), :] = df[columns_to_average].agg(
            reportUtils.geo_mean)
        means.to_csv(chip + '_' + op + '_' +
                     reportUtils.PERF_STATS_REPORT_FUSION_FILE)
    else:
        means = df.groupby(["DataType"])[columns_to_average]\
            .agg(reportUtils.geo_mean)
        means.loc["All"] = df[columns_to_average].agg(reportUtils.geo_mean)
        means.to_csv(chip + '_' + op + '_' +
                     reportUtils.PERF_STATS_REPORT_FUSION_FILE)

    to_highlight = ['Fusion/MLIR']

    with open(chip + "_" + op + '_' + "fusion.html", 'w') as html_output:
        reportUtils.html_report(df, means, "Fusion performance", to_highlight,
                                reportUtils.color_for_speedups, html_output)


# Main function.
if __name__ == '__main__':
    print_all_performance(sys.argv[1], 'conv')
    print_all_performance(sys.argv[1], 'gemm')
