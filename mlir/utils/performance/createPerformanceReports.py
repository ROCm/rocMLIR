#!/usr/bin/env python3

import reportUtils

import pandas as pd
import sys


# Create html reports from .csv files
def print_all_performance(chip, lib='rocBLAS'):

    df = pd.read_csv(chip + '_' + reportUtils.PERF_REPORT_FILE[lib])
    columns_to_average = [
        'MLIR TFlops', f'{lib} TFlops (no MLIR Kernels)', f'MLIR/{lib}'
    ]
    if 'Tuned MLIR TFlops' in df:
        columns_to_average += [
            'Tuned MLIR TFlops', 'Tuned/Untuned', f'Tuned/{lib}'
        ]
        if 'Quick Tuned MLIR TFlops' in df:
            columns_to_average += [
                'Quick Tuned MLIR TFlops', f'Quick Tuned/{lib}',
                'Quick Tuned/Untuned', 'Quick Tuned/Tuned'
            ]
    elif 'Quick Tuned MLIR TFlops' in df:
        columns_to_average += [
            'Quick Tuned MLIR TFlops', f'Quick Tuned/{lib}',
            'Quick Tuned/Untuned'
        ]

    # Only plot the actual averages, not the ratios
    # (This conveniently keeps the old behavior for the no tuning DB case)
    plot_mean = df[columns_to_average[:3]].agg(reportUtils.geo_mean)
    plot_mean.name = "Geo. mean"
    plot_mean = pd.DataFrame(plot_mean).T
    plot_mean[['MLIR TFlops', f'{lib} TFlops (no MLIR Kernels)']]\
        .to_csv(chip + '_' + reportUtils.PERF_PLOT_REPORT_FILE[lib], index=False)

    if lib == 'MIOpen':
        means = df.groupby(["Direction", "DataType", "InputLayout"])[columns_to_average]\
            .agg(reportUtils.geo_mean)
        means.loc["All", "ALL",
                  "ALL"] = df[columns_to_average].agg(reportUtils.geo_mean)
    else:
        means = df.groupby(["DataType"])[columns_to_average]\
            .agg(reportUtils.geo_mean)
        means.loc["All"] = df[columns_to_average].agg(reportUtils.geo_mean)
    means.to_csv(chip + '_' + reportUtils.PERF_STATS_REPORT_FILE[lib])

    to_highlight = [f"MLIR/{lib}"]
    if "Tuned/Untuned" in df:
        to_highlight += [f"Tuned/{lib}", "Tuned/Untuned"]
        if "Quick Tuned/Tuned":
            to_highlight += [
                f"Quick Tuned/{lib}", "Quick Tuned/Untuned",
                "Quick Tuned/Tuned"
            ]
    elif "Quick Tuned/Untuned" in df:
        to_highlight += [f"Quick Tuned/{lib}", "Quick Tuned/Untuned"]
    with open(chip + "_" + f"MLIR_vs_{lib}.html", 'w') as html_output:
        reportUtils.html_report(df, means, f"MLIR vs. {lib} performance",
                                to_highlight, reportUtils.color_for_speedups,
                                html_output)


# Main function.
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(
            "Error: missing chip argument (usage: createPerformanceReports.py <chip> [lib])"
        )
        sys.exit(1)
    chip = sys.argv[1]
    lib = sys.argv[2] if len(sys.argv) > 2 else 'rocBLAS'
    try:
        print_all_performance(chip, lib)
    except FileNotFoundError:
        print(f"Error: No performance report found for {chip}")
        sys.exit(1)
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)
