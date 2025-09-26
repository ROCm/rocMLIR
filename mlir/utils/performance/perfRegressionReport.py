#!/usr/bin/evn python3

import reportUtils
import sys
import pandas as pd
from pathlib import PurePath
from perfRunner import get_num_cu
from typing import Tuple


def load_mlir_data(filename: str):
    df = pd.read_csv(filename, sep=',', header=0, index_col=False)
    columns_dropped = [
        'MIOpen TFlops (no MLIR Kernels)', 'MLIR/MIOpen',
        'MIOpen TFlops (Tuned MLIR Kernels)',
        'MIOpen TFlops (Untuned MLIR Kernels)', 'Tuned/Untuned',
        'Tuned/MIOpen', 'rocBLAS TFlops (no MLIR Kernels)', 'MLIR/rocBLAS',
        'Tuned/rocBLAS', 'Quick Tuned/rocBLAS', 'Quick Tuned/MIOpen',
        'Quick Tuned/Untuned', 'Quick Tuned/Tuned', 'LDSBankConflict (MIOpen)',
        'LDSBankConflict (rocBLAS)'
    ]
    df.drop(columns=columns_dropped, inplace=True, errors='ignore')
    # Work around empty PerfConfig field whin migrating from no tuning to yes tuning
    # Can be removed next time we touch this
    if 'PerfConfig' in df:
        df['PerfConfig'] = df['PerfConfig'].fillna('None')
    if 'numCU' not in df:
        df.insert(4, 'numCU', get_num_cu(df['Chip'][0]))
    return df


def merge_perfconfigs(v: Tuple[str, str]) -> str:
    v1, v2 = v
    if v1 == v2:
        return v1
    return f"{v1} -> {v2}"


def summarize_stat(grouped, func, data):
    ret = grouped.agg(func)
    if ret.index.nlevels == 1:
        ret.loc["All"] = data.agg(func)
    else:
        ret.loc[("All",) * ret.index.nlevels, :] = data.agg(func)
    return ret


def compute_perf_stats(old_df: pd.DataFrame, new_df: pd.DataFrame,
                       old_label: str,
                       new_label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    is_gemm = "TransA" in new_df
    is_attn = "TransQ" in new_df
    parameters = reportUtils.CONV_TEST_PARAMETERS
    if is_gemm:
        parameters = reportUtils.GEMM_TEST_PARAMETERS
    if is_attn:
        parameters = reportUtils.ATTN_TEST_PARAMETERS
    # Ignore perf config in join
    join_cols = parameters[:-1]
    try:
        data = new_df.merge(old_df, on=join_cols, suffixes=('_new', '_old'))
    except KeyError as e:
        print("Missing columns in data, forcing copy: ", e, file=sys.stderr)
        return compute_perf_stats(new_df.copy(), new_df, "forced copy",
                                  new_label)
    if len(data) == 0:
        print(
            "Old and new data have come from disjoint performance runs, ignoring old data",
            file=sys.stderr)
        return compute_perf_stats(new_df.copy(), new_df, "forced copy",
                                  new_label)

    # Clean up PerfConfig columns, as the report generator wants a single PerfConfig
    if "PerfConfig_old" in data and "PerfConfig_new" in data:
        perfconfig_col_pos = data.columns.get_loc("PerfConfig_old")
        zipped = list(
            map(merge_perfconfigs,
                zip(data["PerfConfig_old"], data["PerfConfig_new"])))
        data.insert(perfconfig_col_pos, "PerfConfig", zipped)
        data.drop(columns=["PerfConfig_old", "PerfConfig_new"], inplace=True)

    if "PerfConfig (quick tuned)_old" in data and "PerfConfig (quick tuned)_new" in data:
        perfconfig_col_pos = data.columns.get_loc(
            "PerfConfig (quick tuned)_old")
        zipped = list(
            map(
                merge_perfconfigs,
                zip(data["PerfConfig (quick tuned)_old"],
                    data["PerfConfig (quick tuned)_new"])))
        data.insert(perfconfig_col_pos, "PerfConfig (quick tuned)", zipped)
        data.drop(columns=[
            "PerfConfig (quick tuned)_old", "PerfConfig (quick tuned)_new"
        ],
                  inplace=True)

    if (old_label == new_label):
        old_label += "_old"
        new_label += "_new"
    old_label = f"MLIR TFlops ({old_label})"
    new_label = f"MLIR TFlops ({new_label})"
    old_label_tuned = f"Tuned TFlops ({old_label})"
    new_label_tuned = f"Tuned TFlops ({new_label})"
    old_label_quick_tuned = f"Quick Tuned TFlops ({old_label})"
    new_label_quick_tuned = f"Quick Tuned TFlops ({new_label})"
    data.rename(columns={
        'MLIR TFlops_old': old_label,
        'MLIR TFlops_new': new_label,
        'TFlops_old': old_label,
        'TFlops_new': new_label,
        'Tuned MLIR TFlops_old': old_label_tuned,
        'Tuned MLIR TFlops_new': new_label_tuned,
        "Quick Tuned MLIR TFlops_old": old_label_quick_tuned,
        "Quick Tuned MLIR TFlops_new": new_label_quick_tuned
    },
                inplace=True)
    data['% change'] = 100.0 * (data[new_label] -
                                data[old_label]) / data[old_label]
    has_tuning = False
    has_quick_tuning = False
    if old_label_tuned in data and new_label_tuned in data:
        data['% change (tuned)'] = 100.0 * (
            data[new_label_tuned] -
            data[old_label_tuned]) / data[old_label_tuned]
        has_tuning = True
    if old_label_quick_tuned in data and new_label_quick_tuned in data:
        data['% change (quick tuned)'] = 100.0 * (
            data[new_label_quick_tuned] -
            data[old_label_quick_tuned]) / data[old_label_quick_tuned]
        has_quick_tuning = True
    columns_to_average = ['% change', old_label, new_label]
    if has_tuning:
        columns_to_average += [
            '% change (tuned)', old_label_tuned, new_label_tuned
        ]
    if has_quick_tuning:
        columns_to_average += [
            '% change (quick tuned)', old_label_quick_tuned,
            new_label_quick_tuned
        ]
    statistics = [("Geo. mean", reportUtils.geo_mean), ("Arith. mean", "mean")]
    groups = ["DataType"] if is_gemm or is_attn else [
        "Direction", "DataType", "InputLayout"
    ]
    grouped = data.groupby(groups)[columns_to_average]
    stats = pd.concat(
        {
            name: summarize_stat(grouped, func, data[columns_to_average])
            for name, func in statistics
        },
        axis=0).unstack(level=0)
    stats.drop(columns=[('% change', 'Geo. mean'),
                        ('% change (tuned)', 'Geo. mean'),
                        ('% change (quick tuned)', 'Geo. mean')],
               errors='ignore',
               inplace=True)

    return data, stats


def get_perf_date(stats_path: PurePath, default="???"):
    path = stats_path.with_name('perf-run-date')
    try:
        with open(str(path), "r") as f:
            return f.readline().rstrip()
    except FileNotFoundError:  # Shouldn't happen once things get running
        return default


if __name__ == '__main__':
    chip = sys.argv[1]
    old_data_path = PurePath(sys.argv[2]) if len(sys.argv) >= 3\
        else PurePath('./', 'oldData/', chip + '_' + reportUtils.PERF_REPORT_FILE['MIOpen'])
    new_data_path = PurePath(sys.argv[3]) if len(sys.argv) >= 4\
        else PurePath('./', chip + '_' + reportUtils.PERF_REPORT_FILE['MIOpen'])
    output_path = PurePath(sys.argv[4]) if len(sys.argv) >= 5\
        else PurePath('./', chip + '_' + 'MLIR_Performance_Changes.html')

    try:
        new_df = load_mlir_data(str(new_data_path))
        new_label = get_perf_date(new_data_path, "new")
    except FileNotFoundError:
        print(
            "Could not load current performance data: run perf or provide a path",
            file=sys.stderr)
        sys.exit(1)
    try:
        old_df = load_mlir_data(str(old_data_path))
        old_label = get_perf_date(old_data_path, "old")
    except FileNotFoundError:
        print("Warning: No old performance data, reusing new one",
              file=sys.stderr)
        old_df = new_df.copy()
        old_label = "copy"

    data, summary = compute_perf_stats(old_df, new_df, old_label, new_label)
    is_gemm = ("TransA" in data)
    is_attn = ("TransQ" in data)
    has_tuning = ("% change (tuned)" in data)
    if is_gemm and len(sys.argv) < 5:
        output_path = PurePath(
            './', chip + '_' + 'MLIR_Performance_Changes_Gemm.html')
    if is_attn and len(sys.argv) < 5:
        output_path = PurePath(
            './', chip + '_' + 'MLIR_Performance_Changes_Attention.html')
    with open(output_path, "w") as output_stream:
        to_highlight = ["% change", "% change (tuned)"] if has_tuning \
            else ["% change"]
        reportUtils.html_report(
            data, summary,
            "MLIR Performance Changes, " + ("GEMM" if is_gemm else "Conv"),
            to_highlight, reportUtils.color_for_changes, output_stream)
