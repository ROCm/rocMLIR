import numpy as np
import pandas as pd
import scipy.stats

from typing import Tuple, List

PERF_REPORT_FILE = {
    'rocBLAS': 'mlir_vs_rocblas_perf.csv',
    'CK': 'mlir_vs_ck_perf.csv',
    'MIOpen': 'mlir_vs_miopen_perf.csv'
}
PERF_REPORT_FUSION_FILE = 'mlir_fusion_perf.csv'
PERF_PLOT_REPORT_FILE = 'mlir_vs_miopen_perf_for_plot.csv'
PERF_PLOT_REPORT_FILE = {
    'rocBLAS': 'mlir_vs_rocblas_perf_for_plot.csv',
    'CK': 'mlir_vs_ck_perf_for_plot.csv',
    'MIOpen': 'mlir_vs_miopen_perf_for_plot.csv'
}
PERF_PLOT_REPORT_FUSION_FILE = 'mlir_fusion_perf_for_plot.csv'
PERF_STATS_REPORT_FILE = 'mlir_vs_miopen_perf_means.csv'
PERF_STATS_REPORT_FILE = {
    'rocBLAS': 'mlir_vs_rocblas_perf_means.csv',
    'CK': 'mlir_vs_ck_perf_means.csv',
    'MIOpen': 'mlir_vs_miopen_perf_means.csv'
}
PERF_STATS_REPORT_FUSION_FILE = 'mlir_fusion_perf_means.csv'
MIOPEN_REPORT_FILE = 'miopen_perf.csv'
MIOPEN_TUNED_REPORT_FILE = 'miopen_tuned_perf.csv'
MIOPEN_UNTUNED_REPORT_FILE = 'miopen_untuned_perf.csv'

# In order to prevent issues with the tuning data reporting, 'PerfConfig'
# MUST STAY LAST!
CONV_TEST_PARAMETERS = [
    'Direction', 'DataType', 'Chip', 'numCU', 'FilterLayout', 'InputLayout', 'OutputLayout', 'N',
    'C', 'H', 'W', 'K', 'Y', 'X', 'DilationH', 'DilationW', 'StrideH', 'StrideW', 'PaddingH',
    'PaddingW', 'PerfConfig'
]
GEMM_TEST_PARAMETERS = [
    'DataType', 'OutDataType', 'Chip', 'numCU', 'TransA', 'TransB', 'G', 'M', 'K', 'N',
    'ScaledGemm', 'ScaleADtype', 'ScaleBDtype', 'TransScaleA', 'TransScaleB', 'PerfConfig'
]
ATTN_TEST_PARAMETERS = [
    'DataType', 'Chip', 'numCU', 'TransQ', 'TransK', 'TransV', 'TransO', 'Causal', 'ReturnLSE',
    'SplitKV', 'WithAttnScale', 'WithAttnBias', 'G', 'SeqLenQ', 'SeqLenK', 'NumHeadsQ',
    'NumHeadsKV', 'HeadDimQK', 'HeadDimV', 'PerfConfig'
]
GEMM_GEMM_TEST_PARAMETERS = [
    'DataType', 'Chip', 'numCU', 'TransA', 'TransB', 'TransC', 'TransO', 'G', 'M', 'K', 'N', 'O',
    'PerfConfig'
]
CONV_GEMM_TEST_PARAMETERS = [
    'DataType', 'Chip', 'numCU', 'FilterLayout', 'InputLayout', 'TransC', 'TransO', 'N', 'C', 'H',
    'W', 'K', 'Y', 'X', 'DilationH', 'DilationW', 'StrideH', 'StrideW', 'PaddingH', 'PaddingW', 'O',
    'PerfConfig'
]
ROUND_DIGITS = 2


def geo_mean(data):
    masked_data = np.ma.masked_where(~(np.isfinite(data) & (data > 0)), data)
    if masked_data.count() == 0:
        means = 0
    else:
        means = scipy.stats.gmean(masked_data)
    return means


def color_for_speedups(value):
    if not np.isfinite(value):
        return 'background-color: #ff00ff'

    if value <= 0.7:
        return 'background-color: #ff0000; color: #ffffff'
    elif value <= 0.9:
        return 'background-color: #dddd00'
    elif value >= 1.2:
        return 'background-color: #00ffff'
    elif value >= 1.05:
        return 'background-color: #00cccc'
    else:
        return ''


def color_for_changes(value):
    if not np.isfinite(value):
        return 'background-color: #ff00ff'

    if value <= -30.0:
        return 'background-color: #ff0000; color: #ffffff'
    elif value <= -10.0:
        return 'background-color: #dddd00'
    elif value >= 20.0:
        return 'background-color: #00ffff'
    elif value >= 5.0:
        return 'background-color: #00cccc'
    else:
        return ''


def set_common_styles(styler: 'pd.io.formats.style.Styler', speedup_cols: list, colorizer):
    styler.set_table_styles([{
        'selector': 'tbody tr:nth-child(odd)',
        'props': [('background-color', '#e0e0e0')]
    }, {
        'selector': 'tbody tr:nth-child(even)',
        'props': [('background-color', '#eeeeee')]
    }, {
        'selector': 'table',
        'props': [('background-color', '#dddddd'), ('border-collapse', 'collapse')]
    }, {
        'selector': 'th, td',
        'props': [('padding', '0.5em'), ('text-align', 'center'), ('max-width', '150px')]
    }])
    styler.format(precision=ROUND_DIGITS, na_rep="---")
    for col in speedup_cols:
        if col in styler.columns:
            styler.map(colorizer, subset=[col])


# Adapted from
# https://stackoverflow.com/questions/54405704/check-if-all-values-in-dataframe-column-are-the-same
def unique_cols(df: pd.DataFrame) -> List[str]:
    a: np.array = df.to_numpy()
    return df.columns[(a[0] == a).all(0)]


def clean_data_for_humans(data: pd.DataFrame, title: str)\
        -> Tuple[pd.DataFrame, str, List[str]]:
    is_gemm = "TransA" in data
    is_attention = "TransQ" in data
    parameters = CONV_TEST_PARAMETERS
    if is_gemm:
        parameters = GEMM_TEST_PARAMETERS
    if is_attention:
        parameters = ATTN_TEST_PARAMETERS

    index_cols = {k: k for k in parameters}  # Preserves order
    if all((x in data.columns) for x in {"FilterLayout", "InputLayout", "OutputLayout"}):
        if (((data["FilterLayout"] == "kcyx") & (data["InputLayout"] == "nchw") &
             (data["OutputLayout"] == "nkhw")) |
            ((data["FilterLayout"] == "kyxc") & (data["InputLayout"] == "nhwc") &
             (data["OutputLayout"] == "nhwk"))).all():
            # Layouts are consistent
            to_remove = {"FilterLayout", "OutputLayout"}
            data = data.drop(columns=to_remove, inplace=False)
            for c in to_remove:
                del index_cols[c]

            data.rename(columns={"InputLayout": "Layout"}, inplace=True)
            index_cols["InputLayout"] = "Layout"

    columns_to_drop = unique_cols(data)
    # Do not drop unique columns in attention for now
    # to keep it transparent what we are tracking.
    # We can revisit this if it ever becomes an issue.
    if len(columns_to_drop) > 0 and not is_attention:
        title = title + ": " + ", ".join(f"{c} = {data[c].iloc[0]}" for c in columns_to_drop)
        data = data.drop(columns=columns_to_drop, inplace=False)
        for c in columns_to_drop:
            if c == "Layout" and index_cols.get("InputLayout", "") == "Layout":
                del index_cols["InputLayout"]
            index_cols.pop(c, "")

    return data, title, list(index_cols.values())


def html_report(data: pd.DataFrame,
                stats: pd.DataFrame,
                title: str,
                speedup_cols: list,
                colorizer=color_for_speedups,
                stream=None):
    data, long_title, index_cols = clean_data_for_humans(data, title)
    print(f"""
<!doctype html>
<html lang="en_US">
<head>
<meta charset="utf-8">
<title>{long_title}</title>
<style type="text/css">
caption {{
    caption-side: bottom;
    padding: 0.5em;
}}
</style>
</head>
<body>
<h1>{long_title}</h1>
<h2>Summary</h2>
""",
          file=stream)

    stats_printer = stats.style
    stats_printer.set_caption(f"Summary statistics for {title}")
    set_common_styles(stats_printer, speedup_cols, colorizer)
    print(stats_printer.to_html(), file=stream)

    print("<h2>Details</h2>", file=stream)
    data_printer = data.style
    if len(index_cols) > 0:
        indexed = data.set_index(index_cols)
        data_printer = indexed.style
        data_printer.set_caption(f"{title}: Per-test breakdown")
        set_common_styles(data_printer, speedup_cols, colorizer)
        print(data_printer.to_html(), file=stream)
    print("""
</body>
</html>
""", file=stream)
