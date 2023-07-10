import numpy as np
import pandas as pd
import scipy.stats

from typing import Tuple, List

PERF_REPORT_FILE = 'mlir_vs_miopen_perf.csv'
PERF_REPORT_GEMM_FILE = {'rocBLAS': 'mlir_vs_rocblas_perf.csv', 'CK': 'mlir_vs_ck_perf.csv'}
PERF_REPORT_FUSION_FILE = 'mlir_fusion_perf.csv'
PERF_PLOT_REPORT_FILE = 'mlir_vs_miopen_perf_for_plot.csv'
PERF_PLOT_REPORT_GEMM_FILE = {'rocBLAS': 'mlir_vs_rocblas_perf_for_plot.csv', 'CK' : 'mlir_vs_ck_perf_for_plot.csv'}
PERF_PLOT_REPORT_FUSION_FILE = 'mlir_fusion_perf_for_plot.csv'
PERF_STATS_REPORT_FILE = 'mlir_vs_miopen_perf_means.csv'
PERF_STATS_REPORT_GEMM_FILE = {'rocBLAS': 'mlir_vs_rocblas_perf_means.csv', 'CK' : 'mlir_vs_ck_perf_means.csv'}
PERF_STATS_REPORT_FUSION_FILE = 'mlir_fusion_perf_means.csv'
MIOPEN_REPORT_FILE = 'miopen_perf.csv'
MIOPEN_TUNED_REPORT_FILE = 'miopen_tuned_perf.csv'
MIOPEN_UNTUNED_REPORT_FILE = 'miopen_untuned_perf.csv'

## In order to prevent issues with the tuning data reporting, 'PerfConfig'
## MUST STAY LAST!
CONV_TEST_PARAMETERS = ['Direction', 'DataType', 'Chip', 'FilterLayout',
                        'InputLayout', 'OutputLayout', 'N', 'C', 'H', 'W', 'K', 'Y',
                        'X', 'DilationH', 'DilationW', 'StrideH', 'StrideW',
                        'PaddingH', 'PaddingW', 'PerfConfig']
GEMM_TEST_PARAMETERS = ['DataType', 'OutDataType', 'Chip', 'TransA', 'TransB', 'G', 'M', 'K', 'N', 'PerfConfig']
ROUND_DIGITS = 2

def geoMean(data):
    maskedData = np.ma.masked_where(~(np.isfinite(data) & (data > 0)), data)
    means = scipy.stats.gmean(maskedData)
    return means

def colorForSpeedups(value):
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

def colorForChanges(value):
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

def setCommonStyles(styler: 'pd.io.formats.style.Styler', speedupCols: list, colorizer):
    styler.set_table_styles([
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#e0e0e0')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#eeeeee')]},
        {'selector': 'table', 'props': [('background-color', '#dddddd'), ('border-collapse', 'collapse')]},
        {'selector': 'th, td', 'props': [('padding', '0.5em'), ('text-align', 'center'), ('max-width', '150px')]}])
    styler.format(precision=ROUND_DIGITS, na_rep="FAILED")
    for col in speedupCols:
        if col in styler.columns:
            styler.applymap(colorizer, subset=[col])

# Adapted from
# https://stackoverflow.com/questions/54405704/check-if-all-values-in-dataframe-column-are-the-same
def uniqueCols(df: pd.DataFrame) -> List[str]:
    a: np.array = df.to_numpy()
    return df.columns[(a[0] == a).all(0)]

def cleanDataForHumans(data: pd.DataFrame, title: str)\
        -> Tuple[pd.DataFrame, str, List[str]]:
    isGemm = "TransA" in data
    parameters = GEMM_TEST_PARAMETERS if isGemm else CONV_TEST_PARAMETERS
    indexCols = {k: k for k in parameters} # Preserves order
    if all((x in data.columns) for x in {"FilterLayout", "InputLayout",
                                              "OutputLayout"}):
        if (((data["FilterLayout"] == "kcyx") & (data["InputLayout"] == "nchw") &
              (data["OutputLayout"] == "nkhw")) | ((data["FilterLayout"] == "kyxc") &
              (data["InputLayout"] == "nhwc") & (data["OutputLayout"] == "nhwk")))\
                .all():
            # Layouts are consistent
            TO_REMOVE = {"FilterLayout", "OutputLayout"}
            data = data.drop(columns=TO_REMOVE, inplace=False)
            for c in TO_REMOVE:
                del indexCols[c]

            data.rename(columns={"InputLayout": "Layout"}, inplace=True)
            indexCols["InputLayout"] = "Layout"

    columnsToDrop = uniqueCols(data)
    if len(columnsToDrop) > 0:
        title = title + ": " + ", ".join(f"{c} = {data[c].iloc[0]}"
            for c in columnsToDrop)
        data = data.drop(columns=columnsToDrop, inplace=False)
        for c in columnsToDrop:
            if c == "Layout" and indexCols.get("InputLayout", "") == "Layout":
                del indexCols["InputLayout"]
            indexCols.pop(c, "")

    return data, title, list(indexCols.values())

def htmlReport(data: pd.DataFrame, stats: pd.DataFrame, title: str,
                  speedupCols: list, colorizer=colorForSpeedups, stream=None):
    data, longTitle, indexCols = cleanDataForHumans(data, title)
    print(f"""
<!doctype html>
<html lang="en_US">
<head>
<meta charset="utf-8">
<title>{longTitle}</title>
<style type="text/css">
caption {{
    caption-side: bottom;
    padding: 0.5em;
}}
</style>
</head>
<body>
<h1>{longTitle}</h1>
<h2>Summary</h2>
""", file=stream)

    statsPrinter = stats.style
    statsPrinter.set_caption(f"Summary statistics for {title}")
    setCommonStyles(statsPrinter, speedupCols, colorizer)
    print(statsPrinter.to_html(), file=stream)

    print("<h2>Details</h2>", file=stream)
    dataPrinter = data.style
    if len(indexCols) > 0 :
        indexed = data.set_index(indexCols)
        dataPrinter = indexed.style
        dataPrinter.set_caption(f"{title}: Per-test breakdown")
        setCommonStyles(dataPrinter, speedupCols, colorizer)
        print(dataPrinter.to_html(), file=stream)
    print("""
</body>
</html>
""", file=stream)

