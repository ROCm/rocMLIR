import numpy as np
import pandas as pd
import scipy.stats

from typing import Tuple, List

PERF_REPORT_FILE = 'mlir_vs_miopen_perf.csv'
PERF_PLOT_REPORT_FILE = 'mlir_vs_miopen_perf_for_plot.csv'
PERF_STATS_REPORT_FILE = 'mlir_vs_miopen_perf_means.csv'

TEST_PARAMETERS = ['Direction', 'DataType', 'XDLOPS', 'FilterLayout', 'InputLayout', 'OutputLayout',
                       'N', 'C', 'H', 'W', 'K', 'Y', 'X', 'DilationH', 'DilationW', 'StrideH', 'StrideW',
                       'PaddingH', 'PaddingW']
ROUND_DIGITS = 2

def geoMean(data):
    maskedData = np.ma.masked_where(~(np.isfinite(data) & (data != 0)), data)
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

def setCommonStyles(styler: 'pd.io.formats.style.Styler', speedupCol: str):
    styler.set_table_styles([
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#e0e0e0')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#eeeeee')]},
        {'selector': 'table', 'props': [('background-color', '#dddddd'), ('border-collapse', 'collapse')]},
        {'selector': 'th, td', 'props': [('padding', '0.5em'), ('text-align', 'center')]}])
    styler.set_precision(ROUND_DIGITS)
    styler.set_na_rep("FAILED")
    if speedupCol in styler.columns:
        styler.applymap(colorForSpeedups, subset=[speedupCol])

# Adapted from
# https://stackoverflow.com/questions/54405704/check-if-all-values-in-dataframe-column-are-the-same
def uniqueCols(df: pd.DataFrame) -> List[str]:
    a: np.array = df.to_numpy()
    return df.columns[(a[0] == a).all(0)]

def cleanDataForHumans(data: pd.DataFrame, title: str)\
        -> Tuple[pd.DataFrame, str, List[str]]:
    indexCols = {k: k for k in TEST_PARAMETERS} # Preserves order
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
                  speedupCol: str, stream=None):
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
    setCommonStyles(statsPrinter, speedupCol)
    print(statsPrinter.render(), file=stream)

    print("<h2>Details</h2>", file=stream)
    indexed = data.set_index(indexCols)
    dataPrinter = indexed.style
    dataPrinter.set_caption(f"{title}: Per-test breakdown")
    setCommonStyles(dataPrinter, speedupCol)
    print(dataPrinter.render(), file=stream)
    print("""
</body>
</html>
""", file=stream)


