import numpy as np
import pandas as pd
import scipy.stats

PERF_REPORT_FILE = 'mlir_vs_miopen_perf.csv'
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
        return 'background-color: #00ff00'
    elif value >= 1.05:
        return 'background-color: #00cc00'
    else:
        return ''

def setCommonStyles(styler: 'pd.io.formats.style.Styler', speedupCol: str):
    styler.set_precision(ROUND_DIGITS)
    styler.set_na_rep("FAILED")
    styler.applymap(colorForSpeedups, subset=[speedupCol])

def htmlReport(data: pd.DataFrame, stats: pd.DataFrame, title: str, speedupCol: str, stream=None):
    print(f"""
<!doctype html>
<html lang="en_US">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style type="text/css">
caption {{
    caption-side: bottom;
    padding: 0.5em;
}}
</head>
<body>
<h1>{title}</h1>
<h2>Summary</h2>
""", file=stream)

    statsPrinter = stats.style
    statsPrinter.set_caption(f"Summary statistics for {title}")
    statsPrinter.set_table_styles([
        {'selector': 'th, td', 'props': [('padding', '0.5em'), ('text-align', 'center')]},
        {'selector': 'table', 'props': [('background-color', '#eeeeee'), ('border-collapse', 'collapse')]}])
    setCommonStyles(statsPrinter, speedupCol)
    print(statsPrinter.render(), file=stream)

    print("<h2>Details</h2>", file=stream)
    indexed = data.set_index(TEST_PARAMETERS)
    dataPrinter = indexed.style
    dataPrinter.set_caption(f"{title}: Per-test breakdown")
    dataPrinter.set_table_styles([
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#e0e0e0')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#eeeeee')]},
        {'selector': 'table', 'props': [('background-color', '#dddddd'), ('border-collapse', 'collapse')]},
        {'selector': 'th, td', 'props': [('padding', '0.5em'), ('text-align', 'center')]}])
    setCommonStyles(dataPrinter, speedupCol)
    print(dataPrinter.render(), file=stream)
    print("""
</body>
</html>
""", file=stream)


    