# Python script that analyses .tsv.debug files and gives insights such as
# important metrics (Arithmetic Intensity, Occupancy, Work Imbalance) and
# plots correlation between them with the selected parameters.
#
# Usage: python3 ./testing-metrics.py <debug file(s)> [--n <percent>] [--m <metrics>] [--t <method for threshold>] [--o <output directory>] [--c <num_cus>]
# Arguments:
#       <debug file(s)>               Input file(s) in .tsv.debug format
#       --n <percent>                 Percent of the best perfconfigs to be considered (default=5) - doesn't affect analysis when checking only the best perfConfigs
#       --m <metrics>                 Metrics to be shown (ai, oc, wi, nmk)
#       --t <method for threshold>    Method for calculating threshold (m - max, mn - maxN, qn - quantileN)
#       --o <output directory>        Output directory in case of saving plots
#       --c <num_cus>                  CUs count if data is not collected on the machine on which the script is executed

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import math
import reportUtils
import os
from hip import hip

# TODO use AmdArchDb.py (when it's implemented)

num_eu_per_cu = 4  # may be changed in newer architectures


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def assign_num_cu():
    if args.c:
        return int(args.c)
    else:
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, 0))
        print(
            "Using info from GPU 0 in your system, the data should have be obtained from the same GPU."
        )
        return int(props.multiProcessorCount)


def analyze_gemm_file(file, n):
    df = pd.read_csv(file, sep='\t')

    # make a copy so we do not modify the original list
    gemm_keys = list(reportUtils.GEMM_TEST_PARAMETERS)
        
    # we remove the columns that are not needed for this.
    # this makes sure any new columns added to 
    # GEMM_TEST_PARAMETERS/CONV_TEST_PARAMETERS will be used here.
    gemm_keys.remove('DataType')
    gemm_keys.remove('OutDataType')
    gemm_keys.remove('Chip')
    gemm_keys.remove('numCU')
    gemm_keys.remove('PerfConfig')
    perfconfig_params = ['MPerBlock', 'NPerBlock', 'KPerBlock', 'MPerWave', 'NPerWave', 'kPack', 'splitKFactor', 'forceUnroll', 'ThreadCopyMore']

    assert df["PerfConfig"].str.startswith(
        "v2:").all(), "PerfConfig that doesn't start with v2: found"
    df[perfconfig_params] = df["PerfConfig"].str.replace("v2:", "").str.split(",", expand=True)

    df["ArithmeticIntensity"] = df.apply(
        lambda row: calculate_arithmetic_intensity(row["m"], row["n"], row["k"]), axis=1)
    df["mn_per_wave"] = df.apply(lambda row: (int(row["MPerWave"]) * int(row["NPerWave"])), axis=1)
    df["Occupancy"] = df.apply(lambda row: calculate_occupancy(int(row["m"]), int(row[
        "n"]), int(row["g"]), int(row["m_per_block"]), int(row[
            "n_per_block"]), int(row["mn_per_wave"]), min_num_waves),
                               axis=1)
    df["WorkImbalance"] = df.apply(lambda row: calculate_work_imbalance(
        int(row["m"]), int(row["n"]), int(row["g"]), int(row["m_per_block"]), int(row[
            "n_per_block"]), int(row["mn_per_wave"]), min_num_waves, int(row["split_k_factor"])),
                                   axis=1)

    top_list = []

    for (key, group) in df.groupby(gemm_keys):
        if args.t == "m":
            threshold = group['TFlops'].max()
            top_list.append(group[group['TFlops'] == threshold])
        if args.t == "mn":
            threshold = group[group['TFlops'] >= (group['TFlops'].max() * (1 - n / 100))]
            top_list.append(group[group['TFlops'] >= threshold])
        if args.t == "qn":
            threshold = group['TFlops'].quantile(1 - n / 100.0)
            top_list.append(group[group['TFlops'] >= threshold])

    list = pd.concat(top_list)

    df[[
        'Unnamed: 0', 'DataType', 'OutDataType', 'Chip', 'numCU', 'TransA', 'TransB', 'AccelLayoutA', 'AccelLayoutB', 'g', 'm', 'k',
        'n', 'PerfConfig', 'LDSBankConflict', 'TFlops', 'm_per_block', 'n_per_block', 'KPerBlock',
        'MPerWave', 'NPerWave', 'kPack', 'split_k_factor', 'forceUnroll', 'ThreadCopyMore',
        'ArithmeticIntensity', 'Occupancy', 'WorkImbalance'
    ]] = df[[
        'Unnamed: 0', 'DataType', 'OutDataType', 'Chip', 'numCU', 'TransA', 'TransB', 'AccelLayoutA', 'AccelLayoutB', 'g', 'm', 'k',
        'n', 'PerfConfig', 'LDSBankConflict', 'TFlops', 'm_per_block', 'n_per_block', 'KPerBlock',
        'MPerWave', 'NPerWave', 'kPack', 'split_k_factor', 'forceUnroll', 'ThreadCopyMore',
        'ArithmeticIntensity', 'Occupancy', 'WorkImbalance'
    ]].apply(pd.to_numeric, errors='coerce')

    list[[
        'Unnamed: 0', 'DataType', 'OutDataType', 'Chip', 'numCU', 'TransA', 'TransB', 'AccelLayoutA', 'AccelLayoutB', 'g', 'm', 'k',
        'n', 'PerfConfig', 'LDSBankConflict', 'TFlops', 'm_per_block', 'n_per_block', 'KPerBlock',
        'MPerWave', 'NPerWave', 'kPack', 'split_k_factor', 'forceUnroll', 'ThreadCopyMore',
        'ArithmeticIntensity', 'Occupancy', 'WorkImbalance'
    ]] = list[[
        'Unnamed: 0', 'DataType', 'OutDataType', 'Chip', 'numCU', 'TransA', 'TransB', 'AccelLayoutA', 'AccelLayoutB', 'g', 'm', 'k',
        'n', 'PerfConfig', 'LDSBankConflict', 'TFlops', 'm_per_block', 'n_per_block', 'KPerBlock',
        'MPerWave', 'NPerWave', 'kPack', 'split_k_factor', 'forceUnroll', 'ThreadCopyMore',
        'ArithmeticIntensity', 'Occupancy', 'WorkImbalance'
    ]].apply(pd.to_numeric, errors='coerce')

    params = [
        'm_per_block', 'n_per_block', 'KPerBlock', 'MPerWave', 'NPerWave', 'kPack', 'split_k_factor'
    ]

    if args.m == "ai":
        print(list.corr()['ArithmeticIntensity'])

        fig, axes = plt.subplots(2, 4)
        for ax, param in zip(axes.flat, params):
            ax.scatter(list[param], list['ArithmeticIntensity'], alpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel('ArithmeticIntensity')

        plt.tight_layout()
        plot_output("ArithmeticIntensity_vs_perfconfig_params.png")

    if args.m == "oc":
        print(list.corr()['Occupancy'])

        fig, axes = plt.subplots(2, 4)
        for ax, param in zip(axes.flat, params):
            ax.scatter(list[param], list['Occupancy'], alpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel('Occupancy')

        plt.tight_layout()
        plot_output("Occupancy_vs_perfconfig_params.png")

    if args.m == "wi":
        print(list.corr()['WorkImbalance'])

        fig, axes = plt.subplots(2, 4)
        for ax, param in zip(axes.flat, params):
            ax.scatter(list[param], list['WorkImbalance'], alpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel('WorkImbalance')

        plt.tight_layout()
        plot_output("WorkImbalance_vs_perfconfig_params.png")

    if args.m == "nmk":
        figure, axes = plt.subplots(3, 7)
        for i, nmk in enumerate(['n', 'm', 'k']):
            for j, param in enumerate(params):
                subplot = axes[i, j]
                sns.scatterplot(x=list[param], y=list[nmk], alpha=0.7, ax=subplot)
                subplot.set_xlabel(param)
                subplot.set_ylabel(nmk)
        plot_output("NMK_vs_perfconfig_params.png")

    return pd.concat(top_list)


def analyze_conv_file(file, n):
    # implementation goes here

    raise NotImplementedError("The script is not implemented for analyzing conv files yet.")


def calculate_arithmetic_intensity(m, n, k):
    return (m * n * k) / (m * n + m * k + n * k)  # opPerByte/bytesLoaded


def calculate_occupancy(m,
                        n,
                        g,
                        m_per_block,
                        n_per_block,
                        mn_per_wave,
                        min_num_waves,
                        split_k_factor=1):
    m_tiles = math.ceil(m / m_per_block)
    n_tiles = math.ceil(n / n_per_block)

    workgroups = g * m_tiles * n_tiles * split_k_factor
    waves_per_block = m_per_block * n_per_block // mn_per_wave
    waves = workgroups * waves_per_block

    return waves / min_num_waves


def calculate_work_imbalance(m,
                             n,
                             g,
                             m_per_block,
                             n_per_block,
                             mn_per_wave,
                             min_num_waves,
                             split_k_factor=1):
    m_tiles = math.ceil(m / m_per_block)
    n_tiles = math.ceil(n / n_per_block)
    workgroups = g * m_tiles * n_tiles * split_k_factor
    waves_per_block = m_per_block * n_per_block // mn_per_wave
    waves = workgroups * waves_per_block
    work_imbalance_interm_res = (waves % min_num_waves) / min_num_waves

    return ((1 - (work_imbalance_interm_res)) if work_imbalance_interm_res != 0 else 0)


def plot_output(name):
    if args.o:
        os.makedirs(args.o, exist_ok=True)
        plt.savefig(os.path.join(args.o, name), dpi=300)
        plt.close()
    else:
        plt.show()


def determine_file_type(file):
    with open(file, 'r') as file:
        header = file.readline().strip()

    if "Direction" in header:
        return "conv"
    elif "TransA" in header:
        return "gemm"
    else:
        raise Exception("Invalid file format or support for filetype not implemented yet: {file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze .tsv.debug file")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--n", type=float, default=5)  # percent of configs close to winning
    parser.add_argument("--m", type=str, default="ai")  # plots to be shown: ai, oc, wi, nmk
    parser.add_argument("--t", type=str, default="m")  # threshold formula: m, mn, qn
    parser.add_argument("--o", type=str, default=None)  # Directory in case of saving the plots
    parser.add_argument(
        "--c", type=int, default=None
    )  # num_cus (if data is not collected on the machine on which the script is executed)

    args = parser.parse_args()

    num_cus = assign_num_cu()
    min_num_waves = num_cus * num_eu_per_cu

    row_list = []

    for file in args.files:
        file_type = determine_file_type(file)

        if file_type == "gemm":
            row_list.append(analyze_gemm_file(file, args.n))
        elif file_type == "conv":
            row_list.append(analyze_conv_file(file, args.n))
