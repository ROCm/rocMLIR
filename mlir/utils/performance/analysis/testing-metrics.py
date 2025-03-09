# Python script that analyses .tsv.debug files and gives insights such as
# important metrics (Arithmetic Intensity, Occupancy, Work Imbalance) and
# plots correlation between them with the selected parameters.
#
# Usage: python3 ./testing-metrics.py <debug file(s)> [--n <percent>] [--m <metrics>]
# Arguments:
#       <debug file(s)>               Input file(s) in .tsv.debug format
#       --n <percent>                 Percent of the best perfconfigs to be considered (default=5)
#       --m <metrics>                 Metrics to be shown (ai, oc, wi, nmk)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import math

numEUPerCU = 4 # may be changed in newer architectures
numCUs = 304 # temporary hardcoded

minNumWaves = numCUs * numEUPerCU


def analyze_gemm_file(file, n):
    df = pd.read_csv(file, sep='\t')

    gemm_keys = ['TransA', 'TransB', 'G', 'M', 'K', 'N']
    perfConfig_params = ['MPerBlock', 'NPerBlock', 'KPerBlock', 'MPerWave', 'NPerWave', 'kPack', 'splitKFactor', 'forceUnroll', 'ThreadCopyMore']

    df[perfConfig_params] = df["PerfConfig"].str.replace("v2:", "").str.split(",", expand=True)

    df["ArithmeticIntensity"] = df.apply(lambda row: calculate_arithmetic_intensity(row["M"], row["N"], row["K"]), axis=1)
    df["MNPerWave"] = df.apply(lambda row: (int(row["MPerWave"]) * int(row["NPerWave"])), axis=1)
    df["Occupancy"] = df.apply(lambda row: calculate_occupancy(row["M"], row["N"], row["G"], row["MPerBlock"], row["NPerBlock"], row["MNPerWave"], minNumWaves), axis=1)
    df["WorkImbalance"] = df.apply(lambda row: calculate_work_imbalance(row["M"], row["N"], row["G"], row["MPerBlock"], row["NPerBlock"], row["MNPerWave"], minNumWaves, row["splitKFactor"]), axis=1)

    top_list = []

    for (key, group) in df.groupby(gemm_keys):
        threshold = group['TFlops'].max() # change the grouping method according to the needs
        top_list.append(group[group['TFlops'] == threshold])

    list = pd.concat(top_list)

    df[['Unnamed: 0', 'DataType', 'OutDataType', 'Chip', 'numCU', 'TransA',
       'TransB', 'G', 'M', 'K', 'N', 'PerfConfig', 'LDSBankConflict', 'TFlops',
       'MPerBlock', 'NPerBlock', 'KPerBlock', 'MPerWave', 'NPerWave', 'kPack',
       'splitKFactor', 'forceUnroll', 'ThreadCopyMore', 'ArithmeticIntensity', 'Occupancy', 'WorkImbalance']] = df[['Unnamed: 0', 'DataType', 'OutDataType', 'Chip', 'numCU', 'TransA',
       'TransB', 'G', 'M', 'K', 'N', 'PerfConfig', 'LDSBankConflict', 'TFlops',
       'MPerBlock', 'NPerBlock', 'KPerBlock', 'MPerWave', 'NPerWave', 'kPack',
       'splitKFactor', 'forceUnroll', 'ThreadCopyMore', 'ArithmeticIntensity', 'Occupancy', 'WorkImbalance']].apply(pd.to_numeric, errors='coerce')

    list[['Unnamed: 0', 'DataType', 'OutDataType', 'Chip', 'numCU', 'TransA',
       'TransB', 'G', 'M', 'K', 'N', 'PerfConfig', 'LDSBankConflict', 'TFlops',
       'MPerBlock', 'NPerBlock', 'KPerBlock', 'MPerWave', 'NPerWave', 'kPack',
       'splitKFactor', 'forceUnroll', 'ThreadCopyMore', 'ArithmeticIntensity', 'Occupancy', 'WorkImbalance']] = list[['Unnamed: 0', 'DataType', 'OutDataType', 'Chip', 'numCU', 'TransA',
       'TransB', 'G', 'M', 'K', 'N', 'PerfConfig', 'LDSBankConflict', 'TFlops',
       'MPerBlock', 'NPerBlock', 'KPerBlock', 'MPerWave', 'NPerWave', 'kPack',
       'splitKFactor', 'forceUnroll', 'ThreadCopyMore', 'ArithmeticIntensity', 'Occupancy', 'WorkImbalance']].apply(pd.to_numeric, errors='coerce')   

    params = ['MPerBlock', 'NPerBlock', 'KPerBlock', 'MPerWave', 'NPerWave', 'kPack', 'splitKFactor']

    if args.m == "ai":
        print(list.corr()['ArithmeticIntensity'])

        fig, axes = plt.subplots(2, 4)
        for ax, param in zip(axes.flat, params):
            ax.scatter(list[param], list['ArithmeticIntensity'], alpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel('ArithmeticIntensity')

        plt.tight_layout()
        plt.show()

    if args.m == "oc":        
        print(list.corr()['Occupancy'])

        fig, axes = plt.subplots(2, 4)
        for ax, param in zip(axes.flat, params):
            ax.scatter(list[param], list['Occupancy'], alpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel('Occupancy')

        plt.tight_layout()
        plt.show()

    if args.m == "wi":
        print(list.corr()['WorkImbalance'])

        fig, axes = plt.subplots(2, 4)
        for ax, param in zip(axes.flat, params):
            ax.scatter(list[param], list['WorkImbalance'], alpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel('WorkImbalance')

        plt.tight_layout()
        plt.show()

    if args.m == "nmk":
        figure, axes = plt.subplots(3, 7)
        for i, nmk in enumerate(['N', 'M', 'K']):
            for j, param in enumerate(params):
                subplot = axes[i, j]
                sns.scatterplot(x=list[param], y=list[nmk], alpha=0.7, ax=subplot)
                subplot.set_xlabel(param)
                subplot.set_ylabel(nmk)
        plt.show()

    return pd.concat(top_list)


def analyze_conv_file(file, n):
    # implementation goes here

    top_list = []    
    return pd.concat(top_list)


def calculate_arithmetic_intensity(M, N, K):
    return (M*N*K)/(M*N + M*K + N*K) # opPerByte/bytesLoaded


def calculate_occupancy(M, N, G, MPerBlock, NPerBlock, MNPerWave, minNumWaves):
    MTiles = (int(M) + int(MPerBlock) - 1) / int(MPerBlock)
    NTiles = (int(N) + int(NPerBlock) - 1) / int(NPerBlock)

    WorkGroups = G * MTiles * NTiles
    WavesPerBlock = int(MPerBlock) * int(NPerBlock) / int(MNPerWave)
    Waves = WorkGroups * WavesPerBlock

    return Waves / minNumWaves


def calculate_work_imbalance(M, N, G, MPerBlock, NPerBlock, MNPerWave, minNumWaves, splitKFactor=1):
    MTiles = (int(M) + int(MPerBlock) - 1) / int(MPerBlock)
    NTiles = (int(N) + int(NPerBlock) - 1) / int(NPerBlock)
    WorkGroups = G * MTiles * NTiles * int(splitKFactor)
    WavesPerBlock = int(MPerBlock) * int(NPerBlock) / int(MNPerWave)
    Waves = WorkGroups * WavesPerBlock

    maxWavesPerCU = math.ceil(Waves / minNumWaves)

    return (maxWavesPerCU * minNumWaves) / Waves


def determine_filetype(file):
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
    parser.add_argument("--n", type=float, default=5) # percent of configs close to winning
    parser.add_argument("--m", type=str, default="ai") # plots to be shown: ai, oc, wi, nmk

    args = parser.parse_args()

    row_list = []

    for file in args.files:
        file_type = determine_filetype(file)
        
        if file_type == "gemm":
            row_list.append(analyze_gemm_file(file, args.n))
        elif file_type == "conv":
            row_list.append(analyze_conv_file(file, args.n))
