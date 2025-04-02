# Python script that analyses .tsv.debug files and gives insights such as
# important metrics (Arithmetic Intensity, Occupancy, Work Imbalance) and
# plots correlation between them with the selected parameters.
#
# Usage: python3 ./testing-metrics.py <debug file(s)> [--n <percent>] [--m <metrics>] [--t <method for threshold>] [--o <output directory>] [--c <numCUs>]
# Arguments:
#       <debug file(s)>               Input file(s) in .tsv.debug format
#       --n <percent>                 Percent of the best perfconfigs to be considered (default=5) - doesn't affect analysis when checking only the best perfConfigs
#       --m <metrics>                 Metrics to be shown (ai, oc, wi, nmk)
#       --t <method for threshold>    Method for calculating threshold (m - max, mn - maxN, qn - quantileN)
#       --o <output directory>        Output directory in case of saving plots
#       --c <numCUs>                  CUs count if data is not collected on the machine on which the script is executed

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import math
import os
from hip import hip

# TODO use AmdArchDb.py (when it's implemented)

numEUPerCU = 4 # may be changed in newer architectures

def hipCheck(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def assignNumCu():
    if args.c:
        return int(args.c)
    else:
        props = hip.hipDeviceProp_t()
        hipCheck(hip.hipGetDeviceProperties(props,0))
        print("Using info from GPU 0 in your system, the data should have be obtained from the same GPU.")
        return int(props.multiProcessorCount)


def analyzeGemmFile(file, n):
    df = pd.read_csv(file, sep='\t')

    gemmKeys = ['TransA', 'TransB', 'G', 'M', 'K', 'N']
    perfConfigParams = ['MPerBlock', 'NPerBlock', 'KPerBlock', 'MPerWave', 'NPerWave', 'kPack', 'splitKFactor', 'forceUnroll', 'ThreadCopyMore']

    assert df["PerfConfig"].str.startswith("v2:").all(), "PerfConfig that doesn't start with v2: found"
    df[perfConfigParams] = df["PerfConfig"].str.replace("v2:", "").str.split(",", expand=True)

    df["ArithmeticIntensity"] = df.apply(lambda row: calculateArithmeticIntensity(row["M"], row["N"], row["K"]), axis=1)
    df["MNPerWave"] = df.apply(lambda row: (int(row["MPerWave"]) * int(row["NPerWave"])), axis=1)
    df["Occupancy"] = df.apply(lambda row: calculateOccupancy(int(row["M"]), int(row["N"]), int(row["G"]), int(row["MPerBlock"]), int(row["NPerBlock"]), int(row["MNPerWave"]), minNumWaves), axis=1)
    df["WorkImbalance"] = df.apply(lambda row: calculateWorkImbalance(int(row["M"]), int(row["N"]), int(row["G"]), int(row["MPerBlock"]), int(row["NPerBlock"]), int(row["MNPerWave"]), minNumWaves, int(row["splitKFactor"])), axis=1)

    topList = []

    for (key, group) in df.groupby(gemmKeys):
        if args.t == "m":
            threshold = group['TFlops'].max()
            topList.append(group[group['TFlops'] == threshold])
        if args.t == "mn":
            threshold = group[ group['TFlops'] >= (group['TFlops'].max() * (1 - n / 100))]
            topList.append(group[group['TFlops'] >= threshold])
        if args.t == "qn":
            threshold = group['TFlops'].quantile(1 - n / 100.0)
            topList.append(group[group['TFlops'] >= threshold])

    list = pd.concat(topList)

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
        plotOutput("ArithmeticIntensity_vs_perfConfigParams.png")

    if args.m == "oc":        
        print(list.corr()['Occupancy'])

        fig, axes = plt.subplots(2, 4)
        for ax, param in zip(axes.flat, params):
            ax.scatter(list[param], list['Occupancy'], alpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel('Occupancy')

        plt.tight_layout()
        plotOutput("Occupancy_vs_perfConfigParams.png")

    if args.m == "wi":
        print(list.corr()['WorkImbalance'])

        fig, axes = plt.subplots(2, 4)
        for ax, param in zip(axes.flat, params):
            ax.scatter(list[param], list['WorkImbalance'], alpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel('WorkImbalance')

        plt.tight_layout()
        plotOutput("WorkImbalance_vs_perfConfigParams.png")

    if args.m == "nmk":
        figure, axes = plt.subplots(3, 7)
        for i, nmk in enumerate(['N', 'M', 'K']):
            for j, param in enumerate(params):
                subplot = axes[i, j]
                sns.scatterplot(x=list[param], y=list[nmk], alpha=0.7, ax=subplot)
                subplot.set_xlabel(param)
                subplot.set_ylabel(nmk)
        plotOutput("NMK_vs_perfConfigParams.png")

    return pd.concat(topList)


def analyzeConvFile(file, n):
    # implementation goes here

    raise NotImplementedError("The script is not implemented for analyzing conv files yet.")


def calculateArithmeticIntensity(M, N, K):
    return (M*N*K)/(M*N + M*K + N*K) # opPerByte/bytesLoaded


def calculateOccupancy(M, N, G, MPerBlock, NPerBlock, MNPerWave, minNumWaves, splitKFactor=1):
    MTiles = math.ceil(M/MPerBlock)
    NTiles = math.ceil(N/NPerBlock)

    WorkGroups = G * MTiles * NTiles * splitKFactor
    WavesPerBlock = MPerBlock * NPerBlock // MNPerWave
    Waves = WorkGroups * WavesPerBlock

    return Waves / minNumWaves


def calculateWorkImbalance(M, N, G, MPerBlock, NPerBlock, MNPerWave, minNumWaves, splitKFactor=1):
    MTiles = math.ceil(M/MPerBlock)
    NTiles = math.ceil(N/NPerBlock)
    WorkGroups = G * MTiles * NTiles * splitKFactor
    WavesPerBlock = MPerBlock * NPerBlock // MNPerWave
    Waves = WorkGroups * WavesPerBlock
    WorkImbalanceIntermedResult = (Waves % minNumWaves) / minNumWaves

    return ((1-(WorkImbalanceIntermedResult)) if WorkImbalanceIntermedResult != 0 else 0)


def plotOutput(name):
    if args.o:
        os.makedirs(args.o, exist_ok=True)
        plt.savefig(os.path.join(args.o, name), dpi=300)
        plt.close()
    else:
        plt.show()


def determineFiletype(file):
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
    parser.add_argument("--t", type=str, default="m") # threshold formula: m, mn, qn
    parser.add_argument("--o", type=str, default=None) # Directory in case of saving the plots
    parser.add_argument("--c", type=int, default=None) # numCUs (if data is not collected on the machine on which the script is executed)

    args = parser.parse_args()

    numCUs = assignNumCu()
    minNumWaves = numCUs * numEUPerCU

    rowList = []

    for file in args.files:
        fileType = determineFiletype(file)
        
        if fileType == "gemm":
            rowList.append(analyzeGemmFile(file, args.n))
        elif fileType == "conv":
            rowList.append(analyzeConvFile(file, args.n))
