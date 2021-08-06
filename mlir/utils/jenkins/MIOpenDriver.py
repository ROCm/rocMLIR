#!/usr/bin/env python3

import reportUtils

import csv
from collections import OrderedDict
import getopt
import os
import subprocess
import sys
import math
import itertools
from datetime import date

import numpy as np
import pandas as pd

# global variables.
MLIR_BIN_DIR = './bin'
LLVM_BIN_DIR = './external/llvm-project/llvm/bin'
MLIR_MIOPEN_DRIVER = 'mlir-miopen-driver'
MLIR_ROCM_RUNNER = 'mlir-rocm-runner'
MLIR_ROCM_RUNNER_ARGS = ' --shared-libs=./external/llvm-project/llvm/lib/librocm-runtime-wrappers.so,./external/llvm-project/llvm/lib/libmlir_runner_utils.so --entry-point-result=void'
ROCPROF = '/opt/rocm/bin/rocprof'
MIOPEN_DRIVER = '../MIOpen/build/bin/MIOpenDriver'
BENCHMARKING_RESULT_FILE_NAME = 'results.stats.csv'
CONFIGURATION_FILE_NAME ='../mlir/utils/jenkins/miopen-tests/resnet50-miopen-configs'

DIRECTIONS = ['-F 1', '-F 2', '-F 4']
DATA_TYPES = ['conv', 'convfp16']
LAYOUTS = ['NCHW']
#LAYOUTS = ['NHWC', 'NCHW']

# utility functions.
def getConfigurations(fileName):
    xdlops = False;
    r = subprocess.run(f'/opt/rocm/bin/rocm_agent_enumerator -t GPU|grep gfx908', shell=True)
    if r.returncode == 0:
        xdlops = True

    configFile = open(fileName, 'r')
    lines = configFile.readlines()
    configs = [];
    for direction, datatype, layout, line in itertools.product(DIRECTIONS, DATA_TYPES, LAYOUTS, lines):
        line = line.strip()
        if len(line) > 0 and line[0] != '#':
            oneConfig = f"{datatype} {direction} -f {layout } -I {layout} -O {layout} {line}"
            configs.append(oneConfig)
    return configs

def getNanoSeconds(fileName):
    if not os.path.exists(fileName):
        return np.nan
    with open(fileName, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter = ',')

        result = 0
        for row in reader:
            result += int(row['AverageNs'])
        csv_file.close()
        return result

# convolution configurations.
class ConvConfiguration:
    def computeTFlops(self, ns):
        # NaN will propagate as expected
        return (2.0 * self.n * self.c * self.k * self.ho * self.wo * self.y * self.x) / (float(ns) * 1e-9) / 1e12

    TABLE_COLUMNS = reportUtils.TEST_PARAMETERS + ['TFlops']

    def tableEntry(self, nanoSeconds):
        # Future(kdrewnia): This can just be a dict literal on Python 3.7+
        result = OrderedDict()
        values = [self.direction, self.dataType, self.xdlops, self.filterLayout, self.inputLayout, self.outputLayout,
                   self.n, self.c, self.hi, self.wi, self.k, self.y, self.x, self.dilationH, self.dilationW,
                   self.convStrideH, self.convStrideW, self.paddingH, self.paddingW,
                   self.computeTFlops(nanoSeconds)]
        assert(len(self.TABLE_COLUMNS) == len(values))

        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def generateMlirDriverCommandLine(self):
        direction = {'fwd':'--operation conv2d',
                     'bwd':'--operation conv2d_bwd_data',
                     'wrw':'--operation conv2d_bwd_weight'}[self.direction]

        result = ' '.join([direction,
                           '-t', self.dataType,
                           '-x2' if self.xdlops else '',
                           '--fil_layout', self.filterLayout,
                           '--in_layout', self.inputLayout,
                           '--out_layout', self.outputLayout,
                           '--batchsize', str(self.n),
                           '--in_channels', str(self.c),
                           '--in_h', str(self.hi),
                           '--in_w', str(self.wi),
                           '--out_channels', str(self.k),
                           '--fil_w', str(self.y),
                           '--fil_h', str(self.x),
                           '--dilation_h', str(self.dilationH),
                           '--dilation_w', str(self.dilationW),
                           '--conv_stride_h', str(self.convStrideH),
                           '--conv_stride_w', str(self.convStrideW),
                           '--padding_h', str(self.paddingH),
                           '--padding_w', str(self.paddingW)])

        return result

    def __init__(self, argv, xdlops):
        self.xdlops = xdlops

        mlirFilterLayout={"NCHW":"kcyx", "NHWC":"kyxc"}
        mlirOutputLayout={"NCHW":"nkhw", "NHWC":"nkhw"}
        # determine dataType from argv[1]
        if argv[0] == 'conv':
            self.dataType = 'f32'
        elif argv[0] == 'convfp16':
            self.dataType = 'f16'
        elif argv[0] == 'convbfp16':
            self.dataType = 'bf16'

        try:
            # TBD:
            # implement -m ?
            # implement -t ?
            opts, args = getopt.getopt(argv[1:], "F:f:I:O:n:c:H:W:k:y:x:p:q:l:j:u:v:g:m:t:")
        except getopt.GetOptError:
            print('getopt error')
            sys.exit(-1)

        for opt, arg in opts:
            if opt == '-F':
                # -F
                # 1 fwd only
                # 2 bwd only
                # 4 wrw only
                # TBD:
                # 0 fwd+bwd+wrw
                # 3 fwd+bwd
                # 5 fwd+wrw
                # 6 bwd+wrw
                if int(arg) == 1:
                    self.direction = 'fwd'
                elif int(arg) == 2:
                    self.direction = 'bwd'
                elif int(arg) == 4:
                    self.direction = 'wrw'
            elif opt == '-f':
                self.filterLayout = mlirFilterLayout[arg]
            elif opt == '-I':
                self.inputLayout = arg.lower()
            elif opt == '-O':
                self.outputLayout = mlirOutputLayout[arg]
            elif opt == "-n":
                self.n = int(arg)
            elif opt == '-c':
                self.c = int(arg)
            elif opt == '-H':
                self.hi = int(arg)
            elif opt == '-W':
                self.wi = int(arg)
            elif opt == '-k':
                self.k = int(arg)
            elif opt == '-y':
                self.y = int(arg)
            elif opt == '-x':
                self.x = int(arg)
            elif opt == '-u':
                self.convStrideH = int(arg)
            elif opt == '-v':
                self.convStrideW = int(arg)
            elif opt == '-p':
                self.paddingH = int(arg)
            elif opt == '-q':
                self.paddingW = int(arg)
            elif opt == '-l':
                self.dilationH = int(arg)
            elif opt == '-j':
                self.dilationW = int(arg)
            elif opt == '-g':
                self.group = int(arg)
            else:
                continue

        # Ho and Wo are computed.
        self.ho = math.floor((self.hi + self.paddingH * 2 - (self.y - 1) * self.dilationH - 1 ) / self.convStrideH) + 1
        self.wo = math.floor((self.wi + self.paddingW * 2 - (self.x - 1) * self.dilationW - 1 ) / self.convStrideW) + 1


def runConfigWithMLIR(config):
    # remove the result file generated by rocprof in previous benchmarking
    os.system("rm "+BENCHMARKING_RESULT_FILE_NAME)
    commandLineOptions = config.generateMlirDriverCommandLine()
    print("Running MLIR Benchmark: ", commandLineOptions)
    mlirMIOpenDriverCommand = os.path.join(MLIR_BIN_DIR, MLIR_MIOPEN_DRIVER) + ' -ph -c ' + commandLineOptions
    profilerCommand = ROCPROF + ' --stats ' + os.path.join(LLVM_BIN_DIR, MLIR_ROCM_RUNNER) + MLIR_ROCM_RUNNER_ARGS

    # invoke mlir-miopen-driver.
    p1 = subprocess.Popen(mlirMIOpenDriverCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # pipe to rocprof + mlir-rocm-runner.
    p2 = subprocess.Popen(profilerCommand.split(), stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p1.stdout.close() # Allow p1 to receive a SIGPIPE if p2 exits.
    # get output.
    try:
        outs, errs = p2.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        p2.kill()
        outs, errs = p2.communicate()

def runConfigWithMIOpenDriver(commandLine):
    # remove the result file generated by rocprof in previous benchmarking
    os.system("rm "+BENCHMARKING_RESULT_FILE_NAME)
    MIOpenDriverCommand = MIOPEN_DRIVER + ' ' + ' '.join(commandLine) + ' -V 0'
    print("Running MIOpen Benchmark: ", commandLine)
    profilerCommand = ROCPROF + ' --stats ' + MIOpenDriverCommand
    # invoke rocprof + MIOpenDriver.
    p1 = subprocess.Popen(profilerCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # get output.
    try:
        outs, errs = p1.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        p1.kill()
        outs, errs = p1.communicate()

def output(string, outputFile):
    if outputFile != None:
        outputFile.write(string)
        outputFile.write('\n')

# Benchmarking function.
def benchmarkMLIR(commandLine, xdlops):
    config = ConvConfiguration(commandLine, xdlops)
    #runConfigWithMLIR(commandLine, xdlops)
    runConfigWithMLIR(config)
    # get nanoseconds from rocprof output.
    nanoSeconds = getNanoSeconds(BENCHMARKING_RESULT_FILE_NAME)
    return config.tableEntry(nanoSeconds)

def benchmarkMIOpen(commandLine, xdlops):
    config = ConvConfiguration(commandLine, xdlops)
    if config.inputLayout == 'nchw':
        runConfigWithMIOpenDriver(commandLine)
        # get nanoseconds from rocprof output.
        nanoSeconds = getNanoSeconds(BENCHMARKING_RESULT_FILE_NAME)
    else:
        # skip the test for non-supported layouts.
        # MIOpenDriver currently only support NCHW.
        nanoSeconds = np.nan
    return config.tableEntry(nanoSeconds)

def printPerformance(mlir_df, miopen_df):
    df = mlir_df.merge(miopen_df, on=ConvConfiguration.TABLE_COLUMNS[:-1],
                           suffixes=('', ' (MIOpen)'))
    df.rename(columns={'TFlops': 'MLIR TFlops', 'TFlops (MIOpen)': 'MIOpen TFlops'}, inplace=True)

    df['MLIR/MIOpen'] = df['MLIR TFlops'] / df['MIOpen TFlops']
    df.to_csv(reportUtils.PERF_REPORT_FILE, index=False)

    COLUMNS_TO_AVERAGE = ['MLIR TFlops', 'MIOpen TFlops', 'MLIR/MIOpen']
    plotMean = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
    plotMean.name = "Geo. mean"
    plotMean = pd.DataFrame(plotMean).T

    plotMean[['MLIR TFlops', 'MIOpen TFlops']]\
        .to_csv(reportUtils.PERF_PLOT_REPORT_FILE, index=False)

    means = df.groupby(["Direction", "DataType"])[COLUMNS_TO_AVERAGE]\
        .agg(reportUtils.geoMean)
    means.loc[("All", "All"),:] = df[COLUMNS_TO_AVERAGE].agg(reportUtils.geoMean)
    means.to_csv(reportUtils.PERF_STATS_REPORT_FILE)

    with open("MLIR_vs_MIOpen.html", 'w') as htmlOutput:
        reportUtils.htmlReport(df, means, "MLIR vs. MIOpen performance", "MLIR/MIOpen", htmlOutput)

def generatePerformanceResults(configs, xdlops):
    mlir_df = pd.DataFrame(benchmarkMLIR(testVector.split(sep=' '), xdlops) for testVector in configs)
    miopen_df = pd.DataFrame(benchmarkMIOpen(testVector.split(sep=' '), xdlops) for testVector in configs)

    printPerformance(mlir_df, miopen_df)

# Main function.
if __name__ == '__main__':
    """
usage examples:
  python3 MIOpenDriver.py
  python3 MIOpenDriver.py -o mlir.perf -b
  python3 MIOpenDriver.py -o miopen.perf -bmiopen
  python3 MIOpenDriver.py -o mlir.perf conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
  python3 MIOpenDriver.py -o miopen.perf -miopen conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
    """

    xdlops = False
    configs = getConfigurations(CONFIGURATION_FILE_NAME);
    r = subprocess.run(f'/opt/rocm/bin/rocm_agent_enumerator -t GPU|grep gfx908', shell=True)
    if r.returncode == 0:
        xdlops = True

    if len(sys.argv) == 1:
        # batch benchmark with MLIR and MIOpen.
        generatePerformanceResults(configs, xdlops)
    else:
        if sys.argv[1] == '-o':
            fileName = sys.argv[2]
            sys.argv.pop(1)
            sys.argv.pop(1)
        else:
            fileName = date.today().strftime("perf.%m%d%y")

        if sys.argv[1] == '-b':
            # CSV batch benchmarking mode with MLIR.
            df = pd.DataFrame(benchmarkMLIR(testVector.split(sep=' '), xdlops) for testVector in configs)
        elif sys.argv[1] == '-bmiopen':
            df = pd.DataFrame(benchmarkMIOpen(testVector.split(sep=' '), xdlops) for testVector in configs)
        elif sys.argv[1] == '-miopen':
            # bechmarking one config with MIOpenDriver.
            df = pd.DataFrame([benchmarkMIOpen(sys.argv[2:], xdlops)])
        else:
            # bechmarking one config with MLIR.
            df = pd.DataFrame([benchmarkMLIR(sys.argv[1:], xdlops)])

        df.to_csv(fileName)
        with pd.option_context('precision', reportUtils.ROUND_DIGITS):
            print(df) # for interactive consumption
