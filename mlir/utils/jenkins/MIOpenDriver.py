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
MIOPEN_GEN = 'miopen-gen'
MLIR_MIOPEN_DRIVER = 'mlir-miopen-driver'
MLIR_ROCM_RUNNER = 'mlir-rocm-runner'
MLIR_ROCM_RUNNER_ARGS = ['--shared-libs=./lib/librocm-runtime-wrappers.so,./external/llvm-project/llvm/lib/libmlir_runner_utils.so', '--entry-point-result=void']
ROCPROF = '/opt/rocm/bin/rocprof'
MIOPEN_DRIVER = '../MIOpen/build/bin/MIOpenDriver'
BENCHMARKING_RESULT_FILE_NAME = 'results.stats.csv'
CONFIGURATION_FILE_NAME ='../mlir/utils/jenkins/miopen-tests/resnet50-miopen-configs'

DIRECTIONS = ['-F 1', '-F 2', '-F 4']
DATA_TYPES = ['conv', 'convfp16']
LAYOUTS = ['NHWC', 'NCHW']

# utility functions.
def getConfigurations(fileName):
    configFile = open(fileName, 'r')
    lines = configFile.readlines()
    configs = [];
    # All combinations of conv direction, type and layouts
    for direction, datatype, layout, line in itertools.product(DIRECTIONS, DATA_TYPES, LAYOUTS, lines):
        line = line.strip()
        if len(line) > 0 and line[0] != '#':
            oneConfig = f"{datatype} {direction} -f {layout } -I {layout} -O {layout} {line}"
            configs.append(oneConfig)
    # int8 convolution for fwd direction
    for layout, line in itertools.product(LAYOUTS, lines):
        line = line.strip()
        if len(line) > 0 and line[0] != '#':
            oneConfig = f"convint8 -F 1 -f {layout} -I {layout} -O {layout} {line}"
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

    def __repr__(self):
        return f"""ConvConiguration(dtype={self.dataType!r}, direction={self.direction!r}, layout={self.inputLayout.upper()!r},
                n={self.n!r}, c={self.c!r}, hi={self.hi!r}, wi={self.wi!r}, k={self.k!r}, y={self.y!r}, x={self.x!r},
                convStrideH={self.convStrideH!r}, convStrideW={self.convStrideW!r}, paddingH={self.paddingH!r}, paddingW={self.paddingW!r},
                dilationH={self.dilationH!r}, dilationW={self.dilationW!r}, group={self.group!r}, xdlops={self.xdlops!r})"""

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
                           '--fil_h', str(self.y),
                           '--fil_w', str(self.x),
                           '--dilation_h', str(self.dilationH),
                           '--dilation_w', str(self.dilationW),
                           '--conv_stride_h', str(self.convStrideH),
                           '--conv_stride_w', str(self.convStrideW),
                           '--padding_h', str(self.paddingH),
                           '--padding_w', str(self.paddingW)])

        return result

    MLIR_FILTER_LAYOUTS = {"NCHW": "kcyx", "NHWC": "kyxc"}
    MLIR_OUTPUT_LAYOUTS = {"NCHW": "nkhw", "NHWC": "nhwk"}

    @classmethod
    def fromCommandLine(cls, argv, xdlops):
        # determine dataType from argv[1]
        if argv[0] == 'conv':
            dataType = 'f32'
        elif argv[0] == 'convfp16':
            dataType = 'f16'
        elif argv[0] == 'convbfp16':
            dataType = 'bf16'
        elif argv[0] == 'convint8':
            dataType = 'i8'

        layout = None
        try:
            # TBD:
            # implement -m ?
            # implement -t ?
            opts, args = getopt.getopt(argv[1:], "F:f:I:O:n:c:H:W:k:y:x:p:q:l:j:u:v:g:m:t:")
        except getopt.GetOptError:
            print('getopt error')
            sys.exit(1)

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
                    direction = 'fwd'
                elif int(arg) == 2:
                    direction = 'bwd'
                elif int(arg) == 4:
                    direction = 'wrw'
            elif opt == '-f':
                if layout is not None and layout != arg:
                    raise ValueError("Mixed layouts")
                layout = arg
            elif opt == '-I':
                if layout is not None and layout != arg:
                    raise ValueError("Mixed layouts")
                layout = arg
            elif opt == '-O':
                if layout is not None and layout != arg:
                    raise ValueError("Mixed layouts")
                layout = arg
            elif opt == "-n":
                n = int(arg)
            elif opt == '-c':
                c = int(arg)
            elif opt == '-H':
                hi = int(arg)
            elif opt == '-W':
                wi = int(arg)
            elif opt == '-k':
                k = int(arg)
            elif opt == '-y':
                y = int(arg)
            elif opt == '-x':
                x = int(arg)
            elif opt == '-u':
                convStrideH = int(arg)
            elif opt == '-v':
                convStrideW = int(arg)
            elif opt == '-p':
                paddingH = int(arg)
            elif opt == '-q':
                paddingW = int(arg)
            elif opt == '-l':
                dilationH = int(arg)
            elif opt == '-j':
                dilationW = int(arg)
            elif opt == '-g':
                group = int(arg)
            else:
                continue

        return cls(dataType, direction, layout, n, c, hi, wi, k, y, x,
            convStrideH, convStrideW, paddingH, paddingW, dilationH, dilationW,
            group, xdlops)

    def __init__(self, dtype: str, direction: str, layout: str,
                    n: int, c: int, hi: int, wi: int, k: int, y: int, x: int,
                    convStrideH: int, convStrideW: int, paddingH: int, paddingW: int, dilationH: int, dilationW: int,
                    group: int, xdlops: bool):
        if dtype not in {"f16", "f32", "bf16", "i8"}:
            raise ValueError(f"Invalid datatype: {dtype}")
        if direction not in {"fwd", "bwd", "wrw"}:
            raise ValueError(f"Invalid direction: {direction}")
        if layout not in self.MLIR_OUTPUT_LAYOUTS:
            raise ValueError(f"Invalid layout: {layout}")

        self.dataType = dtype
        self.direction = direction

        self.filterLayout = self.MLIR_FILTER_LAYOUTS[layout]
        self.inputLayout = layout.lower()
        self.outputLayout = self.MLIR_OUTPUT_LAYOUTS[layout]

        self.n = n
        self.c = c
        self.hi = hi
        self.wi = wi
        self.k = k
        self.y = y
        self.x = x

        self.convStrideH = convStrideH
        self.convStrideW = convStrideW
        self.paddingH = paddingH
        self.paddingW = paddingW
        self.dilationH = dilationH
        self.dilationW = dilationW

        self.group = group
        self.xdlops = xdlops
        self.ho = math.floor((self.hi + self.paddingH * 2 - (self.y - 1) * self.dilationH - 1 ) / self.convStrideH) + 1
        self.wo = math.floor((self.wi + self.paddingW * 2 - (self.x - 1) * self.dilationW - 1 ) / self.convStrideW) + 1

def runConfigWithMLIR(config):
    # remove the result file generated by rocprof in previous benchmarking
    os.system("rm "+BENCHMARKING_RESULT_FILE_NAME)
    commandLineOptions = config.generateMlirDriverCommandLine()
    print("Running MLIR Benchmark: ", repr(config))
    miopenGenCommand = os.path.join(MLIR_BIN_DIR, MIOPEN_GEN) + ' -ph ' + commandLineOptions
    mlirMiopenDriverCommand = [os.path.join(MLIR_BIN_DIR, MLIR_MIOPEN_DRIVER), '-c']
    profilerCommand = [ROCPROF, '--stats', os.path.join(MLIR_BIN_DIR, MLIR_ROCM_RUNNER)] + MLIR_ROCM_RUNNER_ARGS

    # invoke miopen-gen.
    p1 = subprocess.Popen(miopenGenCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # pipe to mlir-miopen-driver
    p2 = subprocess.Popen(mlirMiopenDriverCommand, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p1.stdout.close() # Allow p1 to receive a SIGPIPE if p2 exits.
    # pipe to rocprof + mlir-rocm-runner.
    p3 = subprocess.Popen(profilerCommand, stdin=p2.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2.stdout.close() # Allow p2 to receive a SIGPIPE if p3 exits.
    # get output.
    try:
        outs, errs = p3.communicate(timeout=60)
        if len(errs) > 0:
            print("Test printed errors: ", errs.decode('utf-8'))
            print("Failing command line: ", miopenGenCommand)
    except subprocess.TimeoutExpired:
        print("Test timed out: ", miopenGenCommand)
        p3.kill()
        outs, errs = p3.communicate()

def runConfigWithMIOpenDriver(commandLine, envs):
    # remove the result file generated by rocprof in previous benchmarking
    os.system("rm "+BENCHMARKING_RESULT_FILE_NAME)
    profilerCommand = [ROCPROF, '--stats', MIOPEN_DRIVER, *commandLine, '-V', '0']
    print("Running MIOpen Benchmark: ", ' '.join(commandLine))
    # invoke rocprof + MIOpenDriver.
    p1 = subprocess.Popen(profilerCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=envs)
    # get output.
    try:
        outs, errs = p1.communicate(timeout=180)
        if len(errs) > 0:
            print("MIOpen benchmark produced errors: ", errs.decode('utf-8'))
    except subprocess.TimeoutExpired:
        p1.kill()
        print("MIOpen benchmark timed out")
        outs, errs = p1.communicate()

# Benchmarking function.
def benchmarkMLIR(commandLine, xdlops):
    config = ConvConfiguration.fromCommandLine(commandLine, xdlops)
    runConfigWithMLIR(config)
    # get nanoseconds from rocprof output.
    nanoSeconds = getNanoSeconds(BENCHMARKING_RESULT_FILE_NAME)
    return config.tableEntry(nanoSeconds)

def benchmarkMIOpen(commandLine, xdlops, envs=dict()):
    config = ConvConfiguration.fromCommandLine(commandLine, xdlops)
    runConfigWithMIOpenDriver(commandLine, envs)
    # get nanoseconds from rocprof output.
    nanoSeconds = getNanoSeconds(BENCHMARKING_RESULT_FILE_NAME)
    return config.tableEntry(nanoSeconds)

#Generate MLIR vs. MIOpen performance results
def generatePerformanceResults(configs, xdlops):
    mlir_df = pd.DataFrame(benchmarkMLIR(testVector.split(sep=' '), xdlops)
        for testVector in configs)
    miopen_df = pd.DataFrame(benchmarkMIOpen(testVector.split(sep=' '), xdlops)
        for testVector in configs)

    df = mlir_df.merge(miopen_df, on=ConvConfiguration.TABLE_COLUMNS[:-1],
                           suffixes=('', ' (MIOpen)'))
    df.rename(columns={'TFlops': 'MLIR TFlops', 'TFlops (MIOpen)': 'MIOpen TFlops (no MLIR Kernels)'}, inplace=True)

    df['MLIR/MIOpen'] = df['MLIR TFlops'] / df['MIOpen TFlops (no MLIR Kernels)']
    df.to_csv(reportUtils.PERF_REPORT_FILE, index=False)

def getSolverName(testVector, xdlops):
    config = ConvConfiguration.fromCommandLine(testVector.split(sep=' '), xdlops)
    if config.direction == 'fwd':
       solverName = 'ConvMlirIgemmFwd'
    elif config.direction == 'bwd':
       solverName = 'ConvMlirIgemmBwd'
    else:
       solverName = 'ConvMlirIgemmWrW'
    if xdlops == True:
       solverName+='Xdlops'
    return solverName

def benchmarkMIOpenWithMLIRKernels(configs, xdlops, filename):
    solver_names = {testVector : getSolverName(testVector, xdlops) for testVector in configs}

    # Set environment variables for running MIOpenDriver with MLIR kernels
    envs = os.environ.copy()
    envs['MIOPEN_FIND_MODE'] = '1'
    envs['MIOPEN_DRIVER_USE_GPU_REFERENCE'] = '1'
    perf_list = []
    for testVector in configs:
        envs['MIOPEN_DEBUG_FIND_ONLY_SOLVER']=solver_names[testVector]
        perf_list.append(benchmarkMIOpen(testVector.split(sep=' '), xdlops, envs))
    df = pd.DataFrame(perf_list)
    df.to_csv(filename, index=False)

#Tune MIOpen with MLIR kernels
def tuneMLIRKernels(configs, xdlops):
    solver_names = {testVector : getSolverName(testVector, xdlops) for testVector in configs}

    envs = os.environ.copy()
    envs['MIOPEN_FIND_ENFORCE'] = '4'
    envs['MIOPEN_DRIVER_USE_GPU_REFERENCE'] = '1'
    for testVector in configs:
        envs['MIOPEN_DEBUG_FIND_ONLY_SOLVER']=solver_names[testVector]
        commandLine = testVector.split(sep=' ')
        config = ConvConfiguration.fromCommandLine(commandLine, xdlops)
        if config.inputLayout == 'nchw':
            MIOpenDriverCommand = [MIOPEN_DRIVER, *commandLine,'-V', '0']
            print(' '.join(MIOpenDriverCommand))
            p1 = subprocess.Popen(MIOpenDriverCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=envs)
            # get output.
            try:
               outs, errs = p1.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                p1.kill()
                print("MIOpen tuning timed out")
                outs, errs = p1.communicate()

# Main function.
if __name__ == '__main__':
    """
usage examples:
  python3 MIOpenDriver.py
  python3 MIOpenDriver.py -b
  python3 MIOpenDriver.py -bmiopen
  python3 MIOpenDriver.py conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
  python3 MIOpenDriver.py -miopen conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
  python3 MIOpenDriver.py -bmiopen_use_tuned_mlir
  python3 MIOpenDriver.py -bmiopen_use_untuned_mlir
    """

    xdlops = False
    configs = getConfigurations(CONFIGURATION_FILE_NAME);
    r = subprocess.run("/opt/rocm/bin/rocm_agent_enumerator -t GPU | grep -E 'gfx908|gfx90a'", shell=True)
    if r.returncode == 0:
        xdlops = True

    if len(sys.argv) == 1:
        # batch benchmark with MLIR and MIOpen.
        generatePerformanceResults(configs, xdlops)
    elif sys.argv[1] == '-bmiopen_use_tuned_mlir':
        benchmarkMIOpenWithMLIRKernels(configs, xdlops, reportUtils.MIOPEN_TUNED_REPORT_FILE)
    elif sys.argv[1] == '-bmiopen_use_untuned_mlir':
        benchmarkMIOpenWithMLIRKernels(configs, xdlops, reportUtils.MIOPEN_UNTUNED_REPORT_FILE)
    elif sys.argv[1] =='-tuning':
        #tune MLIR kernels
        tuneMLIRKernels(configs, xdlops)
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
