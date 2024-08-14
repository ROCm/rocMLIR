#!/usr/bin/env python3

import csv
from collections import OrderedDict
import getopt
import os
import subprocess
import sys
import math
import itertools
from datetime import date
from pathlib import Path
import glob
import argparse
import re

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

import reportUtils
from perfCommonUtils import Operation, GEMMLibrary

# global variables.
ROCPROF = '/opt/rocm/bin/rocprof'
MIOPENDRIVER = '/opt/rocm/bin/MIOpenDriver'
BENCHMARKING_RESULT_FILE_NAME = 'results.stats.csv'
BENCHMARKING_METRICS_FILE_NAME = 'results.csv'
ROCMLIR_INPUT_METRICS_FILE_NAME = 'rocmlir_metrics.txt'
DIRECTIONS = ['-F 1', '-F 2', '-F 4']
DATA_TYPES = ['conv', 'convfp16', 'convint8']
LAYOUTS = ['NHWC', 'NCHW']

DATA_TYPES_GEMM = ['f32', 'f16', 'i8', 'fp8']
DATA_TYPES_ATTENTION = ['f32', 'f16']
OUTPUT_DATA_TYPES_MAP = {'f32': 'f32', 'f16': 'f16', 'i8': 'i32', 'fp8':'f32',
                         'fp8_fp8': 'f32', 'fp8_bf8': 'f32', 'bf8_fp8': 'f32',
                         'bf8_bf8': 'f32'}

# Compiled regexp object used for extracting elapsed time from MIOpenDriver's output
ELAPSED_TIME_RE = re.compile(r"Elapsed: ([0-9\.]*) ms")
# Compiled regexp object used for extracting target chip from arch
GFX_CHIP_RE = re.compile(r"gfx[0-9a-z]+")
INFO_ARCH_NAME = re.compile(r"Name:\s*(.*)")
INFO_ARCH_CU = re.compile(r"Compute Unit:\s*(.*)")

@dataclass
class MLIRPaths:
    rocmlir_gen_path : str
    rocmlir_driver_path : str
    rocmlir_opt_path : str
    cpu_runner_path : str
    libmlir_rocm_runtime_path : str
    libconv_validation_wrappers_path : str
    libmlir_runtime_utils_path : str
    libmlir_c_runner_utils_path : str
    rocmlir_tuning_driver_path : str
    rocblas_benchmark_driver_path : Optional[str] = None
    ck_gemm_benchmark_driver_path : Optional[str] = None

@dataclass
class Paths:
    """This structure is used to hold paths needed to perform the tests"""
    configuration_file_path : str
    mlir_paths: Optional[MLIRPaths] = None

def find_mlir_build_dir() -> str:
    """
    Finds mlir build dir searching either WORKSPACE dir
    or home dir
    """
    rocmlir_gen_path = None
    candidate_paths = [
        # if the script is run from build dir
        Path('./bin/rocmlir-gen'),
        # if the script is run from source
        Path(__file__).parent.parent.parent.parent / 'build' / 'bin' / 'rocmlir-gen'
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            rocmlir_gen_path = candidate_path

    if not rocmlir_gen_path:
        try:
            # Prioritize the search in the current repo first.
            search_root = str(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
        except subprocess.CalledProcessError:
            # Else look in the home or WORKSPACE directory
            search_root = os.environ.get('WORKSPACE', str(Path.home()))
            assert search_root, "Cant find WORKSPACE env arg or home directory"

        rocmlir_gen_path = glob.glob(search_root + '/**/bin/rocmlir-gen', recursive=True)
        if len(rocmlir_gen_path) != 1:
            # rocmlir_gen not available or ambiguous
            return None
        rocmlir_gen_path = rocmlir_gen_path[0]

    build_dir = Path(rocmlir_gen_path).parent.parent
    return str(build_dir)

def create_paths(config_file_path, mlir_build_dir_path) -> Paths:
    """Creates the composite Paths structure using build dir paths"""

    mlir_paths = None
    if mlir_build_dir_path:
        mlir_bin_dir_path = (Path(mlir_build_dir_path) / 'bin').resolve()
        mlir_bin_dir = str(mlir_bin_dir_path)
        rocblas_benchmark_driver_location = mlir_bin_dir_path / 'rocblas-benchmark-driver'
        ck_gemm_benchmark_driver_location = mlir_bin_dir_path / 'ck-gemm-benchmark-driver'
        llvm_bin_dir = str((Path(mlir_build_dir_path) / 'external/llvm-project/llvm/bin').resolve())
        mlir_lib_dir = str((Path(mlir_build_dir_path) / 'lib').resolve())
        llvm_lib_dir = str((Path(mlir_build_dir_path) / 'external/llvm-project/llvm/lib').resolve())
        mlir_paths = MLIRPaths(rocmlir_gen_path = mlir_bin_dir + '/rocmlir-gen',
            rocmlir_driver_path = mlir_bin_dir + '/rocmlir-driver',
            rocmlir_opt_path = mlir_bin_dir + '/rocmlir-opt',
            cpu_runner_path = llvm_bin_dir + '/mlir-cpu-runner',
            libmlir_rocm_runtime_path =  llvm_lib_dir + '/libmlir_rocm_runtime.so',
            libconv_validation_wrappers_path = mlir_lib_dir + '/libconv-validation-wrappers.so',
            libmlir_runtime_utils_path = llvm_lib_dir + '/libmlir_runner_utils.so',
            libmlir_c_runner_utils_path = llvm_lib_dir + '/libmlir_c_runner_utils.so',
            rocmlir_tuning_driver_path = mlir_bin_dir + '/rocmlir-tuning-driver',
            rocblas_benchmark_driver_path = str(rocblas_benchmark_driver_location) \
              if rocblas_benchmark_driver_location.exists() else None,
            ck_gemm_benchmark_driver_path = str(ck_gemm_benchmark_driver_location) \
              if ck_gemm_benchmark_driver_location.exists() else None)

    return Paths(config_file_path, mlir_paths)

# utility functions.
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

def getMetricArgsForRocprof(arch):
    chip = GFX_CHIP_RE.search(arch).group(0)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(current_dir, ROCMLIR_INPUT_METRICS_FILE_NAME)
    metrics = []
    if (chip not in ["gfx942"]):
       metrics = ['-i', metrics_path]
    return metrics


# Bank conflict functions.The percentage of GPUTime LDS is stalled by bank
# conflicts. Value range: 0% (optimal) to 100% (bad).
def getBankConflict(fileName):
    if not os.path.exists(fileName):
        result = "NaN"
        return result
    with open(fileName, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter = ',')
        header = reader.fieldnames
        if 'LDSBankConflict' not in header:
            return np.nan

        result = []
        for row in reader:
            result.append(float(row['LDSBankConflict']))
        csv_file.close()
        result_average = sum(result) / len(result)
        return result_average

# Tuning databases
MaybeTuningDb = Optional[Dict[Tuple[str, str], str]]
def read_tuning_db(path: Optional[str]) -> MaybeTuningDb:
    try:
        ret = {}
        with open(path, 'r') as dbFile:
            for line in dbFile:
                line = line.strip()
                if line.startswith('#'):
                    continue
                entries = line.split('\t')

                # note: legacy format has 3 entries
                if len(entries) == 3:
                    arch, config, perfConfig = entries
                    ret[arch, config] = perfConfig
                # note: new format has 4 entries
                elif len(entries) == 4:
                    arch, _, config, perfConfig = entries
                    ret[arch, config] = perfConfig
                # note: 5-entry form includes tflops at end
                elif len(entries) == 5:
                    arch, _, config, perfConfig, _ = entries
                    ret[arch, config] = perfConfig
                else:
                    print("Warning: Malformed tuning database entry:", line)
                    continue
        return ret
    except FileNotFoundError:
        if path:
            print("Warning: Failed to find tuning database:", path)
        return None

def getMilliseconds(output):
    result = re.search(r"kernel time: (.*)", output.decode("utf-8"))
    if not result:
        return float('NaN')

    return float(result.group(1))

class PerfConfiguration:
    TABLE_COLUMNS = []

    def computeTFlops(self, ns: int) -> float:
        raise NotImplementedError()

    def tableEntry(self, nanoSeconds):
        raise NotImplementedError()

    def generateMlirDriverCommandLine(self, rocmlir_gen_flags):
        raise NotImplementedError()

    def setPerfConfig(self, perf_config):
        raise NotImplementedError()

    @classmethod
    def fromCommandLine(cls, argv, arch, numCU):
        raise NotImplementedError()

    def toCommandLine(self):
        raise NotImplementedError()

    @classmethod
    def benchmarkExternal(cls, commandLine, paths: Paths, arch, numCU, envs={}):
        raise NotImplementedError()

    EXTERNAL_NAME = "unknown"

# convolution configurations.
def getConvConfigurations(fileName):
    configs = []
    if fileName:
        with open(fileName, 'r') as configFile:
            lines = configFile.readlines()
            # All combinations of conv direction, type and layouts
            for direction, datatype, layout, line in \
                    itertools.product(DIRECTIONS, DATA_TYPES, LAYOUTS, lines):
                line = line.strip()

                # Skip empty lines
                if len(line) == 0 or line[0] == '#':
                    continue
                # Skip int8 non-fwd convolutions
                if datatype == 'convint8' and direction != '-F 1':
                    continue

                # Skip datatype if already in
                datatype = f"{datatype} "
                # check for the presense of a positional arg
                if line[0][0] != "-":
                    datatype = ""

                # Skip direction if already in
                direction = f"{direction} "
                if "-F" in line:
                    direction = ""

                # Skip filter layout if already in
                filter_layout = f"-f {layout} "
                if "-f" in line:
                    filter_layout = ""

                # Skip input layout if already in
                input_layout = f"-I {layout} "
                if "-I" in line:
                    input_layout = ""

                # Skip output layout if already in
                output_layout = f"-O {layout} "
                if "-O" in line:
                    output_layout = ""

                oneConfig = f"{datatype}{direction}{filter_layout}{input_layout}{output_layout}{line}"
                if oneConfig not in configs:
                    configs.append(oneConfig)
    return configs

class ConvConfiguration(PerfConfiguration):
    TABLE_COLUMNS = reportUtils.CONV_TEST_PARAMETERS + ['LDSBankConflict'] + ['TFlops']
    EXTERNAL_NAME = "MIOpen"

    def computeTFlops(self, ns):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        return (2.0 * self.n * self.c * self.k * self.ho * self.wo * self.y * self.x) / (float(ns) * 1e-9) / 1e12

    def tableEntry(self, nanoSeconds):
        # Future(kdrewnia): This can just be a dict literal on Python 3.7+
        bankConflict = getBankConflict(BENCHMARKING_METRICS_FILE_NAME)
        result = OrderedDict()
        values = [self.direction, self.dataType, self.chip, self.numCU, self.filterLayout, self.inputLayout, self.outputLayout,
                   self.n, self.c, self.hi, self.wi, self.k, self.y, self.x, self.dilationH, self.dilationW,
                   self.convStrideH, self.convStrideW, self.paddingH, self.paddingW, self.perfConfig, bankConflict,
                   self.computeTFlops(nanoSeconds)]
        assert(len(self.TABLE_COLUMNS) == len(values))

        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def __repr__(self):
        return f"""ConvConfiguration(dtype={self.dataType!r}, direction={self.direction!r}, layout={self.inputLayout.upper()!r},
                n={self.n!r}, c={self.c!r}, hi={self.hi!r}, wi={self.wi!r}, k={self.k!r}, y={self.y!r}, x={self.x!r},
                convStrideH={self.convStrideH!r}, convStrideW={self.convStrideW!r}, paddingH={self.paddingH!r}, paddingW={self.paddingW!r},
                dilationH={self.dilationH!r}, dilationW={self.dilationW!r}, group={self.group!r}, arch={self.arch!r}, numCU={self.numCU!r}, perf_config={self.perfConfig})"""

    def setPerfConfig(self, perf_config):
        self.perfConfig = perf_config

    def generateMlirDriverCommandLine(self, rocmlir_gen_flags):
        direction = {'fwd':'--operation conv',
                     'bwd':'--operation conv_bwd_data',
                     'wrw':'--operation conv_bwd_weight'}[self.direction]

        result = ' '.join([direction,
                           '-t', self.dataType,
                           '--arch', self.arch,
                           '--num_cu', str(self.numCU),
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
                           '--padding_w', str(self.paddingW),
                           '--kernel-repeats', str(self.MLIR_N_REPEATS),
                           f"--perf_config={self.perfConfig}"])
        result += ' '
        if rocmlir_gen_flags != '':
            result += ' '.join(rocmlir_gen_flags.split())
        return result

    MLIR_FILTER_LAYOUTS = {"NCHW": "kcyx", "NCHWG": "kcyxg", "NHWC": "kyxc", "NHWCG": "kyxcg",
                           "NC01": "kc01", "NC01G": "kc01g", "N01C": "k01c", "N01CG": "k01cg",
                           "GNC01":"gkc01"}
    MLIR_OUTPUT_LAYOUTS = {"NCHW": "nkhw", "NCHWG": "nkhwg", "NHWC": "nhwk", "NHWCG": "nhwkg",
                           "NC01": "nk01", "NC01G": "nk01g", "N01C": "n01k", "N01CG": "n01kg",
                           "NGC01":"ngk01"}
    filterLayoutMap = {'N':'k', 'C':'c', 'H':'0', 'W':'1', 'G':'g'}
    inputLayoutMap = {'N':'n', 'C':'c', 'H':'0', 'W':'1', 'G':'g'}
    outputLayoutMap = {'N':'n', 'C':'k', 'H':'0', 'W':'1', 'G':'g'}

    INVERSE_FILTER_LAYOUTS = {v: k for k, v in MLIR_FILTER_LAYOUTS.items()}
    INVERSE_OUTPUT_LAYOUTS = {v: k for k, v in MLIR_OUTPUT_LAYOUTS.items()}

    @classmethod
    def fromCommandLine(cls, argv, arch, numCU):
        # determine dataType from argv[1]
        # Please keep this in sync with mlir::rock::getTuningProblemStr()
        if argv[0] == 'conv':
            dataType = 'f32'
        elif argv[0] == 'convfp16':
            dataType = 'f16'
        elif argv[0] == 'convbfp16':
            dataType = 'bf16'
        elif argv[0] == 'convint8':
            dataType = 'i8'
        elif argv[0] == 'convfp8_fp8':
            dataType = 'fp8_fp8'
        elif argv[0] == 'convfp8_bf8':
            dataType = 'fp8_bf8'
        elif argv[0] == 'convbf8_fp8':
            dataType = 'bf8_fp8'
        elif argv[0] == 'convbf8_bf8':
            dataType = 'bf8_bf8'

        try:
            # TBD:
            # implement -m ?
            # implement -t ?
            opts, _ = getopt.getopt(argv[1:], "F:f:I:O:n:c:H:W:k:y:x:p:q:l:j:u:v:g:m:t:")
        except getopt.GetoptError:
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
                filterLayout = arg
            elif opt == '-I':
                inputLayout = arg
            elif opt == '-O':
                outputLayout = arg
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

        return cls(dataType, direction, filterLayout, inputLayout, outputLayout, n, c, hi, wi, k, y, x,
            convStrideH, convStrideW, paddingH, paddingW, dilationH, dilationW,
                   group, arch, numCU)

    def toCommandLine(self):
        return (f"conv{ {'f32':'', 'f16':'fp16', 'bf16':'bfp16', 'i8':'int8'}[self.dataType]} "
                + f"-F { {'fwd':1, 'bwd':2, 'wrw':4}[self.direction]} "
                + f"-f {self.INVERSE_FILTER_LAYOUTS[self.filterLayout]} -I {self.inputLayout.upper()} "
                + f"-O {self.INVERSE_OUTPUT_LAYOUTS[self.outputLayout]} "
                + f"-n {self.n} -c {self.c} -H {self.hi} -W {self.wi} -k {self.k} "
                + f"-y {self.y} -x {self.x} -p {self.paddingH} -q {self.paddingW} "
                + f"-u {self.convStrideH} -v {self.convStrideW} -l {self.dilationH} "
                + f"-j {self.dilationW} -m conv -g {self.group} -t 1")

    def __init__(self, dtype: str, direction: str, filterLayout: str, inputLayout:str, outputLayout:str,
                    n: int, c: int, hi: int, wi: int, k: int, y: int, x: int,
                    convStrideH: int, convStrideW: int, paddingH: int, paddingW: int,
                    dilationH: int, dilationW: int, group: int, arch: str, numCU: int):
        if dtype not in {"f16", "f32", "bf16", "i8", "fp8"}:
            raise ValueError(f"Invalid datatype: {dtype}")
        if direction not in {"fwd", "bwd", "wrw"}:
            raise ValueError(f"Invalid direction: {direction}")

        self.MLIR_N_REPEATS = 5
        self.dataType = dtype
        self.direction = direction

        self.filterLayout = ''.join(self.filterLayoutMap.get(c, c).lower() for c in filterLayout)
        self.inputLayout = ''.join(self.inputLayoutMap.get(c, c).lower() for c in inputLayout)
        self.outputLayout = ''.join(self.outputLayoutMap.get(c, c).lower() for c in outputLayout)

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
        self.arch = arch
        self.numCU = numCU
        self.chip = GFX_CHIP_RE.search(arch).group(0)

        self.ho = math.floor((self.hi + self.paddingH * 2 - (self.y - 1) * self.dilationH - 1 ) / self.convStrideH) + 1
        self.wo = math.floor((self.wi + self.paddingW * 2 - (self.x - 1) * self.dilationW - 1 ) / self.convStrideW) + 1

        self.perfConfig = ''

    @classmethod
    def benchmarkExternal(cls, commandLine, paths: Paths, arch, numCU, envs={}):
        os.system("rm -f "+BENCHMARKING_METRICS_FILE_NAME)
        config = cls.fromCommandLine(commandLine, arch, numCU)
        MIOpenDriverCommand = [MIOPENDRIVER, *commandLine, '-V', '0', '-t', '1']
        print("Running MIOpen Benchmark: ", ' '.join(commandLine))
        # invoke MIOpenDriver.
        p1 = subprocess.Popen(MIOpenDriverCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=envs)
        # get output.
        nanoSeconds = np.nan
        try:
            outs, errs = p1.communicate(timeout=300)
            if len(errs) > 0:
                print("MIOpen benchmark produced errors: ", errs.decode('utf-8'))
                if p1.returncode != 0:
                    raise OSError(errs.decode('utf-8'))
            else:
                # convert bytes to str
                outs = outs.decode('utf-8')
                # Extract Elapsed time in ms from the output of MIOpenDriver
                # Use regular expression to match the contents between
                # "Elasped: " (note the space at the end) and "ms"
                elapsedTimeInMs = ELAPSED_TIME_RE.search(outs).group(1)
                nanoSeconds = float(elapsedTimeInMs)*1.0e6
        except subprocess.TimeoutExpired:
            p1.kill()
            print("MIOpen benchmark timed out")
            outs, errs = p1.communicate()
        return config.tableEntry(nanoSeconds)

def getGemmConfigurations(fileName, dataTypes=DATA_TYPES_GEMM, outDataTypeMap=OUTPUT_DATA_TYPES_MAP):
    configs = []

    if fileName:
        with open(fileName, 'r') as configFile:
            lines = configFile.readlines()

            # All combinations of types and transposition (A and B)
            for datatype, transA, transB, line in \
                    itertools.product(DATA_TYPES_GEMM, ['false', 'true'], ['false', 'true'], lines):
                line = line.strip()
                # Skip empty lines
                if len(line) == 0 or line[0] == '#':
                    continue
                if datatype not in dataTypes:
                    continue

                # We need trailing spaces here to account for the concat below
                # Skip type if already in
                dataTypeString = ""
                if "-t " not in line:
                    dataTypeString = f"-t {datatype} "

                # Skip transA if already in
                transAString = ""
                if "-transA " not in line:
                    transAString = f"-transA {transA} "

                # Skip transB if already in
                transBString = ""
                if "-transB " not in line:
                    transBString = f"-transB {transB} "

                # Skip out_datatype if already in
                outDataTypeString = ""
                if "-out_datatype" not in line:
                    outDataTypeString = "-out_datatype " + outDataTypeMap.get(datatype, datatype) + " "

                # Strip to avoid spurious spaces
                oneConfig = f"{dataTypeString}{outDataTypeString}{transAString}{transBString}{line}".strip()
                if oneConfig not in configs:
                    configs.append(oneConfig)
    return configs

def getAttentionConfigurations(fileName):
    bool_space = ['false', 'true']
    default_test_space = {
        "-t": DATA_TYPES_ATTENTION,
        "-transQ": bool_space,
        "-transK": bool_space,
        "-transV": bool_space,
        "-transO": bool_space,
        "-with-attn-scale": bool_space
    }
    configs = []
    if fileName:
        with open(fileName, 'r') as configFile:
            lines = configFile.readlines()
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if len(line) == 0 or line[0] == '#':
                    continue
                test_space = []
                args = []
                for arg in default_test_space.keys():
                    if arg not in line:
                        test_space.append(default_test_space[arg])
                        args.append(arg)
                for test_vector in itertools.product(*test_space):
                    # Strip to avoid spurious spaces
                    oneConfig = line.strip()
                    for arg, value in zip(args, test_vector):
                        oneConfig = f"{arg} {value} {oneConfig}"
                    if oneConfig not in configs:
                        configs.append(oneConfig)
    return configs


class GemmConfiguration(PerfConfiguration):
    TABLE_COLUMNS = reportUtils.GEMM_TEST_PARAMETERS + ['LDSBankConflict'] + ['TFlops']
    def computeTFlops(self, ns):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        return (2.0 * self.g * self.m * self.k * self.n) / (float(ns) * 1e-9) / 1e12

    def tableEntry(self, nanoSeconds):
        # Future(kdrewnia): This can just be a dict literal on Python 3.7+
        bankConflict = getBankConflict(BENCHMARKING_METRICS_FILE_NAME)
        result = OrderedDict()
        values = [self.dataType, self.outDataType, self.chip, self.numCU, self.transA, self.transB, \
                   self.g, self.m, self.k, self.n, self.perfConfig, bankConflict, self.computeTFlops(nanoSeconds)]
        assert(len(self.TABLE_COLUMNS) == len(values))

        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def __repr__(self):
        return f"""GemmConfiguration(dtype={self.dataType!r}, outDataType={self.outDataType!r}, g={self.g!r}, m={self.m!r}, k={self.k!r}, n={self.n!r},
                transA={self.transA!r}, transB={self.transB!r}, arch={self.arch!r}, numCU={self.numCU}, perf_config={self.perfConfig})"""

    def setPerfConfig(self, perf_config):
        self.perfConfig = perf_config

    def generateMlirDriverCommandLine(self, rocmlir_gen_flags):
        result = ' '.join(['-operation', 'gemm',
                           '-t', self.dataType,
                           '-out_datatype', self.outDataType,
                           '--arch', self.arch,
                           '--num_cu', str(self.numCU),
                           '-g', str(self.g),
                           '-m', str(self.m),
                           '-k', str(self.k),
                           '-n', str(self.n),
                           f"-transA={self.transA}",
                           f"-transB={self.transB}",
                           '--kernel-repeats', str(self.MLIR_N_REPEATS),
                           f"--perf_config={self.perfConfig}"])

        result += ' '
        if rocmlir_gen_flags != '':
            result += ' '.join(rocmlir_gen_flags.split())
        return result

    @classmethod
    def fromCommandLine(cls, argv, arch, numCU):
        # Please keep this in sync with mlir::rock::getTuningProblemStr()
        dtype = None
        g = None
        m = None
        k = None
        n = None
        transA = None
        transB = None
        perf_config = ''
        for i in range(0, len(argv), 2):
            opt = argv[i]
            val = argv[i + 1]
            if opt == '-t':
                dtype = val
            elif opt == '-g':
                g = int(val)
            elif opt == '-m':
                m = int(val)
            elif opt == '-k':
                k = int(val)
            elif opt == '-n':
                n = int(val)
            elif opt.endswith("-transA"):
                transA = (val.lower() in ["1", "true"])
            elif opt.endswith("-transB"):
                transB = (val.lower() in ["1", "true"])
            elif opt.endswith("-out_datatype"):
                outDataType =val.lower()
            elif opt.endswith("-perf_config"):
                perf_config = val
            else:
                raise ValueError(f"Unknown GEMM config argument {opt} -> {val}")
        for v in [dtype, outDataType, g, m, k, n, transA, transB]:
            if v is None:
                raise ValueError("Incomplete GEMM configuration")

        return cls(dtype, outDataType, g, m, k, n, transA, transB, arch, numCU, perf_config)

    def toCommandLine(self):
        return (f"-t {self.dataType} -out_datatype {self.outDataType} "
                + f"-transA {str(self.transA).lower()} -transB {str(self.transB).lower()} "
                + f"-g {self.g} -m {self.m} -n {self.n} -k {self.k}")

    def __init__(self, dtype: str, outDataType: str, g: int, m: int, k: int, n: int,
                 transA: bool, transB: bool, arch: str, numCU: int, perf_config: str = ''):
        if dtype not in {"f16", "f32", "bf16", "i8", "fp8"}:
            raise ValueError(f"Invalid datatype: {dtype}")
        self.MLIR_N_REPEATS = 5
        self.dataType = dtype
        self.outDataType = outDataType
        self.g = g
        self.m = m
        self.k = k
        self.n = n
        self.transA = transA
        self.transB = transB
        self.perfConfig = perf_config

        self.arch = arch
        self.chip = GFX_CHIP_RE.search(arch).group(0)
        self.numCU = numCU

class AttentionConfiguration(PerfConfiguration):
    TABLE_COLUMNS = reportUtils.ATTN_TEST_PARAMETERS + ['TFlops']
    def __init__(self, dtype: str, g: int, seq_len_q: int, seq_len_k: int, head_dim_qk: int, head_dim_v: int, with_attn_scale: int,
                 transQ: bool, transK: bool, transV: bool, transO: bool, arch: str, numCU: int, perf_config: str = ''):
        if dtype not in {"f16", "f32"}:
            raise ValueError(f"Invalid datatype: {dtype}")
        self.dataType = dtype
        self.g = g
        self.seq_len_q = seq_len_q
        self.seq_len_k = seq_len_k
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.with_attn_scale = with_attn_scale
        self.transQ = transQ
        self.transK = transK
        self.transV = transV
        self.transO = transO

        self.arch = arch
        self.chip = GFX_CHIP_RE.search(arch).group(0)
        self.numCU = numCU
        self.MLIR_N_REPEATS = 5
        self.perfConfig = perf_config

    def computeTFlops(self, ns, only_matmul_flops=True):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        first_matmul_flops = 2.0 * self.g * self.seq_len_q * self.head_dim_qk * self.seq_len_k
        # max, sub, exp, sum, div
        softmax_flops = 5.0 * self.g * self.seq_len_q * self.seq_len_k
        second_matmul_flops = 2.0 * self.g * self.seq_len_q * self.seq_len_k * self.head_dim_v
        total_flops = first_matmul_flops + second_matmul_flops
        # Weirdly, triton does not account for flops coming from
        # non matmul operations as per FA2 paper. Hence not including
        # by default
        # References:
        # 1) https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
        # 2) Flash-Attention 2 : https://arxiv.org/abs/2307.08691
        if not only_matmul_flops:
            total_flops += softmax_flops
            if self.with_attn_scale:
                total_flops += self.seq_len_q * self.seq_len_k
        return total_flops / (float(ns) * 1e-9) / 1e12

    def tableEntry(self, nanoSeconds):
        result = {}
        values = [
            self.dataType,
            self.chip,
            self.numCU,
            self.transQ,
            self.transK,
            self.transV,
            self.transO,
            self.with_attn_scale,
            self.g,
            self.seq_len_q,
            self.seq_len_k,
            self.head_dim_qk,
            self.head_dim_v,
            self.perfConfig,
            self.computeTFlops(nanoSeconds)
        ]
        assert(len(self.TABLE_COLUMNS) == len(values))
        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def __repr__(self):
        return f"""AttentionConfiguration(dtype={self.dataType!r}, g={self.g!r}, seq_len_q={self.seq_len_q!r}, seq_len_k={self.seq_len_k!r}, head_dim_qk={self.head_dim_qk!r}, head_dim_v={self.head_dim_v!r}, with_attn_scale={self.with_attn_scale!r},
                transQ={self.transQ!r}, transK={self.transK!r}, transV={self.transV!r}, transO={self.transO!r}, arch={self.arch!r}, numCU={self.numCU}, perf_config={self.perfConfig})"""

    def setPerfConfig(self, perf_config):
        self.perfConfig = perf_config

    def generateMlirDriverCommandLine(self, rocmlir_gen_flags):
        result = ' '.join(['-operation', 'attention',
                           '-t', self.dataType,
                           '--arch', self.arch,
                           '--num_cu', str(self.numCU),
                           '-g', str(self.g),
                           '-seq_len_q', str(self.seq_len_q),
                           '-seq_len_k', str(self.seq_len_k),
                           '-head_dim_qk', str(self.head_dim_qk),
                           '-head_dim_v', str(self.head_dim_v),
                           f"-with-attn-scale={self.with_attn_scale}",
                           f"-transQ={self.transQ}",
                           f"-transK={self.transK}",
                           f"-transQ={self.transV}",
                           f"-transK={self.transO}",
                           '--kernel-repeats', str(self.MLIR_N_REPEATS),
                           f"--perf_config={self.perfConfig}"])
        result += ' '
        if rocmlir_gen_flags != '':
            result += ' '.join(rocmlir_gen_flags.split())
        return result

    @classmethod
    def fromCommandLine(cls, argv, arch, numCU):
        # optional defaults
        perf_config = ''
        transQ = False
        transK = False
        transV = False
        transO = False
        with_attn_scale = False
        # Please keep this in sync with mlir::rock::getTuningProblemStr()
        for i in range(0, len(argv), 2):
            opt = argv[i]
            val = argv[i + 1]
            if opt.endswith("-t"):
                dtype = val
            elif opt.endswith("-g"):
                g = int(val)
            elif opt.endswith("-seq_len_q"):
                seq_len_q = int(val)
            elif opt.endswith("-seq_len_k"):
                seq_len_k = int(val)
            elif opt.endswith("-head_dim_qk"):
                head_dim_qk = int(val)
            elif opt.endswith("-head_dim_v"):
                head_dim_v = int(val)
            elif opt.endswith("-with-attn-scale"):
                with_attn_scale = (val.lower() in ["1", "true"])
            elif opt.endswith("-transQ"):
                transQ = (val.lower() in ["1", "true"])
            elif opt.endswith("-transK"):
                transK = (val.lower() in ["1", "true"])
            elif opt.endswith("-transV"):
                transV = (val.lower() in ["1", "true"])
            elif opt.endswith("-transO"):
                transO = (val.lower() in ["1", "true"])
            elif opt.endswith("-perf_config"):
                perf_config = val
            else:
                raise ValueError(f"Unknown Attention config argument {opt} -> {val}")
        for v in [dtype, g, seq_len_q, seq_len_k, head_dim_qk, head_dim_v, with_attn_scale, transQ, transK, transV, transO]:
            if v is None:
                raise ValueError("Incomplete Attention configuration")

        return cls(dtype, g, seq_len_q, seq_len_k, head_dim_qk, head_dim_v, with_attn_scale, transQ, transK, transV, transO, arch, numCU, perf_config)

    def toCommandLine(self):
        return (f"-t {self.dataType} "
                + f"-transQ {str(self.transQ).lower()} -transK {str(self.transK).lower()} "
                + f"-transV {str(self.transV).lower()} -transO {str(self.transO).lower()} "
                + f"-g {self.g} "
                + f"-seq_len_q {str(self.seq_len_q)} -seq_len_k {str(self.seq_len_k)} -head_dim_qk {str(self.head_dim_qk)} -head_dim_v {str(self.head_dim_v)} "
                + f"-with-attn-scale {str(self.with_attn_scale).lower()}")


class RocBLASGemmConfig(GemmConfiguration):
    EXTERNAL_NAME = "rocBLAS"

    @classmethod
    def benchmarkExternal(cls, commandLine, paths: Paths, arch, numCU, envs={}):
        config = cls.fromCommandLine(commandLine, arch, numCU)
        if not paths.mlir_paths.rocblas_benchmark_driver_path:
            raise ValueError("rocblas-benchmark-driver not built")
        benchmarkArgs = config.generateMlirDriverCommandLine("")
        # remove the result file generated by rocprof in previous benchmarking
        os.system("rm -f "+BENCHMARKING_RESULT_FILE_NAME)
        print(f"Running rocBLAS benchmark {config!r}")
        profilerCommand = [paths.mlir_paths.rocblas_benchmark_driver_path] + \
            benchmarkArgs.split()
        p = subprocess.Popen(profilerCommand, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # get output.
        try:
            outs, errs = p.communicate(timeout=60)
            if len(errs) > 0:
                print("Test printed errors: ", errs.decode('utf-8'))
                print("Failing command line: ", profilerCommand)
                if p.returncode != 0:
                    raise OSError(errs.decode('utf-8'))
        except subprocess.TimeoutExpired:
            print("Test timed out: ", profilerCommand)
            p.kill()
            outs, errs = p.communicate()
        milliSeconds = getMilliseconds(outs)
        nanoSeconds = milliSeconds*1e6
        return config.tableEntry(nanoSeconds)

class CKGemmConfig(GemmConfiguration):
    EXTERNAL_NAME = "CK"
    @classmethod
    def benchmarkExternal(cls, commandLine, paths: Paths, arch, numCU, envs={}):
        config = cls.fromCommandLine(commandLine, arch, numCU)
        if not paths.mlir_paths.ck_gemm_benchmark_driver_path:
            raise ValueError("ck-gemm-benchmark-driver not built")
        benchmarkArgs = config.generateMlirDriverCommandLine("")

        print(f"Running CK benchmark {config!r}")

        if arch=="gfx1030" and config.g > 1:
            return config.tableEntry(float('NaN'))

        profilerCommand = [paths.mlir_paths.ck_gemm_benchmark_driver_path] + \
            benchmarkArgs.split()
        p = subprocess.Popen(profilerCommand, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # get output.
        try:
            outs, errs = p.communicate(timeout=60)
            if len(errs) > 0:
                print("Test printed errors: ", errs.decode('utf-8'))
                print("Failing command line: ", profilerCommand)
                if p.returncode != 0:
                    raise OSError(errs.decode('utf-8'))
        except subprocess.TimeoutExpired:
            print("Test timed out: ", profilerCommand)
            p.kill()
            outs, errs = p.communicate()
        milliSeconds = getMilliseconds(outs)
        nanoSeconds = milliSeconds*1e6
        return config.tableEntry(nanoSeconds)

def runConfigWithMLIR(config: PerfConfiguration, paths: Paths, arch, rocmlir_gen_flags, debug=True):
    # remove the result file generated by rocprof in previous benchmarking
    os.system("rm -f "+BENCHMARKING_RESULT_FILE_NAME)
    commandLineOptions = config.generateMlirDriverCommandLine(rocmlir_gen_flags)
    if debug:
        print("Running MLIR Benchmark: ", repr(config))
    rocmlirGenCommand = paths.mlir_paths.rocmlir_gen_path + ' -ph ' + commandLineOptions
    rocmlirDriverCommand = [paths.mlir_paths.rocmlir_driver_path, '-c']
    mlir_cpu_runner_args = [f'--shared-libs={paths.mlir_paths.libmlir_rocm_runtime_path},{paths.mlir_paths.libconv_validation_wrappers_path},{paths.mlir_paths.libmlir_runtime_utils_path},{paths.mlir_paths.libmlir_c_runner_utils_path}', '--entry-point-result=void']
    profilerCommand = [ROCPROF] +  getMetricArgsForRocprof(arch) + ['--stats', '-o', BENCHMARKING_METRICS_FILE_NAME, paths.mlir_paths.cpu_runner_path] + mlir_cpu_runner_args

    # invoke rocmlir-gen.
    p1 = subprocess.Popen(rocmlirGenCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # pipe to rocmlir-driver
    p2 = subprocess.Popen(rocmlirDriverCommand, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p1.stdout.close() # Allow p1 to receive a SIGPIPE if p2 exits.
    # pipe to rocprof + mlir-cpu-runner.
    p3 = subprocess.Popen(profilerCommand, stdin=p2.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2.stdout.close() # Allow p2 to receive a SIGPIPE if p3 exits.
    # get output.
    try:
        _, errs = p3.communicate(timeout=60)
        if len(errs) > 0:
            print("Test printed errors: ", errs.decode('utf-8'))
            print("Failing command line: ", rocmlirGenCommand)
            if p1.returncode != 0:
                raise OSError(errs.decode('utf-8'))
    except subprocess.TimeoutExpired:
        print("Test timed out: ", rocmlirGenCommand)
        p3.kill()
        _, errs = p3.communicate()

# Benchmarking function.
def benchmarkMLIR(commandLine, confClass, paths: Paths, arch, numCU, tuningDb: MaybeTuningDb, rocmlir_gen_flags):
    config = confClass.fromCommandLine(commandLine, arch, numCU)
    configStr = config.toCommandLine()
    if tuningDb:
        if (arch, configStr) in tuningDb:
            config.setPerfConfig(tuningDb[arch, configStr])
        else: # Tuning DB present but doesn't contain config, return N/A
            return config.tableEntry(np.nan)

    runConfigWithMLIR(config, paths, arch, rocmlir_gen_flags)
    # get nanoseconds from rocprof output.
    nanoSeconds = getNanoSeconds(BENCHMARKING_RESULT_FILE_NAME)
    return config.tableEntry(nanoSeconds)

#Generate MLIR vs. MIOpen or rocBLAS performance results
def generatePerformanceResults(configs, confClass, paths: Paths, arch, numCU, tuningDb: MaybeTuningDb, quickTuningDb: MaybeTuningDb, rocmlir_gen_flags):
    # Never pass tuning DB to this run
    mlir_df = pd.DataFrame(benchmarkMLIR(testVector.split(sep=' '), confClass, paths, arch, numCU, None, rocmlir_gen_flags)
        for testVector in configs)
    tuned_df = None
    if tuningDb:
        tuned_df = pd.DataFrame(benchmarkMLIR(testVector.split(sep=' '), confClass, paths, arch, numCU, tuningDb, rocmlir_gen_flags)
            for testVector in configs)
    quick_tuned_df = None
    if quickTuningDb:
        quick_tuned_df = pd.DataFrame(benchmarkMLIR(testVector.split(sep=' '), confClass, paths, arch, numCU, quickTuningDb, rocmlir_gen_flags)
            for testVector in configs)

    external_df = pd.DataFrame(confClass.benchmarkExternal(testVector.split(sep=' '), paths, arch, numCU)
        for testVector in configs)

    externalName = confClass.EXTERNAL_NAME
    df = mlir_df.merge(external_df, on=confClass.TABLE_COLUMNS[:-2],
                           suffixes=('', f" ({externalName})"))
    externalTFlopsCol = f"{externalName} TFlops (no MLIR Kernels)"
    df.rename(columns={'TFlops': 'MLIR TFlops', f"TFlops ({externalName})": externalTFlopsCol}, inplace=True)
    if tuned_df is not None:
        # No need for suffixes, the conflicting columns have been renamed
        # Also note that we're ignoring PerfConfig with the -3
        df = df.merge(tuned_df, on=confClass.TABLE_COLUMNS[:-3],
            suffixes=('', ' (tuned)'))
        df.drop(columns=['PerfConfig'], inplace=True)
        df.rename(columns={'TFlops': 'Tuned MLIR TFlops', 'PerfConfig (tuned)' : 'PerfConfig' }, inplace=True)
    if quick_tuned_df is not None:
        # No need for suffixes, the conflicting columns have been renamed
        # Also note that we're ignoring PerfConfig with the -3
        df = df.merge(quick_tuned_df, on=confClass.TABLE_COLUMNS[:-3],
            suffixes=('', ' (quick tuned)'))
        df.rename(columns={'TFlops': 'Quick Tuned MLIR TFlops'}, inplace=True)

    df[f"MLIR/{externalName}"] = df['MLIR TFlops'] / df[externalTFlopsCol]
    if tuned_df is not None:
        df[f"Tuned/{externalName}"] = df['Tuned MLIR TFlops'] / df[externalTFlopsCol]
        df["Tuned/Untuned"] = df['Tuned MLIR TFlops'] / df['MLIR TFlops']
    if quick_tuned_df is not None:
        df[f"Quick Tuned/{externalName}"] = df['Quick Tuned MLIR TFlops'] / df[externalTFlopsCol]
        df["Quick Tuned/Untuned"] = df['Quick Tuned MLIR TFlops'] / df['MLIR TFlops']
    if tuned_df is not None and quick_tuned_df is not None:
        df["Quick Tuned/Tuned"] = df['Quick Tuned MLIR TFlops'] / df['Tuned MLIR TFlops']
    chip = GFX_CHIP_RE.search(arch).group(0)
    if confClass is RocBLASGemmConfig:
        reportFile = reportUtils.PERF_REPORT_FILE['rocBLAS']
    elif confClass is CKGemmConfig:
        reportFile = reportUtils.PERF_REPORT_FILE['CK']
    else:
        reportFile = reportUtils.PERF_REPORT_FILE['MIOpen']
    df.fillna('NaN', inplace=True)
    df.to_csv(chip + '_' + reportFile, index=False)

def getSolverName(testVector, arch, numCU):
    config = ConvConfiguration.fromCommandLine(testVector.split(sep=' '), arch, numCU)
    if config.direction == 'fwd':
        solverName = 'ConvMlirIgemmFwd'
    elif config.direction == 'bwd':
        solverName = 'ConvMlirIgemmBwd'
    else:
        solverName = 'ConvMlirIgemmWrW'
    if config.chip in ['gfx908', 'gfx90a', 'gfx942']:
        solverName+='Xdlops'
    return solverName

RUNNABLE_TEST_RE = re.compile(r"//\s*RUN\s*:(.*)")
ROCMLIRGEN_RE = re.compile(r"rocmlir-gen.*?-fut\s*(\w+)")
def findRunCommand(filename):
    rocmlirCommand = None
    futName = None
    with open(filename, 'r') as f:
        for line in f:
            hasRun = RUNNABLE_TEST_RE.search(line)
            hasRocmlirGen = ROCMLIRGEN_RE.search(line)
            if hasRun:
                command = hasRun.group(1)
                if not rocmlirCommand:
                    parts = command.split('|')  # Split the command using the "|" separator
                    if 'rocmlir-driver' in parts[0] or 'rocmlir-opt' in parts[0]:
                        rocmlirCommand = parts[0].strip() # Find rocmlir-driver command
                    elif 'rocmlir-driver' in parts[1] or 'rocmlir-opt' in parts[1]:
                        rocmlirCommand = parts[1].strip()

                if hasRocmlirGen and not futName:
                    futName = hasRocmlirGen.group(1)

                if 'runner' in line: # Stop processing lines after finding a runner
                    return rocmlirCommand, futName

    # Not found a "RUN" command or a runner
    print("WARNING: cannot find valid RUN command in ", filename)
    return None, None

# Extract testVector and test function name from the test file
def getFusionTestInfo(filename, paths: Paths):
    testEntry = {}
    rocmlirCommand, futName = findRunCommand(filename)
    if not rocmlirCommand:
        return testEntry
    # rocmlir-gen -fut test -arch gfx90a --clone-harness
    rocmlirgenCommand = [paths.mlir_paths.rocmlir_gen_path, '-fut', futName, '-arch', getChip(), '--clone-harness', filename]
    p0 = subprocess.Popen(rocmlirgenCommand, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if "-migraphx-to-tosa" in rocmlirCommand:
        rocmlirOptCommand = [paths.mlir_paths.rocmlir_opt_path, '-migraphx-to-tosa']
        rocmlirDriverCommand = [paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'highlevel', '-targets', getChip()]
        # rocmlir-opt -migraphx-to-tosa ../mlir/test/fusion/resnet50-e2e/mixr-resnet-fusion-case-1.mlir
        p1 = subprocess.Popen(rocmlirOptCommand, stdin=p0.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # pipe to rocmlir-driver -host-pipeline partition,highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlirDriverCommand, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p1.stdout.close()
    elif "migraphx" in rocmlirCommand:
        rocmlirMigraphxCommand = [paths.mlir_paths.rocmlir_driver_path, '-kernel-pipeline', 'migraphx']
        rocmlirDriverCommand = [paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'migraphx,highlevel', '-targets', getChip()]
        # rocmlir-driver -kernel-pipeline migraphx ../mlir/test/fusion/resnet50-e2e/mixr-resnet-fusion-case-1.mlir
        p1 = subprocess.Popen(rocmlirMigraphxCommand, stdin=p0.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # pipe to rocmlir-driver -host-pipeline partition,highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlirDriverCommand, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p1.stdout.close()
    else:
        rocmlirDriverCommand = [paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'highlevel', '-targets', getChip()]
        # rocmlir-driver -host-pipeline partition,highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlirDriverCommand, stdin=p0.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # pipe to rocmlir_gen --emit-tuning-key
    tuningKey = subprocess.Popen([paths.mlir_paths.rocmlir_gen_path, '--emit-tuning-key', '-'], stdin=p2.stdout,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2.stdout.close()
    output, _ = tuningKey.communicate()
    result = output.decode('utf-8').strip().split('\t')
    testEntry = {'filename' : filename, 'testVector' : result[2], 'futName' : futName}
    return testEntry

def runFusionKernel(filename, rocmlirGenArgs, paths: Paths):
    os.system("rm -f "+BENCHMARKING_RESULT_FILE_NAME)

    rocmlirCommand, futName = findRunCommand(filename)

    # rocmlir-gen -fut test -arch gfx90a --clone-harness
    rocmlirgenCommand = [paths.mlir_paths.rocmlir_gen_path, '-fut', futName, '-arch', getChip(), '--clone-harness', filename]
    p0 = subprocess.Popen(rocmlirgenCommand, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if "-migraphx-to-tosa" in rocmlirCommand:
        rocmlirOptCommand = [paths.mlir_paths.rocmlir_opt_path, '-migraphx-to-tosa', filename]
        rocmlirDriverCommand = [paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'highlevel', '-targets', getChip()]
        # rocmlir-opt -migraphx-to-tosa ../mlir/test/fusion/resnet50-e2e/mixr-resnet-fusion-case-1.mlir
        p1 = subprocess.Popen(rocmlirOptCommand, stdin=p0.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # pipe to rocmlir-driver -host-pipeline partition,highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlirDriverCommand, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p1.stdout.close()
    elif "migraphx" in rocmlirCommand:
        rocmlirMigraphxCommand = [paths.mlir_paths.rocmlir_driver_path, '-kernel-pipeline', 'migraphx']
        rocmlirDriverCommand = [paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'migraphx,highlevel', '-targets', getChip()]
        # rocmlir-driver -kernel-pipeline migraphx ../mlir/test/fusion/resnet50-e2e/mixr-resnet-fusion-case-1.mlir
        p1 = subprocess.Popen(rocmlirMigraphxCommand, stdin=p0.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # pipe to rocmlir-driver -host-pipeline partition,highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlirDriverCommand, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p1.stdout.close()
    else:
        rocmlirDriverCommand = [paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'highlevel', '-targets', getChip()]
        # rocmlir-driver -host-pipeline partition,highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlirDriverCommand, stdin=p0.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    rocmlirGenCommand = [paths.mlir_paths.rocmlir_gen_path] + rocmlirGenArgs
    kernelPipelineCommand = [paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'mhal,runner', '-kernel-pipeline','full']
    mlir_cpu_runner_args = [f'--shared-libs={paths.mlir_paths.libmlir_rocm_runtime_path},{paths.mlir_paths.libconv_validation_wrappers_path},{paths.mlir_paths.libmlir_runtime_utils_path},{paths.mlir_paths.libmlir_c_runner_utils_path}', '--entry-point-result=void']
    profilerCommand = [ROCPROF, '--stats', paths.mlir_paths.cpu_runner_path] + mlir_cpu_runner_args
    # pipe to rocmlir-gen -ph --perf_config
    p3 = subprocess.Popen(rocmlirGenCommand, stdin=p2.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2.stdout.close()
    # pipe to rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full
    p4 = subprocess.Popen(kernelPipelineCommand, stdin=p3.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p3.stdout.close()
    profiling = subprocess.Popen(profilerCommand, stdin=p4.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p4.stdout.close()

    # Get output.
    try:
        _, errs = profiling.communicate(timeout=60)
        if len(errs) > 0:
            print("Test printed errors: ", errs.decode('utf-8'))
            print("Failing: ", filename)
    except subprocess.TimeoutExpired:
        print("Test timed out: ", filename)
        profiling.kill()
        _, errs = profiling.communicate()


# Generate fusion vs. gemm/conv performance results
def benchmarkFusionKernels(test_dir, paths: Paths, arch, numCU, tuningDb: MaybeTuningDb):
    allTests = [] #filename, testVector, futName
    perfResults = {} #associate testVector to config and performances
    chip = GFX_CHIP_RE.search(arch).group(0)

    # Prepare test cases
    for filename in glob.glob(test_dir+'/*.mlir'):
        testEntry = getFusionTestInfo(filename, paths)
        if testEntry:
            allTests.append(testEntry)

    # Profile each test case
    for test in allTests:
        filename = test['filename']
        testVector = test['testVector']
        futName = test['futName']

        print("Profiling:", filename)
        # Sanity check
        if not testVector:
            print("\tCannot find a test vector")
            continue
        if not futName:
            print("\tCannot find rocmlir-gen with -fut")
            continue

        commandLine = testVector.split(sep=' ')
        if commandLine[0].startswith('conv'):
            op = 'conv'
            config = ConvConfiguration.fromCommandLine(commandLine, arch, numCU)
        else:
            op = 'gemm'
            config = GemmConfiguration.fromCommandLine(commandLine, arch, numCU)

        # Find the best perf_config
        bestPerf =""
        if tuningDb:
            if (arch, testVector) in tuningDb:
                bestPerf = tuningDb[arch, testVector]
                config.setPerfConfig(bestPerf)
            else: # Tuning DB present but doesn't contain config, add a NaN entry
                if not testVector in perfResults:
                    oneEntry = config.tableEntry(np.nan)
                    oneEntry['FileName'] = filename
                    oneEntry['MLIR TFlops'] = np.nan
                    oneEntry['Fusion/MLIR'] = np.nan
                    perfResults[testVector] = oneEntry
                continue

        # Run fusion test
        rocmlirGenArgs = ['-ph', '-fut='+futName+'_wrapper', '--perf_config='+bestPerf, '-']
        runFusionKernel(filename, rocmlirGenArgs, paths)
        # Get nanoseconds of fusion test
        nanoSeconds = getNanoSeconds(BENCHMARKING_RESULT_FILE_NAME)
        oneEntry = config.tableEntry(nanoSeconds)
        # Keep the best performance
        if testVector in perfResults and oneEntry['TFlops'] <= perfResults[testVector]['TFlops']:
            continue

        # Run gemm or conv op with the same configuration
        runConfigWithMLIR(config, paths, arch, '')
        # Get nanoseconds of gemm/conv
        nanoSeconds = getNanoSeconds(BENCHMARKING_RESULT_FILE_NAME)
        oneEntry['MLIR TFlops'] = config.computeTFlops(nanoSeconds)
        oneEntry['Fusion/MLIR'] = oneEntry['TFlops']/oneEntry['MLIR TFlops']
        oneEntry['FileName'] = filename
        perfResults[testVector] = oneEntry

    df = pd.DataFrame(perfResults.values())
    df.fillna('NaN', inplace=True)
    df.rename(columns={'TFlops': 'Fusion TFlops'}, inplace=True)
    df.to_csv(chip + '_' + op + '_' + reportUtils.PERF_REPORT_FUSION_FILE, index=False)

#Tune MIOpen with MLIR kernels
def tuneMLIRKernels(configs, arch, numCU):
    solver_names = {
        testVector: getSolverName(testVector, arch, numCU)
        for testVector in configs
    }

    envs = os.environ.copy()
    envs['MIOPEN_FIND_ENFORCE'] = '4'
    envs['MIOPEN_DRIVER_USE_GPU_REFERENCE'] = '1'
    for testVector in configs:
        envs['MIOPEN_DEBUG_FIND_ONLY_SOLVER'] = solver_names[testVector]
        commandLine = testVector.split(sep=' ')
        config = ConvConfiguration.fromCommandLine(commandLine, arch, numCU)
        if config.inputLayout == 'nchw':
            MIOpenDriverCommand = [MIOPENDRIVER, *commandLine, '-V', '0']
            print(' '.join(MIOpenDriverCommand))
            p1 = subprocess.Popen(MIOpenDriverCommand,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  env=envs)
            # get output.
            try:
                _, errs = p1.communicate(timeout=300)
                if len(errs) > 0 and p1.returncode != 0:
                    raise OSError(errs.decode('utf-8'))
            except subprocess.TimeoutExpired:
                p1.kill()
                print("MIOpen tuning timed out")
                _, errs = p1.communicate()

def is_xdlops_present() -> bool:
    """This function checks whether a GPU with xdlops support is present"""
    xdlop_supported_gpus = ['gfx908', 'gfx90a', 'gfx942']
    xdlop_supported_gpus_str = xdlop_supported_gpus[0]
    for gpu in xdlop_supported_gpus[1:]:
        xdlop_supported_gpus_str += '|' + gpu
    r = subprocess.run(f"/opt/rocm/bin/rocm_agent_enumerator -t GPU | grep -q -E '{xdlop_supported_gpus_str}'",
                       check=True, shell=True)
    if r.returncode == 0:
        return True
    return False

def getArch():
    p = subprocess.run(["/opt/rocm/bin/rocm_agent_enumerator", "-name"], check=True,
                       stdout=subprocess.PIPE)
    agents = set(x.decode("utf-8") for x in p.stdout.split())
    if not agents:
        # TODO: Remove this workaround for a bug in rocm_agent_enumerator -name
        # Once https://github.com/RadeonOpenCompute/rocminfo/pull/59 lands
        q = subprocess.run(["/opt/rocm/bin/rocm_agent_enumerator"],
                              check=True, stdout=subprocess.PIPE)
        agents = set(x.decode("utf-8") for x in q.stdout.split() if x != b"gfx000")
    return agents

def parseDataTypes(data_types):
    if not data_types:
        return DATA_TYPES_GEMM, OUTPUT_DATA_TYPES_MAP
    datatypes = []
    outMap = {}
    for dpair in data_types:
        dt = dpair.split('_')
        datatypes.append(dt[0])
        outMap[dt[0]] = dt[0]
        if len(dt) == 2:
            outMap[dt[0]] = dt[1]
        elif dt[0] == 'i8':
            outMap[dt[0]] = 'i32'
        elif dt[0] == 'fp8':
            outMap[dt[0]] = 'f32'
    return datatypes, outMap

def getChip():
    archNames = getArch()
    arch = ','.join(archNames)
    chip = GFX_CHIP_RE.search(arch).group(0)
    return chip

def getNumCU(chip):
    rocminfo = subprocess.check_output("/opt/rocm/bin/rocminfo")
    rocminfoLines = rocminfo.decode("utf-8").split("\n")
    foundChip = False
    for line in rocminfoLines:
        if not foundChip:
            m = INFO_ARCH_NAME.search(line)
            if m and chip in m.group(1).strip():
                foundChip = True
        if foundChip:
            computeUnit = INFO_ARCH_CU.search(line)
            if computeUnit:
                return int(computeUnit.group(1))
    assert False, f"Cannot find number of CUs for {chip}"


def foundExternalTool(paths: Paths, opType: Operation, gemmLibrary : Optional[GEMMLibrary] = None):
    if opType == Operation.GEMM:
        if not paths.mlir_paths:
            return False
        if gemmLibrary == GEMMLibrary.CK and not paths.mlir_paths.ck_gemm_benchmark_driver_path:
            return False
        if gemmLibrary == GEMMLibrary.ROCBLAS and not paths.mlir_paths.rocblas_benchmark_driver_path:
            return False
    return True

# Main function.
def main(args=None):
    """
    usage examples:

    python3 perfRunner.py
    python3 perfRunner.py --batch_all -o=output_file.csv
    python3 perfRunner.py --batch_all -o=output_file.csv -t=tuning_db.tsv
    python3 perfRunner.py -b
    # Uses results from tuning db when running MLIR benchmarks
    python3 perfRunner.py -b -t=tuning_db.tsv
    python3 perfRunner.py --batch_external
    python3 perfRunner.py --operation gemm --external # rocblas tests
    python3 perfRunner.py -- conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
    python3 perfRunner.py --external -- conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
    python3 perfRunner.py --operation gemm [--external] -- -t f32 -transA true -transB true -g 1 -m 1024 -k 769 -n 512
    """
    if args is None:
        args = sys.argv[1:]

    archNames = getArch()
    arch = ','.join(archNames)
    chip = GFX_CHIP_RE.search(arch).group(0)
    numCU = getNumCU(chip)

    root_dir = str(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
    default_conv_configs = root_dir + '/mlir/utils/jenkins/performance/conv-configs'

    parser = argparse.ArgumentParser(
        prog="rocMLIR performance test runner",
        description="A test runner script for MIOpen and MLIR-based kernel generator",
        allow_abbrev=False,
    )

    parser.add_argument("--op", "--operation", choices=['conv', 'gemm', 'fusion', 'attention'],
        default='conv',
        help="Operation to benchmark")

    mutex_arg_group = parser.add_mutually_exclusive_group()
    mutex_arg_group.add_argument(
        "--tuning",
        action="store_true",
        help="Only tune the MLIR kernels"
    )
    mutex_arg_group.add_argument(
        "-b", "--batch_mlir",
        action="store_true",
        help="CSV batch benchmarking mode with MLIR"
    )
    mutex_arg_group.add_argument(
        "--batch_external",
        action="store_true",
        help="CSV batch benchmarking mode with external reference"
    )
    mutex_arg_group.add_argument(
        "--batch_all",
        action="store_true",
        help="CSV batch benchmarking with MLIR and external reference (defalut on no args)"
    )
    mutex_arg_group.add_argument(
        "--external",
        action="store_true",
        help="benchmark a single config externally"
    )

    parser.add_argument(
        "-c", "--configs_file",
        type=str,
        default=default_conv_configs,
        help="File of configurations to test"
    )

    parser.add_argument(
        "-o",
        type=str,
        default=chip + '_' + date.today().strftime("perf.%m%d%y"),
        help="Output file name",
        dest="fileName"
    )
    parser.add_argument(
        "-t", "--tuning_db",
        type=str,
        default=argparse.SUPPRESS,
        help="Tuning database filename"
    )
    parser.add_argument(
        "-qt", "--quick_tuning_db",
        type=str,
        default=argparse.SUPPRESS,
        help="Quick tuning database filename"
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        default="../mlir/test/fusion/resnet50-e2e",
        help="The directory of tests"
    )
    parser.add_argument(
        "--mlir-build-dir",
        type=str,
        default=find_mlir_build_dir(),
        help="The build directory of MLIR based kernel generator",
    )
    parser.add_argument(
        "config",
        type=str,
        nargs='*',
        help="The specific config to test, if you want to test one"
    )

    parser.add_argument(
        "--rocmlir_gen_flags",
        type=str,
        default=argparse.SUPPRESS,
        help="rocmlir-gen flags to toggle each feature"
    )

    parser.add_argument(
        "--external-gemm-library",
        type=str,
        default="rocBLAS",
        help="(rocBLAS | CK) external library to run GEMM routines"
    )

    parser.add_argument(
        '--data-type',
         nargs='+',
         choices=["f32", "f16", "i8", "i8_i32", "i8_i8", "fp8", "fp8_fp8", "fp8_f32"],
         default=["f32", "f16", "i8"],
         help='Force a set of datatypes'
    )

    parsed_args = parser.parse_args(args)

    rocmlir_gen_flags = ''
    if 'rocmlir_gen_flags' in parsed_args:
        rocmlir_gen_flags = parsed_args.rocmlir_gen_flags

    tuningDb = None
    quickTuningDb = None
    if 'tuning_db' in parsed_args:
        tuningDb = read_tuning_db(parsed_args.tuning_db)

    if 'quick_tuning_db' in parsed_args:
        quickTuningDb = read_tuning_db(parsed_args.quick_tuning_db)

    # Impose default behavior when no args have been passed
    if len(args) == 0:
        parsed_args.batch_all = True

    confClass = PerfConfiguration
    opType = Operation.fromName(parsed_args.op)
    if opType == Operation.CONV:
        confClass = ConvConfiguration
        externalLib = None
    elif opType == Operation.GEMM:
        externalLib = GEMMLibrary.fromName(parsed_args.external_gemm_library)
        if externalLib == GEMMLibrary.ROCBLAS:
            confClass = RocBLASGemmConfig
        elif externalLib == GEMMLibrary.CK:
            confClass = CKGemmConfig
    elif opType == Operation.ATTENTION:
        confClass = AttentionConfiguration
        externalLib = None

    configs_path = None if parsed_args.config else parsed_args.configs_file
    paths = create_paths(configs_path, parsed_args.mlir_build_dir)
    configs = None
    if opType == Operation.CONV:
        configs = getConvConfigurations(paths.configuration_file_path)
    elif opType == Operation.GEMM:
        datatypes, outputTypeMap = parseDataTypes(parsed_args.data_type)
        configs = getGemmConfigurations(paths.configuration_file_path, datatypes, outputTypeMap)
    elif opType == Operation.ATTENTION:
        configs = getAttentionConfigurations(paths.configuration_file_path)

    if parsed_args.external or parsed_args.batch_external or parsed_args.batch_all:
        if not foundExternalTool(paths, opType, externalLib):
            raise RuntimeError("External benchmark reference (MIOpen or rocBLAS driver) needed but not found")

    if parsed_args.batch_mlir or parsed_args.batch_all:
        if not paths.mlir_paths:
            raise RuntimeError("MLIR build dir was not provided/found")


    #If no arguments are passed, then benchmark with MLIR and MIOpen
    if parsed_args.batch_all:
        # batch benchmark with MLIR and MIOpen.
        generatePerformanceResults(configs, confClass, paths, arch, numCU, tuningDb, quickTuningDb, rocmlir_gen_flags)
    elif parsed_args.tuning:
        tuneMLIRKernels(configs, arch, numCU)
    elif opType == Operation.FUSION:
        if not parsed_args.mlir_build_dir:
            raise RuntimeError("MLIR build dir was not provided/found")
        else:
            benchmarkFusionKernels(parsed_args.test_dir, paths, arch, numCU, tuningDb)
    else:
        if parsed_args.batch_mlir:
            df = pd.DataFrame(benchmarkMLIR(testVector.split(sep=' '), confClass, paths, arch, numCU, tuningDb, rocmlir_gen_flags) for testVector in configs)
        elif parsed_args.batch_external:
            df = pd.DataFrame(confClass.benchmarkExternal(testVector.split(sep=' '), paths, arch, numCU) for testVector in configs)
        elif parsed_args.external:
            df = pd.DataFrame([confClass.benchmarkExternal(parsed_args.config, paths, arch, numCU)])
        else:
            # Will only reach here with more than 1 unspecified arguments
            # These are arguments are directly passed through to benchmarkMLIR
            if not parsed_args.mlir_build_dir:
                raise RuntimeError("MLIR build dir was not provided/found")
            else:
                if parsed_args.config:
                    df = pd.DataFrame([benchmarkMLIR(parsed_args.config, confClass, paths, arch, numCU, tuningDb, rocmlir_gen_flags)])
                else:
                    df = pd.DataFrame([benchmarkMLIR(config.split(), confClass, paths, arch, numCU, tuningDb, rocmlir_gen_flags) for config in configs])
        df.to_csv(parsed_args.fileName)
        with pd.option_context('display.precision', reportUtils.ROUND_DIGITS):
            print(df) # for interactive consumption

if __name__ == '__main__':
    sys.exit(main())
