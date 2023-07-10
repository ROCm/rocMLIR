#!/usr/bin/env python3

from typing import NamedTuple
import reportUtils
from perfCommonUtils import Operation, GEMMLibrary

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

import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# global variables.
ROCPROF = '/opt/rocm/bin/rocprof'
BENCHMARKING_RESULT_FILE_NAME = 'results.stats.csv'
DIRECTIONS = ['-F 1', '-F 2', '-F 4']
DATA_TYPES = ['conv', 'convfp16', 'convint8']
LAYOUTS = ['NHWC', 'NCHW']

DATA_TYPES_GEMM = ['f32', 'f16', 'i8']
OUTPUT_DATA_TYPES_MAP = {'f32':'f32', 'f16':'f16', 'i8':'i32'}

# Compiled regexp object used for extracting elapsed time from MIOpenDriver's output
ELAPSED_TIME_RE = re.compile(r"Elapsed: ([0-9\.]*) ms")
# Compiled regexp object used for extracting target chip from arch
GFX_CHIP_RE = re.compile(r"gfx[0-9a-z]+")

@dataclass
class MLIRPaths:
    rocmlir_gen_path : str
    rocmlir_driver_path : str
    rocmlir_opt_path : str
    cpu_runner_path : str
    libmlir_rocm_runtime_path : str
    libconv_validation_wrappers_path : str
    libmlir_runtime_utils_path : str
    rocmlir_tuning_driver_path : str
    rocblas_benchmark_driver_path : Optional[str] = None
    ck_benchmark_driver_path : Optional[str] = None

@dataclass
class Paths:
    """This structure is used to hold paths needed to perform the tests"""
    configuration_file_path : str
    mlir_paths: Optional[MLIRPaths] = None
    miopen_driver_path: Optional[str] = None

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


def find_miopen_build_dir() -> str:
    """
    Finds miopen build dir searching either WORKSPACE dir
    or home dir
    """

    miopen_driver_path = None
    candidate_paths = [
        # if the script is run from build dir and assuming MIOpen is under mlir build
        Path('../MIOpen/build/bin/MIOpenDriver'),
        # if the script is run from source and assuming MIOpen is under mlir build
        Path(__file__).parent.parent.parent.parent / 'MIOpen'/ 'build' / 'bin' / 'MIOpenDriver'
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            miopen_driver_path = candidate_path

    if not miopen_driver_path:
        search_root = os.environ.get('WORKSPACE', str(Path.home()))
        assert search_root, "Cant find WORKSPACE env arg or home directory"
        miopen_driver_path = glob.glob(search_root + '/**/bin/MIOpenDriver', recursive=True)
        if len(miopen_driver_path) != 1:
            # MIOpen driver not available or ambiguous
            return None
        miopen_driver_path = miopen_driver_path[0]

    miopen_build_dir = Path(miopen_driver_path).parent.parent
    return str(miopen_build_dir)

def create_paths(config_file_path, mlir_build_dir_path, miopen_build_dir_path) -> Paths:
    """Creates the composite Paths structure using build dir paths"""

    mlir_paths = None
    if mlir_build_dir_path:
        mlir_bin_dir_path = (Path(mlir_build_dir_path) / 'bin').resolve()
        mlir_bin_dir = str(mlir_bin_dir_path)
        rocblas_benchmark_driver_location = mlir_bin_dir_path / 'rocblas-benchmark-driver'
        ck_benchmark_driver_location = mlir_bin_dir_path / 'ck-benchmark-driver'
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
            rocmlir_tuning_driver_path = mlir_bin_dir + '/rocmlir-tuning-driver',
            rocblas_benchmark_driver_path = str(rocblas_benchmark_driver_location) \
              if rocblas_benchmark_driver_location.exists() else None,
            ck_benchmark_driver_path = str(ck_benchmark_driver_location) \
              if ck_benchmark_driver_location.exists() else None)

    miopen_driver_path = None
    if miopen_build_dir_path:
        miopen_driver_location = (Path(miopen_build_dir_path) / 'bin' / 'MIOpenDriver').resolve()
        miopen_driver_path = str(miopen_driver_location) if miopen_driver_location.exists() else None

    return Paths(config_file_path, mlir_paths, miopen_driver_path)

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
                if len(entries) == 3:
                    arch, config, perfConfig = entries
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

    def setPerfConfig(self, perfConfig):
        raise NotImplementedError()

    @classmethod
    def fromCommandLine(cls, argv, arch) -> 'self':
        raise NotImplementedError()

    @classmethod
    def benchmarkExternal(cls, commandLine, paths: Paths, arch, envs=dict()):
        raise NotImplementedError()

    EXTERNAL_NAME = "unknown"

# convolution configurations.
def getConvConfigurations(fileName):
    configs = [];
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
    TABLE_COLUMNS = reportUtils.CONV_TEST_PARAMETERS + ['TFlops']
    EXTERNAL_NAME = "MIOpen"

    def computeTFlops(self, ns):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        return (2.0 * self.n * self.c * self.k * self.ho * self.wo * self.y * self.x) / (float(ns) * 1e-9) / 1e12

    def tableEntry(self, nanoSeconds):
        # Future(kdrewnia): This can just be a dict literal on Python 3.7+
        result = OrderedDict()
        values = [self.direction, self.dataType, self.chip, self.filterLayout, self.inputLayout, self.outputLayout,
                   self.n, self.c, self.hi, self.wi, self.k, self.y, self.x, self.dilationH, self.dilationW,
                   self.convStrideH, self.convStrideW, self.paddingH, self.paddingW, self.perfConfig,
                   self.computeTFlops(nanoSeconds)]
        assert(len(self.TABLE_COLUMNS) == len(values))

        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def __repr__(self):
        return f"""ConvConfiguration(dtype={self.dataType!r}, direction={self.direction!r}, layout={self.inputLayout.upper()!r},
                n={self.n!r}, c={self.c!r}, hi={self.hi!r}, wi={self.wi!r}, k={self.k!r}, y={self.y!r}, x={self.x!r},
                convStrideH={self.convStrideH!r}, convStrideW={self.convStrideW!r}, paddingH={self.paddingH!r}, paddingW={self.paddingW!r},
                dilationH={self.dilationH!r}, dilationW={self.dilationW!r}, group={self.group!r}, arch={self.arch!r}, perf_config={self.perfConfig})"""

    def setPerfConfig(self, perf_config):
        self.perfConfig = perf_config

    def generateMlirDriverCommandLine(self, rocmlir_gen_flags):
        direction = {'fwd':'--operation conv2d',
                     'bwd':'--operation conv2d_bwd_data',
                     'wrw':'--operation conv2d_bwd_weight'}[self.direction]

        result = ' '.join([direction,
                           '-t', self.dataType,
                           '--arch', self.arch,
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

    MLIR_FILTER_LAYOUTS = {"NCHW": "kcyx", "NCHWG": "kcyxg", "NHWC": "kyxc", "NHWCG": "kyxcg"}
    MLIR_OUTPUT_LAYOUTS = {"NCHW": "nkhw", "NCHWG": "nkhwg", "NHWC": "nhwk", "NHWCG": "nhwkg"}
    filterLayoutMap = {'N':'k', 'C':'c', 'H':'y', 'W':'x', 'G':'g'}
    inputLayoutMap = {'N':'n', 'C':'c', 'H':'h', 'W':'w', 'G':'g'}
    outputLayoutMap = {'N':'n', 'C':'k', 'H':'h', 'W':'w', 'G':'g'}

    @classmethod
    def fromCommandLine(cls, argv, arch):
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
                   group, arch)

    def __init__(self, dtype: str, direction: str, filterLayout: str, inputLayout:str, outputLayout:str,
                    n: int, c: int, hi: int, wi: int, k: int, y: int, x: int,
                    convStrideH: int, convStrideW: int, paddingH: int, paddingW: int,
                    dilationH: int, dilationW: int, group: int, arch: str):
        if dtype not in {"f16", "f32", "bf16", "i8"}:
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
        self.chip = GFX_CHIP_RE.search(arch).group(0)

        self.ho = math.floor((self.hi + self.paddingH * 2 - (self.y - 1) * self.dilationH - 1 ) / self.convStrideH) + 1
        self.wo = math.floor((self.wi + self.paddingW * 2 - (self.x - 1) * self.dilationW - 1 ) / self.convStrideW) + 1

        self.perfConfig = ''

    @classmethod
    def benchmarkExternal(cls, commandLine, paths: Paths, arch, envs=dict()):
        config = cls.fromCommandLine(commandLine, arch)
        MIOpenDriverCommand = [paths.miopen_driver_path, *commandLine, '-V', '0', '-t', '1']
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
    outDataType = None

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

class GemmConfiguration(PerfConfiguration):
    TABLE_COLUMNS = reportUtils.GEMM_TEST_PARAMETERS + ['TFlops']
    def computeTFlops(self, ns):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        return (2.0 * self.g * self.m * self.k * self.n) / (float(ns) * 1e-9) / 1e12

    def tableEntry(self, nanoSeconds):
        # Future(kdrewnia): This can just be a dict literal on Python 3.7+
        result = OrderedDict()
        values = [self.dataType, self.outDataType, self.chip, self.transA, self.transB, \
                   self.g, self.m, self.k, self.n, self.perfConfig, self.computeTFlops(nanoSeconds)]
        assert(len(self.TABLE_COLUMNS) == len(values))

        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def __repr__(self):
        return f"""GemmConfiguration(dtype={self.dataType!r}, outDataType={self.outDataType!r}, g={self.g!r}, m={self.m!r}, k={self.k!r}, n={self.n!r},
                transA={self.transA!r}, transB={self.transB!r}, arch={self.arch!r}, perf_config={self.perfConfig})"""

    def setPerfConfig(self, perf_config):
        self.perfConfig = perf_config

    def generateMlirDriverCommandLine(self, rocmlir_gen_flags):
        result = ' '.join(['-operation', 'gemm',
                           '-t', self.dataType,
                           '--arch', self.arch,
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
    def fromCommandLine(cls, argv, arch):
        # Please keep this in sync with mlir::rock::getTuningProblemStr()
        dtype = None
        g = None
        m = None
        k = None
        n = None
        transA = None
        transB = None
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
            else:
                raise ValueError(f"Unknown GEMM config argument {opt} -> {val}")
        for v in [dtype, outDataType, g, m, k, n, transA, transB]:
            if v is None:
                raise ValueError("Incomplete GEMM configuration")

        return cls(dtype, outDataType, g, m, k, n, transA, transB, arch)

    def __init__(self, dtype: str, outDataType: str, g: int, m: int, k: int, n: int,
                 transA: bool, transB: bool, arch: str):
        if dtype not in {"f16", "f32", "bf16", "i8"}:
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
        self.perfConfig = ''

        self.arch = arch
        self.chip = GFX_CHIP_RE.search(arch).group(0)

class RocBLASGemmConfig(GemmConfiguration):
    EXTERNAL_NAME = "rocBLAS"

    @classmethod
    def benchmarkExternal(cls, commandLine, paths: Paths, arch, envs=dict()):
        config = cls.fromCommandLine(commandLine, arch)
        if not paths.mlir_paths.rocblas_benchmark_driver_path:
            raise ValueError("rocblas-benchmark-driver not built")
        benchmarkArgs = config.generateMlirDriverCommandLine("")
        # remove the result file generated by rocprof in previous benchmarking
        os.system("rm -f "+BENCHMARKING_RESULT_FILE_NAME)

        print(f"Running rocBLAS benchmark {config!r}")
        profilerCommand = [ROCPROF, '--stats', \
            paths.mlir_paths.rocblas_benchmark_driver_path] + \
            benchmarkArgs.split()
        p = subprocess.Popen(profilerCommand, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # get output.
        try:
            outs, errs = p.communicate(timeout=60)
            if len(errs) > 0:
                print("Test printed errors: ", errs.decode('utf-8'))
                print("Failing command line: ", profilerCommand)
                if p1.returncode != 0:
                    raise OSError(errs.decode('utf-8'))
        except subprocess.TimeoutExpired:
            print("Test timed out: ", profilerCommand)
            p.kill()
            outs, errs = p.communicate()
        nanoSeconds = getNanoSeconds(BENCHMARKING_RESULT_FILE_NAME)
        return config.tableEntry(nanoSeconds)

class CKGemmConfig(GemmConfiguration):
    EXTERNAL_NAME = "CK"
    @classmethod
    def benchmarkExternal(cls, commandLine, paths: Paths, arch, envs=dict()):
        config = cls.fromCommandLine(commandLine, arch)
        if not paths.mlir_paths.ck_benchmark_driver_path:
            raise ValueError("ck-benchmark-driver not built")
        benchmarkArgs = config.generateMlirDriverCommandLine("")

        print(f"Running CK benchmark {config!r}")

        if arch=="gfx1030" and config.g > 1:
            return config.tableEntry(float('NaN'))

        profilerCommand = [paths.mlir_paths.ck_benchmark_driver_path] + \
            benchmarkArgs.split()
        p = subprocess.Popen(profilerCommand, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # get output.
        try:
            outs, errs = p.communicate(timeout=60)
            if len(errs) > 0:
                print("Test printed errors: ", errs.decode('utf-8'))
                print("Failing command line: ", profilerCommand)
                if p1.returncode != 0:
                    raise OSError(errs.decode('utf-8'))
        except subprocess.TimeoutExpired:
            print("Test timed out: ", profilerCommand)
            p.kill()
            outs, errs = p.communicate()
        milliSeconds = getMilliseconds(outs)
        nanoSeconds = milliSeconds*1e6
        return config.tableEntry(nanoSeconds)

def runConfigWithMLIR(config: PerfConfiguration, paths: Paths, rocmlir_gen_flags, debug=True):
    # remove the result file generated by rocprof in previous benchmarking
    os.system("rm -f "+BENCHMARKING_RESULT_FILE_NAME)
    commandLineOptions = config.generateMlirDriverCommandLine(rocmlir_gen_flags)
    if debug:
        print("Running MLIR Benchmark: ", repr(config))
    rocmlirGenCommand = paths.mlir_paths.rocmlir_gen_path + ' -ph ' + commandLineOptions
    rocmlirDriverCommand = [paths.mlir_paths.rocmlir_driver_path, '-c']
    mlir_cpu_runner_args = [f'--shared-libs={paths.mlir_paths.libmlir_rocm_runtime_path},{paths.mlir_paths.libconv_validation_wrappers_path},{paths.mlir_paths.libmlir_runtime_utils_path}', '--entry-point-result=void']
    profilerCommand = [ROCPROF, '--stats', paths.mlir_paths.cpu_runner_path] + mlir_cpu_runner_args

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
        outs, errs = p3.communicate(timeout=60)
        if len(errs) > 0:
            print("Test printed errors: ", errs.decode('utf-8'))
            print("Failing command line: ", rocmlirGenCommand)
            if p1.returncode != 0:
                raise OSError(errs.decode('utf-8'))
    except subprocess.TimeoutExpired:
        print("Test timed out: ", rocmlirGenCommand)
        p3.kill()
        outs, errs = p3.communicate()

# Benchmarking function.
def benchmarkMLIR(commandLine, confClass, paths: Paths, arch, tuningDb: MaybeTuningDb, rocmlir_gen_flags):
    config = confClass.fromCommandLine(commandLine, arch)
    configStr = ' '.join(commandLine)
    if tuningDb:
        if (arch, configStr) in tuningDb:
            config.setPerfConfig(tuningDb[arch, configStr])
        else: # Tuning DB present but doesn't contain config, return N/A
            return config.tableEntry(np.nan)

    runConfigWithMLIR(config, paths, rocmlir_gen_flags)
    # get nanoseconds from rocprof output.
    nanoSeconds = getNanoSeconds(BENCHMARKING_RESULT_FILE_NAME)
    return config.tableEntry(nanoSeconds)

#Generate MLIR vs. MIOpen or rocBLAS performance results
def generatePerformanceResults(configs, confClass, paths: Paths, arch, tuningDb: MaybeTuningDb, rocmlir_gen_flags):
    # Never pass tuning DB to this run
    mlir_df = pd.DataFrame(benchmarkMLIR(testVector.split(sep=' '), confClass, paths, arch, None, rocmlir_gen_flags)
        for testVector in configs)
    tuned_df = None
    if tuningDb:
        tuned_df = pd.DataFrame(benchmarkMLIR(testVector.split(sep=' '), confClass, paths, arch, tuningDb, rocmlir_gen_flags)
            for testVector in configs)
    external_df = pd.DataFrame(confClass.benchmarkExternal(testVector.split(sep=' '), paths, arch)
        for testVector in configs)

    externalName = confClass.EXTERNAL_NAME
    df = mlir_df.merge(external_df, on=confClass.TABLE_COLUMNS[:-1],
                           suffixes=('', f" ({externalName})"))
    externalTFlopsCol = f"{externalName} TFlops (no MLIR Kernels)"
    df.rename(columns={'TFlops': 'MLIR TFlops', f"TFlops ({externalName})": externalTFlopsCol}, inplace=True)
    if tuned_df is not None:
        # No need for suffixes, the conflicting columns have been renamed
        # Also note that we're ignoring PerfConfig with the -2
        df = df.merge(tuned_df, on=confClass.TABLE_COLUMNS[:-2],
            suffixes=('', ' (tuned)'))
        df.drop(columns=['PerfConfig'], inplace=True)
        df.rename(columns={'TFlops': 'Tuned MLIR TFlops',
            'PerfConfig (tuned)': 'PerfConfig'}, inplace=True)

    df[f"MLIR/{externalName}"] = df['MLIR TFlops'] / df[externalTFlopsCol]
    if tuned_df is not None:
        df[f"Tuned/{externalName}"] = df['Tuned MLIR TFlops'] / df[externalTFlopsCol]
        df["Tuned/Untuned"] = df['Tuned MLIR TFlops'] / df['MLIR TFlops']
    chip = GFX_CHIP_RE.search(arch).group(0)
    if confClass is RocBLASGemmConfig:
        reportFile = reportUtils.PERF_REPORT_GEMM_FILE['rocBLAS']
    elif confClass is CKGemmConfig:
        reportFile = reportUtils.PERF_REPORT_GEMM_FILE['CK']
    else:
        reportFile = reportUtils.PERF_REPORT_FILE
    df.fillna('NaN', inplace=True)
    df.to_csv(chip + '_' + reportFile, index=False)

def getSolverName(testVector, arch):
    config = ConvConfiguration.fromCommandLine(testVector.split(sep=' '), arch)
    if config.direction == 'fwd':
       solverName = 'ConvMlirIgemmFwd'
    elif config.direction == 'bwd':
       solverName = 'ConvMlirIgemmBwd'
    else:
       solverName = 'ConvMlirIgemmWrW'
    if config.chip == 'gfx908' or config.chip == 'gfx90a':
       solverName+='Xdlops'
    return solverName

def benchmarkMIOpenWithMLIRKernels(configs, arch, filename, paths: Paths):
    solver_names = {testVector : getSolverName(testVector, arch) for testVector in configs}

    # Set environment variables for running MIOpenDriver with MLIR kernels
    envs = os.environ.copy()
    envs['MIOPEN_FIND_MODE'] = '1'
    envs['MIOPEN_DRIVER_USE_GPU_REFERENCE'] = '1'
    perf_list = []
    for testVector in configs:
        envs['MIOPEN_DEBUG_FIND_ONLY_SOLVER']=solver_names[testVector]
        perf_list.append(ConvConfiguration.benchmarkExternal(testVector.split(sep=' '), paths, arch, envs))
    df = pd.DataFrame(perf_list)
    chip = GFX_CHIP_RE.search(arch).group(0)
    df.to_csv(chip + '_' + filename, index=False)

RUNNABLE_TEST_RE = re.compile(r"//\s*RUN\s*:(.*)")
ROCMLIRGEN_RE = re.compile(r"rocmlir-gen.*-fut\s*(\w+)")
def findRunCommand(filename):
    firstCommand = None
    futName = None
    with open(filename, 'r') as f:
        for line in f:
            hasRun = RUNNABLE_TEST_RE.search(line)
            hasRocmlirGen = ROCMLIRGEN_RE.search(line)
            if hasRun:
                command = hasRun.group(1)
                if not firstCommand:
                    parts = command.split('|')  # Split the command using the "|" separator
                    firstCommand = parts[0].strip() # Find the first command

                if hasRocmlirGen:
                    futName = hasRocmlirGen.group(1)

                if 'runner' in line: # Stop processing lines after finding a runner
                    return firstCommand, futName

    # Not found a "RUN" command or a runner
    print("WARNING: cannot find valid RUN command in ", filename)
    return None, None

# Extract testVector and test function name from the test file
def getFusionTestInfo(filename, paths: Paths):
    testEntry = {}
    firstCommand, futName = findRunCommand(filename)
    if not firstCommand:
        return testEntry

    if "-migraphx-to-tosa" in firstCommand:
        rocmlirOptCommand = [paths.mlir_paths.rocmlir_opt_path, '-migraphx-to-tosa', filename]
        rocmlirDriverCommand = [paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'partition,highlevel', '-targets', getChip()]
        # rocmlir-opt -migraphx-to-tosa ../mlir/test/fusion/resnet50-e2e/mixr-resnet-fusion-case-1.mlir
        p1 = subprocess.Popen(rocmlirOptCommand, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # pipe to rocmlir-driver -host-pipeline partition,highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlirDriverCommand, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p1.stdout.close()
    else:
        rocmlirDriverCommand = [paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'partition,highlevel', '-targets', getChip(), filename]
        # rocmlir-driver -host-pipeline partition,highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlirDriverCommand, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # pipe to rocmlir_gen --emit-tuning-key
    tuningKey = subprocess.Popen([paths.mlir_paths.rocmlir_gen_path, '--emit-tuning-key', '-'], stdin=p2.stdout,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2.stdout.close()
    output, _ = tuningKey.communicate()
    result = output.decode('utf-8').strip().split('\t')
    testEntry = {'filename' : filename, 'testVector' : result[1], 'futName' : futName}
    return testEntry

def runFusionKernel(filename, rocmlirGenArgs, paths: Paths):
    os.system("rm -f "+BENCHMARKING_RESULT_FILE_NAME)

    firstCommand, _ = findRunCommand(filename)
    if "-migraphx-to-tosa" in firstCommand:
        rocmlirOptCommand = [paths.mlir_paths.rocmlir_opt_path, '-migraphx-to-tosa', filename]
        rocmlirDriverCommand = [paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'partition,highlevel', '-targets', getChip()]
        # rocmlir-opt -migraphx-to-tosa ../mlir/test/fusion/resnet50-e2e/mixr-resnet-fusion-case-1.mlir
        p1 = subprocess.Popen(rocmlirOptCommand, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # pipe to rocmlir-driver -host-pipeline partition,highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlirDriverCommand, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p1.stdout.close()
    else:
        rocmlirDriverCommand = [paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'partition,highlevel', '-targets', getChip(), filename]
        # rocmlir-driver -host-pipeline partition,highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlirDriverCommand, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    rocmlirGenCommand = [paths.mlir_paths.rocmlir_gen_path] + rocmlirGenArgs
    kernelPipelineCommand = [paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'mhal,runner', '-kernel-pipeline','full']
    mlir_cpu_runner_args = [f'--shared-libs={paths.mlir_paths.libmlir_rocm_runtime_path},{paths.mlir_paths.libconv_validation_wrappers_path},{paths.mlir_paths.libmlir_runtime_utils_path}', '--entry-point-result=void']
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
        outs, errs = profiling.communicate(timeout=60)
        if len(errs) > 0:
            print("Test printed errors: ", errs.decode('utf-8'))
            print("Failing: ", filename)
    except subprocess.TimeoutExpired:
        print("Test timed out: ", filename)
        profiling.kill()
        outs, errs = profiling.communicate()


# Generate fusion vs. gemm/conv performance results
def benchmarkFusionKernels(test_dir, paths: Paths, arch, tuningDb: MaybeTuningDb):
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
            confClass = ConvConfiguration
            config = ConvConfiguration.fromCommandLine(commandLine, arch)
        else:
            op = 'gemm'
            confClass = GemmConfiguration
            config = GemmConfiguration.fromCommandLine(commandLine, arch)

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
               continue;

        # Run fusion test
        rocmlirGenArgs = ['-ph', '-fut='+futName, '--perf_config='+bestPerf, '-']
        runFusionKernel(filename, rocmlirGenArgs, paths)
        # Get nanoseconds of fusion test
        nanoSeconds = getNanoSeconds(BENCHMARKING_RESULT_FILE_NAME)
        oneEntry = config.tableEntry(nanoSeconds)
        # Keep the best performance
        if testVector in perfResults and oneEntry['TFlops'] <= perfResults[testVector]['TFlops']:
            continue

        # Run gemm or conv op with the same configuration
        runConfigWithMLIR(config, paths, '')
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
def tuneMLIRKernels(configs, paths: Paths, arch):
    solver_names = {testVector : getSolverName(testVector, arch) for testVector in configs}

    envs = os.environ.copy()
    envs['MIOPEN_FIND_ENFORCE'] = '4'
    envs['MIOPEN_DRIVER_USE_GPU_REFERENCE'] = '1'
    for testVector in configs:
        envs['MIOPEN_DEBUG_FIND_ONLY_SOLVER']=solver_names[testVector]
        commandLine = testVector.split(sep=' ')
        config = ConvConfiguration.fromCommandLine(commandLine, arch)
        if config.inputLayout == 'nchw':
            MIOpenDriverCommand = [paths.miopen_driver_path, *commandLine,'-V', '0']
            print(' '.join(MIOpenDriverCommand))
            p1 = subprocess.Popen(MIOpenDriverCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=envs)
            # get output.
            try:
               outs, errs = p1.communicate(timeout=300)
               if len(errs) > 0 and p1.returncode != 0:
                   raise OSError(errs.decode('utf-8'))
            except subprocess.TimeoutExpired:
                p1.kill()
                print("MIOpen tuning timed out")
                outs, errs = p1.communicate()

def is_xdlops_present() -> bool:
    """This function checks whether a GPU with xdlops support is present"""
    xdlop_supported_gpus = ['gfx908', 'gfx90a']
    xdlop_supported_gpus_str = xdlop_supported_gpus[0]
    for gpu in xdlop_supported_gpus[1:]:
        xdlop_supported_gpus_str += '|' + gpu
    r = subprocess.run(f"/opt/rocm/bin/rocm_agent_enumerator -t GPU | grep -q -E '{xdlop_supported_gpus_str}'", shell=True)
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
        return DATA_TYPES_GEMM, OUTPUT_DATA_TYPE_MAP

    datatypes = []
    outMap = {}
    for dpair in data_types:
        dt = dpair.split('_')
        datatypes.append(dt[0])
        outMap[dt[0]] = 'i32' if dt[0] == 'i8' else dt[0]
        if len(dt) == 2:
            outMap[dt[0]] = dt[1]

    return datatypes, outMap

def getChip():
    archNames = getArch()
    arch = ','.join(archNames)
    chip = GFX_CHIP_RE.search(arch).group(0)
    return chip

def foundExternalTool(paths: Paths, opType: Operation, gemmLibrary : Optional[GEMMLibrary] = None):
    if opType == Operation.CONV and not paths.miopen_driver_path:
        return False
    if opType == Operation.GEMM:
        if not paths.mlir_paths:
            return False
        if gemmLibrary == GEMMLibrary.CK and not paths.mlir_paths.ck_benchmark_driver_path:
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
    python3 perfRunner.py --miopen_use_tuned_mlir
    python3 perfRunner.py --miopen_use_untuned_mlir
    """
    if args is None:
        args = sys.argv[1:]

    archNames = getArch()
    arch = ','.join(archNames)
    chip = GFX_CHIP_RE.search(arch).group(0)

    root_dir = str(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
    default_conv_configs = root_dir + '/mlir/utils/jenkins/miopen-tests/important-conv-configs'

    parser = argparse.ArgumentParser(
        prog="rocMLIR performance test runner",
        description="A test runner script for MIOpen and MLIR-based kernel generator",
        allow_abbrev=False,
    )

    parser.add_argument("--op", "--operation", choices=['conv', 'gemm', 'fusion'],
        default='conv',
        help="Operation to benchmark")

    mutex_arg_group = parser.add_mutually_exclusive_group()
    mutex_arg_group.add_argument(
        "--miopen_use_tuned_mlir",
        action="store_true",
        help="Run the benchmarks using tuned MLIR kernels",
    )
    mutex_arg_group.add_argument(
        "--miopen_use_untuned_mlir",
        action="store_true",
        help="Run the benchmarks using untuned MLIR kernels"
    )
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
        "--miopen-build-dir",
        type=str,
        default=find_miopen_build_dir(),
        help="The build directory of MIOpen",
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
         choices=["f32", "f16", "i8", "i8_i32", "i8_i8"],
         default=["f32", "f16", "i8"],
         help='Force a set of datatypes'
    )

    parsed_args = parser.parse_args(args)

    rocmlir_gen_flags = ''
    if 'rocmlir_gen_flags' in parsed_args:
        rocmlir_gen_flags = parsed_args.rocmlir_gen_flags

    tuningDb = None
    if 'tuning_db' in parsed_args:
        tuningDb = read_tuning_db(parsed_args.tuning_db)

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

    configs_path = None if parsed_args.config else parsed_args.configs_file
    paths = create_paths(configs_path, parsed_args.mlir_build_dir, parsed_args.miopen_build_dir)
    configs = []
    if opType == Operation.CONV:
        configs = getConvConfigurations(paths.configuration_file_path)
    elif opType == Operation.GEMM:
        datatypes, outputTypeMap = parseDataTypes(parsed_args.data_type)
        configs = getGemmConfigurations(paths.configuration_file_path, datatypes, outputTypeMap)

    if parsed_args.external or parsed_args.batch_external or parsed_args.batch_all:
        if not foundExternalTool(paths, opType, externalLib):
            raise RuntimeError("External benchmark reference (MIOpen or rocBLAS driver) needed but not found")

    if parsed_args.miopen_use_tuned_mlir or parsed_args.miopen_use_untuned_mlir \
            or parsed_args.tuning:
        if not paths.miopen_driver_path:
            raise RuntimeError("MIOpen build dir was not provided/found where the test requires it")

    if parsed_args.batch_mlir or parsed_args.batch_all:
        if not paths.mlir_paths:
            raise RuntimeError("MLIR build dir was not provided/found")


    #If no arguments are passed, then benchmark with MLIR and MIOpen
    if parsed_args.batch_all:
        # batch benchmark with MLIR and MIOpen.
        generatePerformanceResults(configs, confClass, paths, arch, tuningDb, rocmlir_gen_flags)
    elif parsed_args.miopen_use_tuned_mlir:
        benchmarkMIOpenWithMLIRKernels(configs, arch, reportUtils.MIOPEN_TUNED_REPORT_FILE, paths)
    elif parsed_args.miopen_use_untuned_mlir:
        benchmarkMIOpenWithMLIRKernels(configs, arch, reportUtils.MIOPEN_UNTUNED_REPORT_FILE, paths)
    elif parsed_args.tuning:
        tuneMLIRKernels(configs, paths, arch)
    elif opType == Operation.FUSION:
        if not parsed_args.mlir_build_dir:
            raise RuntimeError("MLIR build dir was not provided/found")
        else:
            benchmarkFusionKernels(parsed_args.test_dir, paths, arch, tuningDb)
    else:
        if parsed_args.batch_mlir:
            df = pd.DataFrame(benchmarkMLIR(testVector.split(sep=' '), confClass, paths, arch, tuningDb, rocmlir_gen_flags) for testVector in configs)
        elif parsed_args.batch_external:
            df = pd.DataFrame(confClass.benchmarkExternal(testVector.split(sep=' '), paths, arch) for testVector in configs)
        elif parsed_args.external:
            df = pd.DataFrame([confClass.benchmarkExternal(parsed_args.config, paths, arch)])
        else:
            # Will only reach here with more than 1 unspecified arguments
            # These are arguments are directly passed through to benchmarkMLIR
            if not parsed_args.mlir_build_dir:
                raise RuntimeError("MLIR build dir was not provided/found")
            else:
                df = pd.DataFrame([benchmarkMLIR(parsed_args.config, confClass, paths, arch, tuningDb, rocmlir_gen_flags)])
        df.to_csv(parsed_args.fileName)
        with pd.option_context('display.precision', reportUtils.ROUND_DIGITS):
            print(df) # for interactive consumption

if __name__ == '__main__':
    sys.exit(main())
