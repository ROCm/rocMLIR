#!/usr/bin/env python3

from typing import NamedTuple
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
from pathlib import Path
import glob
import argparse

import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from typing import Optional

# global variables.
ROCPROF = '/opt/rocm/bin/rocprof'
BENCHMARKING_RESULT_FILE_NAME = 'results.stats.csv'
DIRECTIONS = ['-F 1', '-F 2', '-F 4']
DATA_TYPES = ['conv', 'convfp16', 'convint8']
LAYOUTS = ['NHWC', 'NCHW']

# Compiled regexp object used for extracting elapsed time from MIOpenDriver's output
ELAPSED_TIME_RE = re.compile(r"Elapsed: (.*)ms")

@dataclass
class MLIRPaths:
    miopen_gen_path: str
    mlir_miopen_driver_path: str
    rocm_runner_path : str
    libmlir_rocm_runtime_path : str
    libconv_validation_wrappers_path : str
    libmlir_runtime_utils_path : str

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
    miopen_gen_path = None
    candidate_paths = [
        # if the script is run from build dir
        Path('./bin/miopen-gen'),
        # if the script is run from source
        Path(__file__).parent.parent.parent.parent / 'build' / 'bin' / 'miopen-gen'
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            miopen_gen_path = candidate_path

    if not miopen_gen_path:
        try:
            # Prioritize the search in the current repo first.
            search_root = str(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
        except subprocess.CalledProcessError:
            # Else look in the home or WORKSPACE directory
            search_root = os.environ.get('WORKSPACE', str(Path.home()))
            assert search_root, "Cant find WORKSPACE env arg or home directory"

        miopen_gen_path = glob.glob(search_root + '/**/bin/miopen-gen', recursive=True)
        if len(miopen_gen_path) == 0:
            # MLIR miopen_gen not available
            return None
        assert len(miopen_gen_path) == 1, "Multiple paths found to contain */bin/miopen-gen"
        miopen_gen_path = miopen_gen_path[0]

    build_dir = Path(miopen_gen_path).parent.parent
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
        if len(miopen_driver_path) == 0:
            # MIOpen driver not available
            return None
        assert len(miopen_driver_path) == 1, "Multiple paths found to contain */bin/MIOpenDriver"
        miopen_driver_path = miopen_driver_path[0]

    miopen_build_dir = Path(miopen_driver_path).parent.parent
    return str(miopen_build_dir)

def create_paths(mlir_build_dir_path, miopen_build_dir_path) -> Paths:
    """Creates the composite Paths structure using build dir paths"""

    mlir_paths = None
    if mlir_build_dir_path:
        mlir_bin_dir = str((Path(mlir_build_dir_path) / 'bin').resolve())
        mlir_lib_dir = str((Path(mlir_build_dir_path) / 'lib').resolve())
        llvm_lib_dir = str((Path(mlir_build_dir_path) / 'external/llvm-project/llvm/lib').resolve())
        mlir_paths = MLIRPaths(miopen_gen_path = mlir_bin_dir + '/miopen-gen',
        mlir_miopen_driver_path = mlir_bin_dir + '/mlir-miopen-driver',
        rocm_runner_path = mlir_bin_dir + '/mlir-rocm-runner',
        libmlir_rocm_runtime_path =  llvm_lib_dir + '/libmlir_rocm_runtime.so',
        libconv_validation_wrappers_path = mlir_lib_dir + '/libconv-validation-wrappers.so',
        libmlir_runtime_utils_path = llvm_lib_dir + '/libmlir_runner_utils.so')

    miopen_driver_path = None
    if miopen_build_dir_path:
        miopen_driver_bin_dir = Path(miopen_build_dir_path) / 'bin'
        miopen_driver_path = str((miopen_driver_bin_dir / 'MIOpenDriver').resolve())

    root_dir = str(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
    configuration_dir = root_dir + '/mlir/utils/jenkins/miopen-tests/resnet50-miopen-configs'

    return Paths(configuration_dir, mlir_paths, miopen_driver_path)

# utility functions.
def getConfigurations(fileName):
    configFile = open(fileName, 'r')
    lines = configFile.readlines()
    configs = [];
    # All combinations of conv direction, type and layouts
    for direction, datatype, layout, line in itertools.product(DIRECTIONS, DATA_TYPES, LAYOUTS, lines):
        line = line.strip()

        # Skip empty lines
        if len(line) == 0 or line[0] == '#':
            continue
        # Skip int8 non-fwd convolutions
        if datatype == 'convint8' and direction != '-F 1':
            continue

        oneConfig = f"{datatype} {direction} -f {layout} -I {layout} -O {layout} {line}"
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
        return f"""ConvConfiguration(dtype={self.dataType!r}, direction={self.direction!r}, layout={self.inputLayout.upper()!r},
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
                    convStrideH: int, convStrideW: int, paddingH: int, paddingW: int,
                    dilationH: int, dilationW: int, group: int, xdlops: bool):
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

def runConfigWithMLIR(config: ConvConfiguration, target: str, paths: Paths):
    # remove the result file generated by rocprof in previous benchmarking
    os.system("rm "+BENCHMARKING_RESULT_FILE_NAME)
    commandLineOptions = config.generateMlirDriverCommandLine()
    print("Running MLIR Benchmark: ", repr(config))
    miopenGenCommand = paths.mlir_paths.miopen_gen_path + ' -ph ' + commandLineOptions
    mlirMiopenDriverCommand = [paths.mlir_paths.mlir_miopen_driver_path, '-c']
    mlir_rocm_runner_args = [f'--target={target}', f'--shared-libs={paths.mlir_paths.libmlir_rocm_runtime_path},{paths.mlir_paths.libconv_validation_wrappers_path},{paths.mlir_paths.libmlir_runtime_utils_path}', '--entry-point-result=void']
    profilerCommand = [ROCPROF, '--stats', paths.mlir_paths.rocm_runner_path] + mlir_rocm_runner_args

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

def runConfigWithMIOpenDriver(commandLine, paths: Paths, envs):
    MIOpenDriverCommand = [paths.miopen_driver_path, *commandLine, '-V', '0']
    print("Running MIOpen Benchmark: ", ' '.join(commandLine))
    # invoke MIOpenDriver.
    p1 = subprocess.Popen(MIOpenDriverCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=envs)
    # get output.
    try:
        outs, errs = p1.communicate(timeout=300)
        if len(errs) > 0:
            print("MIOpen benchmark produced errors: ", errs.decode('utf-8'))
            return np.nan
        else:
            # convert bytes to str
            outs=outs.decode('utf-8')
            # Extract Elapsed time in ms from the output of MIOpenDriver
            # Use regular expression to match the contents between
            # "Elasped: " (note the space at the end) and "ms"
            elapsedTimeInMs = ELAPSED_TIME_RE.search(outs).group(1)
            return float(elapsedTimeInMs)*1.0e6
    except subprocess.TimeoutExpired:
        p1.kill()
        print("MIOpen benchmark timed out")
        outs, errs = p1.communicate()
        ## make sure timeout does not break this script
        return np.nan

# Benchmarking function.
def benchmarkMLIR(commandLine, target: str, xdlops: bool, paths: Paths):
    config = ConvConfiguration.fromCommandLine(commandLine, xdlops)
    runConfigWithMLIR(config, target, paths)
    # get nanoseconds from rocprof output.
    nanoSeconds = getNanoSeconds(BENCHMARKING_RESULT_FILE_NAME)
    return config.tableEntry(nanoSeconds)

def benchmarkMIOpen(commandLine, xdlops, paths: Paths, envs=dict()):
    config = ConvConfiguration.fromCommandLine(commandLine, xdlops)
    # get nanoseconds from MIOpenDriver output
    nanoSeconds = runConfigWithMIOpenDriver(commandLine, paths, envs)
    return config.tableEntry(nanoSeconds)

#Generate MLIR vs. MIOpen performance results
def generatePerformanceResults(configs, target: str, xdlops: bool, paths: Paths):
    mlir_df = pd.DataFrame(benchmarkMLIR(testVector.split(sep=' '), target, xdlops, paths)
        for testVector in configs)
    miopen_df = pd.DataFrame(benchmarkMIOpen(testVector.split(sep=' '), xdlops, paths)
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

def benchmarkMIOpenWithMLIRKernels(configs, xdlops, filename, paths: Paths):
    solver_names = {testVector : getSolverName(testVector, xdlops) for testVector in configs}

    # Set environment variables for running MIOpenDriver with MLIR kernels
    envs = os.environ.copy()
    envs['MIOPEN_FIND_MODE'] = '1'
    envs['MIOPEN_DRIVER_USE_GPU_REFERENCE'] = '1'
    perf_list = []
    for testVector in configs:
        envs['MIOPEN_DEBUG_FIND_ONLY_SOLVER']=solver_names[testVector]
        perf_list.append(benchmarkMIOpen(testVector.split(sep=' '), xdlops, paths, envs))
    df = pd.DataFrame(perf_list)
    df.to_csv(filename, index=False)

#Tune MIOpen with MLIR kernels
def tuneMLIRKernels(configs, xdlops, paths: Paths):
    solver_names = {testVector : getSolverName(testVector, xdlops) for testVector in configs}

    envs = os.environ.copy()
    envs['MIOPEN_FIND_ENFORCE'] = '4'
    envs['MIOPEN_DRIVER_USE_GPU_REFERENCE'] = '1'
    for testVector in configs:
        envs['MIOPEN_DEBUG_FIND_ONLY_SOLVER']=solver_names[testVector]
        commandLine = testVector.split(sep=' ')
        config = ConvConfiguration.fromCommandLine(commandLine, xdlops)
        if config.inputLayout == 'nchw':
            MIOpenDriverCommand = [paths.miopen_driver_path, *commandLine,'-V', '0']
            print(' '.join(MIOpenDriverCommand))
            p1 = subprocess.Popen(MIOpenDriverCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=envs)
            # get output.
            try:
               outs, errs = p1.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                p1.kill()
                print("MIOpen tuning timed out")
                outs, errs = p1.communicate()

def get_target() -> str:
    p = subprocess.run(["/opt/rocm/bin/rocm_agent_enumerator", "-name"],
        stdout=subprocess.PIPE, check=True)
    targets = set(x.decode("utf-8") for x in p.stdout.split())
    # Workaround for bug in rocm_agent_enumerator
    if not targets:
        q = subprocess.run(["/opt/rocm/bin/rocm_agent_enumerator"],
            stdout=subprocess.PIPE, check=True)
        targets = set(x.decode("utf-8") for x in q.stdout.split() if x != b"gfx000")
    if not targets:
        raise RuntimeError("Could not find any GPUs present on this system")
    if len(targets) != 1:
        raise RuntimeError("Found too many different GPU types on this system: ", targets)
    return next(iter(targets))

def is_xdlops_present(target: str) -> bool:
    """This function checks whether a GPU with xdlops support is present"""
    xdlop_supported_gpus = ['gfx908', 'gfx90a']
    return any(target.startswith(gpu) for gpu in xdlop_supported_gpus)

def add_double_dash_convention(args):
    """This function will correct single-dashed multi-char args to be double-dash to the parser"""
    double_dash_args = list()
    for arg in args:
        # Only make non-single char args double dash
        # that already had a single dash
        if len(arg) > 2:
            if arg[0:2] != '--' and arg[0] == '-':
                double_dash_args.append('-' + arg)
        else:
            double_dash_args.append(arg)
    return double_dash_args


# Main function.
def main(args=None):
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
    if args is None:
        args = sys.argv[1:]
    args = add_double_dash_convention(args)

    parser = argparse.ArgumentParser(
        prog="MIOpenDriver Test Runner",
        description="A test runner script for MIOpen and MLIR-based kernel generator",
        allow_abbrev=False,
    )

    mutex_arg_group = parser.add_mutually_exclusive_group()
    mutex_arg_group.add_argument(
        "--bmiopen_use_tuned_mlir",
        action="store_true",
        help="Run the benchmarks using tuned MLIR kernels",
    )
    mutex_arg_group.add_argument(
        "--bmiopen_use_untuned_mlir",
        action="store_true",
        help="Run the benchmarks using untuned MLIR kernels"
    )
    mutex_arg_group.add_argument(
        "--tuning",
        action="store_true",
        help="Only tune the MLIR kernels"
    )
    mutex_arg_group.add_argument(
        "-b",
        action="store_true",
        help="CSV batch benchmarking mode with MLIR"
    )
    mutex_arg_group.add_argument(
        "--bmiopen",
        action="store_true",
        help="CSV batch benchmarking mode with MIOpen"
    )
    mutex_arg_group.add_argument(
        "--miopen",
        action="store_true",
        help="benchmark a single config"
    )

    parser.add_argument(
        "-o",
        type=str,
        default=date.today().strftime("perf.%m%d%y"),
        help="Output file name",
        dest="fileName"
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
        "--force-xdlops",
        type=bool,
        default=None,
        help="Force xdlops usage on or off"
    )
    parsed_args, _ = parser.parse_known_args(args)

    if parsed_args.miopen or parsed_args.bmiopen or parsed_args.bmiopen_use_tuned_mlir or \
       parsed_args.bmiopen_use_untuned_mlir or parsed_args.tuning or len(sys.argv) == 1:
        if not parsed_args.miopen_build_dir:
            raise RuntimeError("MIOpen build dir was not provided/found where the test requires it")

    if parsed_args.b or len(sys.argv) == 1:
        if not parsed_args.mlir_build_dir:
            raise RuntimeError("MLIR build dir was not provided/found")

    paths = create_paths(parsed_args.mlir_build_dir, parsed_args.miopen_build_dir)
    target = get_target()
    xdlops = is_xdlops_present(target)
    if parsed_args.force_xdlops is not None:
        xdlops = parsed_args.force_xdlops
    configs = getConfigurations(paths.configuration_file_path)

    #If no arguments are passed, then benchmark with MLIR and MIOpen
    if len(sys.argv) == 1:
        # batch benchmark with MLIR and MIOpen.
        generatePerformanceResults(configs, target, xdlops, paths)
    elif parsed_args.bmiopen_use_tuned_mlir:
        benchmarkMIOpenWithMLIRKernels(configs, xdlops, reportUtils.MIOPEN_TUNED_REPORT_FILE, paths)
    elif parsed_args.bmiopen_use_untuned_mlir:
        benchmarkMIOpenWithMLIRKernels(configs, xdlops, reportUtils.MIOPEN_UNTUNED_REPORT_FILE, paths)
    elif parsed_args.tuning:
        tuneMLIRKernels(configs, xdlops, paths)
    else:
        if parsed_args.b:
            df = pd.DataFrame(benchmarkMLIR(testVector.split(sep=' '), target, xdlops, paths) for testVector in configs)
        elif parsed_args.bmiopen:
            df = pd.DataFrame(benchmarkMIOpen(testVector.split(sep=' '), target, xdlops, paths) for testVector in configs)
        elif parsed_args.miopen:
            df = pd.DataFrame([benchmarkMIOpen(sys.argv[2:], xdlops, paths)])
        else:
            # Will only reach here with more than 1 unspecified arguments
            # These are arguments are directly passed through to benchmarkMLIR
            if not parsed_args.mlir_build_dir:
                raise RuntimeError("MLIR build dir was not provided/found")
            df = pd.DataFrame([benchmarkMLIR(sys.argv[1:], target, xdlops, paths)])
        df.to_csv(parsed_args.fileName)
        with pd.option_context('display.precision', reportUtils.ROUND_DIGITS):
            print(df) # for interactive consumption

if __name__ == '__main__':
    sys.exit(main())
