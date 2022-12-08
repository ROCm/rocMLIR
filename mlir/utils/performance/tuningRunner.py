#!/usr/bin/env python3

from perfCommonUtils import Operation
import getopt
import subprocess
import sys
from pathlib import Path
import argparse

from typing import Optional
import perfRunner
from perfRunner import PerfConfiguration
from perfRunner import ConvConfiguration
from perfRunner import GemmConfiguration
from perfRunner import Paths

import numpy as np
import pandas as pd

# global variables.
BENCHMARKING_RESULT_FILE_NAME = 'results.stats.csv'

#Run a gemm or conv config with --perf_config
def runMLIRWithPerfConfig(perf_config, config, paths: Paths, rocmlir_gen_flags, debug):
    config.setPerfConfig(perf_config.strip())
    rocmlirGenCommand = paths.mlir_paths.rocmlir_gen_path + ' -ph ' + config.generateMlirDriverCommandLine('')
    rocmlirDriverCommand = [paths.mlir_paths.rocmlir_driver_path, '--kernel-pipeline=applicability', '-']
    if debug:
        print(rocmlirGenCommand)

    p1 = subprocess.Popen(rocmlirGenCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(rocmlirDriverCommand, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p1.stdout.close()
    outs, errs = p2.communicate()
    if p2.returncode != 0:
        nanoSeconds = np.nan
        if debug:
            print(errs.decode('utf-8'))
    else:
        # run with applicable perf_config
        perfRunner.runConfigWithMLIR(config, paths, rocmlir_gen_flags, False)
        # get nanoseconds from rocprof output.
        nanoSeconds = perfRunner.getNanoSeconds(BENCHMARKING_RESULT_FILE_NAME)
        if debug:
            print(perf_config, 'takes', nanoSeconds, 'ns')
    return config.tableEntry(nanoSeconds)

#Tune MLIR Gemm or Convolution kernels
def tuneMLIRKernels(configs, confClass, paths: Paths, arch, rocmlir_gen_flags, debug):
    for testVector in configs:
        commandLine = testVector.split(sep=' ')
        config = confClass.fromCommandLine(commandLine, arch)
        config.MLIR_N_REPEATS=1
        commandLineOptions = config.generateMlirDriverCommandLine(rocmlir_gen_flags)
        rocmlirGenCommand = paths.mlir_paths.rocmlir_gen_path + ' -ph ' + commandLineOptions + ' --emit-tuning-space'
        args = rocmlirGenCommand.split()
        # get tuning space for this config
        p = subprocess.check_output(args).decode()
        tune_df = pd.DataFrame(runMLIRWithPerfConfig(perf_config.strip(), config, paths, rocmlir_gen_flags, debug) for perf_config in p.splitlines())
        if debug: 
            print(tune_df)
        print('Tuning', testVector, ':', tune_df['PerfConfig'][tune_df['TFlops'].idxmin()])

# Main function.
def main(args=None):
    """
    usage examples:

    python3 tuningRunner.py --op gemm -configs_file=../mlir/utils/performance/toy-gemm-configs 
    python3 tuningRunner.py --op gemm --config="-g 3 -m 1024 -k 769 -n 512 -t f32 -transA 0 -transB 0"
    python3 tuningRunner.py --op conv --config="conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1"

    """
    if args is None:
        args = sys.argv[1:]

    archNames = perfRunner.getArch()
    arch = ','.join(archNames)

    root_dir = str(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
    default_conv_configs = root_dir + '/mlir/utils/jenkins/miopen-tests/resnet50-miopen-configs'

    parser = argparse.ArgumentParser(
        prog="rocMLIR tuning runner",
        description="A script for tunning MLIR conv2d or gemm kernels",
        allow_abbrev=False,
    )

    parser.add_argument("--op", "--operation", choices=['conv', 'gemm'],
        default='conv',
        help="Operation for tuning")

    mutex_arg_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "-c", "--configs_file",
        type=str,
        default=default_conv_configs,
        help="File of configurations to test"
    )

    parser.add_argument(
        "--mlir-build-dir",
        type=str,
        default=perfRunner.find_mlir_build_dir(),
        help="The build directory of MLIR based kernel generator",
    )

    parser.add_argument(
        "--miopen-build-dir",
        type=str,
        default=perfRunner.find_miopen_build_dir(),
        help="The build directory of MIOpen",
    )

    parser.add_argument(
        "--config",
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
        "--debug", "-d",
        action='store_true',
        default=False,
        help="Print debug messages on failure or inapplicability")

    parsed_args = parser.parse_args(args)

    rocmlir_gen_flags = ''
    if 'rocmlir_gen_flags' in parsed_args:
        rocmlir_gen_flags = parsed_args.rocmlir_gen_flags

    # Impose default behavior when no args have been passed
    if len(args) == 0:
        parsed_args.batch_both = True

    confClass = PerfConfiguration
    opType = Operation.fromName(parsed_args.op)
    if opType == Operation.CONV:
        confClass = ConvConfiguration
    elif opType == Operation.GEMM:
        confClass = GemmConfiguration

    configs_path = None if parsed_args.config else parsed_args.configs_file
    paths = perfRunner.create_paths(configs_path, parsed_args.mlir_build_dir, parsed_args.miopen_build_dir)
    configs = []

    configs = parsed_args.config
    if not parsed_args.config:
        if opType == Operation.CONV:
           configs = perfRunner.getConvConfigurations(paths.configuration_file_path)
        elif opType == Operation.GEMM:
            configs = perfRunner.getGemmConfigurations(paths.configuration_file_path)
        else:
            raise RuntimeError("Tuning operation was not provided/found")

    if not paths.mlir_paths:
        raise RuntimeError("MLIR build dir was not provided/found")

    if parsed_args.config:
        configs = parsed_args.config

    tuneMLIRKernels(configs, confClass, paths, arch, rocmlir_gen_flags, parsed_args.debug)


if __name__ == '__main__':
    sys.exit(main())
