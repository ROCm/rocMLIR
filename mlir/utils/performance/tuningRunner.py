#!/usr/bin/env python3

from perfCommonUtils import Operation
from dataclasses import dataclass
import enum
import getopt
import os
import subprocess
import sys
from pathlib import Path
import argparse
import glob
import tempfile

from collections import OrderedDict
from typing import Optional
import perfRunner
from perfRunner import PerfConfiguration
from perfRunner import ConvConfiguration
from perfRunner import GemmConfiguration
from perfRunner import AttentionConfiguration
from perfRunner import Paths
from perfRunner import getChip
from perfCommonUtils import CORRECT_RESULT_RE
import reportUtils

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Options:
    debug: bool
    tuningSpaceKind: str
    quiet: bool
    arch: str
    numCU: int
    rocmlir_gen_flags: str
    verifyMode: str
    tflops: bool
    compact_print: bool

def verifyModeFlags(verifyMode: str) -> str:
    if verifyMode == "none":
        return ""
    if verifyMode == "cpu":
        return " -pv"
    if verifyMode == "gpu":
        return " -pv_with_gpu --verifier-keep-perf-config=false"
    raise ValueError("Unknown verification mode", verifyMode)

#Run a gemm or conv config and verify it
def verifyKernelWithPerfConfig(perfConfig, config, paths: Paths, options: Options) -> float:
    print(f"Verifying with perfConfig = {perfConfig}", file=sys.stderr)
    config.setPerfConfig(perfConfig.strip())
    rocmlirGenCommand = paths.mlir_paths.rocmlir_gen_path + \
        verifyModeFlags(options.verifyMode) + \
        ' -print-verify-results=summary ' + \
        config.generateMlirDriverCommandLine(options.rocmlir_gen_flags)
    rocmlirDriverCommand = [paths.mlir_paths.rocmlir_driver_path, '-c']
    mlirCpuRunnerArgs = ['-O2', f'--shared-libs={paths.mlir_paths.libmlir_rocm_runtime_path},{paths.mlir_paths.libconv_validation_wrappers_path},{paths.mlir_paths.libmlir_runtime_utils_path}', '--entry-point-result=void']
    profilerCommand = [perfRunner.ROCPROF, '--stats', paths.mlir_paths.cpu_runner_path] + mlirCpuRunnerArgs

    if options.debug:
        print(rocmlirGenCommand, file=sys.stderr)

    prevdir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.chdir(tmpdir)
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
                outs, errs = p3.communicate(timeout=600)
                outs = outs.decode('utf-8')
                if len(errs) > 0 or not CORRECT_RESULT_RE.search(outs):
                    print(f"""Verification failed:
Output = {outs}
Errors = {errs.decode('utf-8')}""", file=sys.stderr)
                    return np.nan
            except subprocess.TimeoutExpired:
                print("Verification timed out", file=sys.stderr)
                p3.kill()
                outs, errs = p3.communicate()
                return np.nan
            nanoSeconds = perfRunner.getNanoSeconds(perfRunner.BENCHMARKING_RESULT_FILE_NAME)
        finally:
            os.chdir(prevdir)
    return nanoSeconds

def getWinningConfig(tuningOutput, config, allData, options: Options):
    maxTFlops = -np.inf
    winningConfig = "None"
    for i, result in enumerate(tuningOutput):
        result = result.decode('utf-8').strip()
        if not options.quiet and not options.compact_print and i > 0 and i % 100 == 0:
            print(f"Tested {i} configs, best perf {maxTFlops} TFlops on perf_config {winningConfig}", file=sys.stderr)
        if options.debug:
            print(result, file=sys.stderr)
        # Time is in ns
        perfConfig, time = result.split('\t')
        if time == "N/A":
            nanoSeconds = np.nan
        else:
            nanoSeconds = float(time)

        config.setPerfConfig(perfConfig)
        entry = config.tableEntry(nanoSeconds)
        allData.append(entry)
        theseTFlops = entry['TFlops']
        if not np.isnan(theseTFlops) and theseTFlops > maxTFlops:
            maxTFlops = theseTFlops
            winningConfig = perfConfig
            if options.compact_print and not options.quiet:
                print(f"Tested {i} configs, best perf {maxTFlops} TFlops on perf_config {winningConfig}", file=sys.stderr)

    return winningConfig, maxTFlops

#Tune MLIR Gemm or Convolution kernels
def tuneMLIRKernels(configs, confClass, paths: Paths, options: Options):
    allData = []
    winners = {}
    for testVector in configs:
        if not testVector.endswith(".mlir"):
            commandLine = testVector.split(sep=' ')
            config = confClass.fromCommandLine(commandLine, options.arch, options.numCU)
            config.MLIR_N_REPEATS=1
            print("Tuning:", testVector, file=sys.stderr)
            commandLineOptions = config.generateMlirDriverCommandLine(options.rocmlir_gen_flags)
            # Note, we don't need the -ph, this goes to the tuning driver
            kernelGenCommand = paths.mlir_paths.rocmlir_gen_path + ' ' + commandLineOptions
            kernelGen = subprocess.Popen(kernelGenCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            tuningLoop = subprocess.Popen(
                [paths.mlir_paths.rocmlir_tuning_driver_path, f"--tuning-space={options.tuningSpaceKind}"],
                stdin=kernelGen.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            kernelGen.stdout.close()
        else:
            # pipe to rocmlir_gen --emit-tuning-key
            tuningKey = subprocess.Popen(
                [paths.mlir_paths.rocmlir_gen_path, '--emit-tuning-key', testVector],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            output, _ = tuningKey.communicate()
            result = output.decode('utf-8').strip().split('\t')
            print(f"Tuning:{result[2]} from {testVector}", file=sys.stderr)
            commandLine = result[2].split(sep=' ')
            config = confClass.fromCommandLine(commandLine, options.arch, options.numCU)
            tuningLoop = subprocess.Popen([paths.mlir_paths.rocmlir_tuning_driver_path, f"--tuning-space={options.tuningSpaceKind}", testVector],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Tune, printing progress as we go to avoid CI timeouts
        winningConfig, maxTFlops = getWinningConfig(tuningLoop.stdout, config, allData, options)

        if options.verifyMode != "none":
            verifyNs = verifyKernelWithPerfConfig(winningConfig, config, paths, options)
            if np.isnan(verifyNs):
                # Verification failed, abort the loop
                return None, None
            verifyTFlops = config.computeTFlops(verifyNs)
            print(f"Tuned and verified : {testVector} : {winningConfig} with {maxTFlops} TFlops and {verifyTFlops} on verification", file=sys.stderr)
            if options.verifyMode == "gpu":
                print("Note: Verify tflops counts verification kernel", file=sys.stderr)
        else:
            print(f"Tuned : {testVector} : {winningConfig} with {maxTFlops} TFlops", file=sys.stderr)
        if options.tflops:
            winners[testVector] = (winningConfig,maxTFlops)
        else:
            winners[testVector] = winningConfig
    allData = pd.DataFrame(allData)
    return winners, allData

#Extract gemm or conv configurations from fusion tests
def extractFusionConfigs(test_dir, paths: Paths, options: Options):
    allConfigs = []
    opType=Operation.FUSION
    for filename in glob.glob(test_dir + '/*mlir'):
        print("Extract from:", filename, file=sys.stderr)
        testEntry = perfRunner.getFusionTestInfo(filename, paths)
        if not testEntry:
            continue
        testVector = testEntry['testVector']
        if not testVector:
            continue
        # skip if the best config already exists
        if testVector in allConfigs:
            print("An entry already exists in the tuning DB", file=sys.stderr)
            continue
        commandLine = testVector.split(sep=' ')
        if commandLine[0].startswith('conv'):
            if opType == Operation.FUSION:
                opType = Operation.CONV
            elif opType != Operation.CONV:
                print("Invalid config op: ", testVector, file=sys.stderr)
                continue
        else:
            if opType == Operation.FUSION:
                opType = Operation.GEMM
            elif opType != Operation.GEMM:
                print("Invalid config op: ", testVector, file=sys.stderr)
                continue
        allConfigs.append(testVector)

    with open(paths.configuration_file_path, 'w') as outFile:
        for item in allConfigs:
            outFile.write("%s\n" % item)

    return opType

# Main function.
def main(args=None):
    """
    usage examples:

    python3 tuningRunner.py --op gemm -configs_file=../mlir/utils/performance/toy-gemm-configs --output=tuning_db.tsv
    python3 tuningRunner.py --op gemm --config="-g 3 -m 1024 -k 769 -n 512 -t f32 -transA 0 -transB 0"
    python3 tuningRunner.py --op conv --tuning-space=quick --config="conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1"
    python3 tuningRunner.py --op fusion -test_dir=../mlir/test/fusion/resnet50-e2e --output=tuning_db.tsv

    """
    if args is None:
        args = sys.argv[1:]

    archNames = perfRunner.getArch()
    arch = ','.join(archNames)
    numCU = perfRunner.getNumCU(perfRunner.getChip())
    root_dir = str(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
    default_conv_configs = root_dir + '/mlir/utils/jenkins/performance/conv-configs'

    parser = argparse.ArgumentParser(
        prog="rocMLIR tuning runner",
        description="A script for tunning MLIR conv2d or gemm kernels",
        allow_abbrev=False,
    )

    parser.add_argument("--op", "--operation", choices=['conv', 'gemm', 'fusion', 'attention'],
        default='conv',
        help="Operation for tuning")

    parser.add_argument(
        "-c", "--configs_file",
        type=str,
        default=default_conv_configs,
        help="File of configurations to test"
    )

    parser.add_argument("-o", "--output",
        type=str,
        default="tuning_results_local.tsv",
        help="File to output tuning results to. Will append to existing files")

    parser.add_argument(
        "--mlir-build-dir",
        type=str,
        default=perfRunner.find_mlir_build_dir(),
        help="The build directory of MLIR based kernel generator",
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

    parser.add_argument(
        "--tuning-space",
        default="exhaustive",
        choices=["quick", "full", "exhaustive"],
        help="Which space of tuning configs should be used while tuning"
    )
    parser.add_argument(
        "--quiet", "-q",
        action='store_true',
        default=False,
        help="Quiet mode (don't output each test result)")

    parser.add_argument("--verify-mode",
        default="gpu",
        choices=["none", "cpu", "gpu"],
        help="How to verify the winning tuned kernel")

    parser.add_argument("--test_dir",
        default="../mlir/test/fusion/resnet50-e2e",
        type=str,
        help="fusion E2E tests directory")

    parser.add_argument(
        '--data-type',
         nargs='+',
         choices=["f32", "f16", "i8", "i8_i32", "i8_i8", "fp8", "fp8_f32", "fp8_fp8"],
         default=["f32", "f16", "i8"],
         help='Force a set of datatypes'
    )

    parser.add_argument(
        "--tflops",
        action='store_true',
        default=False,
        help="Include the TFlops along with the winning perf-configs")

    parser.add_argument(
        "--compact-print",
        action='store_true',
        default=False,
        help="Print info only when a change happens")

    parsed_args = parser.parse_args(args)

    rocmlir_gen_flags = ''
    if 'rocmlir_gen_flags' in parsed_args:
        rocmlir_gen_flags = parsed_args.rocmlir_gen_flags

    opType = Operation.fromName(parsed_args.op)
    if opType == Operation.FUSION:
        configs_path = "./fusion_config_file"
    else:
        configs_path = None if parsed_args.config else parsed_args.configs_file
    paths = perfRunner.create_paths(configs_path, parsed_args.mlir_build_dir)

    if not paths.mlir_paths:
        raise RuntimeError("MLIR build dir was not provided/found")

    options = Options(arch=arch, numCU=numCU, debug=parsed_args.debug,
        quiet=parsed_args.quiet,
        tuningSpaceKind=parsed_args.tuning_space,
        rocmlir_gen_flags=rocmlir_gen_flags,
        verifyMode=parsed_args.verify_mode,
        tflops=parsed_args.tflops,
        compact_print=parsed_args.compact_print)

    if opType == Operation.FUSION:
        opType = extractFusionConfigs(parsed_args.test_dir, paths, options)

    confClass = PerfConfiguration
    if opType == Operation.CONV:
        confClass = ConvConfiguration
    elif opType == Operation.GEMM:
        confClass = GemmConfiguration
    elif opType == Operation.ATTENTION:
        confClass = AttentionConfiguration
    else:
        raise RuntimeError("Tuning operation was not provided/found")

    if parsed_args.config:
        configs = parsed_args.config
    elif opType == Operation.CONV:
        configs = perfRunner.getConvConfigurations(paths.configuration_file_path)
    elif opType == Operation.GEMM:
        datatypes, outputMap = perfRunner.parseDataTypes(parsed_args.data_type)
        configs = perfRunner.getGemmConfigurations(paths.configuration_file_path, datatypes, outputMap)
    elif opType == Operation.ATTENTION:
        configs = perfRunner.getAttentionConfigurations(paths.configuration_file_path)

    winners, allData = tuneMLIRKernels(configs, confClass, paths, options)

    if winners is None:
        # Tuning aborted, bail
        return

    if parsed_args.debug:
        print(allData, file=sys.stderr)
        allData.to_csv(f"{parsed_args.output}.debug", sep='\t')

    # Note, appending results here to allow multiple config sets
    if parsed_args.output == '-':
        outFile = sys.stdout
    else:
        outFile = open(parsed_args.output, 'a')

    with outFile:
        if parsed_args.tflops:
            print(f"# arch\tnumCUs\ttestVector\tperfConfig\tTFlops ({options.tuningSpaceKind})", file=outFile)
            for testVector, (perfConfig, tflops) in winners.items():
                print(f"Arch = {arch}({numCU} CUs), vector = '{testVector}', \
perfConfig = {perfConfig}, TFlops = {tflops}", file=sys.stderr)
                print(f"{arch}\t{numCU}\t{testVector}\t{perfConfig}\t{tflops}", file=outFile)
        else:
            print(f"# arch\tnumCUs\ttestVector\tperfConfig ({options.tuningSpaceKind})", file=outFile)
            for testVector, perfConfig in winners.items():
                print(f"Arch = {arch}({numCU} CUs), vector = '{testVector}', perfConfig = {perfConfig}", file=sys.stderr)
                print(f"{arch}\t{numCU}\t{testVector}\t{perfConfig}", file=outFile)

if __name__ == '__main__':
    sys.exit(main())
