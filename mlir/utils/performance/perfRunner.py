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
from hip import hip

import reportUtils
from perfCommonUtils import Operation, GEMMLibrary

# global variables.
ROCPROF = '/opt/rocm/bin/rocprofv3'
MIOPENDRIVER = '/opt/rocm/bin/MIOpenDriver'
BENCHMARKING_RESULT_FILE_NAME = 'results'
BENCHMARKING_STATS_FILE_NAME = 'results_kernel_stats.csv'
BENCHMARKING_METRICS_FILE_NAME = 'results_counter_collection.csv'
ROCMLIR_INPUT_METRICS_FILE_NAME = 'rocmlir_metrics.txt'
DIRECTIONS = ['-F 1', '-F 2', '-F 4']
DATA_TYPES = ['conv', 'convfp16', 'convbfp16', 'convfp8', 'convint8']
LAYOUTS = ['NHWC', 'NCHW']

DATA_TYPES_GEMM = ['f32', 'f16', 'bf16', 'i8', 'fp8']
DATA_TYPES_ATTENTION_WMMA = ['i8', 'f16', 'bf16']
DATA_TYPES_ATTENTION_MFMA = ['i8', 'f32', 'f16', 'bf16']
DATA_TYPES_GEMM_GEMM = ['f32', 'f16', 'bf16']
DATA_TYPES_CONV_GEMM = ['f32', 'f16', 'bf16']
OUTPUT_DATA_TYPES_MAP = {
    'f32': 'f32',
    'f16': 'f16',
    'bf16': 'bf16',
    'i8': 'i32',
    'fp8': 'f32',
    'fp8_fp8': 'f32',
    'fp8_bf8': 'f32',
    'bf8_fp8': 'f32',
    'bf8_bf8': 'f32'
}
MLIR_N_REPEATS = 10
WARMUP_ITERATIONS = 1
SLEEP_US = 100

FILTER_LAYOUT_MAP = {'N': 'k', 'C': 'c', 'H': 'y', 'W': 'x', 'G': 'g', '0': '0', '1': '1'}
INPUT_LAYOUT_MAP = {'N': 'n', 'C': 'c', 'H': 'h', 'W': 'w', 'G': 'g', '0': '0', '1': '1'}
OUTPUT_LAYOUT_MAP = {'N': 'n', 'C': 'k', 'H': 'h', 'W': 'w', 'G': 'g', '0': '0', '1': '1'}

# Compiled regexp object used for extracting elapsed time from MIOpenDriver's output
ELAPSED_TIME_RE = re.compile(r"Elapsed: ([0-9\.]*) ms")
# Compiled regexp object used for extracting target chip from arch
GFX_CHIP_RE = re.compile(r"gfx[0-9a-z]+")
INFO_ARCH_NAME = re.compile(r"Name:\s*(.*)")
INFO_ARCH_CU = re.compile(r"Compute Unit:\s*(.*)")


def input_layouts(input_layout):
    return "".join(INPUT_LAYOUT_MAP[char] for char in input_layout)


def output_layouts(output_layout):
    return "".join(OUTPUT_LAYOUT_MAP[char] for char in output_layout)


def filter_layouts(filter_layout):
    return "".join(FILTER_LAYOUT_MAP[char] for char in filter_layout)


def inverse_output_layouts(output_layout):
    map = {v: k for k, v in OUTPUT_LAYOUT_MAP.items()}
    return "".join(map[char] for char in output_layout)


def inverse_input_layouts(input_layout):
    map = {v: k for k, v in INPUT_LAYOUT_MAP.items()}
    return "".join(map[char] for char in input_layout)


def inverse_filter_layouts(filter_layout):
    map = {v: k for k, v in FILTER_LAYOUT_MAP.items()}
    return "".join(map[char] for char in filter_layout)


# This map stores the header to flag mapping and a boolean value denoting
# if the flag require a value
DEBUG_HEADER_TO_FLAG = {
    'C': '-c',
    'Causal': '-causal',
    'Chip': '--arch',
    'DataType': '-t',
    'DilationH': '--dilation_h',
    'DilationW': '--dilation_w',
    'Direction': '-F',
    'FilterLayout': '-f',
    'G': '-g',
    'H': '-h',
    'HeadDimQK': '-head_dim_qk',
    'HeadDimV': '-head_dim_v',
    'InputLayout': '-I',
    'K': '-k',
    'M': '-m',
    'N': '-n',
    'numCU': '--num_cu',
    'NumHeadsKV': '-num_heads_kv',
    'NumHeadsQ': '-num_heads_q',
    'O': '-gemmO',
    'OutDataType': '-out_datatype',
    'OutputLayout': '-O',
    'PaddingH': '--padding_h',
    'PaddingW': '--padding_w',
    'PerfConfig': '--perf_config',
    'ReturnLSE': 'return_lse',
    'SplitKV': '-split_kv',
    'SeqLenK': '-seq_len_k',
    'SeqLenQ': '-seq_len_q',
    'StrideH': '--conv_stride_h',
    'StrideW': '--conv_stride_w',
    'TransA': '-transA',
    'TransB': '-transB',
    'TransK': '-transK',
    'TransO': '-transO',
    'TransQ': '-transQ',
    'TransV': '-transV',
    'W': '-w',
    'WithAttnBias': '-with-attn-bias',
    'WithAttnScale': '-with-attn-scale',
    'X': '-x',
    'Y': '-y',
}


@dataclass
class MLIRPaths:
    rocmlir_gen_path: str
    rocmlir_driver_path: str
    rocmlir_opt_path: str
    cpu_runner_path: str
    libmlir_rocm_runtime_path: str
    libconv_validation_wrappers_path: str
    libmlir_runtime_utils_path: str
    libmlir_c_runner_utils_path: str
    rocmlir_tuning_driver_path: str
    rocblas_benchmark_driver_path: Optional[str] = None
    ck_gemm_benchmark_driver_path: Optional[str] = None


@dataclass
class Paths:
    """This structure is used to hold paths needed to perform the tests"""
    configuration_file_path: str
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
            search_root = str(
                subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
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


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def get_arch() -> str:
    agents = set()
    device_count = hip_check(hip.hipGetDeviceCount())
    for device in range(device_count):
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, device))
        agent = props.gcnArchName.decode('utf-8')
        agents.add(agent)
    if (len(agents) > 1):
        print(
            f"WARNING: Found {len(agents)} different kinds of agents on the same machine :  {', '.join(agents)}"
        )
        print(
            "WARNING: Using the first agent by default. If you want to use a different agent, please set the HIP_VISIBLE_DEVICES environment variable."
        )
    # select first agent by default
    return list(agents)[0]


def get_chip():
    arch = get_arch()
    chip = GFX_CHIP_RE.search(arch).group(0)
    return chip


DATA_TYPES_ATTENTION = None


def initialize_dtypes_attn():
    global DATA_TYPES_ATTENTION
    if get_chip().startswith('gfx9'):
        DATA_TYPES_ATTENTION = DATA_TYPES_ATTENTION_MFMA
    else:
        DATA_TYPES_ATTENTION = DATA_TYPES_ATTENTION_WMMA

    return DATA_TYPES_ATTENTION  # For modules that import this function


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
        mlir_paths = MLIRPaths(
            rocmlir_gen_path=mlir_bin_dir + '/rocmlir-gen',
            rocmlir_driver_path=mlir_bin_dir + '/rocmlir-driver',
            rocmlir_opt_path=mlir_bin_dir + '/rocmlir-opt',
            cpu_runner_path=llvm_bin_dir + '/mlir-runner',
            libmlir_rocm_runtime_path=llvm_lib_dir + '/libmlir_rocm_runtime.so',
            libconv_validation_wrappers_path=mlir_lib_dir + '/libconv-validation-wrappers.so',
            libmlir_runtime_utils_path=llvm_lib_dir + '/libmlir_runner_utils.so',
            libmlir_c_runner_utils_path=llvm_lib_dir + '/libmlir_c_runner_utils.so',
            rocmlir_tuning_driver_path=mlir_bin_dir + '/rocmlir-tuning-driver',
            rocblas_benchmark_driver_path=(str(rocblas_benchmark_driver_location)
                                           if rocblas_benchmark_driver_location.exists() else None),
            ck_gemm_benchmark_driver_path=(str(ck_gemm_benchmark_driver_location)
                                           if ck_gemm_benchmark_driver_location.exists() else None))

    return Paths(config_file_path, mlir_paths)


# utility functions.
def get_nanoseconds(filename):
    if not os.path.exists(filename):
        return np.nan
    with open(filename, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        result = 0
        for row in reader:
            result += int(float(row['AverageNs']))
        csv_file.close()
        return result


def get_profiler_output_path(arch: str, base_out_path):
    chip = GFX_CHIP_RE.search(arch).group(0)
    # TODO (gfx950): check if gfx950 need this
    if (chip not in ["gfx942"]):
        return os.path.join('pmc_1', base_out_path)
    return base_out_path


def get_metric_args_for_rocprof(arch: str):
    chip = GFX_CHIP_RE.search(arch).group(0)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(current_dir, ROCMLIR_INPUT_METRICS_FILE_NAME)
    metrics = []
    # TODO (gfx950): check if gfx950 supports this
    if (chip not in ["gfx942"]):
        metrics = ['-i', metrics_path]
    return metrics


# Bank conflict functions.The percentage of GPUTime LDS is stalled by bank
# conflicts. Value range: 0% (optimal) to 100% (bad).
def get_bank_conflict(filename):
    if not os.path.exists(filename):
        result = "NaN"
        return result
    with open(filename, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        header = reader.fieldnames
        if 'Counter_Name' not in header or 'Counter_Value' not in header:
            return np.nan

        result = []
        for row in reader:
            if row['Counter_Name'] == 'LDSBankConflict':
                result.append(float(row['Counter_Value']))
        csv_file.close()
        result_average = sum(result) / len(result)
        return result_average


def parse_debug_db_row(row) -> str:
    """
    Parses a row from the debug database and returns a formatted string.
    """
    # Before we start, we want to ensure that all of the values in
    # DEBUG_HEADER_TO_FLAG are up to date with what is in GEMM_TEST_PARAMETERS,
    # CONV_TEST_PARAMETERS and ATTN_TEST_PARAMETERS in reportUtils.py
    for param in getattr(reportUtils, "GEMM_TEST_PARAMETERS", []):
        assert param in DEBUG_HEADER_TO_FLAG, f"{param} missing in DEBUG_HEADER_TO_FLAG"
    for param in getattr(reportUtils, "ATTN_TEST_PARAMETERS", []):
        assert param in DEBUG_HEADER_TO_FLAG, f"{param} missing in DEBUG_HEADER_TO_FLAG"
    for param in getattr(reportUtils, "CONV_TEST_PARAMETERS", []):
        assert param in DEBUG_HEADER_TO_FLAG, f"{param} missing in DEBUG_HEADER_TO_FLAG"

    # Filter out Chip and numCU values as they are already accounted for and
    # we do not want to double count them
    args = []
    for key, value in row.items():
        if key in DEBUG_HEADER_TO_FLAG and (key != "Chip") and (key != "numCU"):
            args.extend([DEBUG_HEADER_TO_FLAG[key], str(value).lower()])

    # Filter out any empty strings and join with spaces
    result_str = " ".join(filter(None, args))
    return result_str


# Tuning debug databases
MaybeDebugDb = Optional[Dict[Tuple[str, str, str, str, str], str]]


def read_debug_db(path: str) -> MaybeDebugDb:
    try:
        df = pd.read_csv(path, sep='\t')
        ret = {}
        for _, row in df.iterrows():
            # If this was not a valid config, i.e., it did not generate a
            # TFLOPs value, then we can skip it
            if pd.isna(row.get('TFlops')) or row.get('TFlops') == '':
                continue

            # Extract the required fields
            arch = row['Chip']
            num_cu = str(row['numCU'])
            perf_config = row['PerfConfig']
            tflops = row['TFlops']
            configs = parse_debug_db_row(row)
            ret[(arch, num_cu, configs, perf_config, tflops)] = row

        return ret
    except FileNotFoundError:
        if path:
            print("Warning: Failed to find tuning debug database:", path)
        return None
    except Exception as e:
        print(f"Error reading tuning debug database: {e}")
        return None


# Tuning databases
MaybeTuningDb = Optional[Dict[Tuple[str, str, str], str]]


def read_tuning_db(path: [str]) -> MaybeTuningDb:
    try:
        ret = {}
        with open(path, 'r') as db_file:
            for line in db_file:
                line = line.strip()
                if line.startswith('#'):
                    continue
                entries = line.split('\t')

                # note: legacy format has 3 entries
                if len(entries) == 3:
                    arch, config, perf_config = entries
                    ret[arch, None, config] = perf_config
                # note: new format has 4 entries
                elif len(entries) == 4:
                    arch, num_cu, config, perf_config = entries
                    ret[arch, num_cu, config] = perf_config
                # note: 5-entry form includes tflops at end
                elif len(entries) == 5:
                    arch, num_cu, config, perf_config, _ = entries
                    ret[arch, num_cu, config] = perf_config
                else:
                    print("Warning: Malformed tuning database entry:", line)
                    continue
        return ret
    except FileNotFoundError:
        if path:
            print("Warning: Failed to find tuning database:", path)
        return None


def get_miliseconds(output):
    result = re.search(r"kernel time: (.*)", output.decode("utf-8"))
    if not result:
        return float('NaN')

    return float(result.group(1))


def run_pipeline(proc_specs):
    procs = []
    for proc in proc_specs:
        prev_stdout = procs[-1].stdout if procs else subprocess.DEVNULL
        po = subprocess.Popen(proc,
                              stdin=prev_stdout,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        procs.append(po)
    try:
        # Close intermediate stdout pipes
        for p in procs[:-1]:
            if p.stdout:
                p.stdout.close()

        # Wait for the last process to finish and collect its output
        outs, errs = procs[-1].communicate()
        if procs[-1].returncode != 0:
            raise OSError(str(procs[-1].stderr.read()))

        # Now check all processes for errors
        for i, p in enumerate(procs):
            if p.returncode is None:
                p.wait()
            if p.returncode != 0:
                raise OSError(str(p.stderr.read()))

        return outs, errs
    except Exception as err:
        print(f"Error:  {err}")
        print(f"Failing command:  {' '.join(p.args)}")
        print(f"Failing pipeline:  {' | '.join([' '.join(proc) for proc in proc_specs])}")
        outs, errs = p.communicate()
    return outs, False


class PerfConfiguration:
    TABLE_COLUMNS = []

    def get_total_flops(self):
        raise NotImplementedError()

    def compute_ns_from_tflops(self, tflops):
        """
        Calculate nanoseconds from TFlops value.
        This is the inverse of compute_tflops().

        Args:
            tflops: TFlops value to convert to nanoseconds

        Returns:
            float: Time in nanoseconds
        """
        if tflops == 0 or np.isnan(tflops) or np.isinf(tflops):
            return np.nan

        total_flops = self.get_total_flops()
        return total_flops / (tflops * 1e3)

    def compute_tflops(self, ns: int) -> float:
        raise NotImplementedError()

    def table_entry(self, nanoseconds):
        raise NotImplementedError()

    def generate_mlir_driver_commandline(self, rocmlir_gen_flags):
        raise NotImplementedError()

    def set_perfconfig(self, perf_config):
        raise NotImplementedError()

    @classmethod
    def from_command_line(cls, argv, arch, num_cu):
        raise NotImplementedError()

    def to_command_line(self):
        raise NotImplementedError()

    @classmethod
    def benchmark_external(cls, commandline, paths: Paths, arch, num_cu):
        raise NotImplementedError()

    EXTERNAL_NAME = "unknown"

    def __repr__(self):
        attrs = ', '.join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


# convolution configurations.
def get_conv_configurations(filename):
    configs = []
    if filename:
        with open(filename, 'r') as config_file:
            lines = config_file.readlines()
            # All combinations of conv direction, type and layouts
            for direction, datatype, layout, line in \
                    itertools.product(DIRECTIONS, DATA_TYPES, LAYOUTS, lines):
                line = line.strip()

                # Skip empty lines
                if len(line) == 0 or line[0] == '#':
                    continue

                # Skip unsupported datatypes
                if datatype == 'convfp8':
                    unsupported_chips = {'gfx908', 'gfx90a', 'gfx942', 'gfx1030', 'gfx1101'}
                    if get_chip() in unsupported_chips:
                        continue

                # Skip int8 non-fwd convolutions
                if (datatype == 'convint8' or datatype == 'convfp8') and direction != '-F 1':
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

                one_config = f"{datatype}{direction}{filter_layout}{input_layout}{output_layout}{line}"
                if one_config not in configs:
                    configs.append(one_config)
    return configs


class ConvConfiguration(PerfConfiguration):
    TABLE_COLUMNS = reportUtils.CONV_TEST_PARAMETERS + ['LDSBankConflict'] + ['TFlops']
    EXTERNAL_NAME = "MIOpen"

    def get_total_flops(self):
        return (2.0 * self.n * (self.c // self.group) * self.k * self.ho * self.wo * self.y * self.x)

    def compute_tflops(self, ns):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        assert (self.k % self.group == 0)
        assert (self.c % self.group == 0)
        return self.get_total_flops() / (float(ns) * 1e-9) / 1e12

    def table_entry(self, nanoseconds):
        # Future(kdrewnia): This can just be a dict literal on Python 3.7+
        bank_conflict = get_bank_conflict(
            get_profiler_output_path(self.arch, BENCHMARKING_METRICS_FILE_NAME))
        result = OrderedDict()
        values = [
            self.direction, self.datatype, self.chip, self.num_cu, self.filter_layout,
            self.input_layout, self.output_layout, self.n, self.c, self.hi, self.wi, self.k, self.y,
            self.x, self.dilation_h, self.dilation_w, self.conv_stride_h, self.conv_stride_w,
            self.padding_h, self.padding_w, self.perfconfig, bank_conflict,
            self.compute_tflops(nanoseconds)
        ]
        assert (len(self.TABLE_COLUMNS) == len(values))

        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def set_perfconfig(self, perf_config):
        self.perfconfig = perf_config

    def generate_mlir_driver_commandline(self, rocmlir_gen_flags):
        direction = {
            'fwd': '--operation conv',
            'bwd': '--operation conv_bwd_data',
            'wrw': '--operation conv_bwd_weight'
        }[self.direction]

        result = ' '.join([
            direction, '-t', self.datatype, '--arch', self.arch, '--num_cu',
            str(self.num_cu), '--fil_layout', self.filter_layout, '--in_layout', self.input_layout,
            '--out_layout', self.output_layout, '--batchsize',
            str(self.n), '--in_channels',
            str(self.c), '--in_h',
            str(self.hi), '--in_w',
            str(self.wi), '--out_channels',
            str(self.k), '--fil_h',
            str(self.y), '--fil_w',
            str(self.x), '--dilation_h',
            str(self.dilation_h), '--dilation_w',
            str(self.dilation_w), '--conv_stride_h',
            str(self.conv_stride_h), '--conv_stride_w',
            str(self.conv_stride_w), '--padding_h',
            str(self.padding_h), '--padding_w',
            str(self.padding_w), '--groupsize',
            str(self.group), '--kernel-repeats',
            str(MLIR_N_REPEATS), f"--perf_config={self.perfconfig}"
        ])
        result += ' '
        if rocmlir_gen_flags != '':
            result += ' '.join(rocmlir_gen_flags.split())
        return result

    @classmethod
    def from_command_line(cls, argv, arch, num_cu):
        # Determine if argv[0] is an operation type or a flag
        # Please keep this in sync with mlir::rock::getTuningProblemStr()
        datatype = None

        # Check if argv[0] is an operation type (e.g., 'conv', 'convfp16', etc.)
        if argv[0] == 'conv':
            datatype = 'f32'
        elif argv[0] == 'convfp16':
            datatype = 'f16'
        elif argv[0] == 'convbfp16':
            datatype = 'bf16'
        elif argv[0] == 'convint8':
            datatype = 'i8'
        elif argv[0] == 'convfp8_fp8':
            datatype = 'fp8_fp8'
        elif argv[0] == 'convfp8':
            datatype = 'fp8'
        elif argv[0] == 'convfp8_bf8':
            datatype = 'fp8_bf8'
        elif argv[0] == 'convbf8_fp8':
            datatype = 'bf8_fp8'
        elif argv[0] == 'convbf8_bf8':
            datatype = 'bf8_bf8'

        # If datatype was determined from operation type, skip argv[0] when parsing options
        args_start = 1 if datatype is not None else 0

        try:
            # TBD:
            # implement -m ?
            # Short opts: include both uppercase and lowercase H/W
            short_opts = "F:f:I:O:n:c:H:W:h:w:k:y:x:p:q:l:j:u:v:g:m:t:"
            long_opts = [
                "dilation_h=", "dilation_w=",
                "conv_stride_h=", "conv_stride_w=",
                "padding_h=", "padding_w=",
                "perf_config="
            ]
            opts, _ = getopt.getopt(argv[args_start:], short_opts, long_opts)
        except getopt.GetoptError:
            print('getopt error')
            sys.exit(1)

        # Default setting of num groups to 1
        group = 1

        for opt, arg in opts:
            if opt == '-F':
                # Accept either numeric codes (1/2/4) or strings (fwd/bwd/wrw)
                val = arg.lower()
                num_map = {'1': 'fwd', '2': 'bwd', '4': 'wrw'}
                str_list = ['fwd', 'bwd', 'wrw']
                if val in num_map:
                    direction = num_map[val]
                elif val in str_list:
                    direction = val
                else:
                    raise ValueError(f"Invalid -F argument (expected 1/2/4 or fwd/bwd/wrw): {arg}")
            elif opt == '-f':
                filter_layout = arg
            elif opt == '-I':
                input_layout = arg
            elif opt == '-O':
                output_layout = arg
            elif opt == "-n":
                n = int(arg)
            elif opt == '-c':
                c = int(arg)
            elif opt == '-H' or opt == "-h":
                hi = int(arg)
            elif opt == '-W' or opt == "-w":
                wi = int(arg)
            elif opt == '-k':
                k = int(arg)
            elif opt == '-y':
                y = int(arg)
            elif opt == '-x':
                x = int(arg)
            elif opt == '-u' or opt == '--conv_stride_h':
                conv_stride_h = int(arg)
            elif opt == '-v' or opt == '--conv_stride_w':
                conv_stride_w = int(arg)
            elif opt == '-p' or opt == '--padding_h':
                padding_h = int(arg)
            elif opt == '-q' or opt == '--padding_w':
                padding_w = int(arg)
            elif opt == '-l' or opt == '--dilation_h':
                dilation_h = int(arg)
            elif opt == '-j' or opt == '--dilation_w':
                dilation_w = int(arg)
            elif opt == '-g':
                group = int(arg)
            elif opt == '-t' and datatype is None:
                datatype = arg
            else:
                continue

        return cls(datatype, direction, filter_layout, input_layout, output_layout, n, c, hi, wi, k,
                   y, x, conv_stride_h, conv_stride_w, padding_h, padding_w, dilation_h, dilation_w,
                   group, arch, num_cu)

    def to_command_line(self):
        return (
            f"conv{ {'f32':'', 'f16':'fp16', 'bf16':'bfp16', 'i8':'int8','fp8_fp8':'fp8_fp8', 'fp8': 'fp8'}[self.datatype]} "
            + f"-F { {'fwd':1, 'bwd':2, 'wrw':4}[self.direction]} " +
            f"-f {inverse_filter_layouts(self.filter_layout)} -I {self.input_layout.upper()} " +
            f"-O {inverse_output_layouts(self.output_layout)} " +
            f"-n {self.n} -c {self.c} -H {self.hi} -W {self.wi} -k {self.k} " +
            f"-y {self.y} -x {self.x} -p {self.padding_h} -q {self.padding_w} " +
            f"-u {self.conv_stride_h} -v {self.conv_stride_w} -l {self.dilation_h} " +
            f"-j {self.dilation_w} -m conv -g {self.group} -t 1")

    def __init__(self, dtype: str, direction: str, filter_layout: str, input_layout: str,
                 output_layout: str, n: int, c: int, hi: int, wi: int, k: int, y: int, x: int,
                 conv_stride_h: int, conv_stride_w: int, padding_h: int, padding_w: int,
                 dilation_h: int, dilation_w: int, group: int, arch: str, num_cu: int):
        if dtype not in {"f16", "f32", "bf16", "i8", "fp8_fp8", "fp8"}:
            raise ValueError(f"Invalid datatype: {dtype}")
        if direction not in {"fwd", "bwd", "wrw"}:
            raise ValueError(f"Invalid direction: {direction}")

        self.datatype = dtype
        self.direction = direction

        # Only translate if original string is all uppercase; else assume already translated/lowered.
        self.filter_layout = filter_layouts(filter_layout) if filter_layout.isupper() else filter_layout
        self.input_layout = input_layouts(input_layout) if input_layout.isupper() else input_layout
        self.output_layout = output_layouts(output_layout) if output_layout.isupper() else output_layout

        self.n = n
        self.c = c
        self.hi = hi
        self.wi = wi
        self.k = k
        self.y = y
        self.x = x

        self.conv_stride_h = conv_stride_h
        self.conv_stride_w = conv_stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w

        self.group = group
        self.arch = arch
        self.num_cu = num_cu
        self.chip = GFX_CHIP_RE.search(arch).group(0)

        self.ho = math.floor((self.hi + self.padding_h * 2 -
                              (self.y - 1) * self.dilation_h - 1) / self.conv_stride_h) + 1
        self.wo = math.floor((self.wi + self.padding_w * 2 -
                              (self.x - 1) * self.dilation_w - 1) / self.conv_stride_w) + 1

        self.perfconfig = ''

    @classmethod
    def benchmark_external(cls, commandline, paths: Paths, arch, num_cu):
        if os.path.exists(get_profiler_output_path(arch, BENCHMARKING_METRICS_FILE_NAME)):
            os.remove(get_profiler_output_path(arch, BENCHMARKING_METRICS_FILE_NAME))
        config = cls.from_command_line(commandline, arch, num_cu)
        miopen_driver_cmd = [MIOPENDRIVER, *commandline, '-V', '0', '-t', '1']
        print("Running MIOpen Benchmark: ", ' '.join(commandline))
        # invoke MIOpenDriver.
        outs, noerr = run_pipeline([miopen_driver_cmd])
        nanoseconds = np.nan
        if noerr:
            # convert bytes to str
            outs = outs.decode('utf-8')
            # Extract Elapsed time in ms from the output of MIOpenDriver
            # Use regular expression to match the contents between
            # "Elasped: " (note the space at the end) and "ms"
            elapsed_time_in_ms = ELAPSED_TIME_RE.search(outs).group(1)
            nanoseconds = float(elapsed_time_in_ms) * 1.0e6

        return config.table_entry(nanoseconds)


def get_gemm_configurations(filename,
                            datatypes=DATA_TYPES_GEMM,
                            out_dtype_map=OUTPUT_DATA_TYPES_MAP):
    configs = []

    if filename:
        with open(filename, 'r') as config_file:
            lines = config_file.readlines()

            # All combinations of types and transposition (A and B)
            for datatype, trans_a, trans_b, line in \
                    itertools.product(DATA_TYPES_GEMM, ['false', 'true'], ['false', 'true'], lines):
                line = line.strip()

                # Skip empty lines
                if len(line) == 0 or line[0] == '#':
                    continue
                if datatype not in datatypes:
                    continue

                # Skip unsupported datatypes
                if datatype == 'fp8':
                    unsupported_chips = {'gfx908', 'gfx90a', 'gfx942', 'gfx1030', 'gfx1101'}
                    if get_chip() in unsupported_chips:
                        continue

                # We need trailing spaces here to account for the concat below
                # Skip type if already in
                datatype_string = ""
                if "-t " not in line:
                    datatype_string = f"-t {datatype} "

                # Skip trans_a if already in
                trans_a_string = ""
                if "-transA " not in line:
                    trans_a_string = f"-transA {trans_a} "

                # Skip trans_b if already in
                trans_b_string = ""
                if "-transB " not in line:
                    trans_b_string = f"-transB {trans_b} "

                # Skip out_datatype if already in
                out_dtype_string = ""
                if "-out_datatype" not in line:
                    out_dtype_string = "-out_datatype " + out_dtype_map.get(datatype,
                                                                            datatype) + " "

                # Strip to avoid spurious spaces
                one_config = f"{datatype_string}{out_dtype_string}{trans_a_string}{trans_b_string}{line}".strip(
                )
                if one_config not in configs:
                    configs.append(one_config)
    return configs


def get_conv_gemm_configurations(filename):
    bool_space = ['false', 'true']
    default_test_space = {
        "-t": DATA_TYPES_CONV_GEMM,
        "-f": LAYOUTS,
        "-I": LAYOUTS,
        "-transC": bool_space,
        "-transO": bool_space,
    }
    configs = []
    if filename:
        with open(filename, 'r') as config_file:
            lines = config_file.readlines()
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if len(line) == 0 or line[0] == '#':
                    continue
                test_space = []
                args = []
                for arg in default_test_space.keys():
                    """
                    Next condition checks if a flag is not present in the line. Check with re.search(...)
                    ensures flags are matched exactly and not as substring.

                    - (?<!\S) ensures that flag is not part of another token (e.g. that -t is not part of -transQ)
                    - (?!\S) ensures that flag is followed by a space or line end.
                    - re.escape(arg) ensures that flag, in case it contains special character(s), is matched as it is.
                    """
                    if not re.search(rf"(?<!\S){re.escape(arg)}(?!\S)", line):
                        test_space.append(default_test_space[arg])
                        args.append(arg)
                for test_vector in itertools.product(*test_space):
                    # Strip to avoid spurious spaces
                    one_config = line.strip()
                    for arg, value in zip(args, test_vector):
                        one_config = f"{arg} {value} {one_config}"
                    if one_config not in configs:
                        configs.append(one_config)
    return configs


def get_gemm_gemm_configurations(filename):
    bool_space = ['false', 'true']
    default_test_space = {
        "-t": DATA_TYPES_GEMM_GEMM,
        "-transA": bool_space,
        "-transB": bool_space,
        "-transC": bool_space,
        "-transO": bool_space,
    }
    configs = []
    if filename:
        with open(filename, 'r') as config_file:
            lines = config_file.readlines()
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if len(line) == 0 or line[0] == '#':
                    continue
                test_space = []
                args = []
                for arg in default_test_space.keys():
                    """
                    Next condition checks if a flag is not present in the line. Check with re.search(...)
                    ensures flags are matched exactly and not as substring.

                    - (?<!\S) ensures that flag is not part of another token (e.g. that -t is not part of -transQ)
                    - (?!\S) ensures that flag is followed by a space or line end.
                    - re.escape(arg) ensures that flag, in case it contains special character(s), is matched as it is.
                    """
                    if not re.search(rf"(?<!\S){re.escape(arg)}(?!\S)", line):
                        test_space.append(default_test_space[arg])
                        args.append(arg)
                for test_vector in itertools.product(*test_space):
                    # Strip to avoid spurious spaces
                    one_config = line.strip()
                    for arg, value in zip(args, test_vector):
                        one_config = f"{arg} {value} {one_config}"
                    if one_config not in configs:
                        configs.append(one_config)
    return configs


def get_attn_configurations(filename):
    if DATA_TYPES_ATTENTION is None:
        initialize_dtypes_attn()
    bool_space = ['false', 'true']
    default_test_space = {
        "-t": DATA_TYPES_ATTENTION,
        "-transQ": bool_space,
        "-transK": bool_space,
        "-transV": bool_space,
        "-transO": bool_space,
        "-causal": bool_space,
        "-return_lse": bool_space,
        "-with-attn-scale": bool_space,
        "-with-attn-bias": bool_space
    }

    configs = []
    if filename:
        with open(filename, 'r') as config_file:
            lines = config_file.readlines()
            for line in lines:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue

                test_space = []
                args = []
                for arg in default_test_space.keys():
                    """
                    Next condition checks if a flag is not present in the line. Check with re.search(...)
                    ensures flags are matched exactly and not as substring.

                    - (?<!\S) ensures that flag is not part of another token (e.g. that -t is not part of -transQ)
                    - (?!\S) ensures that flag is followed by a space or line end.
                    - re.escape(arg) ensures that flag, in case it contains special character(s), is matched as it is.
                    """
                    if not re.search(rf"(?<!\S){re.escape(arg)}(?!\S)", line):
                        test_space.append(default_test_space[arg])
                        args.append(arg)

                for test_vector in itertools.product(*test_space):
                    # Strip to avoid spurious spaces
                    one_config = line.strip()
                    for arg, value in zip(args, test_vector):
                        one_config = f"{arg} {value} {one_config}"

                    # Check for valid dtypes
                    found_dtype = re.search(r"-t\s+(\w+)", one_config)
                    if not found_dtype or found_dtype.group(1) not in DATA_TYPES_ATTENTION:
                        continue

                    if one_config not in configs:
                        configs.append(one_config)

    return configs


class GemmConfiguration(PerfConfiguration):
    TABLE_COLUMNS = reportUtils.GEMM_TEST_PARAMETERS + ['LDSBankConflict'] + ['TFlops']

    def get_total_flops(self):
        return 2.0 * self.g * self.m * self.k * self.n

    def compute_tflops(self, ns):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        return self.get_total_flops() / (float(ns) * 1e-9) / 1e12

    def table_entry(self, nanoseconds):
        # Future(kdrewnia): This can just be a dict literal on Python 3.7+
        bank_conflict = get_bank_conflict(
            get_profiler_output_path(self.arch, BENCHMARKING_METRICS_FILE_NAME))
        result = OrderedDict()
        values = [
            self.datatype, self.out_dtype, self.chip, self.num_cu, self.trans_a, self.trans_b,
            self.g, self.m, self.k, self.n, self.perfconfig, bank_conflict,
            self.compute_tflops(nanoseconds)
        ]
        assert (len(self.TABLE_COLUMNS) == len(values))

        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def set_perfconfig(self, perf_config):
        self.perfconfig = perf_config

    def generate_mlir_driver_commandline(self, rocmlir_gen_flags):
        result = ' '.join([
            '-operation', 'gemm', '-t', self.datatype, '-out_datatype', self.out_dtype, '--arch',
            self.arch, '--num_cu',
            str(self.num_cu), '-g',
            str(self.g), '-m',
            str(self.m), '-k',
            str(self.k), '-n',
            str(self.n), f"-transA={self.trans_a}", f"-transB={self.trans_b}", '--kernel-repeats',
            str(MLIR_N_REPEATS), f"--perf_config={self.perfconfig}"
        ])

        result += ' '
        if rocmlir_gen_flags != '':
            result += ' '.join(rocmlir_gen_flags.split())
        return result

    @classmethod
    def from_command_line(cls, argv, arch, num_cu):
        # Please keep this in sync with mlir::rock::getTuningProblemStr()
        dtype = None
        g = None
        m = None
        k = None
        n = None
        trans_a = None
        trans_b = None
        out_dtype = None
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
                trans_a = (val.lower() in ["1", "true"])
            elif opt.endswith("-transB"):
                trans_b = (val.lower() in ["1", "true"])
            elif opt.endswith("-out_datatype"):
                out_dtype = val.lower()
            elif opt.endswith("-perf_config"):
                perf_config = val
            else:
                raise ValueError(f"Unknown GEMM config argument {opt} -> {val}")
        for v in [dtype, out_dtype, g, m, k, n, trans_a, trans_b]:
            if v is None:
                raise ValueError("Incomplete GEMM configuration")

        return cls(dtype, out_dtype, g, m, k, n, trans_a, trans_b, arch, num_cu, perf_config)

    def to_command_line(self):
        return (f"-t {self.datatype} -out_datatype {self.out_dtype} " +
                f"-transA {str(self.trans_a).lower()} -transB {str(self.trans_b).lower()} " +
                f"-g {self.g} -m {self.m} -n {self.n} -k {self.k}")

    def __init__(self,
                 dtype: str,
                 out_dtype: str,
                 g: int,
                 m: int,
                 k: int,
                 n: int,
                 trans_a: bool,
                 trans_b: bool,
                 arch: str,
                 num_cu: int,
                 perf_config: str = ''):
        if dtype not in DATA_TYPES_GEMM:
            raise ValueError(f"Invalid datatype: {dtype}")

        self.datatype = dtype
        self.out_dtype = out_dtype
        self.g = g
        self.m = m
        self.k = k
        self.n = n
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.perfconfig = perf_config

        self.arch = arch
        self.chip = GFX_CHIP_RE.search(arch).group(0)
        self.num_cu = num_cu


class ConvGemmConfiguration(PerfConfiguration):
    TABLE_COLUMNS = reportUtils.CONV_GEMM_TEST_PARAMETERS + ['TFlops']

    def __init__(self,
                 dtype: str,
                 filter_layout: str,
                 input_layout: str,
                 trans_c: bool,
                 trans_o: bool,
                 n: int,
                 c: int,
                 hi: int,
                 wi: int,
                 k: int,
                 y: int,
                 x: int,
                 o: int,
                 conv_stride_h: int,
                 conv_stride_w: int,
                 padding_h: int,
                 padding_w: int,
                 dilation_h: int,
                 dilation_w: int,
                 group: int,
                 arch: str,
                 num_cu: int,
                 perf_config: str = ''):
        if dtype not in DATA_TYPES_CONV_GEMM:
            raise ValueError(f"Invalid datatype for a: {dtype}")

        self.datatype = dtype

        self.filter_layout = filter_layouts(filter_layout)
        self.input_layout = input_layouts(input_layout)
        self.trans_c = trans_c
        self.trans_o = trans_o

        self.n = n
        self.c = c
        self.hi = hi
        self.wi = wi
        self.k = k
        self.y = y
        self.x = x
        self.o = o

        self.conv_stride_h = conv_stride_h
        self.conv_stride_w = conv_stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w

        self.group = group
        self.arch = arch
        self.chip = GFX_CHIP_RE.search(arch).group(0)
        self.num_cu = num_cu
        self.perfconfig = perf_config

        self.ho = math.floor((self.hi + self.padding_h * 2 -
                              (self.y - 1) * self.dilation_h - 1) / self.conv_stride_h) + 1
        self.wo = math.floor((self.wi + self.padding_w * 2 -
                              (self.x - 1) * self.dilation_w - 1) / self.conv_stride_w) + 1

    def get_total_flops(self):
        first_conv_flops = 2.0 * self.n * (self.c // self.group) * self.k * self.ho * self.wo * self.y * self.x
        first_gemm_m = self.k
        first_gemm_n = self.n * self.ho * self.wo
        batch_second_gemm = 1.0
        second_matmul_flops = 2.0 * batch_second_gemm * first_gemm_m * first_gemm_n * self.o
        return first_conv_flops + second_matmul_flops

    def compute_tflops(self, ns):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        assert (self.k % self.group == 0)
        assert (self.c % self.group == 0)

        total_flops = self.get_total_flops()

        return total_flops / (float(ns) * 1e-9) / 1e12

    def table_entry(self, nanoseconds):
        result = {}
        values = [
            self.datatype, self.chip, self.num_cu, self.filter_layout, self.input_layout,
            self.trans_c, self.trans_o, self.n, self.c, self.hi, self.wi, self.k, self.y, self.x,
            self.o, self.dilation_h, self.dilation_w, self.conv_stride_h, self.conv_stride_w,
            self.padding_h, self.padding_w, self.perfconfig,
            self.compute_tflops(nanoseconds)
        ]
        assert (len(self.TABLE_COLUMNS) == len(values))
        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def set_perfconfig(self, perf_config):
        self.perfconfig = perf_config

    def generate_mlir_driver_commandline(self, rocmlir_gen_flags):
        result = ' '.join([
            '-operation', 'conv_gemm', '-t', self.datatype, '--arch', self.arch,
            f'--num_cu={self.num_cu}', f'--fil_layout={self.filter_layout}',
            f'--in_layout={self.input_layout}', f'--transC={self.trans_c}',
            f'--transO={self.trans_o}', f'--batchsize={self.n}', f'--in_channels={self.c}',
            f'--in_h={self.hi}', f'--in_w={self.wi}', f'--out_channels={self.k}',
            f'--fil_h={self.y}', f'--fil_w={self.x}', f'--dilation_h={self.dilation_h}',
            f'--dilation_w={self.dilation_w}', f'--conv_stride_h={self.conv_stride_h}',
            f'--conv_stride_w={self.conv_stride_w}', f'--padding_h={self.padding_h}',
            f'--padding_w={self.padding_w}', f'--groupsize={self.group}', f'--gemmO={self.o}',
            f'--kernel-repeats={MLIR_N_REPEATS}', f"--perf_config={self.perfconfig}"
        ])
        result += ' '
        if rocmlir_gen_flags != '':
            result += ' '.join(rocmlir_gen_flags.split())
        return result

    @classmethod
    def from_command_line(cls, argv, arch, num_cu):
        # optional defaults
        perf_config = ''
        dtype = None
        n = None
        c = None
        hi = None
        wi = None
        k = None
        y = None
        x = None
        o = None
        conv_stride_h = None
        conv_stride_w = None
        padding_h = None
        padding_w = None
        dilation_h = None
        dilation_w = None
        group = None
        filter_layout = None
        input_layout = None
        trans_c = False
        trans_o = False
        # Please keep this in sync with mlir::rock::getTuningProblemStr()
        for i in range(0, len(argv), 2):
            opt = argv[i]
            val = argv[i + 1]
            if opt.endswith("-t"):
                dtype = val
            elif opt.endswith("-n"):
                n = int(val)
            elif opt.endswith("-c"):
                c = int(val)
            elif opt.endswith("-H"):
                hi = int(val)
            elif opt.endswith("-W"):
                wi = int(val)
            elif opt.endswith("-k"):
                k = int(val)
            elif opt.endswith("-y"):
                y = int(val)
            elif opt.endswith("-x"):
                x = int(val)
            elif opt.endswith("-gemmO"):
                o = int(val)
            elif opt == '-u':
                conv_stride_h = int(val)
            elif opt == '-v':
                conv_stride_w = int(val)
            elif opt == '-p':
                padding_h = int(val)
            elif opt == '-q':
                padding_w = int(val)
            elif opt == '-l':
                dilation_h = int(val)
            elif opt == '-j':
                dilation_w = int(val)
            elif opt == '-g':
                group = int(val)
            elif opt == '-f':
                filter_layout = val
            elif opt == '-I':
                input_layout = val
            elif opt.endswith("-transC"):
                trans_c = (val.lower() in ["1", "true"])
            elif opt.endswith("-transO"):
                trans_o = (val.lower() in ["1", "true"])
            elif opt.endswith("-perf_config"):
                perf_config = val
            else:
                raise ValueError(f"Unknown conv+gemm config argument {opt} -> {val}")
        for v in [
                dtype, n, c, hi, wi, k, y, x, o, conv_stride_h, conv_stride_w, padding_h, padding_w,
                dilation_h, dilation_w, group, filter_layout, input_layout, trans_c, trans_o
        ]:
            if v is None:
                raise ValueError("Incomplete conv+gemm configuration")

        return cls(dtype, filter_layout, input_layout, trans_c, trans_o, n, c, hi, wi, k, y, x, o,
                   conv_stride_h, conv_stride_w, padding_h, padding_w, dilation_h, dilation_w,
                   group, arch, num_cu, perf_config)

    def to_command_line(self):
        return (f"-t {self.datatype} " +
                f"-f {inverse_filter_layouts(self.filter_layout)} -I {self.input_layout.upper()} " +
                f"-transC {str(self.trans_c).lower()} -transO {str(self.trans_o).lower()} " +
                f"-n {self.n} -c {self.c} -H {self.hi} -W {self.wi} -k {self.k} " +
                f"-y {self.y} -x {self.x} -p {self.padding_h} -q {self.padding_w} " +
                f"-u {self.conv_stride_h} -v {self.conv_stride_w} -l {self.dilation_h} " +
                f"-j {self.dilation_w} -g {self.group}" + f"-gemmO {str(self.o)}")


class GemmGemmConfiguration(PerfConfiguration):
    TABLE_COLUMNS = reportUtils.GEMM_GEMM_TEST_PARAMETERS + ['TFlops']

    def __init__(self,
                 dtype: str,
                 g: int,
                 m: int,
                 k: int,
                 n: int,
                 o: int,
                 trans_a: bool,
                 trans_b: bool,
                 trans_c: bool,
                 trans_o: bool,
                 arch: str,
                 num_cu: int,
                 perf_config: str = ''):
        if dtype not in DATA_TYPES_GEMM_GEMM:
            raise ValueError(f"Invalid datatype for a: {dtype}")

        self.datatype = dtype
        self.g = g
        self.m = m
        self.k = k
        self.n = n
        self.o = o
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.trans_o = trans_o

        self.arch = arch
        self.chip = GFX_CHIP_RE.search(arch).group(0)
        self.num_cu = num_cu
        self.perfconfig = perf_config

    def get_total_flops(self):
        first_matmul_flops = 2.0 * self.g * self.m * self.k * self.n
        second_matmul_flops = 2.0 * self.g * self.m * self.n * self.o
        return first_matmul_flops + second_matmul_flops

    def compute_tflops(self, ns):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        total_flops = self.get_total_flops()
        return total_flops / (float(ns) * 1e-9) / 1e12

    def table_entry(self, nanoseconds):
        result = {}
        values = [
            self.datatype, self.chip, self.num_cu, self.trans_a, self.trans_b, self.trans_c,
            self.trans_o, self.g, self.m, self.k, self.n, self.o, self.perfconfig,
            self.compute_tflops(nanoseconds)
        ]
        assert (len(self.TABLE_COLUMNS) == len(values))
        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def set_perfconfig(self, perf_config):
        self.perfconfig = perf_config

    def generate_mlir_driver_commandline(self, rocmlir_gen_flags):
        result = ' '.join([
            '-operation', 'gemm_gemm', '-t', self.datatype, '--arch', self.arch, '--num_cu',
            str(self.num_cu), '-g',
            str(self.g), '-m',
            str(self.m), '-k',
            str(self.k), '-n',
            str(self.n), '-gemmO',
            str(self.o), f"-transA={self.trans_a}", f"-transB={self.trans_b}",
            f"-transC={self.trans_c}", f"-transO={self.trans_o}", '--kernel-repeats',
            str(MLIR_N_REPEATS), f"--perf_config={self.perfconfig}"
        ])
        result += ' '
        if rocmlir_gen_flags != '':
            result += ' '.join(rocmlir_gen_flags.split())
        return result

    @classmethod
    def from_command_line(cls, argv, arch, num_cu):
        # optional defaults
        perf_config = ''
        dtype = None
        g = None
        m = None
        k = None
        n = None
        o = None
        trans_a = False
        trans_b = False
        trans_c = False
        trans_o = False
        # Please keep this in sync with mlir::rock::getTuningProblemStr()
        for i in range(0, len(argv), 2):
            opt = argv[i]
            val = argv[i + 1]
            if opt.endswith("-t"):
                dtype = val
            elif opt.endswith("-g"):
                g = int(val)
            elif opt.endswith("-m"):
                m = int(val)
            elif opt.endswith("-k"):
                k = int(val)
            elif opt.endswith("-n"):
                n = int(val)
            elif opt.endswith("-gemmO"):
                o = int(val)
            elif opt.endswith("-transA"):
                trans_a = (val.lower() in ["1", "true"])
            elif opt.endswith("-transB"):
                trans_b = (val.lower() in ["1", "true"])
            elif opt.endswith("-transC"):
                trans_c = (val.lower() in ["1", "true"])
            elif opt.endswith("-transO"):
                trans_o = (val.lower() in ["1", "true"])
            elif opt.endswith("-perf_config"):
                perf_config = val
            else:
                raise ValueError(f"Unknown gemm+gemm config argument {opt} -> {val}")
        for v in [dtype, g, m, k, n, o, trans_a, trans_b, trans_c, trans_o]:
            if v is None:
                raise ValueError("Incomplete gemm+gemm configuration")

        return cls(dtype, g, m, k, n, o, trans_a, trans_b, trans_c, trans_o, arch, num_cu,
                   perf_config)

    def to_command_line(self):
        return (f"-t {self.datatype} " +
                f"-transA {str(self.trans_a).lower()} -transB {str(self.trans_b).lower()} " +
                f"-transC {str(self.trans_c).lower()} -transO {str(self.trans_o).lower()} " +
                f"-g {self.g} " +
                f"-m {str(self.m)} -k {str(self.k)} -n {str(self.n)} -gemmO {str(self.o)}")


class AttentionConfiguration(PerfConfiguration):
    TABLE_COLUMNS = reportUtils.ATTN_TEST_PARAMETERS + ['TFlops']

    def __init__(self,
                 dtype: str,
                 g: int,
                 seq_len_q: int,
                 seq_len_k: int,
                 num_heads_q: int,
                 num_heads_kv: int,
                 head_dim_qk: int,
                 head_dim_v: int,
                 with_attn_scale: bool,
                 with_attn_bias: bool,
                 trans_q: bool,
                 trans_k: bool,
                 trans_v: bool,
                 trans_o: bool,
                 causal: bool,
                 return_lse: bool,
                 split_kv: int,
                 arch: str,
                 num_cu: int,
                 perf_config: str = ''):
        if DATA_TYPES_ATTENTION is None:
            initialize_dtypes_attn()
        if dtype not in DATA_TYPES_ATTENTION:
            raise ValueError(f"Invalid datatype for a: {dtype}")

        self.datatype = dtype
        self.g = g
        self.seq_len_q = seq_len_q
        self.seq_len_k = seq_len_k
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.with_attn_scale = with_attn_scale
        self.with_attn_bias = with_attn_bias
        self.trans_q = trans_q
        self.trans_k = trans_k
        self.trans_v = trans_v
        self.trans_o = trans_o
        self.causal = causal
        self.return_lse = return_lse
        self.split_kv = split_kv

        self.arch = arch
        self.chip = GFX_CHIP_RE.search(arch).group(0)
        self.num_cu = num_cu
        self.perfconfig = perf_config

    def get_total_flops(self, only_matmul_flops):
        # GQA broadcasts so that both num_heads_q == num_heads_kv
        g = self.g * max(self.num_heads_q, self.num_heads_kv)
        first_matmul_flops = 2.0 * g * self.seq_len_q * self.head_dim_qk * self.seq_len_k
        # max, sub, exp, sum, div
        softmax_flops = 5.0 * g * self.seq_len_q * self.seq_len_k
        second_matmul_flops = 2.0 * g * self.seq_len_q * self.seq_len_k * self.head_dim_v
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
                total_flops += g * self.seq_len_q * self.seq_len_k
            if self.with_attn_bias:
                total_flops += g * self.seq_len_q * self.seq_len_k

        return total_flops

    def compute_tflops(self, ns, only_matmul_flops=True):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        total_flops = self.get_total_flops(only_matmul_flops)
        return total_flops / (float(ns) * 1e-9) / 1e12

    def compute_ns_from_tflops(self, tflops, only_matmul_flops=True):
        """
        Calculate nanoseconds from TFlops value.
        This is the inverse of compute_tflops().

        Args:
            tflops: TFlops value to convert to nanoseconds

        Returns:
            float: Time in nanoseconds
        """
        if tflops == 0 or np.isnan(tflops) or np.isinf(tflops):
            return np.nan

        total_flops = self.get_total_flops(only_matmul_flops)
        return total_flops / (tflops * 1e3)

    def table_entry(self, nano_seconds):
        result = {}
        values = [
            self.datatype, self.chip, self.num_cu, self.trans_q, self.trans_k, self.trans_v,
            self.trans_o, self.causal, self.return_lse, self.split_kv, self.with_attn_scale,
            self.with_attn_bias, self.g, self.seq_len_q, self.seq_len_k, self.num_heads_q,
            self.num_heads_kv, self.head_dim_qk, self.head_dim_v, self.perfconfig,
            self.compute_tflops(nano_seconds)
        ]
        assert (len(self.TABLE_COLUMNS) == len(values))
        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def set_perfconfig(self, perf_config):
        self.perfconfig = perf_config

    def generate_mlir_driver_commandline(self, rocmlir_gen_flags, kernel_repeats=MLIR_N_REPEATS):
        result = ' '.join([
            '-operation', 'attention', '-t', self.datatype, '--arch', self.arch, '--num_cu',
            str(self.num_cu), '-g',
            str(self.g), '-seq_len_q',
            str(self.seq_len_q), '-seq_len_k',
            str(self.seq_len_k), '-num_heads_q',
            str(self.num_heads_q), '-num_heads_kv',
            str(self.num_heads_kv), '-head_dim_qk',
            str(self.head_dim_qk), '-head_dim_v',
            str(self.head_dim_v), f"-with-attn-scale={self.with_attn_scale}",
            f"-with-attn-bias={self.with_attn_bias}", f"-transQ={self.trans_q}",
            f"-transK={self.trans_k}", f"-transV={self.trans_v}", f"-transO={self.trans_o}",
            f"-causal={self.causal}", f"-return_lse={self.return_lse}",
            f"-split_kv={self.split_kv}",
            *(['--kernel-repeats', str(kernel_repeats)] if kernel_repeats is not None else []),
            f"--perf_config={self.perfconfig}"
        ])
        result += ' '
        if rocmlir_gen_flags != '':
            result += ' '.join(rocmlir_gen_flags.split())
        return result

    @classmethod
    def from_command_line(cls, argv, arch, num_cu):
        # optional defaults
        perf_config = ''
        dtype = None
        g = None
        seq_len_q = None
        seq_len_k = None
        num_heads_q = 1
        num_heads_kv = 1
        head_dim_qk = None
        head_dim_v = None
        trans_q = False
        trans_k = False
        trans_v = False
        trans_o = False
        causal = False
        return_lse = False
        split_kv = 1
        with_attn_scale = False
        with_attn_bias = False
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
            elif opt.endswith("-num_heads_q"):
                num_heads_q = int(val)
            elif opt.endswith("-num_heads_kv"):
                num_heads_kv = int(val)
            elif opt.endswith("-head_dim_qk"):
                head_dim_qk = int(val)
            elif opt.endswith("-head_dim_v"):
                head_dim_v = int(val)
            elif opt.endswith("-with-attn-scale"):
                with_attn_scale = (val.lower() in ["1", "true"])
            elif opt.endswith("-with-attn-bias"):
                with_attn_bias = (val.lower() in ["1", "true"])
            elif opt.endswith("-transQ"):
                trans_q = (val.lower() in ["1", "true"])
            elif opt.endswith("-transK"):
                trans_k = (val.lower() in ["1", "true"])
            elif opt.endswith("-transV"):
                trans_v = (val.lower() in ["1", "true"])
            elif opt.endswith("-transO"):
                trans_o = (val.lower() in ["1", "true"])
            elif opt.endswith("-causal"):
                causal = (val.lower() in ["1", "true"])
            elif opt.endswith("-return_lse"):
                return_lse = (val.lower() in ["1", "true"])
            elif opt.endswith("-split_kv"):
                split_kv = int(val)
            elif opt.endswith("-perf_config"):
                perf_config = val
            else:
                raise ValueError(f"Unknown Attention config argument {opt} -> {val}")
        for v in [
                dtype, g, seq_len_q, seq_len_k, num_heads_q, num_heads_kv, head_dim_qk, head_dim_v,
                with_attn_scale, with_attn_bias, trans_q, trans_k, trans_v, trans_o, causal,
                return_lse, split_kv
        ]:
            if v is None:
                raise ValueError("Incomplete Attention configuration")

        return cls(dtype, g, seq_len_q, seq_len_k, num_heads_q, num_heads_kv, head_dim_qk,
                   head_dim_v, with_attn_scale, with_attn_bias, trans_q, trans_k, trans_v, trans_o,
                   causal, return_lse, split_kv, arch, num_cu, perf_config)

    def to_command_line(self):
        return (
            f"-t {self.datatype} " +
            f"-transQ {str(self.trans_q).lower()} -transK {str(self.trans_k).lower()} " +
            f"-transV {str(self.trans_v).lower()} -transO {str(self.trans_o).lower()} " +
            f"-causal {str(self.causal).lower()} " +
            f"-return_lse {str(self.return_lse).lower()} " + f"-split_kv {str(self.split_kv)} " +
            f"-g {self.g} " +
            f"-seq_len_q {str(self.seq_len_q)} -seq_len_k {str(self.seq_len_k)} -num_heads_q {str(self.num_heads_q)} -num_heads_kv {str(self.num_heads_kv)} -head_dim_qk {str(self.head_dim_qk)} -head_dim_v {str(self.head_dim_v)} "
            + f"-with-attn-scale {str(self.with_attn_scale).lower()} " +
            f"-with-attn-bias {str(self.with_attn_bias).lower()}")


class RocBLASGemmConfig(GemmConfiguration):
    EXTERNAL_NAME = "rocBLAS"

    @classmethod
    def benchmark_external(cls, commandline, paths: Paths, arch, num_cu):
        config = cls.from_command_line(commandline, arch, num_cu)
        if not paths.mlir_paths.rocblas_benchmark_driver_path:
            raise ValueError("rocblas-benchmark-driver not built")
        benchmark_args = config.generate_mlir_driver_commandline("")
        # remove the result file generated by rocprof in previous benchmarking
        if os.path.exists(get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME)):
            os.remove(get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME))
        print(f"Running rocBLAS benchmark {config!r}")
        profiler_cmd = [paths.mlir_paths.rocblas_benchmark_driver_path] + \
            benchmark_args.split()
        outs, noerr = run_pipeline([profiler_cmd])
        nanoseconds = np.nan
        if noerr:
            miliseconds = get_miliseconds(outs)
            nanoseconds = miliseconds * 1e6

        return config.table_entry(nanoseconds)


class CKGemmConfig(GemmConfiguration):
    EXTERNAL_NAME = "CK"

    @classmethod
    def benchmark_external(cls, commandline, paths: Paths, arch, num_cu):
        config = cls.from_command_line(commandline, arch, num_cu)
        if not paths.mlir_paths.ck_gemm_benchmark_driver_path:
            raise ValueError("ck-gemm-benchmark-driver not built")
        benchmark_args = config.generate_mlir_driver_commandline("")

        print(f"Running CK benchmark {config!r}")

        if arch == "gfx1030" and config.g > 1:
            return config.table_entry(float('NaN'))

        profiler_cmd = [paths.mlir_paths.ck_gemm_benchmark_driver_path] + \
            benchmark_args.split()
        outs, noerr = run_pipeline([profiler_cmd])
        nanoseconds = np.nan
        if noerr:
            miliseconds = get_miliseconds(outs)
            nanoseconds = miliseconds * 1e6

        return config.table_entry(nanoseconds)


def run_config_with_mlir(config: PerfConfiguration,
                         paths: Paths,
                         arch,
                         rocmlir_gen_flags,
                         use_rocprof=False,
                         debug=True):
    # remove the result file generated by rocprof in previous benchmarking
    if os.path.exists(get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME)):
        os.remove(get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME))
    commandline_options = config.generate_mlir_driver_commandline(rocmlir_gen_flags)
    if debug:
        print("Running MLIR Benchmark: ", repr(config))

    nanoseconds = np.nan

    # Use HIP timing via tuning-driver if rocprof is disabled and perfconfig is present
    if not use_rocprof and config.perfconfig:
        if debug:
            print("Using HIP timing for benchmarking")
        rocmlir_gen_cmd = paths.mlir_paths.rocmlir_gen_path + ' ' + commandline_options
        tuning_driver_command = [
            paths.mlir_paths.rocmlir_tuning_driver_path, f'--benchmark-config={config.perfconfig}',
            f'--num-iterations={MLIR_N_REPEATS}', f'--warmup-iterations={WARMUP_ITERATIONS}',
            f'--sleep-us={SLEEP_US}', '--use-median', '-'
        ]
        outs, noerr = run_pipeline([rocmlir_gen_cmd.split(), tuning_driver_command])
        if noerr:
            try:
                _, time = outs.split()
                if time != "N/A":
                    nanoseconds = float(time)
            except ValueError:
                if debug:
                    print(f"Failed to parse timing result: {outs}")
    else:
        if debug:
            print("Using rocprof for benchmarking")
        rocmlir_gen_cmd = paths.mlir_paths.rocmlir_gen_path + ' -ph ' + commandline_options
        rocmlir_driver_cmd = [paths.mlir_paths.rocmlir_driver_path, '-c']
        mlir_cpu_runner_args = [
            f'--shared-libs={paths.mlir_paths.libmlir_rocm_runtime_path},{paths.mlir_paths.libconv_validation_wrappers_path},{paths.mlir_paths.libmlir_runtime_utils_path},{paths.mlir_paths.libmlir_c_runner_utils_path}',
            '--entry-point-result=void'
        ]
        profiler_cmd = [ROCPROF] + get_metric_args_for_rocprof(arch) + [
            '--kernel-trace', '--stats', '-f', 'csv', '-o', BENCHMARKING_RESULT_FILE_NAME, '--',
            paths.mlir_paths.cpu_runner_path
        ] + mlir_cpu_runner_args

        outs, noerr = run_pipeline([rocmlir_gen_cmd.split(), rocmlir_driver_cmd, profiler_cmd])
        if noerr:
            nanoseconds = get_nanoseconds(
                get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME))

    return nanoseconds


# Benchmarking function.
def benchmark_mlir(commandline,
                   conf_class,
                   paths: Paths,
                   arch,
                   num_cu,
                   tuning_db: MaybeTuningDb,
                   rocmlir_gen_flags,
                   use_rocprof=False):
    config = conf_class.from_command_line(commandline, arch, num_cu)
    config_str = config.to_command_line()
    if tuning_db:
        if (arch, config_str) in tuning_db:
            config.set_perfconfig(tuning_db[arch, config_str])
        else:  # Tuning DB present but doesn't contain config, return N/A
            return config.table_entry(np.nan)

    nanoseconds = run_config_with_mlir(config, paths, arch, rocmlir_gen_flags, use_rocprof)
    return config.table_entry(nanoseconds)


# Generate MLIR vs. MIOpen or rocBLAS performance results
def generate_performance_results(configs,
                                 conf_class,
                                 paths: Paths,
                                 arch,
                                 num_cu,
                                 tuning_db: MaybeTuningDb,
                                 quick_tuning_db: MaybeTuningDb,
                                 rocmlir_gen_flags,
                                 use_rocprof=False):
    # Never pass tuning DB to this run
    mlir_df = pd.DataFrame(
        benchmark_mlir(test_vector.split(
            sep=' '), conf_class, paths, arch, num_cu, None, rocmlir_gen_flags, use_rocprof)
        for test_vector in configs)
    tuned_df = None
    if tuning_db:
        tuned_df = pd.DataFrame(
            benchmark_mlir(test_vector.split(sep=' '), conf_class, paths, arch, num_cu, tuning_db,
                           rocmlir_gen_flags, use_rocprof) for test_vector in configs)
    quick_tuned_df = None
    if quick_tuning_db:
        quick_tuned_df = pd.DataFrame(
            benchmark_mlir(test_vector.split(sep=' '), conf_class, paths, arch, num_cu,
                           quick_tuning_db, rocmlir_gen_flags, use_rocprof)
            for test_vector in configs)

    external_df = pd.DataFrame(
        conf_class.benchmark_external(test_vector.split(sep=' '), paths, arch, num_cu)
        for test_vector in configs)

    external_name = conf_class.EXTERNAL_NAME
    df = mlir_df.merge(external_df,
                       on=conf_class.TABLE_COLUMNS[:-2],
                       suffixes=('', f" ({external_name})"))
    external_tflops_col = f"{external_name} TFlops (no MLIR Kernels)"
    df.rename(columns={
        'TFlops': 'MLIR TFlops',
        f"TFlops ({external_name})": external_tflops_col
    },
              inplace=True)
    #     if tuned_df is None and quick_tuned_df is None:
    #         df.drop(columns=['PerfConfig'], inplace=True)
    if tuned_df is not None:
        # No need for suffixes, the conflicting columns have been renamed
        # Also note that we're ignoring PerfConfig with the -3
        df = df.merge(tuned_df, on=conf_class.TABLE_COLUMNS[:-3], suffixes=('', ' (tuned)'))
        df.drop(columns=['PerfConfig'], inplace=True)
        df.rename(columns={
            'TFlops': 'Tuned MLIR TFlops',
            'PerfConfig (tuned)': 'PerfConfig'
        },
                  inplace=True)
    if quick_tuned_df is not None:
        # No need for suffixes, the conflicting columns have been renamed
        # Also note that we're ignoring PerfConfig with the -3
        df = df.merge(quick_tuned_df,
                      on=conf_class.TABLE_COLUMNS[:-3],
                      suffixes=('', ' (quick tuned)'))
        df.rename(columns={'TFlops': 'Quick Tuned MLIR TFlops'}, inplace=True)

    df[f"MLIR/{external_name}"] = df['MLIR TFlops'] / df[external_tflops_col]
    if tuned_df is not None:
        df[f"Tuned/{external_name}"] = df['Tuned MLIR TFlops'] / df[external_tflops_col]
        df["Tuned/Untuned"] = df['Tuned MLIR TFlops'] / df['MLIR TFlops']
    if quick_tuned_df is not None:
        df[f"Quick Tuned/{external_name}"] = df['Quick Tuned MLIR TFlops'] / df[external_tflops_col]
        df["Quick Tuned/Untuned"] = df['Quick Tuned MLIR TFlops'] / df['MLIR TFlops']
    if tuned_df is not None and quick_tuned_df is not None:
        df["Quick Tuned/Tuned"] = df['Quick Tuned MLIR TFlops'] / df['Tuned MLIR TFlops']
    chip = GFX_CHIP_RE.search(arch).group(0)
    if conf_class is RocBLASGemmConfig:
        report_file = reportUtils.PERF_REPORT_FILE['rocBLAS']
    elif conf_class is CKGemmConfig:
        report_file = reportUtils.PERF_REPORT_FILE['CK']
    else:
        report_file = reportUtils.PERF_REPORT_FILE['MIOpen']
    df.fillna(np.nan, inplace=True)
    df.to_csv(chip + '_' + report_file, index=False)


def get_solver_name(test_vector, arch, num_cu):
    config = ConvConfiguration.from_command_line(test_vector.split(sep=' '), arch, num_cu)
    if config.direction == 'fwd':
        solver_name = 'ConvMlirIgemmFwd'
    elif config.direction == 'bwd':
        solver_name = 'ConvMlirIgemmBwd'
    else:
        solver_name = 'ConvMlirIgemmWrW'
    if config.chip in ['gfx908', 'gfx90a', 'gfx942', 'gfx950']:
        solver_name += 'Xdlops'
    return solver_name


RUNNABLE_TEST_RE = re.compile(r"//\s*RUN\s*:(.*)")
ROCMLIRGEN_RE = re.compile(r"rocmlir-gen.*?-fut\s*(\w+)")


def find_run_command(filename):
    rocmlir_cmd = None
    fut_name = None
    with open(filename, 'r') as f:
        for line in f:
            has_run = RUNNABLE_TEST_RE.search(line)
            has_rocmlir_gen = ROCMLIRGEN_RE.search(line)
            if has_run:
                command = has_run.group(1)
                if not rocmlir_cmd:
                    parts = command.split('|')  # Split the command using the "|" separator
                    if 'rocmlir-driver' in parts[0] or 'rocmlir-opt' in parts[0]:
                        rocmlir_cmd = parts[0].strip()  # Find rocmlir-driver command
                    elif 'rocmlir-driver' in parts[1] or 'rocmlir-opt' in parts[1]:
                        rocmlir_cmd = parts[1].strip()

                if has_rocmlir_gen and not fut_name:
                    fut_name = has_rocmlir_gen.group(1)

                if 'runner' in line:  # Stop processing lines after finding a runner
                    return rocmlir_cmd, fut_name

    # Not found a "RUN" command or a runner
    print("WARNING: cannot find valid RUN command in ", filename)
    return None, None


# Extract test_vector and test function name from the test file
def get_fusion_test_info(filename, paths: Paths):
    chip = get_chip()
    test_entry = {}
    rocmlir_cmd, fut_name = find_run_command(filename)
    if not rocmlir_cmd:
        return test_entry
    # rocmlir-gen -fut test -arch gfx90a --clone-harness
    rocmlirgen_cmd = [
        paths.mlir_paths.rocmlir_gen_path, '-fut', fut_name, '-arch', chip, '--clone-harness',
        filename
    ]
    p0 = subprocess.Popen(rocmlirgen_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if "-migraphx-to-tosa" in rocmlir_cmd:
        rocmliropt_cmd = [paths.mlir_paths.rocmlir_opt_path, '-migraphx-to-tosa']
        rocmlir_driver_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'highlevel', '-kernel-pipeline',
            'highlevel', '-targets', chip
        ]
        # rocmlir-opt -migraphx-to-tosa ../mlir/test/fusion/resnet50-e2e/mixr-resnet-fusion-case-1.mlir
        p1 = subprocess.Popen(rocmliropt_cmd,
                              stdin=p0.stdout,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)
        # pipe to rocmlir-driver -host-pipeline highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlir_driver_cmd,
                              stdin=p1.stdout,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)
        p1.stdout.close()
    elif "migraphx" in rocmlir_cmd:
        rocmlir_migraphx_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-kernel-pipeline', 'migraphx,highlevel'
        ]
        rocmlir_driver_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'migraphx,highlevel',
            '-targets', chip
        ]
        # rocmlir-driver -kernel-pipeline migraphx ../mlir/test/fusion/resnet50-e2e/mixr-resnet-fusion-case-1.mlir
        p1 = subprocess.Popen(rocmlir_migraphx_cmd,
                              stdin=p0.stdout,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)
        # pipe to rocmlir-driver -host-pipeline highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlir_driver_cmd,
                              stdin=p1.stdout,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)
        p1.stdout.close()
    else:
        rocmlir_driver_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'highlevel', '-kernel-pipeline',
            'highlevel', '-targets', chip
        ]
        # rocmlir-driver -host-pipeline highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlir_driver_cmd,
                              stdin=p0.stdout,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)

    # pipe to rocmlir_gen --emit-tuning-key
    tuning_key = subprocess.Popen([paths.mlir_paths.rocmlir_gen_path, '--emit-tuning-key', '-'],
                                  stdin=p2.stdout,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
    p2.stdout.close()
    output, _ = tuning_key.communicate()
    result = output.decode('utf-8').strip().split('\t')
    test_entry = {'filename': filename, 'testVector': result[2], 'futName': fut_name}
    return test_entry


def run_fusion_kernel(filename, rocmlir_gen_args, paths: Paths):
    arch = get_arch()
    chip = get_chip()
    if os.path.exists(get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME)):
        os.remove(get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME))

    rocmlir_cmd, fut_name = find_run_command(filename)

    # rocmlir-gen -fut test -arch gfx90a --clone-harness
    rocmlirgen_cmd = [
        paths.mlir_paths.rocmlir_gen_path, '-fut', fut_name, '-arch', chip, '--clone-harness',
        filename
    ]
    commands = [rocmlirgen_cmd]
    if "-migraphx-to-tosa" in rocmlir_cmd:
        rocmliropt_cmd = [paths.mlir_paths.rocmlir_opt_path, '-migraphx-to-tosa', filename]
        commands.append(rocmliropt_cmd)
        rocmlir_driver_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'highlevel', '-kernel-pipeline',
            'highlevel', '-targets', chip
        ]
        commands.append(rocmlir_driver_cmd)
    elif "migraphx" in rocmlir_cmd:
        rocmlir_migraphx_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-kernel-pipeline', 'migraphx,highlevel'
        ]
        commands.append(rocmlir_migraphx_cmd)
        rocmlir_driver_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'migraphx,highlevel',
            '-targets', chip
        ]
        commands.append(rocmlir_driver_cmd)
    else:
        rocmlir_driver_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'highlevel', '-kernel-pipeline',
            'highlevel', '-targets', chip
        ]
        commands.append(rocmlir_driver_cmd)

    rocmlir_gen_cmd = [paths.mlir_paths.rocmlir_gen_path] + rocmlir_gen_args
    commands.append(rocmlir_gen_cmd)
    kernel_pipeline_cmd = [
        paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'mhal,runner', '-kernel-pipeline',
        'full'
    ]
    commands.append(kernel_pipeline_cmd)
    mlir_cpu_runner_args = [
        f'--shared-libs={paths.mlir_paths.libmlir_rocm_runtime_path},{paths.mlir_paths.libconv_validation_wrappers_path},{paths.mlir_paths.libmlir_runtime_utils_path},{paths.mlir_paths.libmlir_c_runner_utils_path}',
        '--entry-point-result=void'
    ]
    profiler_cmd = [ROCPROF] + get_metric_args_for_rocprof(chip) + [
        '--kernel-trace', '--stats', '-f', 'csv', '-o', BENCHMARKING_RESULT_FILE_NAME
    ] + ['--', paths.mlir_paths.cpu_runner_path] + mlir_cpu_runner_args
    commands.append(profiler_cmd)
    outs, noerr = run_pipeline(commands)
    nanoseconds = np.nan
    if noerr:
        nanoseconds = get_nanoseconds(get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME))

    return nanoseconds


# Generate fusion vs. gemm/conv performance results
def benchmark_fusion_kernels(test_dir,
                             paths: Paths,
                             arch,
                             num_cu,
                             tuning_db: MaybeTuningDb,
                             use_rocprof=False):
    all_tests = []  # filename, test_vector, fut_name
    perf_results = {}  # associate test_vector to config and performances
    chip = GFX_CHIP_RE.search(arch).group(0)

    # Prepare test cases
    for filename in glob.glob(test_dir + '/*.mlir'):
        test_entry = get_fusion_test_info(filename, paths)
        if test_entry:
            all_tests.append(test_entry)

    if tuning_db:
        # Force all split-K factors to 1, to avoid trouble because fusion
        # and split-K aren't compatible.  Crude parser approximating
        # InitParamsAccel::visit().
        for (arch, config), perfconfig in tuning_db.items():
            split_perf = perfconfig.split(',')
            if ((perfconfig[0:3] == 'v2:' or perfconfig[0:3] == 'v3:') and int(split_perf[6]) > 1):
                split_perf[6] = '1'
                tuning_db[arch, config] = ','.join(split_perf)

    # Profile each test case
    for test in all_tests:
        filename = test['filename']
        test_vector = test['testVector']
        fut_name = test['futName']

        print("Profiling:", filename)
        # Sanity check
        if not test_vector:
            print("\tCannot find a test vector")
            continue
        if not fut_name:
            print("\tCannot find rocmlir-gen with -fut")
            continue

        commandline = test_vector.split(sep=' ')
        if commandline[0].startswith('conv'):
            op = 'conv'
            config = ConvConfiguration.from_command_line(commandline, arch, num_cu)
        else:
            op = 'gemm'
            config = GemmConfiguration.from_command_line(commandline, arch, num_cu)

        # Find the best perf_config
        best_perf = ""
        if tuning_db:
            config_str = config.to_command_line()
            if (arch, config_str) in tuning_db:
                best_perf = tuning_db[arch, config_str]
                config.set_perfconfig(best_perf)
            else:  # Tuning DB present but doesn't contain config, add a NaN entry
                if test_vector not in perf_results:
                    one_entry = config.table_entry(np.nan)
                    one_entry['MLIR TFlops'] = np.nan
                    one_entry['Fusion/MLIR'] = np.nan
                    one_entry['FileName'] = filename
                    perf_results[test_vector] = one_entry
                continue

        # Run fusion test
        rocmlir_gen_args = [
            '-ph', '-fut=' + fut_name + '_wrapper', '--perf_config=' + best_perf, '-'
        ]
        nanoseconds = run_fusion_kernel(filename, rocmlir_gen_args, paths)
        one_entry = config.table_entry(nanoseconds)
        # Keep the best performance
        if test_vector in perf_results and one_entry['TFlops'] <= perf_results[test_vector][
                'TFlops']:
            continue

        # Run gemm or conv op with the same configuration
        nanoseconds = run_config_with_mlir(config, paths, arch, '', use_rocprof)
        one_entry['MLIR TFlops'] = config.compute_tflops(nanoseconds)
        one_entry['Fusion/MLIR'] = one_entry['TFlops'] / one_entry['MLIR TFlops']
        one_entry['FileName'] = filename
        perf_results[test_vector] = one_entry

    df = pd.DataFrame(perf_results.values())
    df.fillna(np.nan, inplace=True)
    df.rename(columns={'TFlops': 'Fusion TFlops'}, inplace=True)
    df.to_csv(chip + '_' + op + '_' + reportUtils.PERF_REPORT_FUSION_FILE, index=False)


# Tune MIOpen with MLIR kernels
def tune_mlir_kernels(configs, arch, num_cu):
    solver_names = {
        test_vector: get_solver_name(test_vector, arch, num_cu) for test_vector in configs
    }

    envs = os.environ.copy()
    envs['MIOPEN_FIND_ENFORCE'] = '4'
    envs['MIOPEN_DRIVER_USE_GPU_REFERENCE'] = '1'
    for test_vector in configs:
        envs['MIOPEN_DEBUG_FIND_ONLY_SOLVER'] = solver_names[test_vector]
        commandline = test_vector.split(sep=' ')
        config = ConvConfiguration.from_command_line(commandline, arch, num_cu)
        if config.input_layout == 'nchw':
            miopen_driver_cmd = [MIOPENDRIVER, *commandline, '-V', '0']
            print(' '.join(miopen_driver_cmd))
            p1 = subprocess.Popen(miopen_driver_cmd,
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


def parse_data_types(data_types):
    if not data_types:
        return DATA_TYPES_GEMM, OUTPUT_DATA_TYPES_MAP
    datatypes = []
    out_map = {}
    for dpair in data_types:
        dt = dpair.split('_')
        datatypes.append(dt[0])
        out_map[dt[0]] = dt[0]
        if len(dt) == 2:
            out_map[dt[0]] = dt[1]
        elif dt[0] == 'i8':
            out_map[dt[0]] = 'i32'
        elif dt[0] == 'fp8':
            out_map[dt[0]] = 'f32'
    return datatypes, out_map


def get_num_cu(chip):
    try:
        rocminfo = subprocess.check_output("/opt/rocm/bin/rocminfo", stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode('utf-8'))
        raise
    except Exception as e:
        print(f"Exception: {e}")
        raise
    rocminfo_lines = rocminfo.decode("utf-8").split("\n")
    found_chip = False
    for line in rocminfo_lines:
        if not found_chip:
            m = INFO_ARCH_NAME.search(line)
            if m and chip in m.group(1).strip():
                found_chip = True
        if found_chip:
            compute_unit = INFO_ARCH_CU.search(line)
            if compute_unit:
                return int(compute_unit.group(1))
    assert False, f"Cannot find number of CUs for {chip}"


def found_external_tool(paths: Paths,
                        optype: Operation,
                        gemm_library: Optional[GEMMLibrary] = None):
    if optype == Operation.GEMM:
        if not paths.mlir_paths:
            return False
        if gemm_library == GEMMLibrary.CK and not paths.mlir_paths.ck_gemm_benchmark_driver_path:
            return False
        if gemm_library == GEMMLibrary.ROCBLAS and not paths.mlir_paths.rocblas_benchmark_driver_path:
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

    arch = get_arch()
    chip = get_chip()
    num_cu = get_num_cu(chip)
    initialize_dtypes_attn()

    root_dir = str(
        subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
    default_conv_configs = root_dir + '/mlir/utils/jenkins/performance/configs/tier1-conv-configs'

    parser = argparse.ArgumentParser(
        prog="rocMLIR performance test runner",
        description="A test runner script for MIOpen and MLIR-based kernel generator",
        allow_abbrev=False,
    )

    parser.add_argument("--op",
                        "--operation",
                        choices=['conv', 'gemm', 'fusion', 'attention', 'gemm_gemm', 'conv_gemm'],
                        default='conv',
                        help="Operation to benchmark")

    mutex_arg_group = parser.add_mutually_exclusive_group()
    mutex_arg_group.add_argument("--tuning", action="store_true", help="Only tune the MLIR kernels")
    mutex_arg_group.add_argument("-b",
                                 "--batch_mlir",
                                 action="store_true",
                                 help="CSV batch benchmarking mode with MLIR")
    mutex_arg_group.add_argument("--batch_external",
                                 action="store_true",
                                 help="CSV batch benchmarking mode with external reference")
    mutex_arg_group.add_argument(
        "--batch_all",
        action="store_true",
        help="CSV batch benchmarking with MLIR and external reference (defalut on no args)")
    mutex_arg_group.add_argument("--external",
                                 action="store_true",
                                 help="benchmark a single config externally")

    parser.add_argument("-c",
                        "--configs_file",
                        type=str,
                        default=default_conv_configs,
                        help="File of configurations to test")

    parser.add_argument("-o",
                        type=str,
                        default=chip + '_' + date.today().strftime("perf.%m%d%y"),
                        help="Output file name",
                        dest="filename")
    parser.add_argument("-t",
                        "--tuning_db",
                        type=str,
                        default=argparse.SUPPRESS,
                        help="Tuning database filename")
    parser.add_argument("-qt",
                        "--quick_tuning_db",
                        type=str,
                        default=argparse.SUPPRESS,
                        help="Quick tuning database filename")

    parser.add_argument("--test_dir",
                        type=str,
                        default="../mlir/test/fusion/resnet50-e2e",
                        help="The directory of tests")
    parser.add_argument(
        "--mlir-build-dir",
        type=str,
        default=find_mlir_build_dir(),
        help="The build directory of MLIR based kernel generator",
    )
    parser.add_argument("config",
                        type=str,
                        nargs='*',
                        help="The specific config to test, if you want to test one")

    parser.add_argument("--rocmlir_gen_flags",
                        type=str,
                        default=argparse.SUPPRESS,
                        help="rocmlir-gen flags to toggle each feature")

    parser.add_argument("--external-gemm-library",
                        type=str,
                        default="rocBLAS",
                        help="(rocBLAS | CK) external library to run GEMM routines")

    parser.add_argument(
        '--data-type',
        nargs='+',
        choices=["f32", "f16", "i8", "i8_i32", "i8_i8", "fp8", "fp8_fp8", "fp8_f32"],
        default=["f32", "f16", "i8"],
        help='Force a set of datatypes')

    parser.add_argument(
        '--use-rocprof',
        action="store_true",
        help="Use rocprof instead of rocmlir-tuning-driver to collect performance data")

    parsed_args = parser.parse_args(args)

    rocmlir_gen_flags = ''
    if 'rocmlir_gen_flags' in parsed_args:
        rocmlir_gen_flags = parsed_args.rocmlir_gen_flags

    tuning_db = None
    quick_tuning_db = None
    if 'tuning_db' in parsed_args:
        tuning_db = read_tuning_db(parsed_args.tuning_db)

    if 'quick_tuning_db' in parsed_args:
        quick_tuning_db = read_tuning_db(parsed_args.quick_tuning_db)

    # Impose default behavior when no args have been passed
    if len(args) == 0:
        parsed_args.batch_all = True

    conf_class = PerfConfiguration
    optype = Operation.from_name(parsed_args.op)
    if optype == Operation.CONV:
        conf_class = ConvConfiguration
        external_lib = None
    elif optype == Operation.GEMM:
        external_lib = GEMMLibrary.from_name(parsed_args.external_gemm_library)
        if external_lib == GEMMLibrary.ROCBLAS:
            conf_class = RocBLASGemmConfig
        elif external_lib == GEMMLibrary.CK:
            conf_class = CKGemmConfig
    elif optype == Operation.ATTENTION:
        conf_class = AttentionConfiguration
        external_lib = None
    elif optype == Operation.GEMM_GEMM:
        conf_class = GemmGemmConfiguration
        external_lib = None
    elif optype == Operation.CONV_GEMM:
        conf_class = ConvGemmConfiguration
        external_lib = None

    configs_path = None if parsed_args.config else parsed_args.configs_file
    paths = create_paths(configs_path, parsed_args.mlir_build_dir)
    configs = None
    if optype == Operation.CONV:
        configs = get_conv_configurations(paths.configuration_file_path)
    elif optype == Operation.GEMM:
        datatypes, output_type_map = parse_data_types(parsed_args.data_type)
        configs = get_gemm_configurations(paths.configuration_file_path, datatypes, output_type_map)
    elif optype == Operation.ATTENTION:
        configs = get_attn_configurations(paths.configuration_file_path)
    elif optype == Operation.GEMM_GEMM:
        configs = get_gemm_gemm_configurations(paths.configuration_file_path)
    elif optype == Operation.CONV_GEMM:
        configs = get_conv_gemm_configurations(paths.configuration_file_path)

    if parsed_args.external or parsed_args.batch_external or parsed_args.batch_all:
        if not found_external_tool(paths, optype, external_lib):
            raise RuntimeError(
                "External benchmark reference (MIOpen or rocBLAS driver) needed but not found")

    if parsed_args.batch_mlir or parsed_args.batch_all:
        if not paths.mlir_paths:
            raise RuntimeError("MLIR build dir was not provided/found")

    # If no arguments are passed, then benchmark with MLIR and MIOpen
    if parsed_args.batch_all:
        # batch benchmark with MLIR and MIOpen.
        generate_performance_results(configs, conf_class, paths, arch, num_cu, tuning_db,
                                     quick_tuning_db, rocmlir_gen_flags, parsed_args.use_rocprof)
    elif parsed_args.tuning:
        tune_mlir_kernels(configs, arch, num_cu)
    elif optype == Operation.FUSION:
        if not parsed_args.mlir_build_dir:
            raise RuntimeError("MLIR build dir was not provided/found")
        else:
            benchmark_fusion_kernels(parsed_args.test_dir, paths, arch, num_cu, tuning_db,
                                     parsed_args.use_rocprof)
    else:
        if parsed_args.batch_mlir:
            df = pd.DataFrame(
                benchmark_mlir(test_vector.split(sep=' '), conf_class, paths, arch, num_cu,
                               tuning_db, rocmlir_gen_flags, parsed_args.use_rocprof)
                for test_vector in configs)
        elif parsed_args.batch_external:
            df = pd.DataFrame(
                conf_class.benchmark_external(test_vector.split(sep=' '), paths, arch, num_cu)
                for test_vector in configs)
        elif parsed_args.external:
            df = pd.DataFrame(
                [conf_class.benchmark_external(parsed_args.config, paths, arch, num_cu)])
        else:
            # Will only reach here with more than 1 unspecified arguments
            # These are arguments are directly passed through to benchmark_mlir
            if not parsed_args.mlir_build_dir:
                raise RuntimeError("MLIR build dir was not provided/found")
            else:
                if parsed_args.config:
                    df = pd.DataFrame([
                        benchmark_mlir(parsed_args.config, conf_class, paths, arch, num_cu,
                                       tuning_db, rocmlir_gen_flags, parsed_args.use_rocprof)
                    ])
                else:
                    df = pd.DataFrame([
                        benchmark_mlir(config.split(), conf_class, paths, arch, num_cu, tuning_db,
                                       rocmlir_gen_flags, parsed_args.use_rocprof)
                        for config in configs
                    ])
        df.to_csv(parsed_args.filename)
        with pd.option_context('display.precision', reportUtils.ROUND_DIGITS):
            print(df)  # for interactive consumption


if __name__ == '__main__':
    sys.exit(main())
