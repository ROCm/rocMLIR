"""
This script compiles a given config in order to collect the following data
points for each config:
- Blocksize
- Gridsize
- vgpr
- sgpr
- LDS allocated
- Occupancy
- wf_per_wg
- mfma_wmma_instruction

The given config is expected to be a tsv with the following format:
|# arch| numCUs | testVector | perfConfig (exhaustive) |

Usage:
    python3 compileAndCollectTuningData.py --op <operation> <config.tsv>
"""

import argparse
import csv
import os
import re
import sys

from datetime import datetime
from testing_metrics import calculate_gemm_occupancy, calculate_attention_occupancy

# This script expects that ninja ci-performance-scripts has already been run
# so that we have access to perfRunner and perfCommonUtils
import perfRunner
from perfCommonUtils import Operation

# Try to import amd_arch_db; provide actionable error if missing.
try:
    import amd_arch_db
except ModuleNotFoundError as e:
    print(
        "ERROR: Could not import amd_arch_db (pybind11 GPU arch database).\n"
        f"Reason: {e}\n\n"
        "To build it:\n"
        "  1) Manually build amd_arch_db:\n"
        "     ninja amd_arch_db\n"
        "  2) Add the amd_arch_db to PYTHONPATH\n"
    )
    amd_arch_db = None

# Constants for the new result field names
FIELD_NS = 'ns'
FIELD_BLOCKSIZE = 'blocksize'
FIELD_GRIDSIZE = 'gridsize'
FIELD_VGPR_COUNT = 'vgpr_count'
FIELD_VGPR_SPILLS = 'vgpr_spills'
FIELD_SGPR_COUNT = 'sgpr_count'
FIELD_SGPR_SPILLS = 'sgpr_spills'
FIELD_LDS_ALLOCATED = 'lds_allocated'
FIELD_OCCUPANCY = 'occupancy'
FIELD_WF_PER_WG = 'wf_per_wg'
FIELD_MFMA_WMMA_INSTRUCTION = 'mfma_wmma_instruction'

# Define new data fieldnames (the tuning metrics we're adding)
NEW_DATA_FIELDNAMES = [
    FIELD_NS,
    FIELD_BLOCKSIZE,
    FIELD_GRIDSIZE,
    FIELD_VGPR_COUNT,
    FIELD_VGPR_SPILLS,
    FIELD_SGPR_COUNT,
    FIELD_SGPR_SPILLS,
    FIELD_LDS_ALLOCATED,
    FIELD_OCCUPANCY,
    FIELD_WF_PER_WG,
    FIELD_MFMA_WMMA_INSTRUCTION
]


class TuningData:
    """Class to represent tuning data results."""

    def __init__(self):
        self.ns = None
        self.blocksize = None
        self.gridsize = None
        self.vgpr_count = None
        self.vgpr_spills = None
        self.sgpr_count = None
        self.sgpr_spills = None
        self.lds_allocated = None
        self.occupancy = None
        self.wf_per_wg = None
        self.mfma_wmma_instruction = None

    def to_dict(self):
        """Convert to dictionary format for tsv writing."""
        return self.__dict__


def get_perf_config(operation, test_vector, arch, num_cu):
    """
    Get the performance configuration for the given test vector, architecture,
    and number of compute units.

    Args:
        test_vector: The test vector string.
        arch: The architecture string.
        num_cu: The number of compute units.

    Returns:
        str: The performance configuration string.
    """
    conf_class = perfRunner.PerfConfiguration
    if operation == Operation.ATTENTION:
        conf_class = perfRunner.AttentionConfiguration.from_command_line(test_vector.split(sep=' '), arch, num_cu)
    elif operation == Operation.GEMM:
        conf_class = perfRunner.GemmConfiguration.from_command_line(test_vector.split(sep=' '), arch, num_cu)
    elif operation == Operation.CONV:
        conf_class = perfRunner.ConvConfiguration.from_command_line(test_vector.split(sep=' '), arch, num_cu)
    elif operation == Operation.GEMM_GEMM:
        conf_class = perfRunner.GemmGemmConfiguration.from_command_line(test_vector.split(sep=' '), arch, num_cu)

    return conf_class


def compile_config(conf_class, paths, timestamp):
    rocmlir_gen_options = conf_class.generate_mlir_driver_commandline("")

    # Build the rocmlir-gen command
    rocmlir_gen_cmd = [paths.mlir_paths.rocmlir_gen_path] + rocmlir_gen_options.split()

    # Build the rocmlir-driver command
    rocmlir_driver_cmd = [
        paths.mlir_paths.rocmlir_driver_path,
        "-c",
        "--debug-only=convert-rock-to-gpu,serialize-to-isa",
    ]

    commands = [rocmlir_gen_cmd, rocmlir_driver_cmd]
    out, err = perfRunner.run_pipeline(commands)

    return err


def parse_mfma_wmma_instructions(content):
    """
    Parse MFMA and WMMA instructions from the debug output.

    Args:
        content: String content of the debug output file

    Returns:
        list: Unique list of MFMA/WMMA instruction names
    """
    # Pattern to match MFMA and WMMA instructions
    full_pattern = r'\b(v_(?:mfma|wmma)_[a-zA-Z0-9_]+)\b'
    full_matches = re.findall(full_pattern, content, re.IGNORECASE)

    # Remove duplicates and sort for consistent output
    unique_instructions = list(set(full_matches))

    # Assert that there is only one unique instruction
    size = len(unique_instructions)
    assert size <= 1, \
           f"Expected exactly one unique MFMA/WMMA instruction, found: {size}"

    return unique_instructions


def parse_results(debug_output):
    """
    This function parses the generated output file to gather the desired
    information.debug_output will contain all of the output from running
    rocmlir-driver (debug output and assembly output). It will be structured
    something like the following:
    """
    tuning_data = TuningData()

    # Look for blocksize
    blocksize_match = re.search(r'blockSize:\s*(\d+)', debug_output)
    if not blocksize_match:
        raise ValueError("Could not find blockSize in output")
    tuning_data.blocksize = int(blocksize_match.group(1))

    # Look for gridsize
    gridsize_match = re.search(r'gridSize:\s*(\d+)', debug_output)
    if not gridsize_match:
        raise ValueError("Could not find gridSize in output")
    tuning_data.gridsize = int(gridsize_match.group(1))

    # Look for waveSize
    wavesize_match = re.search(r'waveSize:\s*(\d+)', debug_output)
    if not wavesize_match:
        raise ValueError("Could not find waveSize in output")
    tuning_data.wf_per_wg = int(blocksize_match.group(1)) / int(wavesize_match.group(1))

    # Look for lds_allocated
    lds_match = re.search(r'ldsUsage:\s*(\d+)', debug_output)
    if not lds_match:
        raise ValueError("Could not find ldsUsage in output")
    tuning_data.lds_allocated = int(lds_match.group(1))

    # Look for SGPR count
    sgpr_match = re.search(r'\.sgpr_count:\s+(\d+)', debug_output)
    if not sgpr_match:
        raise ValueError("Could not find sgpr_count in output")
    tuning_data.sgpr_count = int(sgpr_match.group(1))

    # Look for VGPR count
    vgpr_match = re.search(r'\.vgpr_count:\s+(\d+)', debug_output)
    if not vgpr_match:
        raise ValueError("Could not find vgpr_count in output")
    tuning_data.vgpr_count = int(vgpr_match.group(1))

    # Look for SGPR spill count
    sgpr_spill_match = re.search(r'\.sgpr_spill_count:\s+(\d+)',
                                 debug_output)
    if not sgpr_spill_match:
        raise ValueError("Could not find sgpr_spill_count in output")
    tuning_data.sgpr_spills = int(sgpr_spill_match.group(1))

    # Look for VGPR spill count
    vgpr_spill_match = re.search(r'\.vgpr_spill_count:\s+(\d+)',
                                 debug_output)
    if not vgpr_spill_match:
        raise ValueError("Could not find vgpr_spill_count in output")
    tuning_data.vgpr_spills = int(vgpr_spill_match.group(1))

    mfma_wmma_instructions = parse_mfma_wmma_instructions(debug_output)
    if mfma_wmma_instructions:
        tuning_data.mfma_wmma_instruction = mfma_wmma_instructions[0]

    return tuning_data


def parse_perf_config(perf_config, num_cu, arch):
    """
    Parse the perfConfig string to extract tuning parameters.

    Format: attn:v1:MPerBlock,NPerBlock,KPerBlock,MPerWave,NPerWave,kPack,
            splitKFactor,forceUnroll,ThreadCopyMore

    Returns:
        dict: Dictionary containing parsed parameters

    TODO: The format of the perfConfig string is subject to changes in the
          future, so we should at a minimum be keeping this in sync with the
          c++ code, but we should als consider making bindings to the c++ code
          that can be called from here.
    """
    try:
        # Split by ':' to separate operation, version, and parameters
        parts = perf_config.split(':')
        if len(parts) < 2:
            raise ValueError(f"Invalid perfConfig format: {perf_config}")

        # The format is either going to have three parts or two parts. Make sure
        # to properly handle the `operation` case
        # - operation:version:parameters
        # - version:parameters
        version = None
        uses_attn = False
        if len(parts) >= 3:
            # If there are three parts, then we assume the first part denoting
            # the operation is going to be equal to `attn`
            if (parts[0] != 'attn'):
                raise ValueError("Invalid perfConfig format. Expected 'attn' "
                                 "to be the first part.")
            params_str = parts[2]
            version = parts[1]
            uses_attn = True
        else:
            # parameters are after the first ':'
            params_str = parts[1]
            version = parts[0]

            # Attention will be the only operation that uses the v1/v2 format
            # for PerfConfig, so in all other cases we should be using v3
            if (version != 'v3'):
                raise ValueError("Invalid perfConfig format. Expected 'v3' "
                                 "format for non-attention ops.")

        # Split parameters by comma
        params = params_str.split(',')
        parsed_params = {}

        # Calculate minNumWaves based on numCUs and numEUPerCU
        arch_info = amd_arch_db.lookup_arch_info(arch)
        num_eu_per_cu = getattr(arch_info, 'num_eu_per_cu')
        parsed_params['minNumWaves'] = int(num_cu) * num_eu_per_cu
        wave_size = getattr(arch_info, 'wave_size')

        def has_feature(info, feature):
            return (int(info.default_features) & int(feature)) != 0

        has_mfma = has_feature(arch_info, amd_arch_db.GemmFeatures.MFMA)
        has_wmma = has_feature(arch_info, amd_arch_db.GemmFeatures.WMMA)

        # Handle v1/v2 attention formats. Note, the only difference between
        # v1 and v2 are the extra parameters added on at the end of the v2
        # format, which we do not need for this specific case.
        if (uses_attn):
            # MPerBlock = MPerBlockG0 * MPerBlockG1
            parsed_params['MPerBlock'] = int(params[0]) * int(params[1])
            parsed_params['NPerBlock'] = int(params[2])
            parsed_params['KPerBlock'] = int(params[3])
            m_per_wave = int(params[4])
            # NPerWave = max(NPerBlock/minNumWaves, mnPerXdl)
            mn_per_xdl = int(params[5])
            n_per_wave = max(parsed_params['NPerBlock'] / parsed_params['minNumWaves'], mn_per_xdl)
            parsed_params['kPack'] = int(params[6])
            parsed_params['splitKFactor'] = int(params[7])
            parsed_params['MNPerWave'] = m_per_wave * n_per_wave
        elif (has_mfma):
            # Handle MFMA v3 formats
            parsed_params['MPerBlock'] = int(params[0])
            parsed_params['NPerBlock'] = int(params[1])
            parsed_params['KPerBlock'] = int(params[2])
            parsed_params['kPack'] = int(params[5])
            parsed_params['splitKFactor'] = int(params[6])
            # For the MFMA v3 formats we have no way of distinguishing between
            # the XdlopsGemmDerivedParamsAttr and the XdlopsGemmParamsAttr.
            # The only difference between these two perfConfigs strings is that
            # one has a NPerWave value and the other has a MnPerXdl value
            # in the 5th position in the params array.
            # We can use # NPerWave = max(NPerBlock/minNumWaves, params[5]) to
            # cover both of the possibilities and still compute a valid
            # NPerWave value
            m_per_wave = int(params[3])
            n_per_wave = max(parsed_params['NPerBlock'] / parsed_params['minNumWaves'], int(params[4]))
            parsed_params['MNPerWave'] = m_per_wave * n_per_wave
        elif (has_wmma):
            # Handle WMMA v3 formats
            parsed_params['MPerBlock'] = int(params[0])
            parsed_params['NPerBlock'] = int(params[1])
            parsed_params['KPerBlock'] = int(params[2])
            parsed_params['kPack'] = int(params[5])
            parsed_params['splitKFactor'] = int(params[6])
            m_per_wave = int(params[3])
            n_per_wave = int(params[4])
            parsed_params['MNPerWave'] = m_per_wave * n_per_wave
        else:
            # Handle non-accel v3 formats
            block_size = int(params[0])
            m_per_block = int(params[1])
            n_per_block = int(params[2])
            parsed_params['MPerBlock'] = m_per_block
            parsed_params['NPerBlock'] = n_per_block
            parsed_params['KPerBlock'] = int(params[3])
            parsed_params['MNPerWave'] = (m_per_block * n_per_block) / (block_size / wave_size)
            # Non-accel does not use kpack, so default to 1
            parsed_params['kPack'] = 1
            parsed_params['splitKFactor'] = int(params[6])

        return parsed_params

    except (ValueError, IndexError) as e:
        print(f"Error parsing perfConfig '{perf_config}': {e}")
        return None


def extract_mng_from_config(conf_class, operation):
    """
    Extract M, N, and G values from the testVector based on the operation type.

    Args:
        conf_class: Configuration class instance of specified operation
        operation: Operation type (e.g., 'attention', 'gemm', 'conv2d')

    Returns:
        tuple: (M, N, G) values based on operation type
    """
    try:
        if operation == Operation.ATTENTION:
            # For attention ops: M = seq_len_q, N = seq_len_k, G = g * num_heads_q
            m = conf_class.seq_len_q
            n = conf_class.seq_len_k
            g = conf_class.g * conf_class.num_heads_q

        elif operation == Operation.GEMM:
            # For GEMM ops: M = m, N = n, G = g
            m = conf_class.m
            n = conf_class.n
            g = conf_class.g

        elif operation == Operation.GEMM_GEMM:
            # For gemm+gemm ops: M = m, N = o (final output dimension), G = g
            m = conf_class.m
            n = conf_class.o
            g = conf_class.g

        elif operation == Operation.CONV:
            # For conv ops: M = k, N = batch_size * output_height * output_width,
            # G = g
            assert conf_class.direction == 'fwd', \
                "Only forward convolution (-F=1) is supported"

            # For group convolution (g > 1), validate the configuration
            if conf_class.group > 1:
                # Check that output channels and input channels are divisible by
                # group size
                assert conf_class.k % conf_class.group == 0, (
                    f"Invalid group conv config - output channels ({conf_class.k}) "
                    f"not divisible by group size ({conf_class.group})"
                )
                assert conf_class.c % conf_class.group == 0, (
                    f"Invalid group conv config - input channels ({conf_class.c}) "
                    f"not divisible by group size ({conf_class.group})"
                )

            # For group convolution: adjust k by dividing by group size. This
            # will also work in non-group conv case of g = 1
            k_per_group = conf_class.k // conf_class.group
            m = k_per_group
            g = conf_class.group
            n = conf_class.n * conf_class.ho * conf_class.wo

        else:
            print(f"Warning: Unknown operation type '{operation}'")
            return None, None, None

    except (ValueError, TypeError) as e:
        print(f"Warning: Error parsing M, N, G values from testVector: {e}")
        return None, None, None

    return m, n, g


def gather_occupancy_parameters(config, perf_config, conf_class, operation):
    '''
    This function gathers all of the parameters that are needed to calculate
    the theoretical occupancy
    '''
    num_cu = config[1]
    parsed_params = parse_perf_config(perf_config, num_cu, config[0])

    if parsed_params is None:
        return [None] * 8  # Return None values if parsing fails

    # Extract the required parameters for occupancy calculation
    [m, n, g] = extract_mng_from_config(conf_class, operation)

    m_per_block = int(parsed_params['MPerBlock'])
    n_per_block = int(parsed_params['NPerBlock'])
    mn_per_wave = int(parsed_params['MNPerWave'])
    min_num_waves = int(parsed_params['minNumWaves'])
    split_k_factor = int(parsed_params['splitKFactor'])

    return [m, n, g, m_per_block, n_per_block, mn_per_wave, min_num_waves, split_k_factor]


def compile_and_collect_data(config, operation, binaries):
    """
    Compile and collect the resulting data points that we are interested in
    """
    # Get current timestamp in a filesystem-friendly format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use the operation class
    op_type = Operation.from_name(operation)

    # Create a performance configuration class instance
    arch = config[0].split(':')[0]
    num_cu = config[1]
    test_vector = config[2]
    perf_config = config[3]
    tflops = config[4]
    conf_class = get_perf_config(op_type, test_vector, arch, num_cu)
    conf_class.set_perfconfig(perf_config)

    # Compile the config
    debug_output = compile_config(conf_class, binaries,
                                  timestamp)
    if isinstance(debug_output, bytes):
        debug_output = debug_output.decode('utf-8')

    # If the debug output is empty, then this means that the compilation
    # pipeline failed. We expect this to happe for some of the invalid configs
    if not debug_output:
        print(f"Warning: Compilation failed for config {config}. "
              "Skipping calculations.")
        return None

    # Parse the results from the compiled config
    results = parse_results(debug_output)

    # Convert the TFLOPs value to seconds
    results.ns = conf_class.compute_ns_from_tflops(tflops)

    # Calculate occupancy using the method in testing_metrics.py
    [m, n, g, m_per_block, n_per_block,
     mn_per_wave, min_num_waves, split_k_factor] = \
        gather_occupancy_parameters(config, perf_config, conf_class, op_type)
    # If any of the parameters are None, we cannot calculate occupancy
    if None in [m, n, g, m_per_block, n_per_block,
                mn_per_wave, min_num_waves, split_k_factor]:
        print("Warning: Could not gather all parameters for occupancy "
              "calculation for config. Skipping occupancy calculation.")
        results.occupancy = None
    elif op_type == Operation.ATTENTION:
        results.occupancy = calculate_attention_occupancy(n, g, m_per_block,
                                                          n_per_block,
                                                          mn_per_wave, min_num_waves)
    else:
        results.occupancy = calculate_gemm_occupancy(m, n, g, m_per_block, n_per_block,
                                                     mn_per_wave, min_num_waves,
                                                     split_k_factor)

    return results


def create_tsv_writer(configs, output_file):
    """
    Create and initialize a TSV writer with headers.

    Args:
        configs: Dictionary of original configuration dictionaries
        output_file: Path to the output file

    Returns:
        tuple: (file_handle, csv.DictWriter) - caller must close file_handle
    """
    # Get the original fieldnames from the first config entry
    first_config_data = next(iter(configs.values()))
    original_fieldnames = list(first_config_data.keys())
    tsv_fieldnames = original_fieldnames + NEW_DATA_FIELDNAMES

    try:
        tsvfile = open(output_file, 'w', newline='', encoding='utf-8')
        writer = csv.DictWriter(tsvfile, fieldnames=tsv_fieldnames, delimiter='\t')
        
        # Write the header
        writer.writeheader()
        tsvfile.flush()  # Ensure header is written immediately
        
        return tsvfile, writer

    except Exception as e:
        print(f"\nError creating output file: {e}")
        sys.exit(1)


def write_result_to_tsv(writer, config, config_data, result, tsvfile):
    """
    Write a single result row to the TSV file.

    Args:
        writer: csv.DictWriter instance
        config: Configuration tuple
        config_data: Original configuration dictionary
        result: TuningData result object (or None)
        tsvfile: File handle for flushing
    """
    try:
        # Start with the original config data
        row = dict(config_data)  # Copy all original fields

        # Add new tuning data
        if result is None:
            row.update({field: None for field in NEW_DATA_FIELDNAMES})
        else:
            result_dict = result.to_dict()
            row.update({field: result_dict.get(field, '') for field in NEW_DATA_FIELDNAMES})

        writer.writerow(row)
        tsvfile.flush()  # Ensure row is written immediately

    except Exception as e:
        print(f"\nError writing result to tsv: {e}")
        sys.exit(1)


def print_progress(current, total):
    """Print a progress bar to stdout."""
    prefix = "Processing Configs"
    percent = (current / total) * 100
    bar_length = 40
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{prefix}: |{bar}| {current}/{total} ({percent:.1f}%)', end='',
          flush=True)
    if current == total:
        print()  # New line when complete


def main():
    """Main function to process configurations and collect tuning data."""
    parser = argparse.ArgumentParser(
        description="Compile configurations and collect tuning data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--op', "--operation", required=True,
                        help='Operation to perform (e.g., "compile")',
                        choices=['conv', 'gemm', 'attention', 'gemm_gemm'])
    parser.add_argument('config_tsv', help='Path to the tuning database file')

    args = parser.parse_args()

    # Get the paths to the rocmlir binaries
    build_bin_dir = os.path.dirname(os.path.abspath(__file__))
    rocmlir_root = os.path.dirname(build_bin_dir)
    paths = perfRunner.create_paths(None, rocmlir_root)

    # Check if the input config tsv file exists
    if not os.path.exists(args.config_tsv):
        print("Error: The specified config tsv file cannot be found.")
        return 1

    # Parse the configuration file
    configs = perfRunner.read_debug_db(args.config_tsv)

    print(f"Found {len(configs)} configurations to process")

    # Create output file and writer (streaming mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"tuning_results_{timestamp}.tsv"
    tsvfile, writer = create_tsv_writer(configs, output_file)

    try:
        # Process each configuration and write immediately (no accumulation)
        config_keys = list(configs.keys())
        total_configs = len(config_keys)
        
        for i, config in enumerate(config_keys):
            print_progress(i, total_configs)
            config_data = configs[config]
            
            # Compile and collect data for this config
            metrics = compile_and_collect_data(config, args.op, paths)
            
            # Write result immediately
            write_result_to_tsv(writer, config, config_data, metrics, tsvfile)

        print_progress(total_configs, total_configs)
        print(f"\nResults written to {output_file}")

    except Exception as e:
        print(f"\nError during processing: {e}")
        return 1
    finally:
        # Always close the file
        tsvfile.close()

    # If we have reached this point without crashing, it means that we have had
    # a successful run and we can return 0.
    return 0


if __name__ == "__main__":
    sys.exit(main())
