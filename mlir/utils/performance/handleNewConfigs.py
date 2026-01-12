""" handleNewConfigs.py

This script processes a list of new MLIR problem configurations and appends unique entries
to their respective configuration files based on type: convolution, GEMM, or attention. It
ensures no duplicate configurations are added and classifies each config line.

Usage:
    Run this script as a standalone program. It will read new configuration lines,
    classify and deduplicate them, and append them to the appropriate config files.
"""

import os
import sys
from typing import Iterable, Optional
from perfCommonUtils import Operation

# Global variables

# Set default paths to configuration files
# If --configs-dir is specified, these paths will be appended to it
CONV_FILE_NAME = "tier1-conv-configs"
GEMM_FILE_NAME = "tier1-gemm-configs"
GEMM_GEMM_FILE_NAME = "tier1-gemmgemm-configs"
CONV_GEMM_FILE_NAME = "tier1-convgemm-configs"
ATTENTION_FILE_NAME = "tier1-attention-configs"

NEW_CONFIGS_DEFAULT = "../../mlir/utils/performance/problem-config-tier-1-models"
CONV_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{CONV_FILE_NAME}"
GEMM_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{GEMM_FILE_NAME}"
GEMM_GEMM_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{GEMM_GEMM_FILE_NAME}"
CONV_GEMM_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{CONV_GEMM_FILE_NAME}"
ATTENTION_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{ATTENTION_FILE_NAME}"

# ---------------------------------------------------


def read_non_empty_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        print(f"Error: {path} does not exist")
        sys.exit(-1)
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def load_existing_configs(filepath):
    """Load existing configs from a file into a set (stripped, ignoring empty lines and comments)."""
    configs = set()
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    configs.add(line)
    else:
        print(f"Error: {filepath} does not exist")
        sys.exit(-1)

    return configs


def detect_conf_type(config) -> Optional[Operation]:
    """Detect config type: returns an Operation enum value or None."""
    # GEMM+GEMM and CONV+GEMM configs have -gemmO
    if "-gemmO" in config:
        if config.startswith("conv"):
            return Operation.CONV_GEMM
        return Operation.GEMM_GEMM
    if config.startswith("conv"):
        return Operation.CONV
    if any(flag in config for flag in ["-transQ", "-seq_len_q", "-head_dim_qk"]):
        return Operation.ATTENTION
    if any(flag in config for flag in ["-transA", "-transB", "-m", "-n", "-k"]):
        return Operation.GEMM
    return None


def _append_configs(path: str, lines: Iterable[str]):
    if not lines:
        return
    with open(path, "a") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--new",
                        type=str,
                        default=NEW_CONFIGS_DEFAULT,
                        help="Path to the file containing new configurations to add")
    parser.add_argument("--configs-dir",
                        type=str,
                        default=None,
                        help="Path to the directory containing the existing configuration files")
    parser.add_argument("--conv",
                        type=str,
                        default=None,
                        help="Path to the file containing existing convolution configurations")
    parser.add_argument("--gemm",
                        type=str,
                        default=None,
                        help="Path to the file containing existing GEMM configurations")
    parser.add_argument("--gemm_gemm",
                        type=str,
                        default=None,
                        help="Path to the file containing existing GEMM_GEMM configurations")
    parser.add_argument("--conv_gemm",
                        type=str,
                        default=None,
                        help="Path to the file containing existing CONV_GEMM configurations")
    parser.add_argument("--attn",
                        type=str,
                        default=None,
                        help="Path to the file containing existing attention configurations")

    return parser.parse_args(argv)


def resolve_paths(args):
    """ Resolve paths to configuration files based on command line arguments.
        Priority: explicit conv/gemm/attn paths > --configs-dir > default paths """
    new_path = args.new or NEW_CONFIGS_DEFAULT

    if args.conv:
        conv_path = args.conv
    elif args.configs_dir:
        conv_path = os.path.join(args.configs_dir, f"{CONV_FILE_NAME}")
    else:
        conv_path = CONV_CONFIGS_DEFAULT

    if args.gemm:
        gemm_path = args.gemm
    elif args.configs_dir:
        gemm_path = os.path.join(args.configs_dir, f"{GEMM_FILE_NAME}")
    else:
        gemm_path = GEMM_CONFIGS_DEFAULT

    if args.gemm_gemm:
        gemm_gemm_path = args.gemm_gemm
    elif args.configs_dir:
        gemm_gemm_path = os.path.join(args.configs_dir, f"{GEMM_GEMM_FILE_NAME}")
    else:
        gemm_gemm_path = GEMM_GEMM_CONFIGS_DEFAULT

    if args.conv_gemm:
        conv_gemm_path = args.conv_gemm
    elif args.configs_dir:
        conv_gemm_path = os.path.join(args.configs_dir, f"{CONV_GEMM_FILE_NAME}")
    else:
        conv_gemm_path = CONV_GEMM_CONFIGS_DEFAULT

    if args.attn:
        attn_path = args.attn
    elif args.configs_dir:
        attn_path = os.path.join(args.configs_dir, f"{ATTENTION_FILE_NAME}")
    else:
        attn_path = ATTENTION_CONFIGS_DEFAULT

    return new_path, conv_path, gemm_path, gemm_gemm_path, conv_gemm_path, attn_path


def main(argv=None):
    args = parse_args(argv)
    new_configs, conv_configs, gemm_configs, gemm_gemm_configs, conv_gemm_configs, attn_configs = resolve_paths(
        args)

    # Load existing configs
    existing_conv = load_existing_configs(conv_configs)
    existing_gemm = load_existing_configs(gemm_configs)
    existing_gemm_gemm = load_existing_configs(gemm_gemm_configs)
    existing_conv_gemm = load_existing_configs(conv_gemm_configs)
    existing_attn = load_existing_configs(attn_configs)

    new_conv: list[str] = []
    new_gemm: list[str] = []
    new_gemm_gemm: list[str] = []
    new_conv_gemm: list[str] = []
    new_attn: list[str] = []
    unrecognized_configs: list[str] = []

    with open(new_configs, "r") as f:
        for line in f:
            config = line.strip()
            if not config or config.startswith("#"):
                continue
            op = detect_conf_type(config)
            if op == Operation.CONV:
                if config not in existing_conv:
                    new_conv.append(config)
                    existing_conv.add(config)
            elif op == Operation.GEMM:
                if config not in existing_gemm:
                    new_gemm.append(config)
                    existing_gemm.add(config)
            elif op == Operation.ATTENTION:
                if config not in existing_attn:
                    new_attn.append(config)
                    existing_attn.add(config)
            elif op == Operation.GEMM_GEMM:
                if config not in existing_gemm_gemm:
                    new_gemm_gemm.append(config)
                    existing_gemm_gemm.add(config)
            elif op == Operation.CONV_GEMM:
                if config not in existing_conv_gemm:
                    new_conv_gemm.append(config)
                    existing_conv_gemm.add(config)
            else:
                print(f"Warning: Could not determine config type for: {config}")
                unrecognized_configs.append(config)

    # Append new configs to the appropriate files
    _append_configs(conv_configs, new_conv)
    _append_configs(gemm_configs, new_gemm)
    _append_configs(gemm_gemm_configs, new_gemm_gemm)
    _append_configs(conv_gemm_configs, new_conv_gemm)
    _append_configs(attn_configs, new_attn)

    print("Added:")
    print(f"    {len(new_conv)} conv configs.")
    print(f"    {len(new_gemm)} gemm configs.")
    print(f"    {len(new_attn)} attention configs.")
    print(f"    {len(new_gemm_gemm)} gemm+gemm configs.")
    print(f"    {len(new_conv_gemm)} conv+gemm configs.")

    if unrecognized_configs:
        print(f"\nWarning: {len(unrecognized_configs)} unrecognized config(s) were skipped.")
        print("Unrecognized configs:")
        for config in unrecognized_configs:
            print(f"    {config}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
