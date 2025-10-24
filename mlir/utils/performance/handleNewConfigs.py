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
import shlex
from typing import Iterable, Set, Optional
from perfRunner import ConvConfiguration, GemmConfiguration, AttentionConfiguration, get_arch, get_chip, get_num_cu

# Global variables

# Set default paths to configuration files
# If --configs-dir is specified, these paths will be appended to it
CONV_FILE_NAME = "tier1-conv-configs"
GEMM_FILE_NAME = "tier1-gemm-configs"
ATTENTION_FILE_NAME = "tier1-attention-configs"

NEW_CONFIGS_DEFAULT = "../../mlir/utils/performance/problem-config-tier-1-models"
CONV_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{CONV_FILE_NAME}"
GEMM_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{GEMM_FILE_NAME}"
ATTENTION_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{ATTENTION_FILE_NAME}"

# Get the architecture and number of CUs from the environment
ARCH = get_arch()
CHIP = get_chip()
NUM_CU = get_num_cu(CHIP)

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


def detect_conf_type(config) -> Optional[str]:
    """Detect config type: returns 'conv', 'gemm', or 'attention'."""
    # TODO: Add support for conv+gemm kernels in the future

    # Conv configs start with conv, convfp16, convbfp16, convfp8, convint8, etc.
    if config.startswith("conv"):
        return "conv"
    # Attention configs have -transQ, -transK, -transV, -transO, -seq_len_q, etc.
    if any(flag in config for flag in ["-transQ", "-seq_len_q", "-head_dim_qk"]):
        return "attention"
    # GEMM configs have -transA, -transB, -m, -n, -k, etc.
    if any(flag in config for flag in ["-transA", "-transB", "-m", "-n", "-k"]):
        return "gemm"

    return None


def _canonicalize_conv_config(config: str) -> str:
    """Converts a conv config to canonical form for deduplication."""
    obj = ConvConfiguration.from_command_line(shlex.split(config), ARCH, NUM_CU)
    return obj.to_command_line()


def _canonicalize_gemm_config(config: str) -> str:
    """Converts a GEMM config to canonical form for deduplication."""
    obj = GemmConfiguration.from_command_line(shlex.split(config), ARCH, NUM_CU)
    return obj.to_command_line()


def _canonicalize_attn_config(config: str) -> str:
    """Converts an attention config to canonical form for deduplication."""
    obj = AttentionConfiguration.from_command_line(shlex.split(config), ARCH, NUM_CU)
    return obj.to_command_line()


def canonical_set(lines: Iterable[str], kind: str) -> Set[str]:
    """Converts a set of configs to canonical form for deduplication."""
    c_set: Set[str] = set()
    for line in lines:
        if kind == "conv":
            c_set.add(_canonicalize_conv_config(line))
        elif kind == "gemm":
            c_set.add(_canonicalize_gemm_config(line))
        elif kind == "attention":
            c_set.add(_canonicalize_attn_config(line))
        else:
            raise ValueError(f"Unknown kind: {kind}")
    return c_set


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

    if args.attn:
        attn_path = args.attn
    elif args.configs_dir:
        attn_path = os.path.join(args.configs_dir, f"{ATTENTION_FILE_NAME}")
    else:
        attn_path = ATTENTION_CONFIGS_DEFAULT

    return new_path, conv_path, gemm_path, attn_path


def main(argv=None):
    args = parse_args(argv)
    new_configs, conv_configs, gemm_configs, attn_configs = resolve_paths(args)

    # Load existing configs
    existing_conv = load_existing_configs(conv_configs)
    existing_gemm = load_existing_configs(gemm_configs)
    existing_attn = load_existing_configs(attn_configs)

    new_conv: list[str] = []
    new_gemm: list[str] = []
    new_attn: list[str] = []
    new_raw = read_non_empty_lines(new_configs)
    for raw in new_raw:
        conf_type = detect_conf_type(raw)
        if not conf_type:
            print(f"Error: Could not determine config type for: {raw}")
            continue
        if conf_type == "conv":
            canon = _canonicalize_conv_config(raw)
            if canon not in existing_conv:
                new_conv.append(raw)
                existing_conv.add(canon)
        elif conf_type == "gemm":
            canon = _canonicalize_gemm_config(raw)
            if canon not in existing_gemm:
                new_gemm.append(raw)
                existing_gemm.add(canon)
        elif conf_type == "attention":
            canon = _canonicalize_attn_config(raw)
            if canon not in existing_attn:
                new_attn.append(raw)
                existing_attn.add(canon)

    # Append new configs to the appropriate files
    _append_configs(conv_configs, new_conv)
    _append_configs(gemm_configs, new_gemm)
    _append_configs(attn_configs, new_attn)

    print(f"Added {len(new_conv)} conv, {len(new_gemm)} gemm, {len(new_attn)} attention configs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
