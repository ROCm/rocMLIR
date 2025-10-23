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
from perfRunner import getArch, getChip, getNumCU

# Global variables

# Set default paths to configuration files
# If --configs-dir is specified, these paths will be appended to it
CONV_FILE_NAME = "tier1-conv-configs"
GEMM_FILE_NAME = "tier1-gemm-configs"
GEMM_GEMM_FILE_NAME = "tier1-gemmgemm-configs"
CONV_GEMM_FILE_NAME = "tier1-convgemm-configs"
ATTENTION_FILE_NAME = "tier1-attention-configs"

NEW_CONFIGS_DEFAULT = f"../../mlir/utils/performance/problem-config-tier-1-models"
CONV_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{CONV_FILE_NAME}"
GEMM_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{GEMM_FILE_NAME}"
GEMM_GEMM_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{GEMM_GEMM_FILE_NAME}"
CONV_GEMM_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{CONV_GEMM_FILE_NAME}"
ATTENTION_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{ATTENTION_FILE_NAME}"

# Get the architecture and number of CUs from the environment
ARCH = getArch()
CHIP = getChip()
NUM_CU = getNumCU(CHIP)

# ---------------------------------------------------

def readNonEmptyLines(path: str) -> list[str]:
    if not os.path.exists(path):
        print(f"Error: {path} does not exist")
        sys.exit(-1)
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

def loadExistingConfigs(filepath):
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

def detectConfigType(config) -> Optional[str]:
    """Detect config type: returns 'conv', 'gemm', or 'attention'."""
    # GEMM+GEMM and CONV+GEMM configs have -gemmO
    if "-gemmO" in config:
        if config.startswith("conv"):
            return "conv_gemm"
        else:
            return "gemm_gemm"
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



def _appendConfigs(path: str, lines: Iterable[str]):
    if not lines:
        return
    with open(path, "a") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")

def parseArgs(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=str, default=NEW_CONFIGS_DEFAULT,
                        help="Path to the file containing new configurations to add")
    parser.add_argument("--configs-dir", type=str, default=None,
                        help="Path to the directory containing the existing configuration files")
    parser.add_argument("--conv", type=str, default=None,
                        help="Path to the file containing existing convolution configurations")
    parser.add_argument("--gemm", type=str, default=None,
                        help="Path to the file containing existing GEMM configurations")
    parser.add_argument("--gemmgemm", type=str, default=None,
                        help="Path to the file containing existing GEMM_GEMM configurations")
    parser.add_argument("--convgemm", type=str, default=None,
                        help="Path to the file containing existing CONV_GEMM configurations")
    parser.add_argument("--attn", type=str, default=None,
                        help="Path to the file containing existing attention configurations")
    
    return parser.parse_args(argv)

def resolvePaths(args):
    """ Resolve paths to configuration files based on command line arguments.
        Priority: explicit conv/gemm/attn paths > --configs-dir > default paths """
    newPath = args.new or NEW_CONFIGS_DEFAULT

    if args.conv:
        convPath = args.conv
    elif args.configs_dir:
        convPath = os.path.join(args.configs_dir, f"{CONV_FILE_NAME}")
    else:
        convPath = CONV_CONFIGS_DEFAULT

    if args.gemm:
        gemmPath = args.gemm
    elif args.configs_dir:
        gemmPath = os.path.join(args.configs_dir, f"{GEMM_FILE_NAME}")
    else:
        gemmPath = GEMM_CONFIGS_DEFAULT
    
    if args.gemmgemm:
        gemmgemmPath = args.gemmgemm
    elif args.configs_dir:
        gemmgemmPath = os.path.join(args.configs_dir, f"{GEMM_GEMM_FILE_NAME}")
    else:
        gemmgemmPath = GEMM_GEMM_CONFIGS_DEFAULT

    if args.convgemm:
        convgemmPath = args.convgemm
    elif args.configs_dir:
        convgemmPath = os.path.join(args.configs_dir, f"{CONV_GEMM_FILE_NAME}")
    else:
        convgemmPath = CONV_GEMM_CONFIGS_DEFAULT
    
    if args.attn:
        attnPath = args.attn
    elif args.configs_dir:
        attnPath = os.path.join(args.configs_dir, f"{ATTENTION_FILE_NAME}")
    else:
        attnPath = ATTENTION_CONFIGS_DEFAULT
    
    return newPath, convPath, gemmPath, gemmgemmPath, convgemmPath, attnPath

def main(argv=None):
    args = parseArgs(argv)
    newConfigs, convConfigs, gemmConfigs, gemmgemmConfigs, convgemmConfigs, attentionConfigs = resolvePaths(args)

    # Load existing configs
    existingConv = loadExistingConfigs(convConfigs)
    existingGemm = loadExistingConfigs(gemmConfigs)
    existingGemmGemm = loadExistingConfigs(gemmgemmConfigs)
    existingConvGemm = loadExistingConfigs(convgemmConfigs)
    existingAttention = loadExistingConfigs(attentionConfigs)

    newConv: list[str] = []
    newGemm: list[str] = []
    newGemmGemm: list[str] = []
    newConvGemm: list[str] = []
    newAttention: list[str] = []

    with open(newConfigs, "r") as f:
        for line in f:
            config = line.strip()
            if not config or config.startswith("#"):
                continue
            configType = detectConfigType(config)
            if configType == "conv":
                if config not in existingConv:
                    newConv.append(config)
                    existingConv.add(config)
            elif configType == "gemm":
                if config not in existingGemm:
                    newGemm.append(config)
                    existingGemm.add(config)
            elif configType == "attention":
                if config not in existingAttention:
                    newAttention.append(config)
                    existingAttention.add(config)
            elif configType == "gemm_gemm":
                if config not in existingGemmGemm:
                    newGemmGemm.append(config)
                    existingGemmGemm.add(config)
            elif configType == "conv_gemm":
                if config not in existingConvGemm:
                    newConvGemm.append(config)
                    existingConvGemm.add(config)
            else:
                print(f"Warning: Could not determine config type for: {config}")

    # Append new configs to the appropriate files
    _appendConfigs(convConfigs, newConv)
    _appendConfigs(gemmConfigs, newGemm)
    _appendConfigs(gemmgemmConfigs, newGemmGemm)
    _appendConfigs(convgemmConfigs, newConvGemm)
    _appendConfigs(attentionConfigs, newAttention)

    print(f"Added:")
    print(f"    {len(newConv)} conv configs.")
    print(f"    {len(newGemm)} gemm configs.")
    print(f"    {len(newAttention)} attention configs.")
    print(f"    {len(newGemmGemm)} gemm+gemm configs.")
    print(f"    {len(newConvGemm)} conv+gemm configs.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
