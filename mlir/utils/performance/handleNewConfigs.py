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
from perfRunner import ConvConfiguration, GemmConfiguration, AttentionConfiguration, getArch, getChip, getNumCU

# Global variables

# Set default paths to configuration files
# If --configs-dir is specified, these paths will be appended to it
CONV_FILE_NAME = "tier1-conv-configs"
GEMM_FILE_NAME = "tier1-gemm-configs"
ATTENTION_FILE_NAME = "tier1-attention-configs"

NEW_CONFIGS_DEFAULT = f"../../mlir/utils/performance/problem-config-tier-1-models"
CONV_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{CONV_FILE_NAME}"
GEMM_CONFIGS_DEFAULT = f"../../mlir/utils/performance/configs/{GEMM_FILE_NAME}"
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
    # TODO: Add support for conv+gemm kernels in the future

    # Conv configs start with conv, convfp16, convbfp16, convfp8, convint8, etc.
    if config.startswith("conv"):
        return "conv"
    # Attention configs have -transQ, -transK, -transV, -transO, -seq_len_q, etc.
    if any(flag in config for flag in ["-transQ", "-seq_len_q", "-head_dim_qk"]):
        return "attention"
    # GEMM configs have -transA, -transB, -out_datatype, -m, -n, -k, etc.
    if any(flag in config for flag in ["-transA", "-transB", "-m", "-n", "-k"]):
        return "gemm"

    return None

def _canonicalizeConvConfig(config: str) -> str:
    """Converts a conv config to canonical form for deduplication."""
    obj = ConvConfiguration.fromCommandLine(shlex.split(config), ARCH, NUM_CU)
    return obj.toCommandLine()

def _canonicalizeGemmConfig(config: str) -> str:
    """Converts a GEMM config to canonical form for deduplication."""
    obj = GemmConfiguration.fromCommandLine(shlex.split(config), ARCH, NUM_CU)
    return obj.toCommandLine()

def _canonicalizeAttentionConfig(config: str) -> str:
    """Converts an attention config to canonical form for deduplication."""
    obj = AttentionConfiguration.fromCommandLine(shlex.split(config), ARCH, NUM_CU)
    return obj.toCommandLine()

def canonicalSet(lines: Iterable[str], kind: str) -> Set[str]:
    """Converts a set of configs to canonical form for deduplication."""
    S: Set[str] = set()
    for line in lines:
        if kind == "conv":
            S.add(_canonicalizeConvConfig(line))
        elif kind == "gemm":
            S.add(_canonicalizeGemmConfig(line))
        elif kind == "attention":
            S.add(_canonicalizeAttentionConfig(line))
        else:
            raise ValueError(f"Unknown kind: {kind}")
    return S

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
    
    if args.attn:
        attnPath = args.attn
    elif args.configs_dir:
        attnPath = os.path.join(args.configs_dir, f"{ATTENTION_FILE_NAME}")
    else:
        attnPath = ATTENTION_CONFIGS_DEFAULT
    
    return newPath, convPath, gemmPath, attnPath

def main(argv=None):
    args = parseArgs(argv)
    newConfigs, convConfigs, gemmConfigs, attentionConfigs = resolvePaths(args)

    # Load existing configs
    existingConv = loadExistingConfigs(convConfigs)
    existingGemm = loadExistingConfigs(gemmConfigs)
    existingAttention = loadExistingConfigs(attentionConfigs)

    newConv: list[str] = []
    newGemm: list[str] = []
    newAttention: list[str] = []
    newRaw = readNonEmptyLines(newConfigs)
    for raw in newRaw:
        configType = detectConfigType(raw)
        if not configType:
            print(f"Error: Could not determine config type for: {raw}")
            continue
        if configType == "conv":
            canon = _canonicalizeConvConfig(raw)
            if canon not in existingConv:
                newConv.append(raw)
                existingConv.add(canon)
        elif configType == "gemm":
            canon = _canonicalizeGemmConfig(raw)
            if canon not in existingGemm:
                newGemm.append(raw)
                existingGemm.add(canon)
        elif configType == "attention":
            canon = _canonicalizeAttentionConfig(raw)
            if canon not in existingAttention:
                newAttention.append(raw)
                existingAttention.add(canon)

    # Append new configs to the appropriate files
    _appendConfigs(convConfigs, newConv)
    _appendConfigs(gemmConfigs, newGemm)
    _appendConfigs(attentionConfigs, newAttention)

    print(f"Added {len(newConv)} conv, {len(newGemm)} gemm, {len(newAttention)} attention configs.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
