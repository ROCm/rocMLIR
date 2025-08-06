""" handleNewConfigs.py

This script processes a list of new MLIR problem configurations and appends unique entries
to their respective configuration files based on type: convolution, GEMM, or attention. It
ensures no duplicate configurations are added and classifies each config line.

Usage:
    Run this script as a standalone program. It will read new configuration lines from `newConfigs`,
    classify and deduplicate them, and append them to the appropriate config files.
"""

import os
import sys

newConfigs = "../../mlir/utils/performance/problem-config-tier-1-models"
convConfigs = "../../mlir/utils/performance/configs/tier1-conv-configs"
gemmConfigs = "../../mlir/utils/performance/configs/tier1-gemm-configs"
attentionConfigs = "../../mlir/utils/performance/configs/tier1-attention-configs"

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

def detectConfigType(config):
    """Detect config type: returns 'conv', 'gemm', or 'attention'."""
    # Conv configs start with conv, convfp16, convbfp16, convfp8, convint8, etc.
    if config.startswith("conv"):
        return "conv"
    # Attention configs have -transQ, -transK, -transV, -transO, -seq_len_q, etc.
    if any(flag in config for flag in ["-transQ", "-seq_len_q", "-head_dim_qk"]):
        return "attention"
    # GEMM configs have -transA, -transB, -out_datatype, -m, -n, -k, etc.
    if any(flag in config for flag in ["-transA", "-transB", "-out_datatype", "-m", "-n", "-k"]):
        return "gemm"
    return None

def main():
    # Load existing configs
    existingConv = loadExistingConfigs(convConfigs)
    existingGemm = loadExistingConfigs(gemmConfigs)
    existingAttention = loadExistingConfigs(attentionConfigs)

    # Prepare to append new unique configs
    newConv, newGemm, newAttention = [], [], []

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
            else:
                print(f"Error: Could not determine config type for: {config}")
                sys.exit(-1)

    # Append new configs to their respective files
    if newConv:
        with open(convConfigs, "a") as f:
            for config in newConv:
                f.write(config + "\n")
    if newGemm:
        with open(gemmConfigs, "a") as f:
            for config in newGemm:
                f.write(config + "\n")
    if newAttention:
        with open(attentionConfigs, "a") as f:
            for config in newAttention:
                f.write(config + "\n")

    print(f"Added {len(newConv)} conv, {len(newGemm)} gemm, {len(newAttention)} attention configs.")

if __name__ == "__main__":
    main()
