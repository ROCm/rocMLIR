#!/usr/bin/env python3
"""Tuning Config Classifier Trainer

Trains an XGBoost classifier per (arch, dtype) on tuning data (.debug files)
to predict whether a given (problem, config) pair will be performant. Exports
the trained models as flat constexpr C++ arrays for zero-dependency inference
in GridwiseGemmParams.cpp.

All features are cross-features combining problem dimensions AND config
parameters. No raw problem dims, no raw config params, no raw device info.
This prevents overfitting and ensures generalization across problem sizes.

Usage:
    python trainClassifier.py /path/to/tuning-data --update
    python trainClassifier.py /path/to/tuning-data --th 0.85 --n-estimators 30 --update
    python trainClassifier.py /path/to/tuning-data  # dry run
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quickTuningGen import (
    GEMM_COLUMNS,
    CONV_COLUMNS,
    ATTENTION_COLUMNS,
    get_target_columns,
    validate_files,
    load_data,
    parse_perfconfig,
    get_instruction_type,
    is_accel,
)

# Must match AccelFeatureIndex enum in GridwiseGemmParams.cpp exactly.
ACCEL_FEATURE_NAMES = [
    'gridSizePerCU',
    'mPadFrac',
    'nPadFrac',
    'kPadFrac',
    'paddingOverhead',
    'numWaves',
    'kIters',
    'ldsElements',
    'kPerBlockPerMnPerXdl',
    'outputSwizzle',
    'wavesPerEU',
    'gridGroupSize',
    'scheduleVersion',
]

# =============================================================================
# PerfConfig Parsing
# =============================================================================


def _derive_legacy_waves(mnPerXdl, mPerBlock, nPerBlock, mPerWave, is_wmma):
    """Replicate C++ handleLegacyNPerWaveOrMnPerXdl for v3 / attn:v2.

    In legacy formats, nPerWave is not stored directly; instead a single
    field is stored and (mPerWave, nPerWave, mnPerXdl) are derived at parse time.
    """
    if is_wmma:
        nPerWave = mnPerXdl  # raw value IS nPerWave; mnPerXdl defaults to 16
        mnPerXdl = 16
    else:
        maxWavesPerWG = 4
        mWaves = min(mPerBlock // max(mPerWave, 1), maxWavesPerWG)
        nWaves = maxWavesPerWG // max(mWaves, 1)
        mPerWave = mPerBlock // max(mWaves, 1)
        nPerWave = max(nPerBlock // max(nWaves, 1), mnPerXdl)
    return mPerWave, nPerWave, mnPerXdl


def parse_tuning_params(perfconfig, is_wmma=False):
    """Extract tuning parameters needed for feature computation.

    Handles all perfconfig formats (see RockDialect.cpp parsers):
      - gemm/conv v3 (11 params): legacy nPerWave derivation
      - gemm/conv v4 (14 params): nPerWave stored directly
      - attention  attn:v2 (11 params): legacy nPerWave derivation
      - attention  attn:v3 (13 params): nPerWave stored directly

    Returns dict with keys for feature extraction.
    """
    try:
        fmt, version, params_str = parse_perfconfig(perfconfig)
        params = list(map(int, params_str))
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"Cannot parse perfconfig: {perfconfig!r}")

    if fmt == 'attn':
        # attn:v2 → 11 params, attn:v3 → 13 params
        if len(params) not in (11, 13):
            raise ValueError(f"Unsupported attention format: attn:v{version} "
                             f"({len(params)} params), expected attn:v2 (11) or attn:v3 (13)")
        idx = 0
        mPerBlock = params[idx]
        idx += 1  # mPerBlockG0
        idx += 1  # mPerBlockG1 (skip)
        nPerBlock = params[idx]
        idx += 1  # nPerBlockG0
        kpackPerBlock = params[idx]
        idx += 1
        mPerWave = params[idx]
        idx += 1

        if version > 2:  # attn:v3: nPerWave, mnPerXdl stored directly
            nPerWave = params[idx]
            idx += 1
            mnPerXdl = params[idx]
            idx += 1
        else:  # attn:v2: legacy derivation
            mPerWave, nPerWave, mnPerXdl = _derive_legacy_waves(params[idx], mPerBlock, nPerBlock,
                                                                mPerWave, is_wmma)
            idx += 1

        kpack = params[idx]
        idx += 1
        splitKFactor = params[idx]
        idx += 1
        scheduleVersion = params[idx]
        idx += 1
        outputSwizzle = params[idx]
        idx += 1
        wavesPerEU = params[idx] if version > 2 else 0
        idx += (1 if version > 2 else 0)
        gridGroupSize = 0  # not in attention formats

    else:
        # v3 → 11 params, v4 → 14 params
        if len(params) not in (11, 14):
            raise ValueError(f"Unsupported accel format: v{version} "
                             f"({len(params)} params), expected v3 (11) or v4 (14)")
        idx = 0
        mPerBlock = params[idx]
        idx += 1
        nPerBlock = params[idx]
        idx += 1
        kpackPerBlock = params[idx]
        idx += 1
        mPerWave = params[idx]
        idx += 1

        if version > 3:  # v4: nPerWave, mnPerXdl stored directly
            nPerWave = params[idx]
            idx += 1
            mnPerXdl = params[idx]
            idx += 1
        else:  # v3: legacy derivation
            mPerWave, nPerWave, mnPerXdl = _derive_legacy_waves(params[idx], mPerBlock, nPerBlock,
                                                                mPerWave, is_wmma)
            idx += 1

        kpack = params[idx]
        idx += 1
        splitKFactor = params[idx]
        idx += 1
        scheduleVersion = params[idx]
        idx += 1
        outputSwizzle = params[idx]
        idx += 1
        if version > 3:  # v4 has wavesPerEU and gridGroupSize
            wavesPerEU = params[idx]
            idx += 1
            gridGroupSize = params[idx]
            idx += 1
        else:
            wavesPerEU = 0
            gridGroupSize = 0

    return {
        'mPerBlock': mPerBlock,
        'nPerBlock': nPerBlock,
        'kpackPerBlock': kpackPerBlock,
        'mPerWave': mPerWave,
        'nPerWave': nPerWave,
        'mnPerXdl': mnPerXdl,
        'kpack': kpack,
        'splitKFactor': splitKFactor,
        'scheduleVersion': scheduleVersion,
        'outputSwizzle': outputSwizzle,
        'wavesPerEU': wavesPerEU,
        'gridGroupSize': gridGroupSize,
    }


# =============================================================================
# Feature Extraction (cross-features only, matching C++ extractAccelFeatures)
# =============================================================================


def mod_1_to_n(x, n):
    """Match the C++ math_util::mod_1_to_n: returns x%n, but n instead of 0."""
    r = x % n
    return n if r == 0 else r


def compute_accel_features(m, n, k, cfg, num_cu):
    """Compute the feature vector matching extractAccelFeatures() in C++.

    Args:
        m, n, k: GEMM dimensions (or equivalent for attention/conv)
        cfg: dict from parse_tuning_params()
        num_cu: number of CUs for this architecture (from the .debug data)

    Returns:
        list of floats with length == len(ACCEL_FEATURE_NAMES)
    """
    mPB = float(cfg['mPerBlock'])
    nPB = float(cfg['nPerBlock'])
    kPB = float(cfg['kpackPerBlock'])
    kPack = float(cfg['kpack'])
    mPW = float(cfg['mPerWave'])
    nPW = float(cfg['nPerWave'])
    mnPerXdl = float(cfg['mnPerXdl'])
    splitK = float(cfg['splitKFactor'])
    kEff = kPB * kPack

    mTiles = np.ceil(m / mPB)
    nTiles = np.ceil(n / nPB)
    kIters = np.ceil(k / kEff) if kEff > 0 else 0
    numWaves = (mPB / mPW) * (nPB / nPW)
    gridSize = mTiles * nTiles * splitK
    numCU = float(num_cu)

    mPadded = mTiles * mPB
    nPadded = nTiles * nPB
    kPadded = kIters * kEff
    originalVolume = m * n * k
    paddedVolume = mPadded * nPadded * kPadded

    mPadFrac = ((int(mPB) - mod_1_to_n(int(m), int(mPB))) % int(mPB)) / mPB if mPB > 0 else 0
    nPadFrac = ((int(nPB) - mod_1_to_n(int(n), int(nPB))) % int(nPB)) / nPB if nPB > 0 else 0
    kPadFrac = ((int(kEff) - mod_1_to_n(int(k), int(kEff))) % int(kEff)) / kEff if kEff > 0 else 0
    paddingOverhead = (paddedVolume / originalVolume - 1.0) if originalVolume > 0 else 0

    return [
        gridSize / numCU,  # gridSizePerCU
        mPadFrac,  # mPadFrac
        nPadFrac,  # nPadFrac
        kPadFrac,  # kPadFrac
        paddingOverhead,  # paddingOverhead
        numWaves,  # numWaves
        kIters,  # kIters
        (mPB + nPB) * kPB * kPack,  # ldsElements
        kEff / mnPerXdl if mnPerXdl > 0 else 0,  # kPerBlockPerMnPerXdl
        float(cfg['outputSwizzle']),  # outputSwizzle
        float(cfg['wavesPerEU']),  # wavesPerEU
        float(cfg['gridGroupSize']),  # gridGroupSize
        float(cfg['scheduleVersion']),  # scheduleVersion
    ]


def get_gemm_dims(row, op):
    """Extract (M, N, K) from a data row depending on the operation type."""
    if op == 'gemm':
        return row['M'], row['N'], row['K']
    elif op == 'conv':
        return (row.get('N', 1) * row.get('H', 1) * row.get('W', 1), row.get('K',
                                                                             1), row.get('C', 1))
    elif op == 'attention':
        return row.get('SeqLenQ', 1), row.get('SeqLenK', 1), row.get('HeadDimQK', 1)
    raise ValueError(f"Unexpected op: {op!r}")


# =============================================================================
# Training
# =============================================================================


def build_dataset(df, op, threshold, arch, dtype):
    """Build feature matrix and labels for a specific (arch, dtype).

    For each (problem, config) pair, label is 1 if TFlops >= threshold * best
    TFlops for that problem, else 0.
    """
    target_cols = get_target_columns(op)
    is_wmma = arch.startswith("gfx1") and dtype != "f32"

    mask = (df['Chip'] == arch) & (df['DataType'] == dtype)
    df_filtered = df[mask]
    if df_filtered.empty:
        return np.array([]), np.array([])

    if 'numCU' not in df_filtered.columns:
        raise ValueError(f"numCU column not found for {arch}_{dtype}")
    num_cu = int(df_filtered['numCU'].iloc[0])
    print(f"    numCU={num_cu}")

    group_cols = target_cols + ['PerfConfig']
    agg = df_filtered.groupby(group_cols, as_index=False)['TFlops'].max()
    # NaN/zero TFlops → 0 so they always label as bad configs
    agg['TFlops'] = agg['TFlops'].fillna(0)
    best = agg.groupby(target_cols, as_index=False)['TFlops'].max()
    best = best.rename(columns={'TFlops': 'BestTFlops'})
    agg = agg.merge(best, on=target_cols)
    agg['label'] = (agg['TFlops'] >= threshold * agg['BestTFlops']).astype(int)

    all_features = []
    all_labels = []

    for _, row in agg.iterrows():
        cfg = parse_tuning_params(row['PerfConfig'], is_wmma=is_wmma)
        if cfg is None:
            raise ValueError(f"Failed to parse PerfConfig: {row['PerfConfig']}")

        m, n, k = get_gemm_dims(row, op)
        features = compute_accel_features(m, n, k, cfg, num_cu)
        all_features.append(features)
        all_labels.append(row['label'])

    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int32)


def train_model(X, y, n_estimators, max_depth):
    """Train an XGBoost classifier."""
    pos = y.sum()
    neg = len(y) - pos
    scale = neg / max(pos, 1)
    print(f"    Samples: {len(y)}, positive: {pos} ({100*pos/len(y):.1f}%), "
          f"scale_pos_weight={scale:.2f}")

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        scale_pos_weight=scale,
        eval_metric='logloss',
        verbosity=0,
    )
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    """Print basic evaluation metrics."""
    y_pred = model.predict(X)
    tp = ((y_pred == 1) & (y == 1)).sum()
    fp = ((y_pred == 1) & (y == 0)).sum()
    fn = ((y_pred == 0) & (y == 1)).sum()
    tn = ((y_pred == 0) & (y == 0)).sum()
    accuracy = (tp + tn) / len(y)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"    Accuracy={accuracy:.4f}  Precision={precision:.4f}  "
          f"Recall={recall:.4f}  TP={tp} FP={fp} FN={fn} TN={tn}")
    if fn > 0:
        print(f"    WARNING: {fn} good configs would be incorrectly filtered out")


# =============================================================================
# Export to C++
# =============================================================================


def extract_trees(model):
    """Extract tree structure from an XGBoost model as flat arrays."""
    booster = model.get_booster()
    dump = booster.get_dump(dump_format='json')
    trees = []
    for tree_json in dump:
        tree_data = json.loads(tree_json)
        nodes = []
        _flatten_xgb_tree(tree_data, nodes)
        trees.append(nodes)
    base_score = float(booster.attr('base_score') or 0.0)
    return trees, base_score


def _flatten_xgb_tree(node, nodes):
    """Recursively flatten an XGBoost JSON tree into a node list."""
    idx = len(nodes)
    nodes.append(None)

    if 'leaf' in node:
        nodes[idx] = (-1, 0.0, -1, -1, node['leaf'])
    else:
        feature_idx = int(node['split'][1:]) if node['split'].startswith('f') else int(
            node['split'])
        threshold = node['split_condition']

        left_idx = len(nodes)
        for child in node['children']:
            if child['nodeid'] == node['yes']:
                _flatten_xgb_tree(child, nodes)
                break

        right_idx = len(nodes)
        for child in node['children']:
            if child['nodeid'] == node['no']:
                _flatten_xgb_tree(child, nodes)
                break

        nodes[idx] = (feature_idx, threshold, left_idx, right_idx, 0.0)

    return idx


def sanitize_name(key):
    """Convert 'gfx942_f16' to a valid C++ identifier fragment."""
    return re.sub(r'[^a-zA-Z0-9]', '_', key)


def format_array(name, dtype, values, indent=""):
    """Format a C++ constexpr array."""
    if dtype == 'float':
        formatted = ', '.join(f'{v:.6f}f' for v in values)
    else:
        formatted = ', '.join(str(int(v)) for v in values)
    return f"{indent}constexpr {dtype} {name}[] = {{{formatted}}};"


def generate_classifier_data(key, trees, base_score, decision_threshold=0.0):
    """Generate C++ constexpr arrays for one (arch, dtype) classifier."""
    prefix = sanitize_name(key)
    lines = [f"// Classifier for {key}: {len(trees)} trees"]

    for i, nodes in enumerate(trees):
        feature_idxs = [n[0] for n in nodes]
        thresholds = [n[1] for n in nodes]
        left_children = [n[2] for n in nodes]
        right_children = [n[3] for n in nodes]
        leaf_values = [n[4] for n in nodes]

        lines.append(format_array(f'{prefix}_feat_{i}', 'int16_t', feature_idxs))
        lines.append(format_array(f'{prefix}_thr_{i}', 'float', thresholds))
        lines.append(format_array(f'{prefix}_left_{i}', 'int16_t', left_children))
        lines.append(format_array(f'{prefix}_right_{i}', 'int16_t', right_children))
        lines.append(format_array(f'{prefix}_leaf_{i}', 'float', leaf_values))

    tree_entries = []
    for i, nodes in enumerate(trees):
        tree_entries.append(f"    {{{len(nodes)}, {prefix}_feat_{i}, {prefix}_thr_{i}, "
                            f"{prefix}_left_{i}, {prefix}_right_{i}, {prefix}_leaf_{i}}}")
    lines.append(f"constexpr TreeEnsembleTree {prefix}_trees[] = {{")
    lines.append(",\n".join(tree_entries))
    lines.append("};")
    lines.append(f"constexpr TreeEnsemble {prefix}_ensemble = "
                 f"{{{prefix}_trees, {len(trees)}, {base_score:.6f}f, {decision_threshold:.6f}f}};")
    lines.append("")

    return "\n".join(lines), prefix


def get_output_path():
    script_dir = Path(__file__).resolve().parent
    return (script_dir.parent.parent.parent /
            "include/mlir/Dialect/Rock/Tuning/QuickTuningClassifier.inc")


def get_generator_path():
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / ".git").exists() or (parent / "mlir").is_dir():
            try:
                return script_path.relative_to(parent)
            except ValueError:
                pass
    return script_path.name


def write_inc_file(classifiers):
    """Write the QuickTuningClassifier.inc file.

    Args:
        classifiers: list of (key, cpp_data_str, prefix) tuples
    """
    path = get_output_path()
    lines = [
        f"// Generated by: {get_generator_path()}",
        "//",
        "// Decision-tree ensemble weights for filtering tuning configurations.",
        "// One classifier per (arch, dtype) combination, looked up at runtime.",
        "//",
        "// All features are cross-features (problem x config interactions).",
        "// No raw problem dimensions, config parameters, or device info.",
        "//",
        "// For XGBoost binary classification, leaf values are raw log-odds scores.",
        "// Prediction: accept if (baseScore + sum_of_tree_outputs) >= decisionThreshold.",
        "",
        "// clang-format off",
        "",
        "#ifdef CLASSIFIER_DATA_GEN",
        "",
    ]

    lines.append("// BEGIN_CLASSIFIER_DATA")
    if classifiers:
        for key, data_str, prefix in classifiers:
            lines.append("")
            lines.append(data_str)
    lines.append("// END_CLASSIFIER_DATA")
    lines.append("")

    lines.append("// BEGIN_CLASSIFIER_TABLE")
    lines.append("constexpr ClassifierEntry classifierTable[] = {")
    if classifiers:
        for key, data_str, prefix in classifiers:
            lines.append(f'    {{{{"{key}"}}, &{prefix}_ensemble}},')
    else:
        lines.append('    {{{"_placeholder_"}, nullptr},')
    lines.append("};")
    lines.append("// END_CLASSIFIER_TABLE")
    lines.append("")
    lines.append("#endif // CLASSIFIER_DATA_GEN")
    lines.append("")
    lines.append("// clang-format on")

    path.write_text("\n".join(lines))
    print(f"\nWrote {path}")
    print(f"  {len(classifiers)} classifier(s)")
    for key, _, _ in classifiers:
        print(f"  - {key}")


# =============================================================================
# Main
# =============================================================================


def train_op(df, op, pargs):
    """Train classifiers for a single op on already-loaded data.

    Returns a list of (key, data_str, prefix) tuples.
    """
    arch_dtype_pairs = sorted(df.groupby(['Chip', 'DataType']).groups.keys())
    print(f"Found {len(arch_dtype_pairs)} (arch, dtype) combinations")

    classifiers = []
    for arch, dtype in arch_dtype_pairs:
        if not is_accel(arch, dtype, op):
            continue

        key = f"{op}_{arch}_{dtype}"
        instr = get_instruction_type(arch, dtype, op)
        print(f"\n=== {key} ({instr}) ===")

        X, y = build_dataset(df, op, pargs.th, arch, dtype)

        if len(X) < pargs.min_samples:
            raise RuntimeError(f"{key}: too few samples ({len(X)})")

        if y.sum() == 0 or y.sum() == len(y):
            raise RuntimeError(f"{key}: all samples have same label")

        print(f"    Training...")
        model = train_model(X, y, pargs.n_estimators, pargs.max_depth)

        print(f"    Evaluating on training data:")
        evaluate_model(model, X, y)

        trees, base_score = extract_trees(model)

        total_nodes = sum(len(t) for t in trees)
        print(f"    Exported: {len(trees)} trees, {total_nodes} total nodes")

        data_str, prefix = generate_classifier_data(key, trees, base_score)
        classifiers.append((key, data_str, prefix))

    return classifiers


def main(args=None):
    parser = argparse.ArgumentParser(
        prog='trainClassifier.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Train tuning config classifiers and export to C++ arrays.',
        epilog='''
Examples:
    %(prog)s /path/to/tuning-data --update
    %(prog)s /path/to/tuning-data --th 0.85 --n-estimators 30 --update
    %(prog)s /path/to/tuning-data  # dry run

Expects directory layout: <tuning-dir>/<arch>/{gemm,conv,attention}/*.debug
''')

    parser.add_argument('tuning_dir',
                        type=Path,
                        metavar='TUNING_DIR',
                        help='Tuning data root dir (layout: <dir>/<arch>/<op>/*.debug)')
    parser.add_argument('--th',
                        type=float,
                        default=0.90,
                        metavar='THRESHOLD',
                        help='Label threshold: good if TFlops >= th * best (default: 0.90)')
    parser.add_argument('--n-estimators',
                        type=int,
                        default=25,
                        help='Number of trees (default: 25)')
    parser.add_argument('--max-depth',
                        type=int,
                        default=10,
                        help='Maximum tree depth (default: 10)')
    parser.add_argument('--min-samples',
                        type=int,
                        default=100,
                        help='Minimum samples to train a classifier (default: 100)')
    parser.add_argument('--update', action='store_true', help='Write QuickTuningClassifier.inc')

    pargs = parser.parse_args(args)

    td = pargs.tuning_dir
    if not td.is_dir():
        parser.error(f"{td} is not a directory")

    ops_and_files = []
    for op in ['gemm', 'conv', 'attention']:
        debug_files = sorted(td.glob(f'*/{op}/*.debug'))
        if debug_files:
            ops_and_files.append((op, [str(f) for f in debug_files]))
            print(f"Found {len(debug_files)} {op} .debug files")
        else:
            print(f"No {op} .debug files found, skipping")

    if not ops_and_files:
        print("No .debug files found in any op subdirectory.")
        return 1

    all_classifiers = []
    for op, files in ops_and_files:
        print(f"\n{'='*60}")
        print(f"Training {op} classifiers")
        print(f"{'='*60}")

        df = load_data(files, no_splitk=False)
        if df.empty:
            print(f"No data for {op}, skipping.")
            continue

        print(f"Loaded {len(df)} rows for {op}")
        classifiers = train_op(df, op, pargs)
        all_classifiers.extend(classifiers)

    print(f"\n{'='*60}")
    print(f"Trained {len(all_classifiers)} classifier(s) total")

    if pargs.update:
        write_inc_file(all_classifiers)
    else:
        print("\nDry run (use --update to write the .inc file)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
