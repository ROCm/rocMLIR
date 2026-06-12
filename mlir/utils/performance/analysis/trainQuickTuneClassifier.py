#!/usr/bin/env python3
"""Quick-Tune Classifier Training Script

Trains XGBoost regressors to predict the performance ratio of quick-tune-list
perfconfigs for a given problem size.  Produces one .ubj model file per
(arch, op, dtype) combination.

Training data is filtered to only include perfconfigs present in the
quick-tune list (obtained via ``rocmlir-gen --emit-tuning-space=quick``).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import perfRunner  # noqa: E402

from quickTuningGen import (  # noqa: E402
    get_target_columns, load_data, parse_perfconfig,
)

DIRECTION_MAP = {'fwd': 0, 'bwd': 1, 'wrw': 2}

# ---------------------------------------------------------------------------
# Quick-tune list via rocmlir-gen
# ---------------------------------------------------------------------------


def get_quick_tune_configs(rocmlir_gen, arch, op, dtype):
    """Run ``rocmlir-gen --emit-tuning-space=quick`` and return perfconfigs."""
    cmd = [
        rocmlir_gen,
        '-p',
        '--arch',
        arch,
        f'--operation={op}',
        '-t',
        dtype,
        '--emit-tuning-space=quick',
    ]
    env = {**os.environ, 'ROCMLIR_QUICK_TUNE_TOP_N': '0'}
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, env=env)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f'    rocmlir-gen failed: {e}', file=sys.stderr)
        return None
    if result.returncode != 0:
        return None
    configs = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return configs if configs else None


# ---------------------------------------------------------------------------
# Feature extraction: raw problem dims + raw perfconfig params.
# Each op type uses its own problem features; perfconfig params are always
# the raw numeric values from the perfconfig string.  Must match the C++
# feature builders in QuickTuningClassifier.cpp exactly.
# ---------------------------------------------------------------------------


def _perfconfig_params(perfconfig):
    """Extract raw numeric params from a perfconfig string."""
    _, _, params = parse_perfconfig(perfconfig)
    return [int(x) for x in params]


def _conv_problem_features(df):
    """Raw conv dimensions as feature columns (14 features)."""
    direction = np.array([DIRECTION_MAP.get(d, 0)
                          for d in df['Direction'].values],
                         dtype=np.float32)
    return np.column_stack([
        df['N'].values, df['C'].values, df['H'].values, df['W'].values,
        df['K'].values, df['Y'].values, df['X'].values,
        df['PaddingH'].values, df['PaddingW'].values,
        df['StrideH'].values, df['StrideW'].values,
        df['DilationH'].values, df['DilationW'].values,
        direction,
    ]).astype(np.float32)


def _gemm_problem_features(df):
    """Raw GEMM dimensions as feature columns (4 features)."""
    return np.column_stack([
        df['G'].values, df['M'].values, df['N'].values, df['K'].values,
    ]).astype(np.float32)


def _attention_problem_features(df):
    """Raw attention dimensions as feature columns (7 features)."""
    return np.column_stack([
        df['G'].values, df['SeqLenQ'].values, df['SeqLenK'].values,
        df['NumHeadsQ'].values, df['NumHeadsKV'].values,
        df['HeadDimQK'].values, df['HeadDimV'].values,
    ]).astype(np.float32)


def build_feature_matrix(df, op):
    """Raw problem dims + raw perfconfig params."""
    pc_raw = np.array([_perfconfig_params(pc) for pc in df['PerfConfig']],
                      dtype=np.float32)
    if op == 'conv':
        problem = _conv_problem_features(df)
    elif op == 'attention':
        problem = _attention_problem_features(df)
    else:
        problem = _gemm_problem_features(df)
    feats = np.column_stack([problem, pc_raw]).astype(np.float32)
    return feats, feats.shape[1]


# ---------------------------------------------------------------------------
# Coverage evaluation
# ---------------------------------------------------------------------------


def _topn_ratios(model, df, problem_cols, op, top_n):
    """For each problem, pick top-N by predicted score and return the best
    actual TFlops ratio found per problem."""
    features, _ = build_feature_matrix(df, op)
    pred = model.predict(features)

    df = df.copy()
    df['_pred'] = pred

    ratios = []
    for _, grp in df.groupby(problem_cols):
        top = grp.nlargest(min(top_n, len(grp)), '_pred')
        ratios.append(top['ratio'].max())
    return np.array(ratios)


def evaluate_coverage(model, df, problem_cols, op, top_n):
    """Returns (mean_ratio, min_ratio) on the given data."""
    ratios = _topn_ratios(model, df, problem_cols, op, top_n)
    return float(ratios.mean()), float(ratios.min())


def cross_validate(df, problem_cols, op, top_n, n_folds=5, model_params=None):
    """K-fold cross-validation split by problem (not by sample).

    Trains on (k-1)/k of the problems, evaluates on the held-out 1/k.
    Returns per-problem coverage ratios for all held-out problems.
    """
    df = df.copy()
    df['_problem_id'] = df.groupby(problem_cols).ngroup()
    unique_problems = np.array(sorted(df['_problem_id'].unique()))

    rng = np.random.RandomState(42)
    shuffled = rng.permutation(unique_problems)
    folds = np.array_split(shuffled, n_folds)

    all_ratios = []

    for fold_idx, test_problem_ids in enumerate(folds):
        test_set = set(test_problem_ids)
        train_mask = ~df['_problem_id'].isin(test_set)
        test_mask = df['_problem_id'].isin(test_set)

        df_train = df[train_mask].reset_index(drop=True)
        df_test = df[test_mask].reset_index(drop=True)

        if df_train.empty or df_test.empty:
            continue

        X_train, _ = build_feature_matrix(df_train, op)
        y_train = df_train['ratio'].values.astype(np.float32)

        fold_model = XGBRegressor(**(model_params or {}))
        fold_model.fit(X_train, y_train)

        fold_ratios = _topn_ratios(fold_model, df_test, problem_cols, op, top_n)
        n_test = len(test_set)
        print(f'      fold {fold_idx+1}/{n_folds}: {n_test} problems, '
              f'mean={float(fold_ratios.mean()):.4f} min={float(fold_ratios.min()):.4f}',
              flush=True)
        all_ratios.extend(fold_ratios.tolist())

    return np.array(all_ratios)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_models(df, op, top_n, output_dir, rocmlir_gen):
    problem_cols = get_target_columns(op)

    summary = []

    for arch in sorted(df['Chip'].unique()):
        df_arch = df[df['Chip'] == arch]

        for dtype in sorted(df_arch['DataType'].unique()):
            key = f'{arch}_{op}_{dtype}'

            print(f'\n  {key}: fetching quick-tune list ...')
            qt_list = get_quick_tune_configs(rocmlir_gen, arch, op, dtype)
            if qt_list is None:
                print('    no quick-tune list available, skipping')
                continue

            qt_set = set(qt_list)
            mask = (df_arch['DataType'] == dtype) & df_arch['PerfConfig'].isin(qt_set)
            df_sub = df_arch[mask].copy()

            if df_sub.empty:
                print('    no matching rows in data, skipping')
                continue

            n_pc_params = len(_perfconfig_params(df_sub['PerfConfig'].iloc[0]))
            bad_rows = df_sub['PerfConfig'].apply(
                lambda pc, n=n_pc_params: len(_perfconfig_params(pc)) != n)
            if bad_rows.any():
                print(f'    dropping {bad_rows.sum()} rows with inconsistent '
                      f'perfconfig param counts')
                df_sub = df_sub[~bad_rows]

            df_best = df_sub.groupby(problem_cols + ['PerfConfig'], as_index=False)['TFlops'].max()
            df_best = df_best[df_best['TFlops'].notna() & (df_best['TFlops'] > 0)]

            max_tf = df_best.groupby(
                problem_cols, as_index=False)['TFlops'].max().rename(columns={'TFlops': '_max_tf'})
            df_labeled = df_best.merge(max_tf, on=problem_cols)
            df_labeled['ratio'] = df_labeled['TFlops'] / df_labeled['_max_tf']

            y = df_labeled['ratio'].values.astype(np.float32)

            features, n_feats = build_feature_matrix(df_labeled, op)

            print(f'    samples: {len(y)}', flush=True)
            print(f'    features: {n_feats} (raw problem + perfconfig)', flush=True)
            n_problems = len(max_tf)
            configs_per_problem = len(y) // n_problems if n_problems else 0
            print(f'    quick-tune configs: {len(qt_set)}, '
                  f'unique problems: {n_problems}, '
                  f'~{configs_per_problem} configs/problem',
                  flush=True)

            model_params = dict(
                objective='reg:squarederror',
                n_estimators=1000,
                max_depth=12,
                learning_rate=0.1,
                subsample=1.0,
                colsample_bytree=1.0,
                n_jobs=min(os.cpu_count() or 1, 16),
                random_state=42,
                verbosity=0,
            )
            model = XGBRegressor(**model_params)

            print('    training (reg:squarederror)...', flush=True)
            model.fit(features, y)

            mean_r, min_r = evaluate_coverage(model, df_labeled, problem_cols, op, top_n)
            print(f'    train top-{top_n}: mean={mean_r:.4f}  min={min_r:.4f}', flush=True)

            print(f'    cross-validating (5-fold by problem)...', flush=True)
            cv_ratios = cross_validate(df_labeled, problem_cols, op, top_n,
                                       n_folds=5, model_params=model_params)
            cv_mean = float(cv_ratios.mean())
            cv_min = float(cv_ratios.min())
            print(f'    CV top-{top_n}:    mean={cv_mean:.4f}  min={cv_min:.4f}', flush=True)

            model_path = Path(output_dir) / f'{key}.ubj'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(str(model_path))
            print(f'    saved: {model_path}')

            summary.append({
                'key': key,
                'samples': len(y),
                'mean_ratio': mean_r,
                'min_ratio': min_r,
                'cv_mean': cv_mean,
                'cv_min': cv_min,
            })

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(args=None):
    parser = argparse.ArgumentParser(
        prog='trainQuickTuneClassifier.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Train XGBoost classifiers for quick-tune perfconfig ranking.',
        epilog='''
Examples:
    %(prog)s --op gemm \\
        /data/gfx942/gemm-exhaustive.tsv.debug --output-dir models/
    %(prog)s --op conv \\
        /data/gfx942/conv-tier1.tsv.debug --output-dir models/
    %(prog)s --op attention \\
        /data/gfx942/attn-exhaustive.tsv.debug --output-dir models/
''')

    parser.add_argument('files',
                        nargs='+',
                        metavar='FILE',
                        help='.debug TSV files produced by tuningRunner.py')
    parser.add_argument('--op',
                        required=True,
                        choices=['gemm', 'conv', 'attention'],
                        help='Operation type')
    parser.add_argument('--top-n',
                        type=int,
                        default=30,
                        help='Number of top candidates for coverage evaluation (default: 30)')
    parser.add_argument('--output-dir',
                        default='models/',
                        help='Output directory for .ubj model files (default: models/)')
    parser.add_argument('--mlir-build-dir',
                        default=perfRunner.find_mlir_build_dir(),
                        metavar='DIR',
                        help='MLIR build directory (auto-detected if omitted)')

    pargs = parser.parse_args(args)

    paths = perfRunner.create_paths(None, pargs.mlir_build_dir)
    if paths.mlir_paths is None:
        parser.error('Cannot find rocmlir-gen. '
                     'Use --mlir-build-dir to specify the build directory.')
    rocmlir_gen = paths.mlir_paths.rocmlir_gen_path

    needed = set(get_target_columns(
        pargs.op)) | {'Chip', 'DataType', 'PerfConfig', 'TFlops'}
    print(f'Loading {len(pargs.files)} file(s)...', flush=True)
    df = load_data(pargs.files, no_splitk=False, usecols=needed)
    if df.empty:
        print('ERROR: no data loaded', file=sys.stderr)
        return 1

    print(f'Loaded {len(df)} rows', flush=True)
    print(f'Architectures: {sorted(df["Chip"].unique())}')
    print(f'Data types:    {sorted(df["DataType"].unique())}')

    results = train_models(df, pargs.op, pargs.top_n, pargs.output_dir, rocmlir_gen)

    if results:
        print(f'\n{"=" * 60}')
        print(f'Trained {len(results)} model(s):')
        for r in results:
            print(f'  {r["key"]:30s}  train: mean={r["mean_ratio"]:.4f} min={r["min_ratio"]:.4f}'
                  f'  |  CV: mean={r["cv_mean"]:.4f} min={r["cv_min"]:.4f}')
    else:
        print('\nNo models trained.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
