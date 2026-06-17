# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""CLI driver for the tuning evaluation harness.

One command evaluates every proposer against the absolute-best recorded TFlops
under a compile-time budget, using a consistent methodology:

  * the *problem set* is the source of truth -- a tier1-style ``--configs-file``
    expanded through the same tuningRunner path that generated the data. Any
    expected problem missing from the tuning-db is logged (a data gap) and
    excluded from scoring;
  * the *tuning-db* (``--tuning-db``, the ``.debug`` files) supplies the oracle:
    (problem, config) -> best recorded TFlops;
  * per-problem candidate pools come from ``rocmlir-gen --emit-tuning-space``;
  * generalization is measured with ``--split``: grouped k-fold over problem
    shapes (default), or holding out an entire arch / dtype shard.

Outputs ``rows.csv`` (per problem/budget), ``summary.csv`` (pooled),
``summary_by_dtype.csv`` and ``diagnostics.csv`` to -o/--output.

Examples:

    # conv, grouped 3-fold CV
    python -m tuning_eval --configs-file configs/tier1-conv-configs \
        --tuning-db 'tier1-conv-*.tsv.debug' --split kfold -o conv_out

    # gemm, hold out a dtype to test transfer
    python -m tuning_eval --configs-file configs/tier1-gemm-configs \
        --tuning-db gemm.debug --split dtype --test-dtype f16
"""

import argparse
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# Allow `python __main__.py` as well as `python -m tuning_eval`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "tuning_eval"

import pandas as pd

from . import config_specs
from .cli import add_model_args, load_corpus, log, resolve_op
from .corpus import Corpus
from .features import DEFAULT_THRESHOLD
from .metrics import evaluate
from .proposers import (ModelProposer, NearestKnownProposer, RandomProposer, SetCoverProposer)
from .splits import Split, held_out_dtype, kfold_problems

# Build dir default: .../rocMLIR/build, derived from this file's location.
_DEFAULT_BUILD = str(Path(__file__).resolve().parents[5] / "build")

_ALL_PROPOSERS = ("random", "set_cover", "nearest", "model")


def _valid(t) -> bool:
    return t is not None and not math.isnan(t)


def _summarize(rows, by=("proposer", "budget")) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # ``n`` counts non-NaN regrets (problems with a valid reference best), which
    # is the population the mean/percentiles are actually computed over -- a NaN
    # regret (no valid best) is excluded from both.
    g = df.groupby(list(by), as_index=False).agg(regret_mean=("regret", "mean"),
                                                 regret_p95=("regret", lambda s: s.quantile(0.95)),
                                                 regret_p99=("regret", lambda s: s.quantile(0.99)),
                                                 regret_max=("regret", "max"),
                                                 coverage=("coverage", "mean"),
                                                 proposal_coverage=("proposal_coverage", "mean"),
                                                 n=("regret", "count"))
    return g.sort_values(["budget", "regret_mean"]).reset_index(drop=True)


def _restrict_to_configs(corpus: Corpus, op: str, args) -> Corpus:
    """Make the ``--configs-file`` spec the source of truth for the problem set.

    Expands the spec *per corpus arch* (the expansion is arch/topology-aware)
    through the same tuningRunner path that produced the data, logs which
    expected problems are missing from the tuning-db, and returns a corpus
    restricted to the expected problems -- each shard kept only against its own
    arch's expected set (so a shape expected for one arch never keeps/drops
    another arch's problems). Extras not in the spec are dropped."""
    expected = config_specs.expected_by_arch(corpus, op, args.configs_file, log)
    if not any(expected.values()):
        raise SystemExit(f"--configs-file {args.configs_file} expanded to no problems for op={op}")
    restricted = corpus.subset(lambda sig:
                               (sig.dtype, sig.problem_key) in expected.get(sig.arch, set()))
    if not restricted.keys():
        raise SystemExit("no expected problem from --configs-file is present in the tuning-db")
    return restricted


def _iter_splits(corpus: Corpus, args) -> Iterator[Tuple[int, Split]]:
    """Yield ``(fold_index, (train, test_sigs))`` for the chosen split."""
    if args.split == "kfold":
        for fold, split in enumerate(kfold_problems(corpus, k=args.folds, seed=args.seed)):
            yield fold, split
    elif args.split == "dtype":
        if not args.test_dtype:
            raise SystemExit("--test-dtype is required for --split dtype")
        yield 0, held_out_dtype(corpus, args.test_dtype)
    else:
        raise SystemExit(f"unknown split: {args.split}")


def _make_proposers(names, args, pool, train):
    """Instantiate (and fit) the requested proposers for one fold.

    A single SetCoverProposer is fitted once and shared with nearest, so the
    ILP is solved at most once per fold. Returns a list of fitted proposers."""
    shared_sc = None
    if "set_cover" in names or "nearest" in names:
        shared_sc = SetCoverProposer(threshold=args.threshold)
        t = time.time()
        shared_sc.fit(train)
        log(f"  set_cover fit {time.time() - t:.0f}s")

    out = []
    for name in names:
        t = time.time()
        if name == "set_cover":
            out.append((shared_sc, 0.0))  # already fitted above
            continue
        if name == "nearest":
            p = NearestKnownProposer(set_cover=shared_sc)
        elif name == "random":
            p = RandomProposer(seed=args.seed, pool_provider=pool)
        elif name == "model":
            p = ModelProposer(seed=args.seed,
                              threshold=args.threshold,
                              pool_provider=pool,
                              max_train_pairs=args.max_train_pairs,
                              group_subsample=not args.no_group_subsample,
                              balanced_class_weight=not args.no_class_weight,
                              n_estimators=args.n_estimators,
                              max_depth=args.max_depth,
                              learning_rate=args.learning_rate)
        else:
            raise SystemExit(f"unknown proposer: {name}")
        p.fit(train)
        out.append((p, time.time() - t))
    return out


def run_diagnostics(corpus: Corpus, pool, out_dir: str) -> None:
    """Per-problem applicability/optimality coverage of the candidate pool.

    Also warms the emit cache so the splits reuse it."""
    log("diagnostics: enumerating candidate pool per problem ...")
    sigs = corpus.sigs()
    diag = []
    t = time.time()
    for i, sig in enumerate(sigs):
        measured = corpus.measured(sig.table_key, sig.problem_key)
        best = max((v for v in measured.values() if _valid(v)), default=float("nan"))
        pool_cfgs = pool(sig)
        pool_set = set(pool_cfgs)
        valid_in_pool = [c for c in pool_cfgs if _valid(measured.get(c, float("nan")))]
        # Emitted configs split three ways against the data: applicable
        # (measured, valid), inapplicable (measured but NaN -- ran and failed),
        # and missing (no row at all -- never tuned / not in the data).
        nan_in_pool = [c for c in pool_cfgs if c in measured and not _valid(measured[c])]
        missing_in_pool = [c for c in pool_cfgs if c not in measured]
        best_in_pool = max((measured[c] for c in valid_in_pool), default=float("nan"))
        argmax_cfg = max((c for c in measured if _valid(measured.get(c))),
                         key=lambda c: measured[c],
                         default=None)
        npool = len(pool_cfgs)
        diag.append({
            "dtype": sig.dtype,
            "pool_size": npool,
            "oracle_size": len(measured),
            "pool_valid_frac": (len(valid_in_pool) / npool) if npool else float("nan"),
            "pool_nan_frac": (len(nan_in_pool) / npool) if npool else float("nan"),
            "pool_missing_frac": (len(missing_in_pool) / npool) if npool else float("nan"),
            "best_cfg_in_pool": (argmax_cfg in pool_set) if argmax_cfg else False,
            "pool_best_ratio": (best_in_pool / best) if _valid(best) and best > 0 else float("nan"),
        })
        if (i + 1) % 50 == 0:
            log(f"  {i + 1}/{len(sigs)} ({time.time() - t:.0f}s)")
    ddf = pd.DataFrame(diag)
    ddf.to_csv(f"{out_dir}/diagnostics.csv", index=False)
    log("diagnostics summary (per dtype): emitted configs split into "
        "valid / nan(inapplicable) / missing(not in data)")
    print(ddf.groupby("dtype").agg(n=("pool_size", "size"),
                                   pool_size=("pool_size", "mean"),
                                   oracle_size=("oracle_size", "mean"),
                                   pool_valid_frac=("pool_valid_frac", "mean"),
                                   pool_nan_frac=("pool_nan_frac", "mean"),
                                   pool_missing_frac=("pool_missing_frac", "mean"),
                                   best_in_pool=("best_cfg_in_pool", "mean"),
                                   pool_best_ratio=("pool_best_ratio", "mean")).to_string(),
          flush=True)


def _record(all_rows, fold_means, rows, fold, budgets, name):
    """Append one proposer's rows and per-budget mean regret to the accumulators."""
    all_rows.extend({**r, "fold": fold} for r in rows)
    fdf = pd.DataFrame(rows)
    for b in budgets:
        fold_means.setdefault((name, b), []).append(fdf[fdf.budget == b]["regret"].mean())


def run_eval(corpus: Corpus, pool, args, out_dir: str) -> Tuple[List[dict], dict]:
    budgets = sorted({int(b) for b in args.budgets.split(",") if b})
    if not budgets:
        raise SystemExit("no budgets given")
    all_rows: List[dict] = []
    fold_means: dict = {}  # (proposer, budget) -> [per-fold mean regret]

    log(f"split={args.split} ...")
    for fold, (train, test_sigs) in _iter_splits(corpus, args):
        if not test_sigs:
            log(f"fold {fold}: empty test set, skipping")
            continue
        n_train = sum(len(train.problem_keys(k)) for k in train.keys())
        log(f"fold {fold}: train problems={n_train} test={len(test_sigs)}")
        for p, fit_s in _make_proposers(args.proposers, args, pool, train):
            t = time.time()
            rows = evaluate(p, corpus, test_sigs, budgets, args.threshold)
            _record(all_rows, fold_means, rows, fold, budgets, p.name)
            log(f"  {p.name}: fit {fit_s:.0f}s eval {time.time() - t:.0f}s")
        pd.DataFrame(all_rows).to_csv(f"{out_dir}/rows.csv", index=False)
    if not all_rows:
        raise SystemExit("evaluation produced no rows (every split test set was empty)")
    return all_rows, fold_means


def report(all_rows: List[dict], fold_means, out_dir: str) -> None:
    log("pooled summary (mean / p95 / p99 / max):")
    cv = _summarize(all_rows)
    cv["regret_std_folds"] = [
        statistics.pstdev(fold_means.get((r.proposer, r.budget), [0])) for r in cv.itertuples()
    ]
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(cv.to_string(index=False), flush=True)
    cv.to_csv(f"{out_dir}/summary.csv", index=False)

    log("per-dtype summary:")
    cvd = _summarize(all_rows, by=("dtype", "proposer", "budget"))
    cvd.to_csv(f"{out_dir}/summary_by_dtype.csv", index=False)
    for dtype in sorted({r["dtype"] for r in all_rows}):
        print(f"--- dtype={dtype} ---", flush=True)
        with pd.option_context("display.max_rows", None, "display.width", 220):
            print(cvd[cvd.dtype == dtype].drop(columns=["dtype"]).to_string(index=False),
                  flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tuning_eval", description="Tuning evaluation harness")
    parser.add_argument("--configs-file",
                        required=True,
                        metavar="PATH",
                        help="tier1 config file (e.g. configs/tier1-<op>-configs); the "
                        "source of truth for the problem set. Expanded via tuningRunner.")
    parser.add_argument("--tuning-db",
                        nargs="+",
                        required=True,
                        metavar="GLOB",
                        help="tuningRunner .debug files (globs allowed); the measurement oracle.")
    parser.add_argument("--op",
                        default=None,
                        choices=["gemm", "conv", "attention"],
                        help="operation; inferred from the tuning-db if omitted. "
                        "Must match when given.")
    parser.add_argument(
        "-o",
        "--output",
        default=".",
        help="output directory; CSVs are written to a per-arch subdir <output>/<arch>")
    parser.add_argument("--split",
                        required=True,
                        choices=["kfold", "dtype"],
                        help="refit proposers per fold and score held-out problems (within "
                        "each arch). There is no arch split: models are per-arch.")
    parser.add_argument("--folds", type=int, default=3, help="number of folds for --split kfold")
    parser.add_argument("--test-dtype", default=None, help="held-out dtype for --split dtype")
    parser.add_argument("--proposers",
                        nargs="+",
                        default=list(_ALL_PROPOSERS),
                        choices=list(_ALL_PROPOSERS))
    parser.add_argument("--budgets", default="1,2,4,8,16,30", help="comma-separated budgets")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-splitk", action="store_true")
    # Candidate pool: always the per-problem applicable tuning space from
    # rocmlir-gen --emit-tuning-space (the source of truth).
    parser.add_argument("--emit-tuning-space-kind",
                        default="exhaustive",
                        choices=["quick", "full", "exhaustive"])
    parser.add_argument("--mlir-build-dir",
                        default=_DEFAULT_BUILD,
                        help=f"rocmlir-gen build dir (default: {_DEFAULT_BUILD})")
    add_model_args(parser, max_train_pairs_default=300000)
    parser.add_argument("--no-diagnostics", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(args)
    op = resolve_op(corpus, args.op)

    # The configs file is the source of truth for which problems to evaluate.
    corpus = _restrict_to_configs(corpus, op, args)

    # Features come from rocmlir-gen --emit-features (the single source of truth
    # the deployed scorer also uses); point it at the same build as the pool.
    from . import features
    features.configure_extractor(args.mlir_build_dir)

    # Candidate pool is always the per-problem applicable tuning space.
    from .tuning_space import EmitTuningSpacePool
    pool = EmitTuningSpacePool(mlir_build_dir=args.mlir_build_dir, kind=args.emit_tuning_space_kind)

    # Process each arch independently (one model / cover scope per arch),
    # writing per-arch CSVs to <output>/<arch> -- equivalent to invoking the
    # tool once per arch, like quickTuningGen's per-(arch, op) grouping.
    for arch, arch_corpus in corpus.by_arch():
        out_dir = str(Path(args.output) / arch)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        log(f"=== arch {arch} -> {out_dir} ===")
        if not args.no_diagnostics:
            run_diagnostics(arch_corpus, pool, out_dir)
        all_rows, fold_means = run_eval(arch_corpus, pool, args, out_dir)
        report(all_rows, fold_means, out_dir)

    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
