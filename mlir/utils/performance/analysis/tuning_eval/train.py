# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Deployment path: fit the model on all provided data and emit its tree table.

Where ``python -m tuning_eval`` *splits* the data to measure proposers, this
fits one ModelProposer per arch on the entire tuning-db (no held-out test set)
and emits each model's decision trees as an embeddable ``.inc`` (the deployment
artifact consumed by ``SmartTuningDb.cpp``; see ``export.py``). Training is
always from scratch on all the given data. Features are computed by
``rocmlir-gen --emit-features`` (the same C++ extractor the deployed scorer
uses), so a built rocmlir-gen is required; no GPU is needed.

Examples:

    # train from scratch and emit the .inc table to the default model dir
    python -m tuning_eval.train --tuning-db 'gemm-*.debug'

    # custom location
    python -m tuning_eval.train --tuning-db 'gemm-*.debug' -o model_gemm
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Allow `python train.py` as well as `python -m tuning_eval.train`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "tuning_eval"

from . import export
from .cli import add_model_args, load_corpus, log, resolve_op
from .features import DEFAULT_THRESHOLD
from .proposers import ModelProposer

# Default model directory, relative to the repo root. Mirrors quickTuningGen.py's
# QuickTuningDb default so the deployable artifacts live alongside the other Rock
# tuning databases.
DEFAULT_OUTPUT_REL_PATH = Path("mlir") / "lib" / "Dialect" / "Rock" / "Tuning" / "Models"


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[5] / DEFAULT_OUTPUT_REL_PATH


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tuning_eval.train",
                                     description="Fit a model and compile it to C")
    parser.add_argument("--tuning-db",
                        nargs="+",
                        required=True,
                        metavar="GLOB",
                        help="tuningRunner .debug files (globs allowed); the training data.")
    parser.add_argument("-o",
                        "--output",
                        type=Path,
                        default=None,
                        metavar="DIR",
                        help="model directory for the emitted C (default: <repo>/%s)" %
                        DEFAULT_OUTPUT_REL_PATH.as_posix())
    parser.add_argument("--op", default=None, choices=["gemm", "conv", "attention"])
    parser.add_argument("--no-splitk", action="store_true")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlir-build-dir",
                        default=None,
                        help="rocmlir-gen build dir for --emit-features (default: auto-discover)")
    add_model_args(parser, max_train_pairs_default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    from . import features
    features.configure_extractor(args.mlir_build_dir)

    corpus = load_corpus(args)
    op = resolve_op(corpus, args.op)
    output = args.output or _default_output_dir()

    # Group by arch and fit one model per arch, independently -- the corpus is
    # never mixed across archs (mirrors quickTuningGen's per-(arch, op) loop).
    arch_corpora = corpus.by_arch()
    log(f"op={op}: fitting one model per arch for {[a for a, _ in arch_corpora]}")
    any_fit = False
    for arch, arch_corpus in arch_corpora:
        model = ModelProposer(threshold=args.threshold,
                              seed=args.seed,
                              n_estimators=args.n_estimators,
                              max_depth=args.max_depth,
                              learning_rate=args.learning_rate,
                              group_subsample=not args.no_group_subsample,
                              balanced_class_weight=not args.no_class_weight,
                              max_train_pairs=args.max_train_pairs)
        n_problems = sum(len(arch_corpus.problem_keys(k)) for k in arch_corpus.keys())
        log(f"[{arch}] fitting on {n_problems} problems ...")
        t = time.time()
        model.fit(arch_corpus)
        if not model.is_fitted():
            log(f"[{arch}] skipped: no classifier (data has only one label class)")
            continue
        written = export.export_model(model, output, arch, op)
        log(f"[{arch}] fit in {time.time() - t:.0f}s; exported to {output}:")
        for p in written:
            log(f"  {p.name}")
        any_fit = True

    if not any_fit:
        raise SystemExit("training produced no models (no arch had two label classes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
