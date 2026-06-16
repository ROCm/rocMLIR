# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Helpers shared by the train (``train.py``) and eval (``__main__.py``) CLIs.

Kept here so the deployment path does not have to import the evaluation driver
just to reuse corpus loading and the model hyperparameter flags.
"""

import argparse
import glob
import time
from typing import List, Optional

from .corpus import Corpus


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def expand_globs(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pat in patterns:
        matched = glob.glob(pat)
        files.extend(matched if matched else [pat])
    return sorted(set(files))


def load_corpus(args) -> Corpus:
    """Expand the --tuning-db globs and load the .debug files into the oracle."""
    files = expand_globs(args.tuning_db)
    if not files:
        raise SystemExit("no tuning-db files matched")
    log(f"loading tuning-db from {len(files)} file(s) ...")
    t = time.time()
    corpus = Corpus.from_debug_files(files, no_splitk=args.no_splitk)
    if not corpus.keys():
        raise SystemExit("tuning-db is empty after loading")
    log(f"loaded in {time.time() - t:.0f}s")
    for k in corpus.keys():
        log(f"  key {k} problems {len(corpus.problem_keys(k))}")
    return corpus


def resolve_op(corpus: Corpus, requested_op: Optional[str]) -> str:
    """Pick the op to work on: inferred from the corpus, validated against --op."""
    corpus_ops = sorted({k[1] for k in corpus.keys()})
    if requested_op is not None and requested_op not in corpus_ops:
        raise SystemExit(f"--op={requested_op} not found in tuning-db (has {corpus_ops})")
    if len(corpus_ops) > 1 and requested_op is None:
        raise SystemExit(f"tuning-db mixes ops {corpus_ops}; pass --op to pick one")
    return requested_op or corpus_ops[0]


def add_model_args(parser: argparse.ArgumentParser, max_train_pairs_default: Optional[int]) -> None:
    """Add the LightGBM hyperparameter flags shared by train and eval."""
    parser.add_argument("--n-estimators", type=int, default=100, help="number of boosting trees")
    parser.add_argument("--max-depth", type=int, default=None, help="max tree depth (None = -1)")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="boosting learning rate")
    parser.add_argument("--no-group-subsample",
                        action="store_true",
                        help="disable group-balanced subsampling of training rows")
    parser.add_argument("--no-class-weight",
                        action="store_true",
                        help="disable balanced class weights in the classifiers")
    parser.add_argument("--max-train-pairs",
                        type=int,
                        default=max_train_pairs_default,
                        help="cap (problem, config) training rows (default: use all)")
