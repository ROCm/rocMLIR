# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Train/test splits for measuring generalization.

All splits are deterministic. ``kfold_problems`` partitions by a hash of the
problem key (never by row, to avoid leaking a problem's other configs into
train) and is the in-distribution evaluator (unseen shapes, same arch/dtype).
``held_out_dtype`` holds out an entire dtype to test whether an arch's model
generalizes to an unseen precision (it serves all of that arch's dtypes). There
is no held-out-*arch* split: models are per-arch, so validating one against a
different arch would be meaningless.
"""

from typing import Iterator, List, Tuple

from .corpus import Corpus, ProblemSig

# quickTuningGen is put on sys.path by the package __init__.
import quickTuningGen

Split = Tuple[Corpus, List[ProblemSig]]


def held_out_dtype(corpus: Corpus, test_dtype: str) -> Split:
    """Train on every other dtype; test on the held-out dtype."""
    train = corpus.subset(lambda sig: sig.dtype != test_dtype)
    test_sigs = [sig for sig in corpus.sigs() if sig.dtype == test_dtype]
    return train, test_sigs


def _fold_of(problem_key: str, k: int, seed: int) -> int:
    """Deterministic fold index for a problem key."""
    return quickTuningGen.hash_problem_key(f"{problem_key}|{seed}") % k


def kfold_problems(corpus: Corpus, k: int = 3, seed: int = 0) -> Iterator[Split]:
    """Yield ``k`` (train, test) splits, partitioning problems by a hash of the
    (shape-only) problem key. Grouping by problem key keeps every dtype of a
    given shape in the same fold, so a shape never straddles the boundary."""
    for fold in range(k):
        train = corpus.subset(lambda sig, f=fold: _fold_of(sig.problem_key, k, seed) != f)
        test_sigs = [sig for sig in corpus.sigs() if _fold_of(sig.problem_key, k, seed) == fold]
        yield train, test_sigs
